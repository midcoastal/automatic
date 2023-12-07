import os
import html
import csv
import time
from collections import namedtuple
import torch
from tqdm import tqdm
import safetensors.torch
import numpy as np
from PIL import Image, PngImagePlugin
from torch.utils.tensorboard import SummaryWriter
from modules import shared, devices, sd_hijack, processing, sd_models, images, sd_samplers, sd_hijack_checkpoint, errors
import modules.textual_inversion.dataset
from modules.textual_inversion.learn_schedule import LearnRateScheduler
from modules.textual_inversion.image_embedding import embedding_to_b64, embedding_from_b64, insert_image_data_embed, extract_image_data_embed, caption_image_overlay
from modules.textual_inversion.ti_logging import save_settings_to_file
from modules.modelloader import directory_files, extension_filter, directory_mtime
from typing import List, Optional, Union
import modules.textual_inversion.loaders

TokenToAdd = namedtuple("TokenToAdd", ["clip_l", "clip_g"])

TextualInversionTemplate = namedtuple("TextualInversionTemplate", ["name", "path"])
textual_inversion_templates = {}


def list_textual_inversion_templates():
    textual_inversion_templates.clear()
    for root, _dirs, fns in os.walk(shared.opts.embeddings_templates_dir):
        for fn in fns:
            path = os.path.join(root, fn)
            textual_inversion_templates[fn] = TextualInversionTemplate(fn, path)
    return textual_inversion_templates


def list_embeddings(*dirs):
    is_ext = extension_filter(['.SAFETENSORS', '.PT' ] + ( ['.PNG', '.WEBP', '.JXL', '.AVIF', '.BIN' ] if shared.backend != shared.Backend.DIFFUSERS else [] ))
    is_not_preview = lambda fp: not next(iter(os.path.splitext(fp))).upper().endswith('.PREVIEW') # pylint: disable=unnecessary-lambda-assignment
    return list(filter(lambda fp: is_ext(fp) and is_not_preview(fp) and os.stat(fp).st_size > 0, directory_files(*dirs)))


class Embedding:
    def __init__(self, vec, name, filename=None, step=None):
        self.vec = vec
        self.name = name
        self.tag = name
        self.step = step
        self.filename = filename
        self.basename = os.path.relpath(filename, shared.opts.embeddings_dir) if filename is not None else None
        self.shape = None
        self.vectors = 0
        self.cached_checksum = None
        self.sd_checkpoint = None
        self.sd_checkpoint_name = None
        self.optimizer_state_dict = None

    def save(self, filename):
        embedding_data = {
            "string_to_token": {"*": 265},
            "string_to_param": {"*": self.vec},
            "name": self.name,
            "step": self.step,
            "sd_checkpoint": self.sd_checkpoint,
            "sd_checkpoint_name": self.sd_checkpoint_name,
        }
        torch.save(embedding_data, filename)
        if shared.opts.save_optimizer_state and self.optimizer_state_dict is not None:
            optimizer_saved_dict = {
                'hash': self.checksum(),
                'optimizer_state_dict': self.optimizer_state_dict,
            }
            torch.save(optimizer_saved_dict, f"{filename}.optim")

    def checksum(self):
        if self.cached_checksum is not None:
            return self.cached_checksum
        def const_hash(a):
            r = 0
            for v in a:
                r = (r * 281 ^ int(v) * 997) & 0xFFFFFFFF
            return r
        self.cached_checksum = f'{const_hash(self.vec.reshape(-1) * 100) & 0xffff:04x}'
        return self.cached_checksum


class DirWithTextualInversionEmbeddings:
    def __init__(self, path):
        self.path = path
        self.mtime = None

    def has_changed(self):
        if not os.path.isdir(self.path):
            return False
        return directory_mtime(self.path) != self.mtime

    def update(self):
        if not os.path.isdir(self.path):
            return
        self.mtime = directory_mtime(self.path)


class EmbeddingDatabase:
    def __init__(self):
        self.ids_lookup = {}
        self.word_embeddings = {}
        self.skipped_embeddings = {}
        self.expected_shape = -1
        self.embedding_dirs = {}
        self.previously_displayed_embeddings = ()
        self.embeddings_used = []

    def add_embedding_dir(self, path):
        self.embedding_dirs[path] = DirWithTextualInversionEmbeddings(path)

    def clear_embedding_dirs(self):
        self.embedding_dirs.clear()

    def register_embedding(self, embedding, model):
        self.word_embeddings[embedding.name] = embedding
        if hasattr(model, 'cond_stage_model'):
            ids = model.cond_stage_model.tokenize([embedding.name])[0]
        elif hasattr(model, 'tokenizer'):
            ids = model.tokenizer.convert_tokens_to_ids(embedding.name)
        if type(ids) != list:
            ids = [ids]
        first_id = ids[0]
        if first_id not in self.ids_lookup:
            self.ids_lookup[first_id] = []
        self.ids_lookup[first_id] = sorted(self.ids_lookup[first_id] + [(ids, embedding)], key=lambda x: len(x[0]), reverse=True)
        return embedding

    def get_expected_shape(self):
        if shared.backend == shared.Backend.DIFFUSERS:
            return 0
        if shared.sd_model is None:
            shared.log.error('Model not loaded')
            return 0
        vec = shared.sd_model.cond_stage_model.encode_embedding_init_text(",", 1)
        return vec.shape[1]
    
    def load_diffusers_embedding(
        self, 
        filename: Union[str, List[str]],
        path: Optional[Union[str, List[str]]] = None,
    ):
        _loaded_pre = len(self.word_embeddings)
        embeddings_to_load = []
        loaded_embeddings = {}
        skipped_embeddings = []
        if shared.sd_model is not None:
            pipe = shared.sd_model
            tokenizer   = getattr(pipe, 'tokenizer',   None)
            tokenizer_2 = getattr(pipe, 'tokenizer_2', None)
            clip_l = getattr(pipe, 'text_encoder',   None) # clip_l
            clip_g = getattr(pipe, 'text_encoder_2', None) # clip_g
            filenames = (
                [filename]
                if not isinstance(filename, list)
                else filename
            )
            exts = [".SAFETENSORS", ".PT"]
            filename_paths = zip(filenames, len(filenames) * [path] if (isinstance(path, str) or path is None) else path)
            model_type = None
            if clip_l and tokenizer:
                if clip_g is None and tokenizer_2 is None:
                    model_type = 'SD'
                elif clip_g and tokenizer_2:
                    model_type = 'SD-XL'
                else:
                    model_type = 'UNDEFINED'
                try:
                    unk_token_id = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
                    for filename, path in filename_paths:
                        if path is None:
                            path = filename
                            filename = os.path.basename(path)
                        fn, ext = os.path.splitext(filename)
                        name = os.path.basename(fn)
                        embedding = Embedding(vec=None, name=name, filename=path)
                        try:
                            ext  = ext.upper()
                            _, _ext = os.path.splitext(path)
                            _ext = _ext.upper()
                            if ext != _ext:
                                raise ValueError(f'filename and path extensions do not match: `{ext}` != `{_ext}`')
                            if ext not in exts:
                                raise ValueError(f'extension `{ext}` is invalid, expected one of: {exts}')
                            if name in tokenizer.get_vocab() or f"{name}_1" in tokenizer.get_vocab():
                                raise ValueError(f'token already exists in the tokenizer vocabulary: `{name}`')
                            embeddings_to_load.append(embedding)
                        except Exception as e:
                            skipped_embeddings.append(embedding)
                            continue
                        embeddings_to_load = sorted(embeddings_to_load, key=lambda e: exts.index(os.path.splitext(e.filename)[1].upper()))

                    if model_type == 'SD':
                        loaded_filenames = pipe.load_textual_inversion(
                            [embedding.filename for embedding in embeddings_to_load],
                            token=[embedding.name for embedding in embeddings_to_load],
                            tokenizer=tokenizer,
                            text_encoder=clip_l
                        )
                        _len = len(embeddings_to_load)
                        for embedding in embeddings_to_load.copy():
                            if embedding.filename in loaded_filenames:
                                loaded_embeddings[embedding.name] = embedding
                                embeddings_to_load.remove(embedding)
                    tokens_to_add = {}
                    tokenizer_vocab = tokenizer.get_vocab()
                    for embedding in embeddings_to_load:
                        try:
                            name = embedding.name
                            if name in tokenizer_vocab:
                                raise Exception(f'token `{name}` already in Model Vocabulary')
                            if name in tokens_to_add or name in loaded_embeddings:
                                raise Exception('duplicate Embedding Token')
                            embeddings_dict = {}
                            _, ext = os.path.splitext(embedding.filename)
                            ext = ext.upper()
                            if ext in ['.SAFETENSORS']:
                                with safetensors.torch.safe_open(embedding.filename, framework="pt") as f: # type: ignore
                                    for k in f.keys():
                                        embeddings_dict[k] = f.get_tensor(k)
                                """
                                # The following note has been here a while (as of 11/05/23), go or no-go?
                                # alternatively could disable load_textual_inversion and load everything here
                                elif ext.lower() in ['.PT', '.BIN']:
                                    data = torch.load(path, map_location="cpu")
                                    embedding.tag = data.get('name', None)
                                    embedding.step = data.get('step', None)
                                    embedding.sd_checkpoint = data.get('sd_checkpoint', None)
                                    embedding.sd_checkpoint_name = data.get('sd_checkpoint_name', None)
                                    param_dict = data.get('string_to_param', None)
                                    embeddings_dict['clip_l'] = []
                                    for tokens in param_dict.values():
                                        for vec in tokens:
                                            embeddings_dict['clip_l'].append(vec)
                                """
                            else:
                                raise Exception(f'extension {ext} not supported')
                                continue
                            if 'clip_l' not in embeddings_dict:
                                raise ValueError(f'Invalid Embedding, dict missing required key `clip_l`')
                            if 'clip_g' in embeddings_dict:
                                embedding_type = 'SD-XL'
                            else:
                                embedding_type = 'SD'
                            if embedding_type != model_type:
                                raise ValueError(f'Unable to load `{embedding_type}` Embedding into `{model_type}` Model')
                            did_add = False
                            _tokens_to_add = {}
                            for i in range(len(embeddings_dict["clip_l"])):
                                if (len(clip_l.get_input_embeddings().weight.data[0]) == len(embeddings_dict["clip_l"][i])):
                                    token = name if i == 0 else f"{name}_{i}"
                                    if token in tokenizer_vocab:
                                        raise RuntimeError(f'Multi-Vector Embedding would add pre-existing Token in Vocabulary: {token}')
                                    if token in tokens_to_add:
                                        raise RuntimeError(f'Multi-Vector Embedding would add duplicate Token to Add: {token}')
                                    _tokens_to_add[token] = TokenToAdd(
                                        embeddings_dict["clip_l"][i],
                                        embeddings_dict["clip_g"][i] if 'clip_g' in embeddings_dict else None
                                    )
                            if not _tokens_to_add:
                                raise ValueError('no valid tokens to add')
                            tokens_to_add.update(_tokens_to_add)
                            loaded_embeddings[name] = embedding
                        except Exception as e:
                            continue
                    if len(tokens_to_add) > 0:
                        _tokenizer_len = len(tokenizer)
                        num_added = tokenizer.add_tokens([k for k in tokens_to_add.keys()])
                        clip_l.resize_token_embeddings(len(tokenizer))
                        if model_type == 'SD-XL':
                            tokenizer_2.add_tokens([k for k in tokens_to_add.keys()]) # type: ignore
                            clip_g.resize_token_embeddings(len(tokenizer)) # type: ignore
                        for token, data in tokens_to_add.items():
                            token_id = tokenizer.convert_tokens_to_ids(token)
                            if token_id > unk_token_id:
                                clip_l.get_input_embeddings().weight.data[token_id] = data.clip_l
                                if model_type == 'SD-XL':
                                    clip_g.get_input_embeddings().weight.data[token_id] = data.clip_g # type: ignore
                except Exception as e:
                    errors.display(e, f'Embedding Load Failure')
        for name, embedding in loaded_embeddings.items():
            if not embedding:
                continue
            self.register_embedding(embedding, shared.sd_model)
            if embedding in embeddings_to_load:
                embeddings_to_load.remove(embedding)
        skipped_embeddings.extend(embeddings_to_load)
        for embedding in skipped_embeddings:
            if loaded_embeddings.get(embedding.name, None) == embedding:
                continue
            self.skipped_embeddings[embedding.name] = embedding
        return len(self.word_embeddings) - _loaded_pre
        
    def load_from_file(self, path, filename):
        name, ext = os.path.splitext(filename)
        ext = ext.upper()
        if shared.backend == shared.Backend.DIFFUSERS:
            self.load_diffusers_embedding(filename, path)
            return

        if ext in ['.PNG', '.WEBP', '.JXL', '.AVIF']:
            if '.preview' in filename.lower():
                return
            embed_image = Image.open(path)
            if hasattr(embed_image, 'text') and 'sd-ti-embedding' in embed_image.text:
                data = embedding_from_b64(embed_image.text['sd-ti-embedding'])
            else:
                data = extract_image_data_embed(embed_image)
                if not data: # if data is None, means this is not an embeding, just a preview image
                    return
        elif ext in ['.BIN', '.PT']:
            data = torch.load(path, map_location="cpu")
        elif ext in ['.SAFETENSORS']:
            data = safetensors.torch.load_file(path, device="cpu")
        else:
            return

        # textual inversion embeddings
        if 'string_to_param' in data:
            param_dict = data['string_to_param']
            param_dict = getattr(param_dict, '_parameters', param_dict)  # fix for torch 1.12.1 loading saved file from torch 1.11
            assert len(param_dict) == 1, 'embedding file has multiple terms in it'
            emb = next(iter(param_dict.items()))[1]
        # diffuser concepts
        elif type(data) == dict and type(next(iter(data.values()))) == torch.Tensor:
            if len(data.keys()) != 1:
                self.skipped_embeddings[name] = Embedding(None, name=name, filename=path)
                return
            emb = next(iter(data.values()))
            if len(emb.shape) == 1:
                emb = emb.unsqueeze(0)
        else:
            raise RuntimeError(f"Couldn't identify {filename} as textual inversion embedding")

        vec = emb.detach().to(devices.device, dtype=torch.float32)
        # name = data.get('name', name)
        embedding = Embedding(vec=vec, name=name, filename=path)
        embedding.tag = data.get('name', None)
        embedding.step = data.get('step', None)
        embedding.sd_checkpoint = data.get('sd_checkpoint', None)
        embedding.sd_checkpoint_name = data.get('sd_checkpoint_name', None)
        embedding.vectors = vec.shape[0]
        embedding.shape = vec.shape[-1]
        if self.expected_shape == -1 or self.expected_shape == embedding.shape:
            self.register_embedding(embedding, shared.sd_model)
        else:
            self.skipped_embeddings[name] = embedding

    def load_from_dir(self, embdir):
        if sd_models.model_data.sd_model is None:
            shared.log.info('Skipping embeddings load: model not loaded')
            return
        if not os.path.isdir(embdir.path):
            return
        file_paths = list_embeddings(embdir.path)
        if shared.backend == shared.Backend.DIFFUSERS:
            self.load_diffusers_embedding(file_paths)
        else:
            for file_path in file_paths:
                try:
                    if os.stat(file_path).st_size == 0:
                        continue
                    fn = os.path.basename(file_path)
                    self.load_from_file(file_path, fn)
                except Exception as e:
                    errors.display(e, f'embedding load {fn}')
                    continue

    def load_textual_inversion_embeddings(self, force_reload=False):
        if shared.sd_model is None:
            return
        t0 = time.time()
        if not force_reload:
            need_reload = False
            for embdir in self.embedding_dirs.values():
                if embdir.has_changed():
                    need_reload = True
                    break
            if not need_reload:
                return
        self.ids_lookup.clear()
        self.word_embeddings.clear()
        self.skipped_embeddings.clear()
        self.embeddings_used.clear()
        self.expected_shape = self.get_expected_shape()
        for embdir in self.embedding_dirs.values():
            self.load_from_dir(embdir)
            embdir.update()

        # re-sort word_embeddings because load_from_dir may not load in alphabetic order.
        # using a temporary copy so we don't reinitialize self.word_embeddings in case other objects have a reference to it.
        sorted_word_embeddings = {e.name: e for e in sorted(self.word_embeddings.values(), key=lambda e: e.name.lower())}
        self.word_embeddings.clear()
        self.word_embeddings.update(sorted_word_embeddings)

        displayed_embeddings = (tuple(self.word_embeddings.keys()), tuple(self.skipped_embeddings.keys()))
        if self.previously_displayed_embeddings != displayed_embeddings:
            self.previously_displayed_embeddings = displayed_embeddings
            t1 = time.time()
            shared.log.info(f"Load embeddings: loaded={len(self.word_embeddings)} skipped={len(self.skipped_embeddings)} time={t1-t0:.2f}")


    def find_embedding_at_position(self, tokens, offset):
        token = tokens[offset]
        possible_matches = self.ids_lookup.get(token, None)
        if possible_matches is None:
            return None, None
        for ids, embedding in possible_matches:
            if tokens[offset:offset + len(ids)] == ids:
                return embedding, len(ids)
        return None, None


def create_embedding(name, num_vectors_per_token, overwrite_old, init_text='*'):
    cond_model = shared.sd_model.cond_stage_model
    with devices.autocast():
        cond_model([""])  # will send cond model to GPU if lowvram/medvram is active
    #cond_model expects at least some text, so we provide '*' as backup.
    embedded = cond_model.encode_embedding_init_text(init_text or '*', num_vectors_per_token)
    vec = torch.zeros((num_vectors_per_token, embedded.shape[1]), device=devices.device)
    #Only copy if we provided an init_text, otherwise keep vectors as zeros
    if init_text:
        for i in range(num_vectors_per_token):
            vec[i] = embedded[i * int(embedded.shape[0]) // num_vectors_per_token]
    # Remove illegal characters from name.
    name = "".join( x for x in name if (x.isalnum() or x in "._- "))
    fn = os.path.join(shared.opts.embeddings_dir, f"{name}.pt")
    if not overwrite_old and os.path.exists(fn):
        shared.log.warning(f"Embedding already exists: {fn}")
    else:
        embedding = Embedding(vec=vec, name=name, filename=fn)
        embedding.step = 0
        embedding.save(fn)
        shared.log.info(f'Created embedding: {fn} vectors {num_vectors_per_token} init {init_text}')
    return fn


def write_loss(log_directory, filename, step, epoch_len, values):
    if shared.opts.training_write_csv_every == 0:
        return
    if step % shared.opts.training_write_csv_every != 0:
        return
    write_csv_header = False if os.path.exists(os.path.join(log_directory, filename)) else True
    with open(os.path.join(log_directory, filename), "a+", newline='', encoding='utf-8') as fout:
        csv_writer = csv.DictWriter(fout, fieldnames=["step", "epoch", "epoch_step", *(values.keys())])
        if write_csv_header:
            csv_writer.writeheader()
        epoch = (step - 1) // epoch_len
        epoch_step = (step - 1) % epoch_len
        csv_writer.writerow({
            "step": step,
            "epoch": epoch,
            "epoch_step": epoch_step,
            **values,
        })


def tensorboard_setup(log_directory):
    os.makedirs(os.path.join(log_directory, "tensorboard"), exist_ok=True)
    return SummaryWriter(
            log_dir=os.path.join(log_directory, "tensorboard"),
            flush_secs=shared.opts.training_tensorboard_flush_every)


def tensorboard_add(tensorboard_writer, loss, global_step, step, learn_rate, epoch_num):
    tensorboard_add_scaler(tensorboard_writer, "Loss/train", loss, global_step)
    tensorboard_add_scaler(tensorboard_writer, f"Loss/train/epoch-{epoch_num}", loss, step)
    tensorboard_add_scaler(tensorboard_writer, "Learn rate/train", learn_rate, global_step)
    tensorboard_add_scaler(tensorboard_writer, f"Learn rate/train/epoch-{epoch_num}", learn_rate, step)


def tensorboard_add_scaler(tensorboard_writer, tag, value, step):
    tensorboard_writer.add_scalar(tag=tag, scalar_value=value, global_step=step)


def tensorboard_add_image(tensorboard_writer, tag, pil_image, step):
    # Convert a pil image to a torch tensor
    img_tensor = torch.as_tensor(np.array(pil_image, copy=True))
    img_tensor = img_tensor.view(pil_image.size[1], pil_image.size[0], len(pil_image.getbands()))
    img_tensor = img_tensor.permute((2, 0, 1))
    tensorboard_writer.add_image(tag, img_tensor, global_step=step)


def validate_train_inputs(model_name, learn_rate, batch_size, gradient_step, data_root, template_file, template_filename, steps, save_model_every, create_image_every, name="embedding"):
    assert model_name, f"{name} not selected"
    assert learn_rate, "Learning rate is empty or 0"
    assert isinstance(batch_size, int), "Batch size must be integer"
    assert batch_size > 0, "Batch size must be positive"
    assert isinstance(gradient_step, int), "Gradient accumulation step must be integer"
    assert gradient_step > 0, "Gradient accumulation step must be positive"
    assert data_root, "Dataset directory is empty"
    assert os.path.isdir(data_root), "Dataset directory doesn't exist"
    assert os.listdir(data_root), "Dataset directory is empty"
    assert template_filename, "Prompt template file not selected"
    assert template_file, f"Prompt template file {template_filename} not found"
    assert os.path.isfile(template_file.path), f"Prompt template file {template_filename} doesn't exist"
    assert steps, "Max steps is empty or 0"
    assert isinstance(steps, int), "Max steps must be integer"
    assert steps > 0, "Max steps must be positive"
    assert isinstance(save_model_every, int), "Save {name} must be integer"
    assert save_model_every >= 0, "Save {name} must be positive or 0"
    assert isinstance(create_image_every, int), "Create image must be integer"
    assert create_image_every >= 0, "Create image must be positive or 0"


def train_embedding(id_task, embedding_name, learn_rate, batch_size, gradient_step, data_root, log_directory, training_width, training_height, varsize, steps, clip_grad_mode, clip_grad_value, shuffle_tags, tag_drop_out, latent_sampling_method, use_weight, create_image_every, save_embedding_every, template_filename, save_image_with_stored_embedding, preview_from_txt2img, preview_prompt, preview_negative_prompt, preview_steps, preview_sampler_index, preview_cfg_scale, preview_seed, preview_width, preview_height): # pylint: disable=unused-argument

    shared.log.debug(f'train_embedding: embedding_name={embedding_name}|learn_rate={learn_rate}|batch_size={batch_size}|gradient_step={gradient_step}|data_root={data_root}|log_directory={log_directory}|training_width={training_width}|training_height={training_height}|varsize={varsize}|steps={steps}|clip_grad_mode={clip_grad_mode}|clip_grad_value={clip_grad_value}|shuffle_tags={shuffle_tags}|tag_drop_out={tag_drop_out}|latent_sampling_method={latent_sampling_method}|use_weight={use_weight}|create_image_every={create_image_every}|save_embedding_every={save_embedding_every}|template_filename={template_filename}|save_image_with_stored_embedding={save_image_with_stored_embedding}|preview_from_txt2img={preview_from_txt2img}|preview_prompt={preview_prompt}|preview_negative_prompt={preview_negative_prompt}|preview_steps={preview_steps}|preview_sampler_index={preview_sampler_index}|preview_cfg_scale={preview_cfg_scale}|preview_seed={preview_seed}|preview_width={preview_width}|preview_height={preview_height}')
    save_embedding_every = save_embedding_every or 0
    create_image_every = create_image_every or 0
    template_file = textual_inversion_templates.get(template_filename, None)
    validate_train_inputs(embedding_name, learn_rate, batch_size, gradient_step, data_root, template_file, template_filename, steps, save_embedding_every, create_image_every, name="embedding")
    if log_directory is None or log_directory == '':
        log_directory = f"{os.path.join(shared.cmd_opts.data_dir, 'train/log/embeddings')}"
    template_file = template_file.path

    shared.state.job = "train"
    shared.state.textinfo = "Initializing textual inversion training..."
    shared.state.job_count = steps

    filename = os.path.join(shared.opts.embeddings_dir, f'{embedding_name}.pt')

    if log_directory == '':
        log_directory = f"{os.path.join(shared.cmd_opts.data_dir, 'train/log/embeddings')}"
    log_directory = os.path.join(log_directory, embedding_name)
    unload = shared.opts.unload_models_when_training

    if save_embedding_every > 0:
        embedding_dir = os.path.join(log_directory, "embeddings")
        os.makedirs(embedding_dir, exist_ok=True)
    else:
        embedding_dir = None

    if create_image_every > 0:
        images_dir = os.path.join(log_directory, "images")
        os.makedirs(images_dir, exist_ok=True)
    else:
        images_dir = None

    if create_image_every > 0 and save_image_with_stored_embedding:
        images_embeds_dir = os.path.join(log_directory, "image_embeddings")
        os.makedirs(images_embeds_dir, exist_ok=True)
    else:
        images_embeds_dir = None

    hijack = sd_hijack.model_hijack
    embedding = hijack.embedding_db.word_embeddings[embedding_name]
    checkpoint = sd_models.select_checkpoint()
    initial_step = embedding.step or 0
    if initial_step >= steps:
        shared.state.textinfo = "Model has already been trained beyond specified max steps"
        return embedding, filename
    scheduler = LearnRateScheduler(learn_rate, steps, initial_step)
    clip_grad = torch.nn.utils.clip_grad_value_ if clip_grad_mode == "value" else \
        torch.nn.utils.clip_grad_norm_ if clip_grad_mode == "norm" else \
        None
    if clip_grad:
        clip_grad_sched = LearnRateScheduler(clip_grad_value, steps, initial_step, verbose=False)
    # dataset loading may take a while, so input validations and early returns should be done before this
    shared.state.textinfo = f"Preparing dataset from {html.escape(data_root)}..."
    old_parallel_processing_allowed = shared.parallel_processing_allowed

    if shared.opts.training_enable_tensorboard:
        tensorboard_writer = tensorboard_setup(log_directory)

    pin_memory = shared.opts.pin_memory
    # init dataset
    ds = modules.textual_inversion.dataset.PersonalizedBase(data_root=data_root, width=training_width, height=training_height, repeats=shared.opts.training_image_repeats_per_epoch, placeholder_token=embedding_name, model=shared.sd_model, cond_model=shared.sd_model.cond_stage_model, device=devices.device, template_file=template_file, batch_size=batch_size, gradient_step=gradient_step, shuffle_tags=shuffle_tags, tag_drop_out=tag_drop_out, latent_sampling_method=latent_sampling_method, varsize=varsize, use_weight=use_weight)

    if shared.opts.save_training_settings_to_txt:
        save_settings_to_file(log_directory, {**dict(model_name=checkpoint.model_name, model_hash=checkpoint.shorthash, num_of_dataset_images=len(ds), num_vectors_per_token=len(embedding.vec)), **locals()})
    latent_sampling_method = ds.latent_sampling_method
    # init dataloader
    dl = modules.textual_inversion.dataset.PersonalizedDataLoader(ds, latent_sampling_method=latent_sampling_method, batch_size=ds.batch_size, pin_memory=pin_memory)
    if unload:
        shared.parallel_processing_allowed = False
        shared.sd_model.first_stage_model.to(devices.cpu)

    embedding.vec.requires_grad = True
    optimizer = torch.optim.AdamW([embedding.vec], lr=scheduler.learn_rate, weight_decay=0.0)
    if shared.opts.save_optimizer_state:
        optimizer_state_dict = None
        if os.path.exists(f"{filename}.optim"):
            optimizer_saved_dict = torch.load(f"{filename}.optim", map_location='cpu')
            if embedding.checksum() == optimizer_saved_dict.get('hash', None):
                optimizer_state_dict = optimizer_saved_dict.get('optimizer_state_dict', None)
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)
            shared.log.info("Load existing optimizer from checkpoint")
        else:
            shared.log.info("No saved optimizer exists in checkpoint")

    scaler = torch.cuda.amp.GradScaler()

    batch_size = ds.batch_size
    gradient_step = ds.gradient_step
    # n steps = batch_size * gradient_step * n image processed
    steps_per_epoch = len(ds) // batch_size // gradient_step
    max_steps_per_epoch = len(ds) // batch_size - (len(ds) // batch_size) % gradient_step
    loss_step = 0
    _loss_step = 0 #internal
    last_saved_file = "<none>"
    last_saved_image = "<none>"
    forced_filename = "<none>"
    embedding_yet_to_be_embedded = False
    is_training_inpainting_model = shared.sd_model.model.conditioning_key in {'hybrid', 'concat'}
    img_c = None

    pbar = tqdm(total=steps - initial_step)
    try:
        sd_hijack_checkpoint.add()
        for _i in range((steps-initial_step) * gradient_step):
            if scheduler.finished:
                break
            if shared.state.interrupted:
                break
            for j, batch in enumerate(dl):
                # works as a drop_last=True for gradient accumulation
                if j == max_steps_per_epoch:
                    break
                scheduler.apply(optimizer, embedding.step)
                if scheduler.finished:
                    break
                if shared.state.interrupted:
                    break
                if clip_grad:
                    clip_grad_sched.step(embedding.step)
                with devices.autocast():
                    x = batch.latent_sample.to(devices.device, non_blocking=pin_memory)
                    if use_weight:
                        w = batch.weight.to(devices.device, non_blocking=pin_memory)
                    c = shared.sd_model.cond_stage_model(batch.cond_text)
                    if is_training_inpainting_model:
                        if img_c is None:
                            img_c = processing.txt2img_image_conditioning(shared.sd_model, c, training_width, training_height)
                        cond = {"c_concat": [img_c], "c_crossattn": [c]}
                    else:
                        cond = c
                    if use_weight:
                        loss = shared.sd_model.weighted_forward(x, cond, w)[0] / gradient_step
                        del w
                    else:
                        loss = shared.sd_model.forward(x, cond)[0] / gradient_step
                    del x
                    _loss_step += loss.item()

                scaler.scale(loss).backward()
                # go back until we reach gradient accumulation steps
                if (j + 1) % gradient_step != 0:
                    continue
                if clip_grad:
                    clip_grad(embedding.vec, clip_grad_sched.learn_rate)

                scaler.step(optimizer)
                scaler.update()
                embedding.step += 1
                pbar.update()
                optimizer.zero_grad(set_to_none=True)
                loss_step = _loss_step
                _loss_step = 0
                steps_done = embedding.step + 1
                epoch_num = embedding.step // steps_per_epoch

                description = f"Training textual inversion step {embedding.step} loss: {loss_step:.5f} lr: {scheduler.learn_rate:.5f}"
                pbar.set_description(description)
                if embedding_dir is not None and steps_done % save_embedding_every == 0:
                    # Before saving, change name to match current checkpoint.
                    embedding_name_every = f'{embedding_name}-{steps_done}'
                    last_saved_file = os.path.join(embedding_dir, f'{embedding_name_every}.pt')
                    save_embedding(embedding, optimizer, checkpoint, embedding_name_every, last_saved_file, remove_cached_checksum=True)
                    embedding_yet_to_be_embedded = True

                write_loss(log_directory, f"{embedding_name}.csv", embedding.step, steps_per_epoch, { "loss": f"{loss_step:.7f}", "learn_rate": scheduler.learn_rate })

                if images_dir is not None and steps_done % create_image_every == 0:
                    forced_filename = f'{embedding_name}-{steps_done}'
                    last_saved_image = os.path.join(images_dir, forced_filename)
                    shared.sd_model.first_stage_model.to(devices.device)

                    p = processing.StableDiffusionProcessingTxt2Img(
                        sd_model=shared.sd_model,
                        do_not_save_grid=True,
                        do_not_save_samples=True,
                        do_not_reload_embeddings=True,
                    )

                    if preview_from_txt2img:
                        p.prompt = preview_prompt
                        p.negative_prompt = preview_negative_prompt
                        p.steps = preview_steps
                        p.sampler_name = sd_samplers.samplers[preview_sampler_index].name
                        p.cfg_scale = preview_cfg_scale
                        p.seed = preview_seed
                        p.width = preview_width
                        p.height = preview_height
                    else:
                        p.prompt = batch.cond_text[0]
                        p.steps = 20
                        p.width = training_width
                        p.height = training_height

                    preview_text = p.prompt
                    processed = processing.process_images(p)
                    image = processed.images[0] if len(processed.images) > 0 else None

                    if unload:
                        shared.sd_model.first_stage_model.to(devices.cpu)

                    if image is not None:
                        shared.state.assign_current_image(image)
                        last_saved_image, _last_text_info = images.save_image(image, images_dir, "", p.seed, p.prompt, shared.opts.samples_format, processed.infotexts[0], p=p, forced_filename=forced_filename, save_to_dirs=False)
                        last_saved_image += f", prompt: {preview_text}"
                        if shared.opts.training_enable_tensorboard and shared.opts.training_tensorboard_save_images:
                            tensorboard_add_image(tensorboard_writer, f"Validation at epoch {epoch_num}", image, embedding.step)

                    if save_image_with_stored_embedding and os.path.exists(last_saved_file) and embedding_yet_to_be_embedded:
                        last_saved_image_chunks = os.path.join(images_embeds_dir, f'{embedding_name}-{steps_done}.png')
                        info = PngImagePlugin.PngInfo()
                        data = torch.load(last_saved_file)
                        info.add_text("sd-ti-embedding", embedding_to_b64(data))
                        title = f"<{data.get('name', '???')}>"
                        try:
                            vectorSize = list(data['string_to_param'].values())[0].shape[0]
                        except Exception:
                            vectorSize = '?'
                        checkpoint = sd_models.select_checkpoint()
                        footer_left = checkpoint.model_name
                        footer_mid = f'[{checkpoint.shorthash}]'
                        footer_right = f'{vectorSize}v {steps_done}s'
                        captioned_image = caption_image_overlay(image, title, footer_left, footer_mid, footer_right)
                        captioned_image = insert_image_data_embed(captioned_image, data)
                        captioned_image.save(last_saved_image_chunks, "PNG", pnginfo=info)
                        embedding_yet_to_be_embedded = False

                    last_saved_image, _last_text_info = images.save_image(image, images_dir, "", p.seed, p.prompt, shared.opts.samples_format, processed.infotexts[0], p=p, forced_filename=forced_filename, save_to_dirs=False)
                    last_saved_image += f", prompt: {preview_text}"

                shared.state.job_no = embedding.step
                shared.state.textinfo = f"""
<p>
Loss: {loss_step:.7f}<br/>
Step: {steps_done}<br/>
Last prompt: {html.escape(batch.cond_text[0])}<br/>
Last saved embedding: {html.escape(last_saved_file)}<br/>
Last saved image: {html.escape(last_saved_image)}<br/>
</p>
"""
        filename = os.path.join(shared.opts.embeddings_dir, f'{embedding_name}.pt')
        save_embedding(embedding, optimizer, checkpoint, embedding_name, filename, remove_cached_checksum=True)
    except Exception as e:
        errors.display(e, 'embedding train')
    finally:
        pbar.leave = False
        pbar.close()
        shared.sd_model.first_stage_model.to(devices.device)
        shared.parallel_processing_allowed = old_parallel_processing_allowed
        sd_hijack_checkpoint.remove()
    return embedding, filename


def save_embedding(embedding, optimizer, checkpoint, embedding_name, filename, remove_cached_checksum=True):
    old_embedding_name = embedding.name
    old_sd_checkpoint = embedding.sd_checkpoint if hasattr(embedding, "sd_checkpoint") else None
    old_sd_checkpoint_name = embedding.sd_checkpoint_name if hasattr(embedding, "sd_checkpoint_name") else None
    old_cached_checksum = embedding.cached_checksum if hasattr(embedding, "cached_checksum") else None
    try:
        embedding.sd_checkpoint = checkpoint.shorthash
        embedding.sd_checkpoint_name = checkpoint.model_name
        if remove_cached_checksum:
            embedding.cached_checksum = None
        embedding.name = embedding_name
        embedding.optimizer_state_dict = optimizer.state_dict()
        embedding.save(filename)
    except Exception:
        embedding.sd_checkpoint = old_sd_checkpoint
        embedding.sd_checkpoint_name = old_sd_checkpoint_name
        embedding.name = old_embedding_name
        embedding.cached_checksum = old_cached_checksum
        raise
