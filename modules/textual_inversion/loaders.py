
import os
import torch
import safetensors.torch
from modules import shared, errors
from collections import namedtuple
from modules.patches import patch_method, add_method
from typing import Dict, List, Optional, Union
from diffusers.loaders import TEXT_INVERSION_NAME_SAFE, TEXT_INVERSION_NAME, TextualInversionLoaderMixin, logger, remove_hook_from_module, nn, CpuOffload, AlignDevicesHook
from diffusers.utils import DIFFUSERS_CACHE, HF_HUB_OFFLINE, _get_model_file 
from diffusers import loaders as diffusers_loaders
from contextlib import contextmanager

debug = shared.log.env('SD_LOAD_TI_DEBUG').prefix(f'[{__name__}]')

_suppress_exceptions = False
_exceptions_handler = None

@contextmanager
def suppress_exceptions(suppress=True, *, handler=None):
    global _suppress_exceptions, _exceptions_handler
    old_exceptions = _suppress_exceptions
    old_handler = _exceptions_handler
    _suppress_exceptions = suppress
    _exceptions_handler = handler
    try:
        yield
    finally:
        _suppress_exceptions = old_exceptions
        _exceptions_handler = old_handler

@contextmanager
def supressable_exceptions():
    try:
        yield
    except Exception as e:
        global _suppress_exceptions, _exceptions_handler
        if not _suppress_exceptions:
            raise e
        if callable(_exceptions_handler):
            _exceptions_handler(e)
    
@patch_method(diffusers_loaders)
def load_textual_inversion_state_dicts(pretrained_model_name_or_paths, **kwargs):
    _debug = debug.prefix('load_textual_inversion_state_dicts ')
    _debug('start')

    cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
    force_download = kwargs.pop("force_download", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
    use_auth_token = kwargs.pop("use_auth_token", None)
    revision = kwargs.pop("revision", None)
    subfolder = kwargs.pop("subfolder", None)
    weight_name = kwargs.pop("weight_name", None)
    use_safetensors = kwargs.pop("use_safetensors", None)

    allow_pickle = False
    if use_safetensors is None:
        use_safetensors = True
        allow_pickle = True

    user_agent = {
        "file_type": "text_inversion",
        "framework": "pytorch",
    }
    state_dicts = []
    for pretrained_model_name_or_path in pretrained_model_name_or_paths:
        _debug(f'- {pretrained_model_name_or_path}')
        with supressable_exceptions():
            if not isinstance(pretrained_model_name_or_path, (dict, torch.Tensor)):
                # 3.1. Load textual inversion file
                model_file = None

                # Let's first try to load .safetensors weights
                if (use_safetensors and weight_name is None) or (
                    weight_name is not None and weight_name.endswith(".safetensors")
                ):
                    try:
                        _debug('load as .safetensors')
                        model_file = _get_model_file(
                            pretrained_model_name_or_path,
                            weights_name=weight_name or TEXT_INVERSION_NAME_SAFE,
                            cache_dir=cache_dir,
                            force_download=force_download,
                            resume_download=resume_download,
                            proxies=proxies,
                            local_files_only=local_files_only,
                            use_auth_token=use_auth_token,
                            revision=revision,
                            subfolder=subfolder,
                            user_agent=user_agent,
                        )
                        state_dict = safetensors.torch.load_file(model_file, device="cpu")
                    except Exception as e:
                        if not allow_pickle:
                            raise e

                        model_file = None

                if model_file is None:
                    _debug('load as NOT .safetensors')
                    model_file = _get_model_file(
                        pretrained_model_name_or_path,
                        weights_name=weight_name or TEXT_INVERSION_NAME,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                    )
                    state_dict = torch.load(model_file, map_location="cpu")
            else:
                _debug('is (dict, torch.Tensor)')
                state_dict = pretrained_model_name_or_path
            state_dicts.append(state_dict)

    return state_dicts

@patch_method(TextualInversionLoaderMixin)
@staticmethod
def _extend_tokens_and_embeddings(tokens, embeddings, tokenizer):
    _debug = debug.prefix('pipe._extend_tokens_and_embeddings ')
    _debug('start')
    all_tokens = []
    all_embeddings = []

    for token, embedding in zip(tokens, embeddings):
        _debug(f'IN loop')
        _each_debug = _debug.prefix(f'token {token} ')
        with supressable_exceptions():
            if f"{token}_1" in tokenizer.get_vocab():
                _each_debug('already exists')
                multi_vector_tokens = [token]
                i = 1
                while f"{token}_{i}" in tokenizer.added_tokens_encoder:
                    multi_vector_tokens.append(f"{token}_{i}")
                    i += 1

                raise ValueError(
                    f"Multi-vector Token {multi_vector_tokens} already in tokenizer vocabulary. Please choose a different token name or remove the {multi_vector_tokens} and embedding from the tokenizer and text encoder."
                )

            is_multi_vector = len(embedding.shape) > 1 and embedding.shape[0] > 1
            if is_multi_vector:
                _debug('is multi-vector')
                all_tokens += [token] + [f"{token}_{i}" for i in range(1, embedding.shape[0])]
                all_embeddings += [e for e in embedding]  # noqa: C416
            else:
                _debug('is not multi-vector')
                all_tokens += [token]
                all_embeddings += [embedding[0]] if len(embedding.shape) > 1 else [embedding]
    
    _debug(f'haves {len(tokens)} Tokens and {len(embeddings)} Embeddings')

    return all_tokens, all_embeddings

@patch_method(TextualInversionLoaderMixin)
def _retrieve_tokens_and_embeddings(tokens, state_dicts, tokenizer):
    _debug = debug.prefix('pipe._retrieve_tokens_and_embeddings ')
    _debug('start')
    all_tokens = []
    all_embeddings = []
    for token, state_dict in zip(tokens, state_dicts):
        with supressable_exceptions():
            _loop_debug = _debug.prefix(f'token {token} ')
            if isinstance(state_dict, torch.Tensor):
                _loop_debug('torch.Tensor')
                if token is None:
                    raise ValueError(
                        "You are trying to load a textual inversion embedding that has been saved as a PyTorch tensor. Make sure to pass the name of the corresponding token in this case: `token=...`."
                    )
                loaded_token = token
                embedding = state_dict
            elif len(state_dict) == 1:
                _loop_debug('diffusers')
                # diffusers
                loaded_token, embedding = next(iter(state_dict.items()))
            elif "string_to_param" in state_dict:
                _loop_debug('A1111')
                # A1111
                loaded_token = state_dict["name"]
                embedding = state_dict["string_to_param"]["*"]
            else:
                _loop_debug('UNKNOWN')
                raise ValueError(
                    f"Loaded state dictonary is incorrect: {state_dict}. \n\n"
                    "Please verify that the loaded state dictionary of the textual embedding either only has a single key or includes the `string_to_param`"
                    " input key."
                )

            if token is not None and loaded_token != token:
                logger.info(f"The loaded token: {loaded_token} is overwritten by the passed token {token}.")
            else:
                token = loaded_token

            if token in tokenizer.get_vocab():
                raise ValueError(
                    f"Token {token} already in tokenizer vocabulary. Please choose a different token name or remove {token} and embedding from the tokenizer and text encoder."
                )

            all_tokens.append(token)
            all_embeddings.append(embedding)

    return all_tokens, all_embeddings

@add_method(TextualInversionLoaderMixin)
def filter_embedding_shapes(self, embeddings_tokens: list, embeddings_to_test: list, text_encoder: Optional["PreTrainedModel"] = None) -> list:
    text_encoder = text_encoder or getattr(self, "text_encoder", None)
    expected_emb_dim = text_encoder.get_input_embeddings().weight.shape[-1]
    good_embeddings = []
    good_tokens = []
    for token, embedding in zip(embeddings_tokens, embeddings_to_test):
        with supressable_exceptions():
            embs=[embedding]
            if len(embedding.shape) > 1:
                if embedding.shape[0] > 1:
                    embs=[e for e in embedding]
                else:
                    embs=[embs[0]]
            if any(expected_emb_dim != emb.shape[-1] for emb in embs):
                raise ValueError(
                    "Loaded embeddings are of incorrect shape. Expected each textual inversion embedding "
                    "to be of shape {expected_emb_dim.shape[-1]}, but are {embeddings.shape[-1]} "
                )
            good_embeddings.append(embedding)
            good_tokens.append(token)

    return good_tokens, good_embeddings
#patches.patch(__name__, TextualInversionLoaderMixin, 'filter_embedding_shapes', filter_embedding_shapes)
#TextualInversionLoaderMixin.filter_embedding_shapes=filter_embedding_shapes


@patch_method(TextualInversionLoaderMixin)
def load_textual_inversion(
    self: TextualInversionLoaderMixin,
    pretrained_model_name_or_path: Union[str, List[str], Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
    token: Optional[Union[str, List[str]]] = None,
    tokenizer: Optional["PreTrainedTokenizer"] = None,  # noqa: F821
    text_encoder: Optional["PreTrainedModel"] = None,  # noqa: F821
    **kwargs,
):
    try:
        _debug = debug.prefix('pipe.load_textual_inversion ')
        _debug('start')
        _debug(f'pipe.load_textual_inversion: {pretrained_model_name_or_path}')
        
        # 1. Set correct tokenizer and text encoder
        tokenizer = tokenizer or getattr(self, "tokenizer", None)
        text_encoder = text_encoder or getattr(self, "text_encoder", None)
        loaded_model_names_or_paths = {}

        # 2. Normalize inputs
        pretrained_model_name_or_paths = (
            [pretrained_model_name_or_path]
            if not isinstance(pretrained_model_name_or_path, list)
            else pretrained_model_name_or_path
        )
        tokens = len(pretrained_model_name_or_paths) * [token] if (isinstance(token, str) or token is None) else token
        token_data = {}

        # Build a unique list of Shape-Token/Embedding-Shape pairs, mapped with an associated `name_or_path` for reporting
        expected_emb_dim = text_encoder.get_input_embeddings().weight.shape[-1]
        for token, name_or_path in zip(tokens, pretrained_model_name_or_paths ):
            with supressable_exceptions():
                _embedding_data = { 
                    shape_token: (embedding_shape, name_or_path )
                    for shape_token, embedding_shape in zip(
                        # 5. Extend tokens and embeddings for multi vector
                        *self._extend_tokens_and_embeddings(
                            # 4. Retrieve tokens and embeddings
                            *TextualInversionLoaderMixin._retrieve_tokens_and_embeddings(
                                [token], 
                                # 3. Load state dicts of textual embeddings
                                diffusers_loaders.load_textual_inversion_state_dicts(
                                    [name_or_path], 
                                    cache_dir=shared.opts.diffusers_dir, 
                                    local_files_only=True
                                ), 
                                tokenizer
                            )
                        )
                    )
                }
                # 6. Make sure all embeddings have the correct size
                if any(expected_emb_dim != embedding_shape.shape[-1] for embedding_shape, _ in _embedding_data.values):
                    raise ValueError(
                        "Loaded embeddings are of incorrect shape. Expected each textual inversion embedding "
                        "to be of shape {input_embeddings.shape[-1]}, but are {embeddings.shape[-1]} "
                    )
                token_data.update(_embedding_data)
        
        # 7. Now we can be sure that loading the embedding matrix works
        # < Unsafe code:
        
        # 7.1 Offload all hooks in case the pipeline was cpu offloaded before make sure, we offload and onload again
        is_model_cpu_offload = False
        is_sequential_cpu_offload = False
        
        for _, component in self.components.items():
            if isinstance(component, nn.Module):
                if hasattr(component, "_hf_hook"):
                    is_model_cpu_offload = isinstance(getattr(component, "_hf_hook"), CpuOffload)
                    is_sequential_cpu_offload = isinstance(getattr(component, "_hf_hook"), AlignDevicesHook)
                    logger.info(
                        "Accelerate hooks detected. Since you have called `load_textual_inversion()`, the previous hooks will be first removed. Then the textual inversion parameters will be loaded and the hooks will be applied again."
                    )
                    remove_hook_from_module(component, recurse=is_sequential_cpu_offload)
        
        # 7.2 save expected device and dtype
        device = text_encoder.device
        dtype = text_encoder.dtype

        # 7.2 Add Tokens to the Tokenizer
        _initial_tokenizer_size = len(tokenizer)
        tokens_to_add = [token for token in token_data]
        tokens_added = tokenizer.add_tokens(tokens_to_add)
        tokenizer_size = len(tokenizer)
        
        _debug(f'Added {tokens_added} Tokens: tokenizer_size={tokenizer_size}, _initial_tokenizer_size={_initial_tokenizer_size}')
        _debug.debug(f'Added Tokens: tokens={tokens_to_add}')

        # 7.3 Increase token embedding matrix
        text_encoder.resize_token_embeddings(tokenizer_size)

        input_embeddings = text_encoder.get_input_embeddings().weight

        unk_token_id = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

        # 7.4 Load token and embedding
        for token_id, token in zip(tokenizer.convert_tokens_to_ids(tokens_to_add), tokens_to_add):
                #with supressable_exceptions():
                if token_id <= unk_token_id:
                    raise RuntimeError(f'Processed Shape-Token `{token}` does not resolve to a new Token ID')
                embedding = token_data[token][0]
                path = token_data[token][1]
                input_embeddings.data[token_id] = embedding
                loaded_model_names_or_paths[path] = True
                _debug.debug(f"Loaded textual inversion embedding for {token}: {path}")
        
        input_embeddings.to(dtype=dtype, device=device)

        # 7.5 Offload the model again
        if is_model_cpu_offload:
            self.enable_model_cpu_offload()
        elif is_sequential_cpu_offload:
            self.enable_sequential_cpu_offload()
    except Exception as e:
        errors.display(e, 'load textual inversions')
    
    return [ name_or_path for name_or_path in loaded_model_names_or_paths ]
    # / Unsafe Code >