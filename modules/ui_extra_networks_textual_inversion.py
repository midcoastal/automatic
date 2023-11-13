import json
import os
import time
from modules import shared, sd_hijack, sd_models, ui_extra_networks
from modules.textual_inversion.textual_inversion import Embedding, list_embeddings

debug = shared.log.env('SD_ENPTI_DEBUG').prefix(f'[{__name__}]')

class ExtraNetworksPageTextualInversion(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Embedding')
        self.allow_negative_prompt = True
        self.empty_search_extensions=['.ckpt', '.safetensors', '.pt']

    def refresh(self):
        if sd_models.model_data.sd_model is None:
            return
        if shared.backend == shared.Backend.ORIGINAL:
            sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)
        elif hasattr(sd_models.model_data.sd_model, 'embedding_db'):
            sd_models.model_data.sd_model.embedding_db.load_textual_inversion_embeddings(force_reload=True)

    def list_items(self):
        list_debug = debug.prefix('list_items ')
        list_debug('start')

        if sd_models.model_data.sd_model is None:
            embeddings = list([Embedding(vec=0, name=os.path.basename(fn), filename=fn) for fn in list_embeddings(shared.opts.embeddings_dir)])
        elif shared.backend == shared.Backend.ORIGINAL:
            embeddings = list(sd_hijack.model_hijack.embedding_db.word_embeddings.values())
        elif hasattr(sd_models.model_data.sd_model, 'embedding_db'):
            embeddings = list(sd_models.model_data.sd_model.embedding_db.word_embeddings.values())
        else:
            embeddings = []
        embeddings = sorted(embeddings, key=lambda emb: emb.filename)
        t0 = time.time()
        list_debug(f'embeddings.start {len(embeddings)}')
        for embedding in embeddings:
            try:
                path, _ext = os.path.splitext(embedding.filename)
                tags = {}
                if embedding.tag is not None:
                    tags[embedding.tag]=1
                name = os.path.splitext(embedding.basename)[0]
                list_debug(f'item: {name}')
                item = {
                    "type": 'Embedding',
                    "name": name,
                    "filename": embedding.filename,
                    "preview": self.find_preview(path),
                    "description": self.find_description(path),
                    "info": self.find_info(path),
                    "search_term": self.search_terms_from_path(name),
                    "prompt": json.dumps(os.path.splitext(embedding.name)[0]),
                    "local_preview": f"{path}.{shared.opts.samples_format}",
                    "tags": tags,
                }
                list_debug(f'item: {name}')
                yield item
            except Exception as e:
                shared.log.debug(f"Extra networks error: type=embedding file={embedding.filename} {e}")
        list_debug(f'embeddings.end {time.time()-t0}')

    def allowed_directories_for_previews(self):
        return list(sd_hijack.model_hijack.embedding_db.embedding_dirs)
