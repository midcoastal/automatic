import os
import io
import re
import time
import json
import html
import base64
import urllib.parse
import threading
import filetype
from datetime import datetime
from types import SimpleNamespace
from html.parser import HTMLParser
from collections import OrderedDict
import queue
import gradio as gr
from PIL import Image
from starlette.responses import FileResponse, JSONResponse
from modules import paths, shared, scripts, files_cache, errors
from modules.ui_components import ToolButton
import modules.ui_symbols as symbols


allowed_dirs = [os.path.abspath('html/'),os.path.abspath('models/')]
reference_path = os.path.join('models', 'Reference')
thumb_queue = queue.Queue()
thumb_exts = ("jpg", "jpeg", "png", "webp", "tiff", "jp2")
refresh_time = 0
extra_pages = shared.extra_networks
debug = shared.log.trace if os.environ.get('SD_EN_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: EN')
card_full = '''
    <div class='card' onclick={card_click} title='{name}' data-tab='{tabname}' data-page='{page}' data-name='{name}' data-filename='{filename}' data-tags='{tags}' data-mtime='{mtime}' data-size='{size}' data-search='{search}'>
        <div class='overlay'>
            <div class='tags'></div>
            <div class='name'>{title}</div>
        </div>
        <div class='actions'>
            <span class='details' title="Get details" onclick="showCardDetails(event)">&#x1f6c8;</span>
            <div class='additional'><ul></ul></div>
        </div>
        <img class='preview' src='{preview}' loading='lazy'></img>
    </div>
'''
card_list = '''
    <div class='card card-list' onclick={card_click} title='{name}' data-tab='{tabname}' data-page='{page}' data-name='{name}' data-filename='{filename}' data-tags='{tags}' data-mtime='{mtime}' data-size='{size}' data-search='{search}'>
        <div style='display: flex'>
            <span class='details' title="Get details" onclick="showCardDetails(event)">&#x1f6c8;</span>&nbsp;
            <div class='name'>{title}</div>&nbsp;
            <div class='tags tags-list'></div>
        </div>
    </div>
'''
html_format = '''
    <style>
        #{page_name}_cards .preview {{
            width: {width}px;
            height: {height}px;
            object-fit: {fit};
        }}
    </style>
    <div id='{page_name}_subdirs' class='extra-network-subdirs'>
        {subdirs_html}
    </div>
    <div id='{page_name}_cards' class='extra-network-cards'>
        {cards_html}
    </div>
'''


def in_allowed_dir(filepath):
    abs_filepath = os.path.abspath(filepath)
    return any(abs_filepath.startswith(os.path.join(os.path.abspath(allowed_dir),'')) for allowed_dir in allowed_dirs)


def link_preview(filename):
    thumb_queue.put(filename)
    return _link_preview(filename)


def _link_preview(filename):
    preview = f"./sd_extra_networks/thumb?filename={urllib.parse.quote(os.path.relpath(filename))}"
    return preview


no_preview_link = _link_preview('html/card-no-preview.png')


def resolve_preview(file = None, queue_thumb=False):
    try:
        if file and os.path.exists(file):
            file = os.path.abspath(file)
            if file.startswith(reference_path):
                return file
            diffusers_dir = os.path.abspath(shared.opts.diffusers_dir)
            diffusers_pre = os.path.join(diffusers_dir, 'models--')
            if file.startswith(diffusers_pre):
                file = file[len(diffusers_pre):file.find(os.path.sep,len(diffusers_pre))]
                return f'{os.path.join(reference_path, file)}.{shared.opts.samples_format}'
            base = os.path.splitext(file)[0]
            for mid in ['.thumb.', '.preview.', '.' ]:
                for fn in [f'{base}{mid}{ext}' for ext in thumb_exts]:
                    if os.path.exists(fn) and os.path.isfile(fn):
                        if mid != '.thumb.' and queue_thumb:
                            thumb_queue.put(fn)
                        return fn
    except Exception as e:
        debug(f'Error looking up preview:\n{file}\n{e}')
    return None

def thumbs_worker():
    html_path = os.path.join(os.path.abspath('html'),'')
    skip_paths = [html_path, os.path.abspath(reference_path)]
    created = set()
    processed = set()
    start_time = time.time()
    timeout = 60
    last_report = 0
    last_report_count = 0
    report_interval = 60
    debug(f'EN thumbs_worker(): startup report_interval={report_interval} skip_paths={skip_paths}')
    while True:
        try:
            file = os.path.abspath(thumb_queue.get(timeout=timeout))
            if file not in processed:
                processed.add(file)
                preview_file = resolve_preview(file)
                if preview_file and in_allowed_dir(preview_file) and os.path.isfile(preview_file) and not any(preview_file.startswith(skip) for skip in skip_paths):
                    base, ext = os.path.splitext(preview_file)
                    if not base.endswith('.thumb'):
                        if base.endswith('.preview'):
                            base = base[:-8]
                        thumb_file = f'{base}.thumb.{ext}'
                        if not os.path.exists(thumb_file): # thumbnail doesn't exists, likely, since we could't find it above.
                            try:
                                img = Image.open(preview_file)
                                if img and img.width > 1024 or img.height > 1024 or os.path.getsize(preview_file) > 65536:
                                    img = img.convert('RGB')
                                    img.thumbnail((512, 512), Image.Resampling.HAMMING)
                                    img.save(thumb_file, quality=50)
                                    img.close()
                                    created.add(thumb_file)
                            except Exception as e:
                                img = None
                                #os.remove(file)
                                shared.log.warning(f'Extra network error creating thumbnail: {preview_file} {e}')
                thumb_queue.task_done()
                report_status = 'running'
        except queue.Empty:
            report_status = 'idle'
        finally:
            cur_time = time.time()
            report_time = cur_time-last_report
            if report_time >= report_interval:
                processed_count = len(processed)
                report_count = processed_count - last_report_count
                if report_count:
                    debug(f'EN thumbs_worker(): process_report created={len(created)} processed={processed_count} performance={report_count/report_time:.2f}/s run_time={cur_time-start_time:.2f}s state={report_status}')
                    last_report = cur_time
                    last_report_count = processed_count


threading.Thread(target=thumbs_worker, daemon=True).start()


def init_api(app):

    def fetch_file(filename: str = ""):
        abs_filename = os.path.abspath(filename)
        no_preview_path = os.path.abspath('html/card-no-preview.png')
        status_code = 404
        headers = {
            "Accept-Ranges": "bytes",
            "Cache-Control": "max-age=60480, stale-if-error=86400, must-revalidate"
        }
        try:
            if not in_allowed_dir(abs_filename):
                status_code=403
                debug(f'File not in one of allowed directories: {filename}')
            else:
                file_path = None
                if os.path.splitext(filename)[1].lower() in thumb_exts:
                    file_path = abs_filename
                else:
                    file_path = resolve_preview(filename, queue_thumb=True)
            if file_path:
                return FileResponse(file_path, headers=headers|{"File-Path": os.path.relpath(file_path), 'content-type': filetype.guess_mime(file_path)})
        except BaseException as e:
            status_code = 500
            debug(f'Error Returning "{file_path}": {e}')
        return FileResponse(no_preview_path, headers=headers|{"File-Path": os.path.relpath(no_preview_path)}, status_code=status_code)

    def get_metadata(page: str = "", item: str = ""):
        page = next(iter([x for x in shared.extra_networks if x.name == page]), None)
        if page is None:
            return JSONResponse({ 'metadata': 'none' })
        metadata = page.metadata.get(item, 'none')
        if metadata is None:
            metadata = ''
        # shared.log.debug(f"Extra networks metadata: page='{page}' item={item} len={len(metadata)}")
        return JSONResponse({"metadata": metadata})

    def get_info(page: str = "", item: str = ""):
        page = next(iter([x for x in get_pages() if x.name == page]), None)
        if page is None:
            return JSONResponse({ 'info': 'none' })
        item = next(iter([x for x in page.items if x['name'] == item]), None)
        if item is None:
            return JSONResponse({ 'info': 'none' })
        info = page.find_info(item['filename'])
        if info is None:
            info = {}
        # shared.log.debug(f"Extra networks info: page='{page.name}' item={item['name']} len={len(info)}")
        return JSONResponse({"info": info})

    def get_desc(page: str = "", item: str = ""):
        page = next(iter([x for x in get_pages() if x.name == page]), None)
        if page is None:
            return JSONResponse({ 'description': 'none' })
        item = next(iter([x for x in page.items if x['name'] == item]), None)
        if item is None:
            return JSONResponse({ 'description': 'none' })
        desc = page.find_description(item['filename'])
        if desc is None:
            desc = ''
        # shared.log.debug(f"Extra networks desc: page='{page.name}' item={item['name']} len={len(desc)}")
        return JSONResponse({"description": desc})

    app.add_api_route("/sd_extra_networks/thumb", fetch_file, methods=["GET"])
    app.add_api_route("/sd_extra_networks/metadata", get_metadata, methods=["GET"])
    app.add_api_route("/sd_extra_networks/info", get_info, methods=["GET"])
    app.add_api_route("/sd_extra_networks/description", get_desc, methods=["GET"])


class ExtraNetworksPage:
    def __init__(self, title):
        self.title = title
        self.name = title.lower()
        self.allow_negative_prompt = False
        self.metadata = {}
        self.info = {}
        self.html = ''
        self.items = []
        self.refresh_time = 0
        self.page_time = 0
        self.list_time = 0
        self.info_time = 0
        self.desc_time = 0
        self.preview_time = 0
        self.dirs = {}
        self.view = shared.opts.extra_networks_view
        self.card = card_full if shared.opts.extra_networks_view == 'gallery' else card_list

    def refresh(self):
        pass

    def create_xyz_grid(self):
        xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ == "xyz_grid.py"][0].module

        def add_prompt(p, opt, x):
            for item in [x for x in self.items if x["name"] == opt]:
                try:
                    p.prompt = f'{p.prompt} {eval(item["prompt"])}' # pylint: disable=eval-used
                except Exception as e:
                    shared.log.error(f'Cannot evaluate extra network prompt: {item["prompt"]} {e}')

        if not any(self.title in x.label for x in xyz_grid.axis_options):
            if self.title == 'Model':
                return
            opt = xyz_grid.AxisOption(f"[Network] {self.title}", str, add_prompt, choices=lambda: [x["name"] for x in self.items])
            xyz_grid.axis_options.append(opt)

    def link_preview(self, filename):
        return link_preview(filename)

    def is_empty(self, folder):
        return any(files_cache.list_files(folder, ext_filter=['.ckpt', '.safetensors', '.pt', '.json']))

    def create_items(self, tabname):
        if self.refresh_time is not None and self.refresh_time > refresh_time: # cached results
            return
        t0 = time.time()
        try:
            self.items = list(self.list_items())
            self.refresh_time = time.time()
        except Exception as e:
            self.items = []
            shared.log.error(f'Extra networks error listing items: class={self.__class__.__name__} tab={tabname} {e}')
            if os.environ.get('SD_EN_DEBUG', None):
                errors.display(e, f'Extra networks error listing items: class={self.__class__.__name__} tab={tabname}')
        for item in self.items:
            if item is None:
                continue
            self.metadata[item["name"]] = item.get("metadata", {})
        t1 = time.time()
        debug(f'EN create-items: page={self.name} items={len(self.items)} time={t1-t0:.2f}')
        self.list_time += t1-t0


    def create_page(self, tabname, skip = False):
        debug(f'EN create-page: {self.name}')
        if self.page_time > refresh_time and len(self.html) > 0: # cached page
            return self.html
        self_name_id = self.name.replace(" ", "_")
        if skip:
            return f"<div id='{tabname}_{self_name_id}_subdirs' class='extra-network-subdirs'></div><div id='{tabname}_{self_name_id}_cards' class='extra-network-cards'>Extra network page not ready<br>Click refresh to try again</div>"
        subdirs = {}
        allowed_folders = [os.path.abspath(x) for x in self.allowed_directories_for_previews() if os.path.exists(x)]
        for parentdir, dirs in {d: files_cache.walk(d, cached=True, recurse=files_cache.not_hidden) for d in allowed_folders}.items():
            for tgt in dirs:
                tgt = tgt.path
                if reference_path in tgt:
                    subdirs['Reference'] = 1
                if shared.backend == shared.Backend.DIFFUSERS and shared.opts.diffusers_dir in tgt:
                    subdirs[os.path.basename(shared.opts.diffusers_dir)] = 1
                if 'models--' in tgt:
                    continue
                subdir = tgt[len(parentdir):].replace("\\", "/")
                while subdir.startswith("/"):
                    subdir = subdir[1:]
                if not subdir:
                    continue
                # if not self.is_empty(tgt):
                subdirs[subdir] = 1
        debug(f"Extra networks: page='{self.name}' subfolders={list(subdirs)}")
        subdirs = OrderedDict(sorted(subdirs.items()))
        if self.name == 'model':
            subdirs['Reference'] = 1
            subdirs[os.path.basename(shared.opts.diffusers_dir)] = 1
            subdirs.move_to_end(os.path.basename(shared.opts.diffusers_dir))
            subdirs.move_to_end('Reference')
        if self.name == 'style' and shared.opts.extra_networks_styles:
            subdirs['built-in'] = 1
        subdirs_html = "<button class='lg secondary gradio-button custom-button search-all' onclick='extraNetworksSearchButton(event)'>All</button><br>"
        subdirs_html += "".join(f"<button class='lg secondary gradio-button custom-button' onclick='extraNetworksSearchButton(event)'>{html.escape(subdir)}</button><br>" for subdir in subdirs if subdir != '')
        self.create_items(tabname)
        self.create_xyz_grid()
        if len(self.items) > 0 and self.items[0].get('mtime', None) is not None:
            self.items.sort(key=lambda x: x["mtime"], reverse=True)
        self.html = html_format.format(
            page_name=f'{tabname}_{self_name_id}',
            width=shared.opts.extra_networks_card_size,
            height=shared.opts.extra_networks_card_size if shared.opts.extra_networks_card_square else 'auto',
            fit=shared.opts.extra_networks_card_fit,
            subdirs_html=subdirs_html,
            cards_html=''.join(str(self.create_html(item, tabname)) for item in self.items)
        )
        self.page_time = time.time()
        shared.log.debug(f"Extra networks: page='{self.name}' items={len(self.items)} subfolders={len(subdirs)} tab={tabname} folders={self.allowed_directories_for_previews()} list={self.list_time:.2f} thumb={self.preview_time:.2f} desc={self.desc_time:.2f} info={self.info_time:.2f} workers={shared.max_workers}")
        return self.html

    def list_items(self):
        raise NotImplementedError

    def allowed_directories_for_previews(self):
        return []

    def create_html(self, item, tabname):
        try:
            args = {
                "tabname": tabname,
                "page": self.name,
                "name": item["name"],
                "title": os.path.basename(item["name"].replace('_', ' ')),
                "filename": item["filename"],
                "tags": '|'.join([item.get("tags")] if isinstance(item.get("tags", {}), str) else list(item.get("tags", {}).keys())),
                "preview": html.escape(str(item.get("preview", no_preview_link))),
                "width": shared.opts.extra_networks_card_size,
                "height": shared.opts.extra_networks_card_size if shared.opts.extra_networks_card_square else 'auto',
                "fit": shared.opts.extra_networks_card_fit,
                "prompt": item.get("prompt", None),
                "search": item.get("search_term", ""),
                "description": item.get("description") or "",
                "card_click": item.get("onclick", '"' + html.escape(f'return cardClicked({item.get("prompt", None)}, {"true" if self.allow_negative_prompt else "false"})') + '"'),
                "mtime": item.get("mtime", 0),
                "size": item.get("size", 0),
            }
            alias = item.get("alias", None)
            if alias is not None:
                args['title'] += f'\nAlias: {alias}'
            return self.card.format(**args)
        except Exception as e:
            shared.log.error(f'Extra networks item error: page={tabname} item={item["name"]} {e}')
            return ""

    def find_preview_files(self, source_path):
        source_path = os.path.abspath(source_path)
        preview_file = resolve_preview(source_path)
        previews = []
        if preview_file:
            base, ext = os.path.splitext(preview_file)
            base, preview_type = os.path.splitext(base)
            if preview_type == 'thumb' and os.path.exists(f'{base}.preview.{ext}'):
                preview_file = f'{base}.preview.{ext}'
            previews.append(preview_file)
        base, _ext = os.path.splitext(source_path)
        for preview_file in files_cache.directory_files(os.path.dirname(base),recursive=False):
            if preview_file.startswith(f'{base}.image.'):
                shared.log.debug(f'{source_path} -> {preview_file}')
                _base, ext = os.path.splitext(preview_file.lower())
                shared.log.debug(f'{ext} -> {thumb_exts}')
                if ext[1:] in thumb_exts:
                    previews.append(preview_file)
        shared.log.debug(f'Found Previews ({source_path}):\n{previews}')
        return previews


    def find_preview_file(self, path):
        #debug(f'EN find_preview_file({path})')
        preview_file = resolve_preview(path, queue_thumb=True)
        if not preview_file:
            preview_file = 'html/card-no-preview.png'
        else:
            preview_file = os.path.relpath(preview_file)
        return preview_file

    def find_preview(self, filename):
        #debug(f'EN find_preview_file({filename})')
        t0 = time.time()
        #preview_file = self.find_preview_file(filename)
        preview_link = self.link_preview(filename)
        self.preview_time += time.time() - t0
        return preview_link

    def update_all_previews(self, items):
        t0 = time.time()
        diffusers_dir = os.path.abspath(shared.opts.diffusers_dir)
        diffusers_pre = os.path.join(diffusers_dir, 'models--')
        for item in items:
            if item.get('preview', None) is not None:
                continue
            file = os.path.abspath(item['filename'])
            if file.startswith(diffusers_pre):
                model_name = file[len(diffusers_pre):file.find(os.path.sep,len(diffusers_pre))]
                file = f'{model_name}.{shared.opts.samples_format}'
                item['local_preview'] = f'{os.path.join(f"{diffusers_pre}{model_name}", file)}'
                file = os.path.join(reference_path, file)
            if item.get('local_preview', None) is None:
                item['local_preview'] = f'{os.path.splitext(file)[1]}.{shared.opts.samples_format}'
            item['preview'] = self.link_preview(file)
        self.preview_time += time.time() - t0


    def find_description(self, path, info=None):
        t0 = time.time()
        class HTMLFilter(HTMLParser):
            text = ""
            def handle_data(self, data):
                self.text += data
            def handle_endtag(self, tag):
                if tag == 'p':
                    self.text += '\n'

        fn = os.path.splitext(path)[0] + '.txt'
        if os.path.exists(fn):
            try:
                with open(fn, "r", encoding="utf-8", errors="replace") as f:
                    txt = f.read()
                    txt = re.sub('[<>]', '', txt)
                    return txt
            except OSError:
                pass
        if info is None:
            info = self.find_info(path)
        desc = info.get('description', '') or ''
        f = HTMLFilter()
        f.feed(desc)
        t1 = time.time()
        self.desc_time += t1-t0
        return f.text

    def find_info(self, path):
        data = {}
        if shared.cmd_opts.no_metadata:
            return data
        fn = os.path.splitext(path)[0] + '.json'
        if os.path.exists(fn):
            t0 = time.time()
            data = shared.readfile(fn, silent=True)
            if type(data) is list:
                data = data[0]
            t1 = time.time()
            self.info_time += t1-t0
        return data


def initialize():
    shared.extra_networks.clear()


def register_page(page: ExtraNetworksPage):
    # registers extra networks page for the UI; recommend doing it in on_before_ui() callback for extensions
    debug(f'EN register-page: {page}')
    if page in shared.extra_networks:
        debug(f'EN register-page: {page} already registered')
        return
    shared.extra_networks.append(page)
    # allowed_dirs.clear()
    # for pg in shared.extra_networks:
    for folder in page.allowed_directories_for_previews():
        if folder not in allowed_dirs:
            allowed_dirs.append(os.path.abspath(folder))


def register_pages():
    from modules.ui_extra_networks_textual_inversion import ExtraNetworksPageTextualInversion
    from modules.ui_extra_networks_hypernets import ExtraNetworksPageHypernetworks
    from modules.ui_extra_networks_checkpoints import ExtraNetworksPageCheckpoints
    from modules.ui_extra_networks_styles import ExtraNetworksPageStyles
    from modules.ui_extra_networks_vae import ExtraNetworksPageVAEs
    debug('EN register-pages')
    register_page(ExtraNetworksPageCheckpoints())
    register_page(ExtraNetworksPageStyles())
    register_page(ExtraNetworksPageTextualInversion())
    register_page(ExtraNetworksPageHypernetworks())
    register_page(ExtraNetworksPageVAEs())


def get_pages(title=None):
    pages = []
    if 'All' in shared.opts.extra_networks:
        pages = shared.extra_networks
    else:
        titles = [page.title for page in shared.extra_networks]
        if title is None:
            for page in shared.opts.extra_networks:
                try:
                    idx = titles.index(page)
                    pages.append(shared.extra_networks[idx])
                except ValueError:
                    continue
        else:
            try:
                idx = titles.index(title)
                pages.append(shared.extra_networks[idx])
            except ValueError:
                pass
    return pages


class ExtraNetworksUi:
    def __init__(self):
        self.tabname: str = None
        self.pages: list[str] = None
        self.visible: gr.State = None
        self.state: gr.Textbox = None
        self.details: gr.Group = None
        self.details_tabs: gr.Group = None
        self.details_text: gr.Group = None
        self.tabs: gr.Tabs = None
        self.gallery: gr.Gallery = None
        self.description: gr.Textbox = None
        self.search: gr.Textbox = None
        self.button_details: gr.Button = None
        self.button_refresh: gr.Button = None
        self.button_scan: gr.Button = None
        self.button_view: gr.Button = None
        self.button_quicksave: gr.Button = None
        self.button_save: gr.Button = None
        self.button_sort: gr.Button = None
        self.button_apply: gr.Button = None
        self.button_close: gr.Button = None
        self.button_model: gr.Checkbox = None
        self.details_components: list = []
        self.last_item: dict = None
        self.last_page: ExtraNetworksPage = None
        self.state: gr.State = None


def create_ui(container, button_parent, tabname, skip_indexing = False):
    debug(f'EN create-ui: {tabname}')
    ui = ExtraNetworksUi()
    ui.tabname = tabname
    ui.pages = []
    ui.state = gr.Textbox('{}', elem_id=f"{tabname}_extra_state", visible=False)
    ui.visible = gr.State(value=False) # pylint: disable=abstract-class-instantiated
    ui.details = gr.Group(elem_id=f"{tabname}_extra_details", elem_classes=["extra-details"], visible=False)
    ui.tabs = gr.Tabs(elem_id=f"{tabname}_extra_tabs")
    ui.button_details = gr.Button('Details', elem_id=f"{tabname}_extra_details_btn", visible=False)
    state = {}
    if shared.cmd_opts.profile:
        import cProfile
        pr = cProfile.Profile()
        pr.enable()

    def get_item(state, params = None):
        if params is not None and type(params) == dict:
            page = next(iter([x for x in get_pages() if x.title == 'Style']), None)
            item = page.create_style(params)
        else:
            if state is None or not hasattr(state, 'page') or not hasattr(state, 'item'):
                return None, None
            page = next(iter([x for x in get_pages() if x.title == state.page]), None)
            if page is None:
                return None, None
            item = next(iter([x for x in page.items if x["name"] == state.item]), None)
            if item is None:
                return page, None
        item = SimpleNamespace(**item)
        ui.last_item = item
        ui.last_page = page
        return page, item

    # main event that is triggered when js updates state text field with json values, used to communicate js -> python
    def state_change(state_text):
        try:
            nonlocal state
            state = SimpleNamespace(**json.loads(state_text))
        except Exception as e:
            shared.log.error(f'Extra networks state error: {e}')
            return
        _page, _item = get_item(state)
        # shared.log.debug(f'Extra network: op={state.op} page={page.title if page is not None else None} item={item.filename if item is not None else None}')

    def toggle_visibility(is_visible):
        is_visible = not is_visible
        return is_visible, gr.update(visible=is_visible), gr.update(variant=("secondary-down" if is_visible else "secondary"))

    with ui.details:
        details_close = ToolButton(symbols.close, elem_id=f"{tabname}_extra_details_close", elem_classes=['extra-details-close'])
        details_close.click(fn=lambda: gr.update(visible=False), inputs=[], outputs=[ui.details])
        with gr.Row():
            with gr.Column(scale=1):
                text = gr.HTML('<div>title</div>')
                ui.details_components.append(text)
            with gr.Column(scale=1):
                img = gr.Gallery(value=None, show_label=False, interactive=False, container=False, show_download_button=False, preview=True, elem_id=f"{tabname}_extra_details_img", elem_classes=['extra-details-img'])
                ui.details_components.append(img)
                with gr.Row():
                    btn_save_img = gr.Button('Replace', elem_classes=['small-button'])
                    btn_delete_img = gr.Button('Delete', elem_classes=['small-button'])
        with gr.Group(elem_id=f"{tabname}_extra_details_tabs", visible=False) as ui.details_tabs:
            with gr.Tabs():
                with gr.Tab('Description', elem_classes=['extra-details-tabs']):
                    desc = gr.Textbox('', show_label=False, lines=8, placeholder="Extra network description...")
                    ui.details_components.append(desc)
                    with gr.Row():
                        btn_save_desc = gr.Button('Save', elem_classes=['small-button'], elem_id=f'{tabname}_extra_details_save_desc')
                        btn_delete_desc = gr.Button('Delete', elem_classes=['small-button'], elem_id=f'{tabname}_extra_details_delete_desc')
                        btn_close_desc = gr.Button('Close', elem_classes=['small-button'], elem_id=f'{tabname}_extra_details_close_desc')
                        btn_close_desc.click(fn=lambda: gr.update(visible=False), _js='refeshDetailsEN', inputs=[], outputs=[ui.details])
                with gr.Tab('Model metadata', elem_classes=['extra-details-tabs']):
                    info = gr.JSON({}, show_label=False)
                    ui.details_components.append(info)
                    with gr.Row():
                        btn_save_info = gr.Button('Save', elem_classes=['small-button'], elem_id=f'{tabname}_extra_details_save_info')
                        btn_delete_info = gr.Button('Delete', elem_classes=['small-button'], elem_id=f'{tabname}_extra_details_delete_info')
                        btn_close_info = gr.Button('Close', elem_classes=['small-button'], elem_id=f'{tabname}_extra_details_close_info')
                        btn_close_info.click(fn=lambda: gr.update(visible=False), _js='refeshDetailsEN', inputs=[], outputs=[ui.details])
                with gr.Tab('Embedded metadata', elem_classes=['extra-details-tabs']):
                    meta = gr.JSON({}, show_label=False)
                    ui.details_components.append(meta)
        with gr.Group(elem_id=f"{tabname}_extra_details_text", elem_classes=["extra-details-text"], visible=False) as ui.details_text:
            description = gr.Textbox(label='Description', lines=1, placeholder="Style description...")
            prompt = gr.Textbox(label='Prompt', lines=2, placeholder="Prompt...")
            negative = gr.Textbox(label='Negative prompt', lines=2, placeholder="Negative prompt...")
            extra = gr.Textbox(label='Parameters', lines=2, placeholder="Generation parameters overrides...")
            wildcards = gr.Textbox(label='Wildcards', lines=2, placeholder="Wildcard prompt replacements...")
            ui.details_components += [description, prompt, negative, extra, wildcards]
            with gr.Row():
                btn_save_style = gr.Button('Save', elem_classes=['small-button'], elem_id=f'{tabname}_extra_details_save_style')
                btn_delete_style = gr.Button('Delete', elem_classes=['small-button'], elem_id=f'{tabname}_extra_details_delete_style')
                btn_close_style = gr.Button('Close', elem_classes=['small-button'], elem_id=f'{tabname}_extra_details_close_style')
                btn_close_style.click(fn=lambda: gr.update(visible=False), _js='refeshDetailsEN', inputs=[], outputs=[ui.details])

    with ui.tabs:
        def ui_tab_change(page):
            scan_visible = page in ['Model', 'Lora', 'Hypernetwork', 'Embedding']
            save_visible = page in ['Style']
            model_visible = page in ['Model']
            return [gr.update(visible=scan_visible), gr.update(visible=save_visible), gr.update(visible=model_visible)]

        ui.button_refresh = ToolButton(symbols.refresh, elem_id=f"{tabname}_extra_refresh")
        ui.button_scan = ToolButton(symbols.scan, elem_id=f"{tabname}_extra_scan", visible=True)
        ui.button_quicksave = ToolButton(symbols.book, elem_id=f"{tabname}_extra_quicksave", visible=False)
        ui.button_save = ToolButton(symbols.book, elem_id=f"{tabname}_extra_save", visible=False)
        ui.button_sort = ToolButton(symbols.sort, elem_id=f"{tabname}_extra_sort", visible=True)
        ui.button_view = ToolButton(symbols.view, elem_id=f"{tabname}_extra_view", visible=True)
        ui.button_close = ToolButton(symbols.close, elem_id=f"{tabname}_extra_close", visible=True)
        ui.button_model = ToolButton(symbols.refine, elem_id=f"{tabname}_extra_model", visible=True)
        ui.search = gr.Textbox('', show_label=False, elem_id=f"{tabname}_extra_search", placeholder="Search...", elem_classes="textbox", lines=2, container=False)
        ui.description = gr.Textbox('', show_label=False, elem_id=f"{tabname}_description", elem_classes=["textbox", "extra-description"], lines=2, interactive=False, container=False)

        if ui.tabname == 'txt2img': # refresh only once
            global refresh_time # pylint: disable=global-statement
            refresh_time = time.time()
        if not skip_indexing:
            import concurrent
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                for page in get_pages():
                    executor.submit(page.create_items, ui.tabname)
        for page in get_pages():
            page.create_page(ui.tabname, skip_indexing)
            with gr.Tab(page.title, id=page.title.lower().replace(" ", "_"), elem_classes="extra-networks-tab") as tab:
                page_html = gr.HTML(page.html, elem_id=f'{tabname}{page.name}_extra_page', elem_classes="extra-networks-page")
                ui.pages.append(page_html)
                tab.select(ui_tab_change, _js="getENActivePage", inputs=[ui.button_details], outputs=[ui.button_scan, ui.button_save, ui.button_model])
    if shared.cmd_opts.profile:
        errors.profile(pr, 'ExtraNetworks')
        pr.disable()
        # ui.tabs.change(fn=ui_tab_change, inputs=[], outputs=[ui.button_scan, ui.button_save])

    def fn_save_img(image):
        if ui.last_item is None or ui.last_item.local_preview is None:
            return 'html/card-no-preview.png'
        images = list(ui.gallery.temp_files) # gallery cannot be used as input component so looking at most recently registered temp files
        if len(images) < 1:
            shared.log.warning(f'Extra network no image: item={ui.last_item.name}')
            return 'html/card-no-preview.png'
        try:
            images.sort(key=lambda f: os.path.getmtime(f), reverse=True)
            image = Image.open(images[0])
        except Exception as e:
            shared.log.error(f'Extra network error opening image: item={ui.last_item.name} {e}')
            return 'html/card-no-preview.png'
        fn_delete_img(image)
        if image.width > 512 or image.height > 512:
            image = image.convert('RGB')
            image.thumbnail((512, 512), Image.Resampling.HAMMING)
        try:
            image.save(ui.last_item.local_preview, quality=50)
            shared.log.debug(f'Extra network save image: item={ui.last_item.name} filename="{ui.last_item.local_preview}"')
        except Exception as e:
            shared.log.error(f'Extra network save image: item={ui.last_item.name} filename="{ui.last_item.local_preview}" {e}')
        return image

    def fn_delete_img(_image):
        preview_extensions = ["jpg", "jpeg", "png", "webp", "tiff", "jp2"]
        fn = os.path.splitext(ui.last_item.filename)[0]
        for file in [f'{fn}{mid}{ext}' for ext in preview_extensions for mid in ['.thumb.', '.preview.', '.']]:
            if os.path.exists(file):
                os.remove(file)
                shared.log.debug(f'Extra network delete image: item={ui.last_item.name} filename="{file}"')
        return 'html/card-no-preview.png'

    def fn_save_desc(desc):
        if hasattr(ui.last_item, 'type') and ui.last_item.type == 'Style':
            params = ui.last_page.parse_desc(desc)
            if params is not None:
                fn_save_info(params)
        else:
            fn = os.path.splitext(ui.last_item.filename)[0] + '.txt'
            with open(fn, 'w', encoding='utf-8') as f:
                f.write(desc)
            shared.log.debug(f'Extra network save desc: item={ui.last_item.name} filename="{fn}"')
        return desc

    def fn_delete_desc(desc):
        if ui.last_item is None:
            return desc
        fn = os.path.splitext(ui.last_item.filename)[0] + '.txt'
        if os.path.exists(fn):
            shared.log.debug(f'Extra network delete desc: item={ui.last_item.name} filename="{fn}"')
            os.remove(fn)
            return ''
        return desc

    def fn_save_info(info):
        fn = os.path.splitext(ui.last_item.filename)[0] + '.json'
        shared.writefile(info, fn, silent=True)
        shared.log.debug(f'Extra network save info: item={ui.last_item.name} filename="{fn}"')
        return info

    def fn_delete_info(info):
        if ui.last_item is None:
            return info
        fn = os.path.splitext(ui.last_item.filename)[0] + '.json'
        if os.path.exists(fn):
            shared.log.debug(f'Extra network delete info: item={ui.last_item.name} filename="{fn}"')
            os.remove(fn)
            return ''
        return info

    def fn_save_style(info, description, prompt, negative, extra, wildcards):
        if not isinstance(info, dict) or isinstance(info, list):
            shared.log.warning(f'Extra network save style skip: item={ui.last_item.name} not a dict: {type(info)}')
            return info
        if ui.last_item is None:
            return info
        fn = os.path.splitext(ui.last_item.filename)[0] + '.json'
        if hasattr(ui.last_item, 'type') and ui.last_item.type == 'Style':
            info.update(**{ 'description': description, 'prompt': prompt, 'negative': negative, 'extra': extra, 'wildcards': wildcards })
            shared.writefile(info, fn, silent=True)
            shared.log.debug(f'Extra network save style: item={ui.last_item.name} filename="{fn}"')
        return info

    def fn_delete_style(info):
        if ui.last_item is None:
            return info
        fn = os.path.splitext(ui.last_item.filename)[0] + '.json'
        if os.path.exists(fn):
            shared.log.debug(f'Extra network delete style: item={ui.last_item.name} filename="{fn}"')
            os.remove(fn)
            return {}
        return info

    btn_save_img.click(fn=fn_save_img, _js='closeDetailsEN', inputs=[img], outputs=[img])
    btn_delete_img.click(fn=fn_delete_img, _js='closeDetailsEN', inputs=[img], outputs=[img])
    btn_save_desc.click(fn=fn_save_desc, _js='closeDetailsEN', inputs=[desc], outputs=[desc])
    btn_delete_desc.click(fn=fn_delete_desc, _js='closeDetailsEN', inputs=[desc], outputs=[desc])
    btn_save_info.click(fn=fn_save_info, _js='closeDetailsEN', inputs=[info], outputs=[info])
    btn_delete_info.click(fn=fn_delete_info, _js='closeDetailsEN', inputs=[info], outputs=[info])
    btn_save_style.click(fn=fn_save_style, _js='closeDetailsEN', inputs=[info, description, prompt, negative, extra, wildcards], outputs=[info])
    btn_delete_style.click(fn=fn_delete_style, _js='closeDetailsEN', inputs=[info], outputs=[info])

    def show_details(text, img, desc, info, meta, description, prompt, negative, parameters, wildcards, params, _dummy1=None, _dummy2=None):
        page, item = get_item(state, params)
        if item is not None and hasattr(item, 'name'):
            stat = os.stat(item.filename) if os.path.exists(item.filename) else None
            desc = item.description
            fullinfo = shared.readfile(os.path.splitext(item.filename)[0] + '.json', silent=True)
            if 'modelVersions' in fullinfo: # sanitize massive objects
                fullinfo['modelVersions'] = []
            info = fullinfo
            meta = page.metadata.get(item.name, {}) or {}
            if type(meta) is str:
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            if ui.last_item.preview.startswith('data:'):
                b64str = ui.last_item.preview.split(',',1)[1]
                img = [Image.open(io.BytesIO(base64.b64decode(b64str)))]
            elif hasattr(item, 'local_preview') and os.path.exists(item.local_preview):
                img = [item.local_preview]
            else:
                img = page.find_preview_files(item.filename)
            lora = ''
            model = ''
            style = ''
            note = ''
            if not os.path.exists(item.filename):
                note = f'<br>Target filename: {item.filename}'
            if page.title == 'Model':
                merge = len(list(meta.get('sd_merge_models', {})))
                if merge > 0:
                    model += f'<tr><td>Merge models</td><td>{merge} recipes</td></tr>'
                if meta.get('modelspec.architecture', None) is not None:
                    model += f'''
                        <tr><td>Architecture</td><td>{meta.get('modelspec.architecture', 'N/A')}</td></tr>
                        <tr><td>Title</td><td>{meta.get('modelspec.title', 'N/A')}</td></tr>
                        <tr><td>Resolution</td><td>{meta.get('modelspec.resolution', 'N/A')}</td></tr>
                    '''
            if page.title == 'Lora':
                try:
                    tags = getattr(item, 'tags', {})
                    tags = [f'{name}:{tags[name]}' for i, name in enumerate(tags)]
                    tags = ' '.join(tags)
                except Exception:
                    tags = ''
                try:
                    triggers = ' '.join(info.get('tags', []))
                except Exception:
                    triggers = ''
                lora = f'''
                    <tr><td>Model tags</td><td>{tags}</td></tr>
                    <tr><td>User tags</td><td>{triggers}</td></tr>
                    <tr><td>Base model</td><td>{meta.get('ss_sd_model_name', 'N/A')}</td></tr>
                    <tr><td>Resolution</td><td>{meta.get('ss_resolution', 'N/A')}</td></tr>
                    <tr><td>Training images</td><td>{meta.get('ss_num_train_images', 'N/A')}</td></tr>
                    <tr><td>Comment</td><td>{meta.get('ss_training_comment', 'N/A')}</td></tr>
                '''
            if page.title == 'Style':
                description = item.description
                prompt = item.prompt
                negative = item.negative
                parameters = item.extra
                wildcards = item.wildcards
                style = f'''
                    <tr><td>Name</td><td>{item.name}</td></tr>
                    <tr><td>Description</td><td>{item.description}</td></tr>
                    <tr><td>Preview Embedded</td><td>{item.preview.startswith('data:')}</td></tr>
                '''
                # desc = f'Name: {os.path.basename(item.name)}\nDescription: {item.description}\nPrompt: {item.prompt}\nNegative: {item.negative}\nExtra: {item.extra}\n'
            text = f'''
                <h2 style="border-bottom: 1px solid var(--button-primary-border-color); margin: 0em 0px 1em 0 !important">{item.name}</h2>
                <table style="width: 100%; line-height: 1.3em;"><tbody>
                    <tr><td>Type</td><td>{page.title}</td></tr>
                    <tr><td>Alias</td><td>{getattr(item, 'alias', 'N/A')}</td></tr>
                    <tr><td>Filename</td><td>{item.filename}</td></tr>
                    <tr><td>Hash</td><td>{getattr(item, 'hash', 'N/A')}</td></tr>
                    <tr><td>Size</td><td>{round(stat.st_size/1024/1024, 2) if stat is not None else 'N/A'} MB</td></tr>
                    <tr><td>Last modified</td><td>{datetime.fromtimestamp(stat.st_mtime) if stat is not None else 'N/A'}</td></tr>
                    <tr><td style="border-top: 1px solid var(--button-primary-border-color);"></td><td></td></tr>
                    {lora}
                    {model}
                    {style}
                </tbody></table>
                {note}
            '''
        return [
            text, # gr.html
            img, # gr.image
            desc, # gr.textbox
            info, # gr.json
            meta, # gr.json
            description, # gr.textbox
            prompt, # gr.textbox
            negative, # gr.textbox
            parameters, # gr.textbox
            wildcards, # gr.textbox
            gr.update(visible=item is not None), # details ui visible
            gr.update(visible=page is not None and page.title != 'Style'), # details ui tabs visible
            gr.update(visible=page is not None and page.title == 'Style'), # details ui text visible
        ]

    def ui_refresh_click(title):
        pages = []
        for page in get_pages():
            if title is None or title == '' or title == page.title or len(page.html) == 0:
                page.page_time = 0
                page.refresh_time = 0
                page.refresh()
                page.create_page(ui.tabname)
                shared.log.debug(f"Refreshing Extra networks: page='{page.title}' items={len(page.items)} tab={ui.tabname}")
            pages.append(page.html)
        ui.search.update(value = ui.search.value)
        return pages

    def ui_view_cards(title):
        pages = []
        for page in get_pages():
            if title is None or title == '' or title == page.title or len(page.html) == 0:
                shared.opts.extra_networks_view = page.view
                page.view = 'gallery' if page.view == 'list' else 'list'
                page.card = card_full if page.view == 'gallery' else card_list
                page.html = ''
                page.create_page(ui.tabname)
                shared.log.debug(f"Refreshing Extra networks: page='{page.title}' items={len(page.items)} tab={ui.tabname} view={page.view}")
            pages.append(page.html)
        ui.search.update(value = ui.search.value)
        return pages

    def ui_scan_click(title):
        from modules import ui_models
        if ui_models.search_metadata_civit is not None:
            ui_models.search_metadata_civit(True, title)
        return ui_refresh_click(title)

    def ui_save_click():
        from modules import generation_parameters_copypaste
        filename = os.path.join(paths.data_path, "params.txt")
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf8") as file:
                prompt = file.read()
        else:
            prompt = ''
        params = generation_parameters_copypaste.parse_generation_parameters(prompt)
        res = show_details(text=None, img=None, desc=None, info=None, meta=None, parameters=None, description=None, prompt=None, negative=None, wildcards=None, params=params)
        return res

    def ui_quicksave_click(name):
        from modules import generation_parameters_copypaste
        fn = os.path.join(paths.data_path, "params.txt")
        if os.path.exists(fn):
            with open(fn, "r", encoding="utf8") as file:
                prompt = file.read()
        else:
            prompt = ''
        params = generation_parameters_copypaste.parse_generation_parameters(prompt)
        fn = os.path.join(shared.opts.styles_dir, os.path.splitext(name)[0] + '.json')
        prompt = params.get('Prompt', '')
        item = {
            "name": name,
            "description": '',
            "prompt": prompt,
            "negative": params.get('Negative prompt', ''),
            "extra": '',
        }
        shared.writefile(item, fn, silent=True)
        if len(prompt) > 0:
            shared.log.debug(f"Extra network quick save style: item={name} filename='{fn}'")
        else:
            shared.log.warning(f"Extra network quick save model: item={name} filename='{fn}' prompt is empty")

    def ui_sort_cards(msg):
        shared.log.debug(f'Extra networks: {msg}')
        return msg

    dummy = gr.State(value=False) # pylint: disable=abstract-class-instantiated
    button_parent.click(fn=toggle_visibility, inputs=[ui.visible], outputs=[ui.visible, container, button_parent])
    ui.button_close.click(fn=toggle_visibility, inputs=[ui.visible], outputs=[ui.visible, container])
    ui.button_sort.click(fn=ui_sort_cards, _js='sortExtraNetworks', inputs=[ui.search], outputs=[ui.description])
    ui.button_view.click(fn=ui_view_cards, inputs=[ui.search], outputs=ui.pages)
    ui.button_refresh.click(fn=ui_refresh_click, _js='getENActivePage', inputs=[ui.search], outputs=ui.pages)
    ui.button_scan.click(fn=ui_scan_click, _js='getENActivePage', inputs=[ui.search], outputs=ui.pages)
    ui.button_save.click(fn=ui_save_click, inputs=[], outputs=ui.details_components + [ui.details])
    ui.button_quicksave.click(fn=ui_quicksave_click, _js="() => prompt('Prompt name', '')", inputs=[ui.search], outputs=[])
    ui.button_details.click(show_details, _js="getCardDetails", inputs=ui.details_components + [dummy, dummy, dummy], outputs=ui.details_components + [ui.details, ui.details_tabs, ui.details_text])
    ui.state.change(state_change, inputs=[ui.state], outputs=[])
    return ui


def setup_ui(ui, gallery):
    ui.gallery = gallery
