import os
import time
import shutil
import importlib
from typing import Dict, List, Union, Callable
from urllib.parse import urlparse
import PIL.Image as Image
import rich.progress as p
from modules import shared, errors
from modules.upscaler import Upscaler, UpscalerLanczos, UpscalerNearest, UpscalerNone
from modules.paths import script_path, models_path

diffuser_repos = []

def walk(top, onerror:Callable=None, *, recurse:Union[bool,Callable]=True):
    # A near-exact copy of `os.path.walk()`, trimmed slightly. Probably not nessesary for most people's collections, but makes a difference on really large datasets.
    nondirs = []
    walk_dirs = []
    try:
        scandir_it = os.scandir(top)
    except OSError as error:
        if callable(onerror):
            onerror(error, top)
        return
    with scandir_it:
        while True:
            try:
                try:
                    entry = next(scandir_it)
                except StopIteration:
                    break
            except OSError as error:
                if callable(onerror):
                    onerror(error, top)
                return
            try:
                is_dir = entry.is_dir()
            except OSError:
                is_dir = False
            if not is_dir:
                nondirs.append(entry.name)
            else:
                try:
                    if entry.is_symlink() and not os.path.exists(entry.path):
                        raise NotADirectoryError('Broken Symlink')
                    walk_dirs.append(entry.path)
                except OSError as error:
                    if callable(onerror):
                        onerror(error, entry.path)
    if recurse:
        # Recurse into sub-directories
        for new_path in walk_dirs:
            if os.path.basename(new_path).startswith('models--'):
                continue
            if callable(recurse) and not recurse(new_path):
                continue
            yield from walk(new_path, onerror, recurse=recurse)
    # Yield after recursion if going bottom up
    yield top, nondirs, walk_dirs


def download_civit_meta(model_path: str, model_id):
    fn = os.path.splitext(model_path)[0] + '.json'
    if os.path.exists(fn):
        return ''
    url = f'https://civitai.com/api/v1/models/{model_id}'
    r = shared.req(url)
    if r.status_code == 200:
        try:
            shared.writefile(r.json(), fn, silent=True)
            msg = f'CivitAI download: id={model_id} url={url} file={fn}'
            shared.log.info(msg)
            return msg
        except Exception as e:
            msg = f'CivitAI download error: id={model_id} url={url} file={fn} {e}'
            shared.log.error(msg)
            return msg
    return ''

def download_civit_preview(model_path: str, preview_url: str):
    ext = os.path.splitext(preview_url)[1]
    preview_file = os.path.splitext(model_path)[0] + ext
    if os.path.exists(preview_file):
        return ''
    res = f'CivitAI download: url={preview_url} file={preview_file}'
    r = shared.req(preview_url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    block_size = 16384 # 16KB blocks
    written = 0
    img = None
    shared.state.begin('civitai-download-preview')
    try:
        with open(preview_file, 'wb') as f:
            with p.Progress(p.TextColumn('[cyan]{task.description}'), p.DownloadColumn(), p.BarColumn(), p.TaskProgressColumn(), p.TimeRemainingColumn(), p.TimeElapsedColumn(), p.TransferSpeedColumn(), console=shared.console) as progress:
                task = progress.add_task(description="Download starting", total=total_size)
                for data in r.iter_content(block_size):
                    written = written + len(data)
                    f.write(data)
                    progress.update(task, advance=block_size, description="Downloading")
        if written < 1024: # min threshold
            os.remove(preview_file)
            raise ValueError(f'removed invalid download: bytes={written}')
        img = Image.open(preview_file)
    except Exception as e:
        os.remove(preview_file)
        res += f' error={e}'
        shared.log.error(f'CivitAI download error: url={preview_url} file={preview_file} {e}')
    shared.state.end()
    if img is None:
        return res
    shared.log.info(f'{res} size={total_size} image={img.size}')
    img.close()
    return res


download_pbar = None

def download_civit_model_thread(model_name, model_url, model_path, model_type, preview):
    import hashlib
    sha256 = hashlib.sha256()
    sha256.update(model_name.encode('utf-8'))
    temp_file = sha256.hexdigest()[:8] + '.tmp'

    if model_type == 'LoRA':
        model_file = os.path.join(shared.opts.lora_dir, model_path, model_name)
        temp_file = os.path.join(shared.opts.lora_dir, model_path, temp_file)
    else:
        model_file = os.path.join(shared.opts.ckpt_dir, model_path, model_name)
        temp_file = os.path.join(shared.opts.ckpt_dir, model_path, temp_file)

    res = f'CivitAI download: name={model_name} url={model_url} path={model_path} temp={temp_file}'
    if os.path.isfile(model_file):
        res += ' already exists'
        shared.log.warning(res)
        return res

    headers = {}
    starting_pos = 0
    if os.path.isfile(temp_file):
        starting_pos = os.path.getsize(temp_file)
        res += f' resume={round(starting_pos/1024/1024)}Mb'
        headers = {'Range': f'bytes={starting_pos}-'}

    r = shared.req(model_url, headers=headers, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    res += f' size={round((starting_pos + total_size)/1024/1024)}Mb'
    shared.log.info(res)
    shared.state.begin('civitai-download-model')
    block_size = 16384 # 16KB blocks
    written = starting_pos
    global download_pbar # pylint: disable=global-statement
    if download_pbar is None:
        download_pbar = p.Progress(p.TextColumn('[cyan]{task.description}'), p.DownloadColumn(), p.BarColumn(), p.TaskProgressColumn(), p.TimeRemainingColumn(), p.TimeElapsedColumn(), p.TransferSpeedColumn(), p.TextColumn('[cyan]{task.fields[name]}'), console=shared.console)
    with download_pbar:
        task = download_pbar.add_task(description="Download starting", total=starting_pos+total_size, name=model_name)
        try:
            with open(temp_file, 'ab') as f:
                for data in r.iter_content(block_size):
                    written = written + len(data)
                    f.write(data)
                    download_pbar.update(task, description="Download", completed=written)
            if written < 1024 * 1024: # min threshold
                os.remove(temp_file)
                raise ValueError(f'removed invalid download: bytes={written}')
            if preview is not None:
                preview_file = os.path.splitext(model_file)[0] + '.jpg'
                preview.save(preview_file)
                res += f' preview={preview_file}'
        except Exception as e:
            shared.log.error(f'{res} {e}')
        finally:
            download_pbar.stop_task(task)
            download_pbar.remove_task(task)
    if starting_pos+total_size != written:
        shared.log.warning(f'{res} written={round(written/1024/1024)}Mb incomplete download')
    else:
        os.rename(temp_file, model_file)
    shared.state.end()
    return res


def download_civit_model(model_url: str, model_name: str, model_path: str, model_type: str, preview):
    import threading
    thread = threading.Thread(target=download_civit_model_thread, args=(model_name, model_url, model_path, model_type, preview))
    thread.start()
    return f'CivitAI download: name={model_name} url={model_url} path={model_path}'


def download_diffusers_model(hub_id: str, cache_dir: str = None, download_config: Dict[str, str] = None, token = None, variant = None, revision = None, mirror = None, custom_pipeline = None):
    if hub_id is None or len(hub_id) == 0:
        return None
    from diffusers import DiffusionPipeline
    import huggingface_hub as hf
    shared.state.begin('huggingface-download-model')
    if download_config is None:
        download_config = {
            "force_download": False,
            "resume_download": True,
            "cache_dir": shared.opts.diffusers_dir,
            "load_connected_pipeline": True,
        }
    if cache_dir is not None:
        download_config["cache_dir"] = cache_dir
    if variant is not None and len(variant) > 0:
        download_config["variant"] = variant
    if revision is not None and len(revision) > 0:
        download_config["revision"] = revision
    if mirror is not None and len(mirror) > 0:
        download_config["mirror"] = mirror
    if custom_pipeline is not None and len(custom_pipeline) > 0:
        download_config["custom_pipeline"] = custom_pipeline
    shared.log.debug(f"Diffusers downloading: {hub_id} {download_config}")
    if token is not None and len(token) > 2:
        shared.log.debug(f"Diffusers authentication: {token}")
        hf.login(token)
    pipeline_dir = None

    ok = True
    try:
        pipeline_dir = DiffusionPipeline.download(hub_id, **download_config)
    except Exception as e:
        ok = False
        shared.log.warning(f"Diffusers download error: {hub_id} {e}")
    if not ok:
        try:
            download_config.pop('load_connected_pipeline')
            download_config.pop('variant')
            pipeline_dir = hf.snapshot_download(hub_id, **download_config)
        except Exception as e:
            shared.log.warning(f"Diffusers hub download error: {hub_id} {e}")

    if pipeline_dir is None:
        shared.log.error(f"Diffusers no pipeline folder: {hub_id}")
        return None
    try:
        # TODO diffusers is this real error?
        model_info_dict = hf.model_info(hub_id).cardData if pipeline_dir is not None else None # pylint: disable=no-member
    except Exception:
        model_info_dict = None
    if model_info_dict is not None and "prior" in model_info_dict: # some checkpoints need to be downloaded as "hidden" as they just serve as pre- or post-pipelines of other pipelines
        download_dir = DiffusionPipeline.download(model_info_dict["prior"][0], **download_config)
        model_info_dict["prior"] = download_dir
        with open(os.path.join(download_dir, "hidden"), "w", encoding="utf-8") as f: # mark prior as hidden
            f.write("True")
    if pipeline_dir is not None:
        shared.writefile(model_info_dict, os.path.join(pipeline_dir, "model_info.json"))
    shared.state.end()
    return pipeline_dir


def load_diffusers_models(model_path: str, command_path: str = None):
    t0 = time.time()
    places = []
    places.append(model_path)
    if command_path is not None and command_path != model_path:
        places.append(command_path)
    diffuser_repos.clear()
    output = []
    for place in places:
        if not os.path.isdir(place):
            continue
        try:
            """
            import huggingface_hub as hf
            res = hf.scan_cache_dir(cache_dir=place)
            for r in list(res.repos):
                cache_path = os.path.join(r.repo_path, "snapshots", list(r.revisions)[-1].commit_hash)
                diffuser_repos.append({ 'name': r.repo_id, 'filename': r.repo_id, 'path': cache_path, 'size': r.size_on_disk, 'mtime': r.last_modified, 'hash': list(r.revisions)[-1].commit_hash, 'model_info': str(os.path.join(cache_path, "model_info.json")) })
                if not os.path.isfile(os.path.join(cache_path, "hidden")):
                    output.append(str(r.repo_id))
            """
            for folder in os.listdir(place):
                try:
                    if "--" not in folder:
                        continue
                    _, name = folder.split("--", maxsplit=1)
                    name = name.replace("--", "/")
                    snapshots = os.listdir(os.path.join(place, folder, "snapshots"))
                    if len(snapshots) == 0:
                        shared.log.warning(f"Diffusers folder has no snapshots: location={place} folder={folder} name={name}")
                        continue
                    commit = snapshots[-1]
                    folder = os.path.join(place, folder, 'snapshots', commit)
                    mtime = os.path.getmtime(folder)
                    info = os.path.join(folder, "model_info.json")
                    diffuser_repos.append({ 'name': name, 'filename': name, 'path': folder, 'hash': commit, 'mtime': mtime, 'model_info': info })
                    if os.path.exists(os.path.join(folder, 'hidden')):
                        continue
                    output.append(name)
                except Exception as e:
                    shared.log.error(f"Error analyzing diffusers model: {place}/{folder} {e}")
        except Exception as e:
            shared.log.error(f"Error listing diffusers: {place} {e}")
    shared.log.debug(f'Scanning diffusers cache: {model_path} {command_path} items={len(output)} time={time.time()-t0:.2f}s')
    return output


def find_diffuser(name: str):
    import huggingface_hub as hf
    if name in diffuser_repos:
        return name
    if shared.cmd_opts.no_download:
        return None
    hf_api = hf.HfApi()
    hf_filter = hf.ModelFilter(
        model_name=name,
        # task='text-to-image',
        library=['diffusers'],
    )
    models = list(hf_api.list_models(filter=hf_filter, full=True, limit=20, sort="downloads", direction=-1))
    shared.log.debug(f'Searching diffusers models: {name} {len(models) > 0}')
    if len(models) > 0:
        return models[0].modelId
    return None


modelloader_directories = {}
cache_last = 0
cache_time = 1

def _clean_directory(dir:str, *, recursive:Union[bool,Callable]=True):
    is_dirty = True
    if dir in modelloader_directories:
        if not is_directory(dir):
            is_dirty = True
            _delete_directory()
        else:
            is_dirty = os.path.getmtime(dir) != modelloader_directories[dir][0]
            for _dir in modelloader_directories[dir][2].copy():
                if not is_directory(dir):
                    is_dirty = True
                    _delete_directory(_dir)
                    if _dir in modelloader_directories[dir][2]:
                        modelloader_directories[dir][2].remove(_dir)
                elif recursive and (not callable(recursive) or recursive(_dir)):
                    is_dirty = _clean_directory(_dir) or is_dirty
    return is_dirty


def _delete_directory(dir:str):
    global modelloader_directories
    if dir in modelloader_directories:
        for _dir in modelloader_directories[dir][2]:
            _delete_directory(_dir)
        del modelloader_directories[dir]

def is_directory(dir):
    return dir and os.path.exists(dir) and os.path.isdir(dir)

def directory_has_changed(dir:str, *, recursive:Union[bool,Callable]=True) -> bool: # pylint: disable=redefined-builtin
    try:
        dir = real_path(dir)
        if not is_directory(dir):
            return True
        global modelloader_directories
        if dir not in modelloader_directories:
            return True
        if os.path.getmtime(dir) != modelloader_directories[dir][0]:
            return True
        if recursive:
            return any([directory_has_changed(child, recursive=recursive) for child in modelloader_directories[dir][2] if (not callable(recursive) or recursive(child))])
    except Exception as e:
        shared.log.error(f"Filesystem Error: {e.__class__.__name__}({e})")
        return True
    return False

def refresh_directory(dir:str, *, recursive:Union[bool,Callable]=True) -> dict[str,tuple[float,list[str]]]: # pylint: disable=redefined-builtin
    global modelloader_directories
    res = {}
    t0 = time.time()
    #shared.log.debug(f'reloading dir: {dir}')
    for _dir, _files, _dirs in walk(dir, lambda e, path: shared.log.debug(f"FS walk error: {e} {path}"), recurse=recursive):
        try:
            if _dir in modelloader_directories:
                for dead in modelloader_directories[_dir][2]:
                    if dead not in _dirs:
                        _delete_directory(dead)
            res[_dir] = modelloader_directories[_dir] = (os.path.getmtime(_dir), [os.path.join(_dir, fn) for fn in _files], _dirs)
        except Exception as e:
            shared.log.error(f"Filesystem Error: {e.__class__.__name__}({e})")
            del modelloader_directories[_dir]
    #shared.log.debug(f'reloading dir took {time.time()-t0}s: {dir}')
    return res

def directory_directories(dir:str, *, recursive:Union[bool,Callable]=True) -> dict[str,tuple[float,list[str]]]: # pylint: disable=redefined-builtin
    res = {}
    dir = real_path(dir)
    if not is_directory(dir):
        _delete_directory(dir)
    else:
        global modelloader_directories
        if dir not in modelloader_directories or _clean_directory(dir, recursive=False):
            res = refresh_directory(dir, recursive=recursive)
        elif dir in modelloader_directories:
            res[dir] = modelloader_directories[dir]
            if recursive:
                for _dir in list(modelloader_directories[dir][2]).copy():
                    try:
                        if not os.path.isdir(_dir):
                            _delete_directory(_dir)
                        elif _dir in modelloader_directories and (not callable(recursive) or recursive(_dir)):
                            res.update(directory_directories(_dir, recursive=recursive))
                    except Exception:
                        _delete_directory(_dir)
    return res


def directory_mtime(dir:str, *, recursive:Union[bool,Callable]=True) -> float: # pylint: disable=redefined-builtin
    return float(max(0, *[dat[0] for dat in directory_directories(dir, recursive=recursive).values()]))


def directories_file_paths(directories:dict) -> list[str]:
    return sum([dat[1] for dat in directories.values()],[])


def unique_directories(directories:list[str], *, recursive:Union[bool,Callable]=True) -> list[str]:
    '''Ensure no empty, or duplicates'''
    '''If we are going recursive, then directories that are children of other directories are redundant'''
    directories = [ dir for dir in unique_paths(directories) if os.path.isdir(dir) ]
    if len(directories) > 1 and recursive:
        _directories = sorted([ dir for dir in directories], key=len, reverse=True)
        while _directories:
            dir = _directories.pop()
            _dir = os.path.join(dir, '')
            for child_dir in _directories:
                if not child_dir.startswith(_dir):
                    continue
                remove = bool(recursive)
                if remove and callable(recursive):
                    _child_dir = child_dir[len(_dir):]
                    if _child_dir:
                        _next_dir = dir
                        for sub_dir in _child_dir.split(_dir[:1]):
                            _next_dir = os.path.join(_child_dir, sub_dir)
                            try:
                                if recursive(_next_dir):
                                    continue
                            except Exception:
                                pass
                            remove = False
                            break
                if remove:
                    if child_dir in directories:
                        directories.remove(child_dir)
                    if child_dir in _directories:
                        _directories.remove(child_dir)
    return directories

def real_path(path:str) -> Union[str, None]:
    try:
        return os.path.abspath(os.path.expanduser(path))
    except Exception:
        pass
    return None

def unique_paths(paths:list[str]) -> list[str]:
    return [ k for k in { real_path(fp): True for fp in paths if fp }.keys() if k ]


def directory_files(*directories:list[str], recursive:Union[bool,Callable]=True) -> list[str]:
    return sum([[*directories_file_paths(directory_directories(dir, recursive=recursive))] for dir in unique_directories(directories, recursive=recursive)],[])


def extension_filter(ext_filter=None, ext_blacklist=None):
    if ext_filter:
        ext_filter = [*map(str.upper, ext_filter)]
    if ext_blacklist:
        ext_blacklist = [*map(str.upper, ext_blacklist)]
    def filter(fp:str): # pylint: disable=redefined-builtin
        return (not ext_filter or any(fp.upper().endswith(ew) for ew in ext_filter)) and (not ext_blacklist or not any(fp.upper().endswith(ew) for ew in ext_blacklist))
    return filter


def download_url_to_file(url: str, dst: str):
    # based on torch.hub.download_url_to_file
    import uuid
    import tempfile
    from urllib.request import urlopen, Request
    from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn

    file_size = None
    req = Request(url, headers={"User-Agent": "sdnext"})
    u = urlopen(req) # pylint: disable=R1732
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length") # pylint: disable=R1732
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])
    dst = os.path.expanduser(dst)
    for _seq in range(tempfile.TMP_MAX):
        tmp_dst = dst + '.' + uuid.uuid4().hex + '.partial'
        try:
            f = open(tmp_dst, 'w+b') # pylint: disable=R1732
        except FileExistsError:
            continue
        break
    else:
        shared.log.error('Error downloading: url={url} no usable temporary filename found')
        return
    try:
        with Progress(TextColumn('[cyan]{task.description}'), BarColumn(), TaskProgressColumn(), TimeRemainingColumn(), TimeElapsedColumn(), console=shared.console) as progress:
            task = progress.add_task(description="Downloading", total=file_size)
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                progress.update(task, advance=len(buffer))
        f.close()
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


def load_file_from_url(url: str, *, model_dir: str, progress: bool = True, file_name = None): # pylint: disable=unused-argument
    """Download a file from url into model_dir, using the file present if possible. Returns the path to the downloaded file."""
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        shared.log.info(f'Downloading: url="{url}" file={cached_file}')
        download_url_to_file(url, cached_file)
    return cached_file

def list_files(*paths: List[str], ext_filter=None, ext_blacklist=None, recursive:Union[bool,Callable]=True) -> list:
    return [*filter(extension_filter(ext_filter, ext_blacklist), directory_files(*paths,recursive=recursive))]

def load_models(model_path: str, model_url: str = None, command_path: str = None, ext_filter=None, download_name=None, ext_blacklist=None) -> list:
    """
    A one-and done loader to try finding the desired models in specified directories.
    @param download_name: Specify to download from model_url immediately.
    @param model_url: If no other models are found, this will be downloaded on upscale.
    @param model_path: The location to store/find models in.
    @param command_path: A command-line argument to search for models in first.
    @param ext_filter: An optional list of filename extensions to filter by
    @return: A list of paths containing the desired model(s)
    """
    places = unique_directories([model_path, command_path])
    output = []
    try:
        output:list = list_files(*places, ext_filter=ext_filter, ext_blacklist=ext_blacklist)
        if model_url is not None and len(output) == 0:
            if download_name is not None:
                dl = load_file_from_url(model_url, model_dir=places[0], progress=True, file_name=download_name)
                output.append(dl)
            else:
                output.append(model_url)
    except Exception as e:
        errors.display(e,f"Error listing models: {places}")
    return output


def friendly_name(file: str):
    if "http" in file:
        file = urlparse(file).path
    file = os.path.basename(file)
    model_name, _extension = os.path.splitext(file)
    return model_name


def friendly_fullname(file: str):
    if "http" in file:
        file = urlparse(file).path
    file = os.path.basename(file)
    return file


def cleanup_models():
    # This code could probably be more efficient if we used a tuple list or something to store the src/destinations
    # and then enumerate that, but this works for now. In the future, it'd be nice to just have every "model" scaler
    # somehow auto-register and just do these things...
    root_path = script_path
    src_path = models_path
    dest_path = os.path.join(models_path, "Stable-diffusion")
    # move_files(src_path, dest_path, ".ckpt")
    # move_files(src_path, dest_path, ".safetensors")
    src_path = os.path.join(root_path, "ESRGAN")
    dest_path = os.path.join(models_path, "ESRGAN")
    move_files(src_path, dest_path)
    src_path = os.path.join(models_path, "BSRGAN")
    dest_path = os.path.join(models_path, "ESRGAN")
    move_files(src_path, dest_path, ".pth")
    src_path = os.path.join(root_path, "gfpgan")
    dest_path = os.path.join(models_path, "GFPGAN")
    move_files(src_path, dest_path)
    src_path = os.path.join(root_path, "SwinIR")
    dest_path = os.path.join(models_path, "SwinIR")
    move_files(src_path, dest_path)
    src_path = os.path.join(root_path, "repositories/latent-diffusion/experiments/pretrained_models/")
    dest_path = os.path.join(models_path, "LDSR")
    move_files(src_path, dest_path)
    src_path = os.path.join(root_path, "SCUNet")
    dest_path = os.path.join(models_path, "SCUNet")
    move_files(src_path, dest_path)


def move_files(src_path: str, dest_path: str, ext_filter: str = None):
    try:
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        if os.path.exists(src_path):
            for file in os.listdir(src_path):
                fullpath = os.path.join(src_path, file)
                if os.path.isfile(fullpath):
                    if ext_filter is not None:
                        if ext_filter not in file:
                            continue
                    shared.log.warning(f"Moving {file} from {src_path} to {dest_path}.")
                    try:
                        shutil.move(fullpath, dest_path)
                    except Exception:
                        pass
            if len(os.listdir(src_path)) == 0:
                shared.log.info(f"Removing empty folder: {src_path}")
                shutil.rmtree(src_path, True)
    except Exception:
        pass


def load_upscalers():
    # We can only do this 'magic' method to dynamically load upscalers if they are referenced, so we'll try to import any _model.py files before looking in __subclasses__
    modules_dir = os.path.join(shared.script_path, "modules", "postprocess")
    for file in os.listdir(modules_dir):
        if "_model.py" in file:
            model_name = file.replace("_model.py", "")
            full_model = f"modules.postprocess.{model_name}_model"
            try:
                importlib.import_module(full_model)
            except Exception as e:
                shared.log.error(f'Error loading upscaler: {model_name} {e}')
    datas = []
    commandline_options = vars(shared.cmd_opts)
    # some of upscaler classes will not go away after reloading their modules, and we'll end up with two copies of those classes. The newest copy will always be the last in the list, so we go from end to beginning and ignore duplicates
    used_classes = {}
    for cls in reversed(Upscaler.__subclasses__()):
        classname = str(cls)
        if classname not in used_classes:
            used_classes[classname] = cls
    names = []
    for cls in reversed(used_classes.values()):
        name = cls.__name__
        cmd_name = f"{name.lower().replace('upscaler', '')}_models_path"
        commandline_model_path = commandline_options.get(cmd_name, None)
        scaler = cls(commandline_model_path)
        scaler.user_path = commandline_model_path
        scaler.model_download_path = commandline_model_path or scaler.model_path
        datas += scaler.scalers
        names.append(name[8:])
    shared.sd_upscalers = sorted(datas, key=lambda x: x.name.lower() if not isinstance(x.scaler, (UpscalerNone, UpscalerLanczos, UpscalerNearest)) else "") # Special case for UpscalerNone keeps it at the beginning of the list.
    shared.log.debug(f"Loaded upscalers: total={len(shared.sd_upscalers)} downloaded={len([x for x in shared.sd_upscalers if x.data_path is not None and os.path.isfile(x.data_path)])} user={len([x for x in shared.sd_upscalers if x.custom])} {names}")
