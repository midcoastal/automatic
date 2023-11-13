import os
import json
import mimetypes
from functools import reduce

import gradio as gr
import gradio.routes
import gradio.utils
import numpy as np
from PIL import Image
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call, wrap_gradio_call

from modules import sd_hijack, sd_models, script_callbacks, ui_extensions, deepbooru, extra_networks, ui_common, ui_postprocessing, ui_loadsave, ui_train, ui_models
from modules.ui_components import FormRow, FormGroup, ToolButton, FormHTML
from modules.paths import script_path, data_path
from modules.shared import opts, cmd_opts, log
from modules.dml import directml_override_opts
from modules import prompt_parser
from modules import timer
import modules.ui_symbols as symbols
import modules.generation_parameters_copypaste as parameters_copypaste
import modules.hypernetworks.ui
import modules.scripts
import modules.shared
import modules.errors
import modules.styles
import modules.extras
import modules.theme
import modules.textual_inversion.ui
import modules.sd_samplers

debug = log.env('SD_UI_DEBUG').prefix(__name__)

modules.errors.install()
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')
log = modules.shared.log
ui_system_tabs = None
switch_values_symbol = symbols.switch
detect_image_size_symbol = symbols.detect
paste_symbol = symbols.paste
clear_prompt_symbol = symbols.clear
restore_progress_symbol = symbols.apply
folder_symbol = symbols.folder
extra_networks_symbol = symbols.networks
apply_style_symbol = symbols.apply
save_style_symbol = symbols.save


if not cmd_opts.share and not cmd_opts.listen:
    # fix gradio phoning home
    gradio.utils.version_check = lambda: None
    gradio.utils.get_local_ip_address = lambda: '127.0.0.1'


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


sample_img2img = "assets/stable-samples/img2img/sketch-mountains-input.jpg"
sample_img2img = sample_img2img if os.path.exists(sample_img2img) else None
paste_function = None


def create_output_panel(tabname, outdir): # pylint: disable=unused-argument # outdir is used by extensions
    a, b, c, _d, e = ui_common.create_output_panel(tabname)
    return a, b, c, e

def plaintext_to_html(text): # may be referenced by extensions
    return ui_common.plaintext_to_html(text)

def infotext_to_html(text): # may be referenced by extensions
    return ui_common.infotext_to_html(text)


def send_gradio_gallery_to_image(x):
    if len(x) == 0:
        return None
    return parameters_copypaste.image_from_url_text(x[0])


def add_style(name: str, prompt: str, negative_prompt: str):
    if name is None:
        return [gr_show() for x in range(4)]
    style = modules.styles.Style(name, prompt, negative_prompt)
    modules.shared.prompt_styles.styles[style.name] = style
    modules.shared.prompt_styles.save_styles(modules.shared.opts.styles_dir)
    return [gr.Dropdown.update(visible=True, choices=list(modules.shared.prompt_styles.styles)) for _ in range(2)]


def calc_resolution_hires(width, height, hr_scale, hr_resize_x, hr_resize_y, hr_upscaler):
    from modules import processing, devices
    if hr_upscaler == "None":
        return "Hires resize: None"
    p = processing.StableDiffusionProcessingTxt2Img(width=width, height=height, enable_hr=True, hr_scale=hr_scale, hr_resize_x=hr_resize_x, hr_resize_y=hr_resize_y)
    p.init_hr()
    with devices.autocast():
        p.init([""], [0], [0])
    return f"Hires resize: from <span class='resolution'>{p.width}x{p.height}</span> to <span class='resolution'>{p.hr_resize_x or p.hr_upscale_to_x}x{p.hr_resize_y or p.hr_upscale_to_y}</span>"


def resize_from_to_html(width, height, scale_by):
    target_width = int(width * scale_by)
    target_height = int(height * scale_by)
    if not target_width or not target_height:
        return "Hires resize: no image selected"
    return f"Hires resize: from <span class='resolution'>{width}x{height}</span> to <span class='resolution'>{target_width}x{target_height}</span>"


def apply_styles(prompt, prompt_neg, styles):
    prompt = modules.shared.prompt_styles.apply_styles_to_prompt(prompt, styles)
    prompt_neg = modules.shared.prompt_styles.apply_negative_styles_to_prompt(prompt_neg, styles)
    return [gr.Textbox.update(value=prompt), gr.Textbox.update(value=prompt_neg), gr.Dropdown.update(value=[])]


def parse_style(styles):
    return styles.split('|')


def process_interrogate(interrogation_function, mode, ii_input_files, ii_input_dir, ii_output_dir, *ii_singles):
    if mode in {0, 1, 3, 4}:
        return [interrogation_function(ii_singles[mode]), None]
    if mode == 2:
        return [interrogation_function(ii_singles[mode]["image"]), None]
    if mode == 5:
        if len(ii_input_files) > 0:
            images = [f.name for f in ii_input_files]
        else:
            if not os.path.isdir(ii_input_dir):
                log.error(f"Input directory not found: {ii_input_dir}")
                return [gr.update(), None]
            images = modules.shared.listfiles(ii_input_dir)
        if ii_output_dir != "":
            os.makedirs(ii_output_dir, exist_ok=True)
        else:
            ii_output_dir = ii_input_dir
        for image in images:
            img = Image.open(image)
            filename = os.path.basename(image)
            left, _ = os.path.splitext(filename)
            print(interrogation_function(img), file=open(os.path.join(ii_output_dir, f"{left}.txt"), 'a', encoding='utf-8')) # pylint: disable=consider-using-with
    return [gr.update(), None]


def interrogate(image):
    prompt = modules.shared.interrogator.interrogate(image.convert("RGB"))
    return gr.update() if prompt is None else prompt


def interrogate_deepbooru(image):
    prompt = deepbooru.model.tag(image)
    return gr.update() if prompt is None else prompt


def create_seed_inputs(tab):
    with gr.Accordion(open=False, label="Seed", elem_id=f"{tab}_seed_group", elem_classes=["small-accordion"]):
        with FormRow(elem_id=f"{tab}_seed_row", variant="compact"):
            seed = gr.Number(label='Initial seed', value=-1, elem_id=f"{tab}_seed", container=True)
            random_seed = ToolButton(symbols.random, elem_id=f"{tab}_random_seed", label='Random seed')
            reuse_seed = ToolButton(symbols.reuse, elem_id=f"{tab}_reuse_seed", label='Reuse seed')
        with FormRow(visible=True, elem_id=f"{tab}_subseed_row", variant="compact"):
            subseed = gr.Number(label='Variation seed', value=-1, elem_id=f"{tab}_subseed", container=True)
            random_subseed = ToolButton(symbols.random, elem_id=f"{tab}_random_subseed")
            reuse_subseed = ToolButton(symbols.reuse, elem_id=f"{tab}_reuse_subseed")
            subseed_strength = gr.Slider(label='Variation strength', value=0.0, minimum=0, maximum=1, step=0.01, elem_id=f"{tab}_subseed_strength")
        with FormRow(visible=False):
            seed_resize_from_w = gr.Slider(minimum=0, maximum=4096, step=8, label="Resize seed from width", value=0, elem_id=f"{tab}_seed_resize_from_w")
            seed_resize_from_h = gr.Slider(minimum=0, maximum=4096, step=8, label="Resize seed from height", value=0, elem_id=f"{tab}_seed_resize_from_h")
        random_seed.click(fn=lambda: [-1, -1], show_progress=False, inputs=[], outputs=[seed, subseed])
        random_subseed.click(fn=lambda: -1, show_progress=False, inputs=[], outputs=[subseed])
        return seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w


def connect_clear_prompt(button): # pylint: disable=unused-argument
    pass


def connect_reuse_seed(seed: gr.Number, reuse_seed: gr.Button, generation_info: gr.Textbox, dummy_component, is_subseed):
    """ Connects a 'reuse (sub)seed' button's click event so that it copies last used
        (sub)seed value from generation info the to the seed field. If copying subseed and subseed strength
        was 0, i.e. no variation seed was used, it copies the normal seed value instead."""
    def copy_seed(gen_info_string: str, index):
        res = -1
        try:
            gen_info = json.loads(gen_info_string)
            index -= gen_info.get('index_of_first_image', 0)

            if is_subseed and gen_info.get('subseed_strength', 0) > 0:
                all_subseeds = gen_info.get('all_subseeds', [-1])
                res = all_subseeds[index if 0 <= index < len(all_subseeds) else 0]
            else:
                all_seeds = gen_info.get('all_seeds', [-1])
                res = all_seeds[index if 0 <= index < len(all_seeds) else 0]
        except json.decoder.JSONDecodeError:
            if gen_info_string != '':
                log.error(f"Error parsing JSON generation info: {gen_info_string}")
        return [res, gr_show(False)]

    reuse_seed.click(fn=copy_seed, _js="(x, y) => [x, selected_gallery_index()]", show_progress=False, inputs=[generation_info, dummy_component], outputs=[seed, dummy_component])


def update_token_counter(text, steps):
    try:
        text, _ = extra_networks.parse_prompt(text)
        _, prompt_flat_list, _ = prompt_parser.get_multicond_prompt_list([text])
        prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompt_flat_list, steps)
    except Exception:
        # a parsing error can happen here during typing, and we don't want to bother the user with
        # messages related to it in console
        prompt_schedules = [[[steps, text]]]
    flat_prompts = reduce(lambda list1, list2: list1+list2, prompt_schedules)
    prompts = [prompt_text for step, prompt_text in flat_prompts]
    if modules.shared.backend == modules.shared.Backend.ORIGINAL:
        token_count, max_length = max([sd_hijack.model_hijack.get_prompt_lengths(prompt) for prompt in prompts], key=lambda args: args[0])
    elif modules.shared.backend == modules.shared.Backend.DIFFUSERS:
        if modules.shared.sd_model is not None and hasattr(modules.shared.sd_model, 'tokenizer'):
            tokenizer = modules.shared.sd_model.tokenizer
            if tokenizer is None:
                token_count = 0
                max_length = 75
            else:
                has_bos_token = tokenizer.bos_token_id is not None
                has_eos_token = tokenizer.eos_token_id is not None
                ids = [modules.shared.sd_model.tokenizer(prompt) for prompt in prompts]
                if len(ids) > 0 and hasattr(ids[0], 'input_ids'):
                    ids = [x.input_ids for x in ids]
                token_count = max([len(x) for x in ids]) - int(has_bos_token) - int(has_eos_token)
                max_length = tokenizer.model_max_length - int(has_bos_token) - int(has_eos_token)
        else:
            token_count = 0
            max_length = 75
    return f"<span class='gr-box gr-text-input'>{token_count}/{max_length}</span>"


def create_toprow(is_img2img):
    id_part = "img2img" if is_img2img else "txt2img"
    debug('create_toprow - id_part')
    with gr.Row(elem_id=f"{id_part}_toprow", variant="compact"):
        with gr.Column(elem_id=f"{id_part}_prompt_container", scale=6):
            with gr.Row():
                with gr.Column(scale=80):
                    with gr.Row():
                        prompt = gr.Textbox(elem_id=f"{id_part}_prompt", label="Prompt", show_label=False, lines=3, placeholder="Prompt", elem_classes=["prompt"])
            with gr.Row():
                with gr.Column(scale=80):
                    with gr.Row():
                        negative_prompt = gr.Textbox(elem_id=f"{id_part}_neg_prompt", label="Negative prompt", show_label=False, lines=3, placeholder="Negative prompt", elem_classes=["prompt"])
        button_interrogate = None
        button_deepbooru = None
        if is_img2img:
            with gr.Column(scale=1, elem_classes="interrogate-col"):
                button_interrogate = gr.Button('Interrogate\nCLIP', elem_id="interrogate")
                button_deepbooru = gr.Button('Interrogate\nDeepBooru', elem_id="deepbooru")
        with gr.Column(scale=1, elem_id=f"{id_part}_actions_column"):
            with gr.Row(elem_id=f"{id_part}_generate_box"):
                submit = gr.Button('Generate', elem_id=f"{id_part}_generate", variant='primary')
            with gr.Row(elem_id=f"{id_part}_generate_line2"):
                interrupt = gr.Button('Stop', elem_id=f"{id_part}_interrupt")
                interrupt.click(fn=lambda: modules.shared.state.interrupt(), _js="requestInterrupt", inputs=[], outputs=[])
                skip = gr.Button('Skip', elem_id=f"{id_part}_skip")
                skip.click(fn=lambda: modules.shared.state.skip(), inputs=[], outputs=[])
                pause = gr.Button('Pause', elem_id=f"{id_part}_pause")
                pause.click(fn=lambda: modules.shared.state.pause(), _js='checkPaused', inputs=[], outputs=[])
            with gr.Row(elem_id=f"{id_part}_tools"):
                button_paste = gr.Button(value='Restore', variant='secondary', elem_id="paste") # symbols.paste
                button_clear = gr.Button(value='Clear', variant='secondary', elem_id=f"{id_part}_clear_prompt_btn") # symbols.clear
                button_extra = gr.Button(value='Networks', variant='secondary', elem_id=f"{id_part}_extra_networks_btn") # symbols.networks
                button_clear.click(fn=lambda *x: ['', ''], inputs=[prompt, negative_prompt], outputs=[prompt, negative_prompt], show_progress=False)
            with gr.Row(elem_id=f"{id_part}_counters"):
                token_counter = gr.HTML(value="<span>0/75</span>", elem_id=f"{id_part}_token_counter", elem_classes=["token-counter"])
                token_button = gr.Button(visible=False, elem_id=f"{id_part}_token_button")
                negative_token_counter = gr.HTML(value="<span>0/75</span>", elem_id=f"{id_part}_negative_token_counter", elem_classes=["token-counter"])
                negative_token_button = gr.Button(visible=False, elem_id=f"{id_part}_negative_token_button")
            with gr.Row(elem_id=f"{id_part}_styles_row"):
                prompt_styles = gr.Dropdown(label="Styles", elem_id=f"{id_part}_styles", choices=[style.name for style in modules.shared.prompt_styles.styles.values()], value=[], multiselect=True)
                prompt_styles_btn_select = gr.Button('Select', elem_id=f"{id_part}_styles_select", visible=False)
                prompt_styles_btn_select.click(_js="applyStyles", fn=parse_style, inputs=[prompt_styles], outputs=[prompt_styles])
                prompt_styles_btn_apply = ToolButton(symbols.apply, elem_id=f"{id_part}_extra_apply", visible=False)
                prompt_styles_btn_apply.click(fn=apply_styles, inputs=[prompt, negative_prompt, prompt_styles], outputs=[prompt, negative_prompt, prompt_styles])
    return prompt, prompt_styles, negative_prompt, submit, button_interrogate, button_deepbooru, button_paste, button_extra, token_counter, token_button, negative_token_counter, negative_token_button


def setup_progressbar(*args, **kwargs): # pylint: disable=unused-argument
    pass


def apply_setting(key, value):
    if value is None:
        return gr.update()
    if modules.shared.cmd_opts.freeze:
        return gr.update()
    # dont allow model to be swapped when model hash exists in prompt
    if key == "sd_model_checkpoint" and opts.disable_weights_auto_swap:
        return gr.update()
    if key == "sd_model_checkpoint":
        ckpt_info = sd_models.get_closet_checkpoint_match(value)
        if ckpt_info is not None:
            value = ckpt_info.title
        else:
            return gr.update()
    comp_args = opts.data_labels[key].component_args
    if comp_args and isinstance(comp_args, dict) and comp_args.get('visible') is False:
        return gr.update()
    valtype = type(opts.data_labels[key].default)
    oldval = opts.data.get(key, None)
    opts.data[key] = valtype(value) if valtype != type(None) else value
    if oldval != value and opts.data_labels[key].onchange is not None:
        opts.data_labels[key].onchange()
    opts.save(modules.shared.config_filename)
    return getattr(opts, key)


def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id):
    return ui_common.create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id)


def create_sampler_and_steps_selection(choices, tabname):
    def set_sampler_original_options(sampler_options, sampler_algo):
        opts.data['schedulers_brownian_noise'] = 'brownian noise' in sampler_options
        opts.data['schedulers_discard_penultimate'] = 'discard penultimate sigma' in sampler_options
        opts.data['schedulers_sigma'] = sampler_algo
        opts.save(modules.shared.config_filename, silent=True)

    def set_sampler_diffuser_options(sampler_options):
        opts.data['schedulers_use_karras'] = 'karras' in sampler_options
        opts.data['schedulers_use_thresholding'] = 'dynamic thresholding' in sampler_options
        opts.data['schedulers_use_loworder'] = 'low order' in sampler_options
        opts.save(modules.shared.config_filename, silent=True)

    with FormRow(elem_classes=['flex-break']):
        sampler_index = gr.Dropdown(label='Sampling method', elem_id=f"{tabname}_sampling", choices=[x.name for x in choices], value='Default', type="index")
        steps = gr.Slider(minimum=0, maximum=99, step=1, label="Sampling steps", elem_id=f"{tabname}_steps", value=20)
    if modules.shared.backend == modules.shared.Backend.ORIGINAL:
        with FormRow(elem_classes=['flex-break']):
            choices = ['brownian noise', 'discard penultimate sigma']
            values = []
            values += ['brownian noise'] if opts.data.get('schedulers_brownian_noise', False) else []
            values += ['discard penultimate sigma'] if opts.data.get('schedulers_discard_penultimate', True) else []
            sampler_options = gr.CheckboxGroup(label='Sampler options', choices=choices, value=values, type='value')
        with FormRow(elem_classes=['flex-break']):
            opts.data['schedulers_sigma'] = opts.data.get('schedulers_sigma', 'default')
            sampler_algo = gr.Radio(label='Sigma algorithm', choices=['default', 'karras', 'exponential', 'polyexponential'], value=opts.data['schedulers_sigma'], type='value')
        sampler_options.change(fn=set_sampler_original_options, inputs=[sampler_options, sampler_algo], outputs=[])
        sampler_algo.change(fn=set_sampler_original_options, inputs=[sampler_options, sampler_algo], outputs=[])
    else:
        with FormRow(elem_classes=['flex-break']):
            choices = ['karras', 'dynamic thresholding', 'low order']
            values = []
            values += ['karras'] if opts.data.get('schedulers_use_karras', True) else []
            values += ['dynamic thresholding'] if opts.data.get('schedulers_use_thresholding', False) else []
            values += ['low order'] if opts.data.get('schedulers_use_loworder', True) else []
            sampler_options = gr.CheckboxGroup(label='Sampler options', choices=choices, value=values, type='value')
        sampler_options.change(fn=set_sampler_diffuser_options, inputs=[sampler_options], outputs=[])

    return steps, sampler_index


def get_value_for_setting(key):
    value = getattr(opts, key)
    info = opts.data_labels[key]
    args = info.component_args() if callable(info.component_args) else info.component_args or {}
    args = {k: v for k, v in args.items() if k not in {'precision'}}
    return gr.update(value=value, **args)


def ordered_ui_categories():
    return ['dimensions', 'sampler', 'seed', 'denoising', 'cfg', 'checkboxes', 'accordions', 'override_settings', 'scripts'] # TODO: a1111 compatibility item, not implemented


def create_override_settings_dropdown(tabname, row): # pylint: disable=unused-argument
    dropdown = gr.Dropdown([], label="Override settings", visible=False, elem_id=f"{tabname}_override_settings", multiselect=True)
    dropdown.change(fn=lambda x: gr.Dropdown.update(visible=len(x) > 0), inputs=[dropdown], outputs=[dropdown])
    return dropdown


def create_ui(startup_timer = None):
    debug('create_ui')
    if startup_timer is None:
        timer.startup = timer.Timer()
    reload_javascript()
    parameters_copypaste.reset()

    import modules.txt2img # pylint: disable=redefined-outer-name
    modules.scripts.scripts_current = modules.scripts.scripts_txt2img
    modules.scripts.scripts_txt2img.initialize_scripts(is_img2img=False)
    with gr.Blocks(analytics_enabled=False) as txt2img_interface:
        txt2img_prompt, txt2img_prompt_styles, txt2img_negative_prompt, submit, _interrogate, _deepbooru, txt2img_paste, extra_networks_button, token_counter, token_button, negative_token_counter, negative_token_button = create_toprow(is_img2img=False)
        dummy_component = gr.Label(visible=False)
        txt_prompt_img = gr.File(label="", elem_id="txt2img_prompt_image", file_count="single", type="binary", visible=False)
        with FormRow(variant='compact', elem_id="txt2img_extra_networks", visible=False) as extra_networks_ui:
            from modules import ui_extra_networks
            extra_networks_ui = ui_extra_networks.create_ui(extra_networks_ui, extra_networks_button, 'txt2img', skip_indexing=opts.extra_network_skip_indexing)
            timer.startup.record('ui-extra-networks')

        with gr.Row(elem_id="txt2img_interface", equal_height=False):
            with gr.Column(variant='compact', elem_id="txt2img_settings"):

                with FormRow():
                    width = gr.Slider(minimum=64, maximum=4096, step=8, label="Width", value=512, elem_id="txt2img_width")
                    height = gr.Slider(minimum=64, maximum=4096, step=8, label="Height", value=512, elem_id="txt2img_height")
                    res_switch_btn = ToolButton(value=symbols.switch, elem_id="txt2img_res_switch_btn", label="Switch dims")

                with FormGroup(elem_classes="settings-accordion"):
                    with gr.Accordion(open=False, label="Sampler", elem_id="txt2img_sampler", elem_classes=["small-accordion"]):
                        with FormRow(elem_id="txt2img_row_sampler"):
                            modules.sd_samplers.set_samplers()
                            steps, sampler_index = create_sampler_and_steps_selection(modules.sd_samplers.samplers, "txt2img")

                    with gr.Accordion(open=False, label="Batch", elem_id="txt2img_batch", elem_classes=["small-accordion"]):
                        with FormRow(elem_id="txt2img_row_batch"):
                            batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id="txt2img_batch_count")
                            batch_size = gr.Slider(minimum=1, maximum=32, step=1, label='Batch size', value=1, elem_id="txt2img_batch_size")
                            batch_switch_btn = ToolButton(value=symbols.switch, elem_id="txt2img_batch_switch_btn", label="Switch dims")

                    seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w = create_seed_inputs('txt2img')

                    with gr.Accordion(open=False, label="Advanced", elem_id="txt2img_advanced", elem_classes=["small-accordion"]):
                        with FormRow():
                            cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.1, label='CFG scale', value=6.0, elem_id="txt2img_cfg_scale")
                            clip_skip = gr.Slider(label='CLIP skip', value=1, minimum=1, maximum=14, step=1, elem_id='txt2img_clip_skip', interactive=True)
                        with FormRow():
                            image_cfg_scale = gr.Slider(minimum=1.1, maximum=30.0, step=0.1, label='Secondary CFG scale', value=6.0, elem_id="txt2img_image_cfg_scale")
                            diffusers_guidance_rescale = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Guidance rescale', value=0.7, elem_id="txt2img_image_cfg_rescale")
                        with FormRow():
                            full_quality = gr.Checkbox(label='Full quality', value=True, elem_id="txt2img_full_quality")
                            restore_faces = gr.Checkbox(label='Face restore', value=False, visible=len(modules.shared.face_restorers) > 1, elem_id="txt2img_restore_faces")
                            tiling = gr.Checkbox(label='Tiling', value=False, elem_id="txt2img_tiling")

                    with gr.Accordion(open=False, label="Second pass", elem_id="txt2img_second_pass", elem_classes=["small-accordion"]):
                        with FormGroup():
                            with FormRow(elem_id="sampler_selection_txt2img_alt_row1"):
                                enable_hr = gr.Checkbox(label='Enable second pass', value=False, elem_id="txt2img_enable_hr")
                            with FormRow(elem_id="sampler_selection_txt2img_alt_row1"):
                                latent_index = gr.Dropdown(label='Secondary sampler', elem_id="txt2img_sampling_alt", choices=[x.name for x in modules.sd_samplers.samplers], value='Default', type="index")
                                denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Denoising strength', value=0.5, elem_id="txt2img_denoising_strength")
                            with FormRow(elem_id="txt2img_hires_finalres", variant="compact"):
                                hr_final_resolution = FormHTML(value="", elem_id="txtimg_hr_finalres", label="Upscaled resolution", interactive=False)
                            with FormRow(elem_id="txt2img_hires_fix_row1", variant="compact"):
                                hr_upscaler = gr.Dropdown(label="Upscaler", elem_id="txt2img_hr_upscaler", choices=[*modules.shared.latent_upscale_modes, *[x.name for x in modules.shared.sd_upscalers]], value=modules.shared.latent_upscale_default_mode)
                                hr_force = gr.Checkbox(label='Force Hires', value=False, elem_id="txt2img_hr_force")
                            with FormRow(elem_id="txt2img_hires_fix_row2", variant="compact"):
                                hr_second_pass_steps = gr.Slider(minimum=0, maximum=99, step=1, label='Hires steps', elem_id="txt2img_steps_alt", value=20)
                                hr_scale = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, label="Upscale by", value=2.0, elem_id="txt2img_hr_scale")
                            with FormRow(elem_id="txt2img_hires_fix_row3", variant="compact"):
                                hr_resize_x = gr.Slider(minimum=0, maximum=4096, step=8, label="Resize width to", value=0, elem_id="txt2img_hr_resize_x")
                                hr_resize_y = gr.Slider(minimum=0, maximum=4096, step=8, label="Resize height to", value=0, elem_id="txt2img_hr_resize_y")
                        with FormGroup(visible=modules.shared.backend == modules.shared.Backend.DIFFUSERS):
                            with FormRow(elem_id="txt2img_refiner_row1", variant="compact"):
                                refiner_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Refiner start', value=0.8, elem_id="txt2img_refiner_start")
                                refiner_steps = gr.Slider(minimum=0, maximum=99, step=1, label="Refiner steps", elem_id="txt2img_refiner_steps", value=5)
                            with FormRow(elem_id="txt2img_refiner_row3", variant="compact"):
                                refiner_prompt = gr.Textbox(value='', label='Secondary Prompt')
                            with FormRow(elem_id="txt2img_refiner_row4", variant="compact"):
                                refiner_negative = gr.Textbox(value='', label='Secondary negative prompt')

                with FormRow(elem_id="txt2img_override_settings_row") as row:
                    override_settings = create_override_settings_dropdown('txt2img', row)

                custom_inputs = modules.scripts.scripts_txt2img.setup_ui()

            hr_resolution_preview_inputs = [width, height, hr_scale, hr_resize_x, hr_resize_y, hr_upscaler]
            for preview_input in hr_resolution_preview_inputs:
                preview_input.change(
                    fn=calc_resolution_hires,
                    _js="onCalcResolutionHires",
                    inputs=hr_resolution_preview_inputs,
                    outputs=[hr_final_resolution],
                    show_progress=False,
                )

            txt2img_gallery, generation_info, html_info, _html_info_formatted, html_log = ui_common.create_output_panel("txt2img")
            connect_reuse_seed(seed, reuse_seed, generation_info, dummy_component, is_subseed=False)
            connect_reuse_seed(subseed, reuse_subseed, generation_info, dummy_component, is_subseed=True)

            txt2img_args = dict(
                fn=wrap_gradio_gpu_call(modules.txt2img.txt2img, extra_outputs=[None, '', '']),
                _js="submit_txt2img",
                inputs=[
                    dummy_component,
                    txt2img_prompt, txt2img_negative_prompt,
                    txt2img_prompt_styles,
                    steps,
                    sampler_index, latent_index,
                    full_quality, restore_faces, tiling,
                    batch_count, batch_size,
                    cfg_scale, image_cfg_scale,
                    diffusers_guidance_rescale,
                    clip_skip,
                    seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w,
                    height, width,
                    enable_hr, denoising_strength,
                    hr_scale, hr_upscaler, hr_force, hr_second_pass_steps, hr_resize_x, hr_resize_y,
                    refiner_steps, refiner_start, refiner_prompt, refiner_negative,
                    override_settings,
                ] + custom_inputs,
                outputs=[
                    txt2img_gallery,
                    generation_info,
                    html_info,
                    html_log,
                ],
                show_progress=False,
            )
            txt2img_prompt.submit(**txt2img_args)
            submit.click(**txt2img_args)

            res_switch_btn.click(lambda w, h: (h, w), inputs=[width, height], outputs=[width, height], show_progress=False)
            batch_switch_btn.click(lambda w, h: (h, w), inputs=[batch_count, batch_size], outputs=[batch_count, batch_size], show_progress=False)
            txt_prompt_img.change(fn=modules.images.image_data, inputs=[txt_prompt_img], outputs=[txt2img_prompt, txt_prompt_img])

            txt2img_paste_fields = [
                (txt2img_prompt, "Prompt"),
                (txt2img_negative_prompt, "Negative prompt"),
                # (txt2img_prompt_styles, "Styles"),
                (steps, "Steps"),
                (seed, "Seed"),
                (sampler_index, "Sampler"),
                (cfg_scale, "CFG scale"),
                (width, "Size-1"),
                (height, "Size-2"),
                (subseed, "Variation seed"),
                (subseed_strength, "Variation strength"),
                (clip_skip, "Clip skip"),
                (latent_index, "Latent sampler"),
                (latent_index, "Secondary sampler"),
                (denoising_strength, "Denoising strength"),
                (refiner_steps, "Refiner steps"),
                (refiner_start, "Refiner start"),
                (full_quality, "Full quality"),
                (restore_faces, "Face restoration"),
                (batch_size, "Batch size"),
                (batch_count, "Batch count"),
                (seed_resize_from_w, "Seed resize from-1"),
                (seed_resize_from_h, "Seed resize from-2"),
                (enable_hr, "Second pass"),
                (hr_force, "Hires force"),
                (hr_scale, "Hires upscale"),
                (hr_upscaler, "Hires upscaler"),
                (hr_second_pass_steps, "Hires steps"),
                (hr_resize_x, "Hires resize-1"),
                (hr_resize_y, "Hires resize-2"),
                (diffusers_guidance_rescale, "CFG rescale"),
                (image_cfg_scale, "Image CFG scale"),
                (refiner_steps, "Refiner steps"),
                (refiner_start, "Refiner start"),
                (tiling, "Tiling"),
                (refiner_negative, "Negative2"),
                (refiner_prompt, "Prompt2"),
                *modules.scripts.scripts_txt2img.infotext_fields
            ]
            parameters_copypaste.add_paste_fields("txt2img", None, txt2img_paste_fields, override_settings)
            parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(paste_button=txt2img_paste, tabname="txt2img", source_text_component=txt2img_prompt, source_image_component=None))

            token_button.click(fn=wrap_queued_call(update_token_counter), inputs=[txt2img_prompt, steps], outputs=[token_counter])
            negative_token_button.click(fn=wrap_queued_call(update_token_counter), inputs=[txt2img_negative_prompt, steps], outputs=[negative_token_counter])

            ui_extra_networks.setup_ui(extra_networks_ui, txt2img_gallery)

    timer.startup.record("ui-txt2img")

    import modules.img2img # pylint: disable=redefined-outer-name
    modules.scripts.scripts_current = modules.scripts.scripts_img2img
    modules.scripts.scripts_img2img.initialize_scripts(is_img2img=True)
    with gr.Blocks(analytics_enabled=False) as img2img_interface:
        img2img_prompt, img2img_prompt_styles, img2img_negative_prompt, submit, img2img_interrogate, img2img_deepbooru, img2img_paste, extra_networks_button, token_counter, token_button, negative_token_counter, negative_token_button = create_toprow(is_img2img=True)
        img2img_prompt_img = gr.File(label="", elem_id="img2img_prompt_image", file_count="single", type="binary", visible=False)

        with FormRow(variant='compact', elem_id="img2img_extra_networks", visible=False) as extra_networks_ui:
            from modules import ui_extra_networks
            extra_networks_ui_img2img = ui_extra_networks.create_ui(extra_networks_ui, extra_networks_button, 'img2img', skip_indexing=opts.extra_network_skip_indexing)

        with FormRow(elem_id="img2img_interface", equal_height=False):
            with gr.Column(variant='compact', elem_id="img2img_settings"):
                copy_image_buttons = []
                copy_image_destinations = {}

                def copy_image(img):
                    return img['image'] if isinstance(img, dict) and 'image' in img else img

                def add_copy_image_controls(tab_name, elem):
                    with gr.Row(variant="compact", elem_id=f"img2img_copy_to_{tab_name}"):
                        for title, name in zip(['➠ Image', '➠ Sketch', '➠ Inpaint', '➠ Inpaint sketch'], ['img2img', 'sketch', 'inpaint', 'inpaint_sketch']):
                            if name == tab_name:
                                gr.Button(title, interactive=False)
                                copy_image_destinations[name] = elem
                                continue
                            button = gr.Button(title)
                            copy_image_buttons.append((button, name, elem))

                with gr.Tabs(elem_id="mode_img2img"):
                    img2img_selected_tab = gr.State(0) # pylint: disable=abstract-class-instantiated
                    with gr.TabItem('Image', id='img2img', elem_id="img2img_img2img_tab") as tab_img2img:
                        init_img = gr.Image(label="Image for img2img", elem_id="img2img_image", show_label=False, source="upload", interactive=True, type="pil", tool="editor", image_mode="RGBA", height=512)
                        add_copy_image_controls('img2img', init_img)

                    with gr.TabItem('Sketch', id='img2img_sketch', elem_id="img2img_img2img_sketch_tab") as tab_sketch:
                        sketch = gr.Image(label="Image for img2img", elem_id="img2img_sketch", show_label=False, source="upload", interactive=True, type="pil", tool="color-sketch", image_mode="RGBA", height=512)
                        add_copy_image_controls('sketch', sketch)

                    with gr.TabItem('Inpaint', id='inpaint', elem_id="img2img_inpaint_tab") as tab_inpaint:
                        init_img_with_mask = gr.Image(label="Image for inpainting with mask", show_label=False, elem_id="img2maskimg", source="upload", interactive=True, type="pil", tool="sketch", image_mode="RGBA", height=512)
                        add_copy_image_controls('inpaint', init_img_with_mask)

                    with gr.TabItem('Inpaint sketch', id='inpaint_sketch', elem_id="img2img_inpaint_sketch_tab") as tab_inpaint_color:
                        inpaint_color_sketch = gr.Image(label="Color sketch inpainting", show_label=False, elem_id="inpaint_sketch", source="upload", interactive=True, type="pil", tool="color-sketch", image_mode="RGBA", height=512)
                        inpaint_color_sketch_orig = gr.State(None) # pylint: disable=abstract-class-instantiated
                        add_copy_image_controls('inpaint_sketch', inpaint_color_sketch)

                        def update_orig(image, state):
                            if image is not None:
                                same_size = state is not None and state.size == image.size
                                has_exact_match = np.any(np.all(np.array(image) == np.array(state), axis=-1))
                                edited = same_size and has_exact_match
                                return image if not edited or state is None else state
                            return state

                        inpaint_color_sketch.change(update_orig, [inpaint_color_sketch, inpaint_color_sketch_orig], inpaint_color_sketch_orig)

                    with gr.TabItem('Inpaint upload', id='inpaint_upload', elem_id="img2img_inpaint_upload_tab") as tab_inpaint_upload:
                        init_img_inpaint = gr.Image(label="Image for img2img", show_label=False, source="upload", interactive=True, type="pil", elem_id="img_inpaint_base")
                        init_mask_inpaint = gr.Image(label="Mask", source="upload", interactive=True, type="pil", elem_id="img_inpaint_mask")

                    with gr.TabItem('Batch', id='batch', elem_id="img2img_batch_tab") as tab_batch:
                        hidden = '<br>Disabled when launched with --hide-ui-dir-config.' if modules.shared.cmd_opts.hide_ui_dir_config else ''
                        gr.HTML(
                            "<p style='padding-bottom: 1em;' class=\"text-gray-500\">Upload images or process images in a directory" +
                            "<br>Add inpaint batch mask directory to enable inpaint batch processing"
                            f"{hidden}</p>"
                        )
                        img2img_batch_files = gr.Files(label="Batch Process", interactive=True, elem_id="img2img_image_batch")
                        img2img_batch_input_dir = gr.Textbox(label="Inpaint batch input directory", **modules.shared.hide_dirs, elem_id="img2img_batch_input_dir")
                        img2img_batch_output_dir = gr.Textbox(label="Inpaint batch output directory", **modules.shared.hide_dirs, elem_id="img2img_batch_output_dir")
                        img2img_batch_inpaint_mask_dir = gr.Textbox(label="Inpaint batch mask directory", **modules.shared.hide_dirs, elem_id="img2img_batch_inpaint_mask_dir")

                    img2img_tabs = [tab_img2img, tab_sketch, tab_inpaint, tab_inpaint_color, tab_inpaint_upload, tab_batch]

                    for i, tab in enumerate(img2img_tabs):
                        tab.select(fn=lambda tabnum=i: tabnum, inputs=[], outputs=[img2img_selected_tab])

                for button, name, elem in copy_image_buttons:
                    button.click(fn=copy_image, inputs=[elem], outputs=[copy_image_destinations[name]])
                    button.click(fn=lambda: None, _js=f"switch_to_{name.replace(' ', '_')}", inputs=[], outputs=[])

                with FormGroup(elem_classes="settings-accordion"):
                    with gr.Accordion(open=False, label="Sampler", elem_classes=["small-accordion"], elem_id="img2img_sampling_group"):
                        modules.sd_samplers.set_samplers()
                        steps, sampler_index = create_sampler_and_steps_selection(modules.sd_samplers.samplers_for_img2img, "img2img")

                    with gr.Accordion(open=False, label="Resize", elem_classes=["small-accordion"], elem_id="img2img_resize_group"):
                        with FormRow():
                            resize_mode = gr.Radio(label="Resize mode", elem_id="resize_mode", choices=["None", "Resize fixed", "Crop and resize", "Resize and fill", "Latent upscale"], type="index", value="None")

                        with FormRow():
                            with gr.Column(elem_id="img2img_column_size", scale=4):
                                selected_scale_tab = gr.State(value=0) # pylint: disable=abstract-class-instantiated

                                with gr.Tabs():
                                    with gr.Tab(label="Resize to") as tab_scale_to:
                                        with FormRow():
                                            with gr.Column(elem_id="img2img_column_size", scale=4):
                                                with FormRow():
                                                    width = gr.Slider(minimum=64, maximum=4096, step=8, label="Width", value=512, elem_id="img2img_width")
                                                    height = gr.Slider(minimum=64, maximum=4096, step=8, label="Height", value=512, elem_id="img2img_height")
                                            with gr.Column(elem_id="img2img_column_dim", scale=1, elem_classes="dimensions-tools"):
                                                with FormRow():
                                                    res_switch_btn = ToolButton(value=symbols.switch, elem_id="img2img_res_switch_btn")
                                                    detect_image_size_btn = ToolButton(value=symbols.detect, elem_id="img2img_detect_image_size_btn")

                                    with gr.Tab(label="Resize by") as tab_scale_by:
                                        scale_by = gr.Slider(minimum=0.05, maximum=4.0, step=0.05, label="Scale", value=1.0, elem_id="img2img_scale")

                                        with FormRow():
                                            scale_by_html = FormHTML(resize_from_to_html(0, 0, 0.0), elem_id="img2img_scale_resolution_preview")
                                            gr.Slider(label="Unused", elem_id="img2img_unused_scale_by_slider")
                                            button_update_resize_to = gr.Button(visible=False, elem_id="img2img_update_resize_to")

                                    on_change_args = dict(
                                        fn=resize_from_to_html,
                                        _js="currentImg2imgSourceResolution",
                                        inputs=[dummy_component, dummy_component, scale_by],
                                        outputs=scale_by_html,
                                        show_progress=False,
                                    )

                                    scale_by.release(**on_change_args)
                                    button_update_resize_to.click(**on_change_args)

                                    for component in [init_img, sketch]:
                                        component.change(fn=lambda: None, _js="updateImg2imgResizeToTextAfterChangingImage", inputs=[], outputs=[], show_progress=False)

                            tab_scale_to.select(fn=lambda: 0, inputs=[], outputs=[selected_scale_tab])
                            tab_scale_by.select(fn=lambda: 1, inputs=[], outputs=[selected_scale_tab])

                    with gr.Accordion(open=False, label="Batch", elem_classes=["small-accordion"], elem_id="img2img_batch_group"):
                        with FormRow(elem_id="img2img_column_batch"):
                            batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id="img2img_batch_count")
                            batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id="img2img_batch_size")

                    seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w = create_seed_inputs('img2img')

                    with gr.Accordion(open=False, label="Denoise", elem_classes=["small-accordion"], elem_id="img2img_denoise_group"):
                        with FormRow():
                            denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Denoising strength', value=0.75, elem_id="img2img_denoising_strength")
                            refiner_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Denoise start', value=0.0, elem_id="img2img_refiner_start")

                    with gr.Accordion(open=False, label="Advanced", elem_classes=["small-accordion"], elem_id="img2img_advanced_group"):
                        with FormRow():
                            cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG scale', value=6.0, elem_id="img2img_cfg_scale")
                            image_cfg_scale = gr.Slider(minimum=0, maximum=30.0, step=0.05, label='Image CFG scale', value=1.5, elem_id="img2img_image_cfg_scale")
                        with FormRow():
                            clip_skip = gr.Slider(label='CLIP skip', value=1, minimum=1, maximum=4, step=1, elem_id='img2img_clip_skip', interactive=True)
                            diffusers_guidance_rescale = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Guidance rescale', value=0.7, elem_id="txt2img_image_cfg_rescale")
                        with FormRow(elem_classes="img2img_checkboxes_row", variant="compact"):
                            full_quality = gr.Checkbox(label='Full quality', value=True, elem_id="img2img_full_quality")
                            restore_faces = gr.Checkbox(label='Restore faces', value=False, visible=len(modules.shared.face_restorers) > 1, elem_id="img2img_restore_faces")
                            tiling = gr.Checkbox(label='Tiling', value=False, elem_id="img2img_tiling")

                    with FormGroup(elem_id="inpaint_controls", visible=False) as inpaint_controls:
                        with FormRow():
                            mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=4, elem_id="img2img_mask_blur")
                            mask_alpha = gr.Slider(label="Mask transparency", visible=False, elem_id="img2img_mask_alpha")
                        with FormRow():
                            with gr.Column():
                                inpainting_mask_invert = gr.Radio(label='Mask mode', choices=['Inpaint masked', 'Inpaint not masked'], value='Inpaint masked', type="index", elem_id="img2img_mask_mode")
                            with gr.Column():
                                inpainting_fill = gr.Radio(label='Masked content', choices=['fill', 'original', 'latent noise', 'latent nothing'], value='original', type="index", elem_id="img2img_inpainting_fill")
                        with FormRow():
                            with gr.Column():
                                inpaint_full_res = gr.Radio(label="Inpaint area", choices=["Whole picture", "Only masked"], type="index", value="Whole picture", elem_id="img2img_inpaint_full_res")
                            with gr.Column():
                                inpaint_full_res_padding = gr.Slider(label='Masked padding', minimum=0, maximum=256, step=4, value=32, elem_id="img2img_inpaint_full_res_padding")

                        def select_img2img_tab(tab):
                            return gr.update(visible=tab in [2, 3, 4]), gr.update(visible=tab == 3)

                        for i, elem in enumerate(img2img_tabs):
                            elem.select(fn=lambda tab=i: select_img2img_tab(tab), inputs=[], outputs=[inpaint_controls, mask_alpha]) # pylint: disable=cell-var-from-loop

                with FormRow(elem_id="img2img_override_settings_row") as row:
                    override_settings = create_override_settings_dropdown('img2img', row)

                with FormGroup(elem_id="img2img_script_container"):
                    custom_inputs = modules.scripts.scripts_img2img.setup_ui()

            img2img_gallery, generation_info, html_info, _html_info_formatted, html_log = ui_common.create_output_panel("img2img")

            connect_reuse_seed(seed, reuse_seed, generation_info, dummy_component, is_subseed=False)
            connect_reuse_seed(subseed, reuse_subseed, generation_info, dummy_component, is_subseed=True)

            img2img_prompt_img.change(
                fn=modules.images.image_data,
                inputs=[
                    img2img_prompt_img
                ],
                outputs=[
                    img2img_prompt,
                    img2img_prompt_img
                ]
            )

            img2img_args = dict(
                fn=wrap_gradio_gpu_call(modules.img2img.img2img, extra_outputs=[None, '', '']),
                _js="submit_img2img",
                inputs=[
                    dummy_component, dummy_component,
                    img2img_prompt, img2img_negative_prompt,
                    img2img_prompt_styles,
                    init_img,
                    sketch,
                    init_img_with_mask,
                    inpaint_color_sketch,
                    inpaint_color_sketch_orig,
                    init_img_inpaint,
                    init_mask_inpaint,
                    steps,
                    sampler_index, latent_index,
                    mask_blur, mask_alpha,
                    inpainting_fill,
                    full_quality, restore_faces, tiling,
                    batch_count, batch_size,
                    cfg_scale, image_cfg_scale,
                    diffusers_guidance_rescale,
                    refiner_steps,
                    refiner_start,
                    clip_skip,
                    denoising_strength,
                    seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w,
                    selected_scale_tab,
                    height, width,
                    scale_by,
                    resize_mode,
                    inpaint_full_res, inpaint_full_res_padding, inpainting_mask_invert,
                    img2img_batch_files, img2img_batch_input_dir, img2img_batch_output_dir, img2img_batch_inpaint_mask_dir,
                    override_settings,
                ] + custom_inputs,
                outputs=[
                    img2img_gallery,
                    generation_info,
                    html_info,
                    html_log,
                ],
                show_progress=False,
            )
            img2img_prompt.submit(**img2img_args)
            submit.click(**img2img_args)

            interrogate_args = dict(
                _js="get_img2img_tab_index",
                inputs=[
                    dummy_component,
                    img2img_batch_files,
                    img2img_batch_input_dir,
                    img2img_batch_output_dir,
                    init_img,
                    sketch,
                    init_img_with_mask,
                    inpaint_color_sketch,
                    init_img_inpaint,
                ],
                outputs=[img2img_prompt, dummy_component],
            )
            img2img_interrogate.click(fn=lambda *args: process_interrogate(interrogate, *args), **interrogate_args)
            img2img_deepbooru.click(fn=lambda *args: process_interrogate(interrogate_deepbooru, *args), **interrogate_args)

            res_switch_btn.click(lambda w, h: (h, w), inputs=[width, height], outputs=[width, height], show_progress=False)

            detect_image_size_btn.click(
                fn=lambda w, h, _: (w or gr.update(), h or gr.update()),
                _js="currentImg2imgSourceResolution",
                inputs=[dummy_component, dummy_component, dummy_component],
                outputs=[width, height],
                show_progress=False,
            )

            token_button.click(fn=wrap_queued_call(update_token_counter), inputs=[img2img_prompt, steps], outputs=[token_counter])
            negative_token_button.click(fn=wrap_queued_call(update_token_counter), inputs=[img2img_negative_prompt, steps], outputs=[negative_token_counter])

            ui_extra_networks.setup_ui(extra_networks_ui_img2img, img2img_gallery)
            img2img_paste_fields = [
                (img2img_prompt, "Prompt"),
                (img2img_negative_prompt, "Negative prompt"),
                # (img2img_prompt_styles, "Styles"),
                (steps, "Steps"),
                (seed, "Seed"),
                (sampler_index, "Sampler"),
                (cfg_scale, "CFG scale"),
                (width, "Size-1"),
                (height, "Size-2"),
                (subseed, "Variation seed"),
                (subseed_strength, "Variation strength"),
                (full_quality, "Full quality"),
                (clip_skip, "Clip skip"),
                (latent_index, "Latent sampler"),
                (latent_index, "Secondary sampler"),
                (denoising_strength, "Denoising strength"),
                (restore_faces, "Face restoration"),
                (batch_size, "Batch size"),
                (batch_count, "Batch count"),
                (seed_resize_from_w, "Seed resize from-1"),
                (seed_resize_from_h, "Seed resize from-2"),
                (resize_mode, "Resize mode"),
                (image_cfg_scale, "Image CFG scale"),
                (diffusers_guidance_rescale, "CFG rescale"),
                (tiling, "Tiling"),
                (mask_blur, "Mask blur"),
                # TODO scale_by add to paste fields
                (scale_by, "UNKNOWN"),
                # from txt2img
                (hr_force, "Hires force"),
                (hr_scale, "Hires upscale"),
                (hr_upscaler, "Hires upscaler"),
                (hr_second_pass_steps, "Hires steps"),
                (hr_second_pass_steps, "Hires steps"),
                (hr_resize_x, "Hires resize-1"),
                (hr_resize_y, "Hires resize-2"),
                (refiner_steps, "Refiner steps"),
                (refiner_start, "Refiner start"),
                (refiner_prompt, "Prompt2"),
                (refiner_negative, "Negative2"),
                *modules.scripts.scripts_img2img.infotext_fields
            ]
            parameters_copypaste.add_paste_fields("img2img", init_img, img2img_paste_fields, override_settings)
            parameters_copypaste.add_paste_fields("inpaint", init_img_with_mask, img2img_paste_fields, override_settings)
            parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                paste_button=img2img_paste, tabname="img2img", source_text_component=img2img_prompt, source_image_component=None,
            ))

    timer.startup.record("ui-img2img")

    modules.scripts.scripts_current = None

    with gr.Blocks(analytics_enabled=False) as extras_interface:
        ui_postprocessing.create_ui()
        timer.startup.record("ui-extras")

    with gr.Blocks(analytics_enabled=False) as train_interface:
        ui_train.create_ui([txt2img_prompt, txt2img_negative_prompt, steps, sampler_index, cfg_scale, seed, width, height])
        timer.startup.record("ui-train")

    with gr.Blocks(analytics_enabled=False) as models_interface:
        ui_models.create_ui()
        timer.startup.record("ui-models")

    def create_setting_component(key, is_quicksettings=False):
        def fun():
            return opts.data[key] if key in opts.data else opts.data_labels[key].default

        info = opts.data_labels[key]
        t = type(info.default)
        args = (info.component_args() if callable(info.component_args) else info.component_args) or {}
        if info.component is not None:
            comp = info.component
        elif t == str:
            comp = gr.Textbox
        elif t == int:
            comp = gr.Number
        elif t == bool:
            comp = gr.Checkbox
        else:
            raise ValueError(f'bad options item type: {t} for key {key}')
        elem_id = f"setting_{key}"

        if not is_quicksettings:
            dirtyable_setting = gr.Group(elem_classes="dirtyable", visible=args.get("visible", True))
            dirtyable_setting.__enter__()
            dirty_indicator = gr.Button("", elem_classes="modification-indicator", elem_id="modification_indicator_" + key)

        if info.refresh is not None:
            if is_quicksettings:
                res = comp(label=info.label, value=fun(), elem_id=elem_id, **args)
                ui_common.create_refresh_button(res, info.refresh, info.component_args, f"refresh_{key}")
            else:
                with FormRow():
                    res = comp(label=info.label, value=fun(), elem_id=elem_id, **args)
                    ui_common.create_refresh_button(res, info.refresh, info.component_args, f"refresh_{key}")
        elif info.folder is not None:
            with FormRow():
                res = comp(label=info.label, value=fun(), elem_id=elem_id, elem_classes="folder-selector", **args)
                # ui_common.create_browse_button(res, f"folder_{key}")
        else:
            try:
                res = comp(label=info.label, value=fun(), elem_id=elem_id, **args)
            except Exception as e:
                log.error(f'Error creating setting: {key} {e}')
                res = None

        if res is not None and not is_quicksettings:
            res.change(fn=None, inputs=res, _js=f'(val) => markIfModified("{key}", val)')
            dirty_indicator.click(fn=lambda: getattr(opts, key), outputs=res, show_progress=False)
            dirtyable_setting.__exit__()

        return res

    def create_dirty_indicator(key, keys_to_reset, **kwargs):
        def get_opt_values():
            return [getattr(opts, _key) for _key in keys_to_reset]

        elements_to_reset = [component_dict[_key] for _key in keys_to_reset if component_dict[_key] is not None]
        indicator = gr.Button("", elem_classes="modification-indicator", elem_id=f"modification_indicator_{key}", **kwargs)
        indicator.click(fn=get_opt_values, outputs=elements_to_reset, show_progress=False)
        return indicator

    loadsave = ui_loadsave.UiLoadsave(cmd_opts.ui_config)
    components = []
    component_dict = {}
    modules.shared.settings_components = component_dict

    script_callbacks.ui_settings_callback()
    opts.reorder()

    def run_settings(*args):
        changed = []
        for key, value, comp in zip(opts.data_labels.keys(), args, components):
            if comp == dummy_component:
                continue
            if not opts.same_type(value, opts.data_labels[key].default):
                log.error(f'Setting bad value: {key}={value} expecting={type(opts.data_labels[key].default).__name__}')
                continue
            if opts.set(key, value):
                changed.append(key)
        if cmd_opts.use_directml:
            directml_override_opts()
        if cmd_opts.use_openvino:
            if not modules.shared.opts.cuda_compile:
                modules.shared.log.warning("OpenVINO: Enabling Torch Compile")
                modules.shared.opts.cuda_compile = True
            if modules.shared.opts.cuda_compile_backend != "openvino_fx":
                modules.shared.log.warning("OpenVINO: Setting Torch Compiler backend to OpenVINO FX")
                modules.shared.opts.cuda_compile_backend = "openvino_fx"
            if modules.shared.opts.sd_backend != "diffusers":
                modules.shared.log.warning("OpenVINO: Setting backend to Diffusers")
                modules.shared.opts.sd_backend = "diffusers"
        try:
            opts.save(modules.shared.config_filename)
            if len(changed) > 0:
                log.info(f'Settings: changed={len(changed)} {changed}')
        except RuntimeError:
            log.error(f'Settings failed: change={len(changed)} {changed}')
            return opts.dumpjson(), f'{len(changed)} Settings changed without save: {", ".join(changed)}'
        return opts.dumpjson(), f'{len(changed)} Settings changed{": " if len(changed) > 0 else ""}{", ".join(changed)}'

    def run_settings_single(value, key):
        if not opts.same_type(value, opts.data_labels[key].default):
            return gr.update(visible=True), opts.dumpjson()
        if not opts.set(key, value):
            return gr.update(value=getattr(opts, key)), opts.dumpjson()
        if cmd_opts.use_directml:
            directml_override_opts()
        opts.save(modules.shared.config_filename)
        log.debug(f'Setting changed: key={key}, value={value}')
        return get_value_for_setting(key), opts.dumpjson()

    with gr.Blocks(analytics_enabled=False) as settings_interface:
        with gr.Row(elem_id="system_row"):
            restart_submit = gr.Button(value="Restart server", variant='primary', elem_id="restart_submit")
            shutdown_submit = gr.Button(value="Shutdown server", variant='primary', elem_id="shutdown_submit")
            unload_sd_model = gr.Button(value='Unload checkpoint', variant='primary', elem_id="sett_unload_sd_model")
            reload_sd_model = gr.Button(value='Reload checkpoint', variant='primary', elem_id="sett_reload_sd_model")

        with gr.Tabs(elem_id="system") as system_tabs:
            global ui_system_tabs # pylint: disable=global-statement
            ui_system_tabs = system_tabs
            with gr.TabItem("Settings", id="system_settings", elem_id="tab_settings"):
                with gr.Row(elem_id="settings_row"):
                    settings_submit = gr.Button(value="Apply settings", variant='primary', elem_id="settings_submit")
                    preview_theme = gr.Button(value="Preview theme", variant='primary', elem_id="settings_preview_theme")
                    defaults_submit = gr.Button(value="Restore defaults", variant='primary', elem_id="defaults_submit")
                with gr.Row():
                    _settings_search = gr.Text(label="Search", elem_id="settings_search")

                result = gr.HTML(elem_id="settings_result")
                quicksettings_names = opts.quicksettings_list
                quicksettings_names = {x: i for i, x in enumerate(quicksettings_names) if x != 'quicksettings'}
                quicksettings_list = []

                previous_section = []
                tab_item_keys = []
                current_tab = None
                current_row = None
                with gr.Tabs(elem_id="settings"):
                    for i, (k, item) in enumerate(opts.data_labels.items()):
                        section_must_be_skipped = item.section[0] is None
                        if previous_section != item.section and not section_must_be_skipped:
                            elem_id, text = item.section
                            if current_tab is not None and len(previous_section) > 0:
                                create_dirty_indicator(previous_section[0], tab_item_keys)
                                tab_item_keys = []
                                current_row.__exit__()
                                current_tab.__exit__()
                            current_tab = gr.TabItem(elem_id=f"settings_{elem_id}", label=text)
                            current_tab.__enter__()
                            current_row = gr.Column(variant='compact')
                            current_row.__enter__()
                            previous_section = item.section
                        if k in quicksettings_names and not modules.shared.cmd_opts.freeze:
                            quicksettings_list.append((i, k, item))
                            components.append(dummy_component)
                        elif section_must_be_skipped:
                            components.append(dummy_component)
                        else:
                            component = create_setting_component(k)
                            component_dict[k] = component
                            tab_item_keys.append(k)
                            components.append(component)
                    if current_tab is not None and len(previous_section) > 0:
                        create_dirty_indicator(previous_section[0], tab_item_keys)
                        tab_item_keys = []
                        current_row.__exit__()
                        current_tab.__exit__()

                    request_notifications = gr.Button(value='Request browser notifications', elem_id="request_notifications", visible=False)
                    with gr.TabItem("Show all pages", elem_id="settings_show_all_pages"):
                        create_dirty_indicator("show_all_pages", [], interactive=False)

            with gr.TabItem("User interface", id="system_config", elem_id="tab_config"):
                loadsave.create_ui()
                create_dirty_indicator("tab_defaults", [], interactive=False)

            with gr.TabItem("Change log", id="change_log", elem_id="system_tab_changelog"):
                with open('CHANGELOG.md', 'r', encoding='utf-8') as f:
                    md = f.read()
                gr.Markdown(md)

            with gr.TabItem("Licenses", id="system_licenses", elem_id="system_tab_licenses"):
                gr.HTML(modules.shared.html("licenses.html"), elem_id="licenses", elem_classes="licenses")
                create_dirty_indicator("tab_licenses", [], interactive=False)

        def unload_sd_weights():
            modules.sd_models.unload_model_weights(op='model')
            modules.sd_models.unload_model_weights(op='refiner')

        def reload_sd_weights():
            modules.sd_models.reload_model_weights()

        unload_sd_model.click(fn=unload_sd_weights, inputs=[], outputs=[])
        reload_sd_model.click(fn=reload_sd_weights, inputs=[], outputs=[])
        request_notifications.click(fn=lambda: None, inputs=[], outputs=[], _js='function(){}')
        preview_theme.click(fn=None, _js='previewTheme', inputs=[], outputs=[])

    timer.startup.record("ui-settings")

    interfaces = [
        (txt2img_interface, "From Text", "txt2img"),
        (img2img_interface, "From Image", "img2img"),
        (extras_interface, "Process Image", "process"),
        (train_interface, "Train", "train"),
        (models_interface, "Models", "models"),
    ]
    interfaces += script_callbacks.ui_tabs_callback()
    interfaces += [(settings_interface, "System", "system")]
    extensions_interface = ui_extensions.create_ui()
    interfaces += [(extensions_interface, "Extensions", "extensions")]
    timer.startup.record("ui-extensions")

    modules.shared.tab_names = []
    for _interface, label, _ifid in interfaces:
        modules.shared.tab_names.append(label)

    with gr.Blocks(theme=modules.theme.gradio_theme, analytics_enabled=False, title="SD.Next") as demo:
        with gr.Row(elem_id="quicksettings", variant="compact"):
            for _i, k, _item in sorted(quicksettings_list, key=lambda x: quicksettings_names.get(x[1], x[0])):
                component = create_setting_component(k, is_quicksettings=True)
                component_dict[k] = component

        parameters_copypaste.connect_paste_params_buttons()

        with gr.Tabs(elem_id="tabs") as tabs:
            for interface, label, ifid in interfaces:
                if interface is None:
                    continue
                # if label in modules.shared.opts.hidden_tabs or label == '':
                #    continue
                with gr.TabItem(label, id=ifid, elem_id=f"tab_{ifid}"):
                    interface.render()
            for interface, _label, ifid in interfaces:
                if interface is None:
                    continue
                if ifid in ["extensions", "system"]:
                    continue
                loadsave.add_block(interface, ifid)
            loadsave.add_component(f"webui/Tabs@{tabs.elem_id}", tabs)
            loadsave.setup_ui()
        if opts.notification_audio_enable and os.path.exists(os.path.join(script_path, opts.notification_audio_path)):
            gr.Audio(interactive=False, value=os.path.join(script_path, opts.notification_audio_path), elem_id="audio_notification", visible=False)

        text_settings = gr.Textbox(elem_id="settings_json", value=lambda: opts.dumpjson(), visible=False)
        components = [c for c in components if c is not None]
        settings_submit.click(
            fn=wrap_gradio_call(run_settings, extra_outputs=[gr.update()]),
            inputs=components,
            outputs=[text_settings, result],
        )
        defaults_submit.click(fn=lambda: modules.shared.restore_defaults(restart=True), _js="restart_reload")
        restart_submit.click(fn=lambda: modules.shared.restart_server(restart=True), _js="restart_reload")
        shutdown_submit.click(fn=lambda: modules.shared.restart_server(restart=False), _js="restart_reload")

        for _i, k, _item in quicksettings_list:
            component = component_dict[k]
            info = opts.data_labels[k]
            change_handler = component.release if hasattr(component, 'release') else component.change
            change_handler(
                fn=lambda value, k=k: run_settings_single(value, key=k),
                inputs=[component],
                outputs=[component, text_settings],
                show_progress=info.refresh is not None,
            )

        button_set_checkpoint = gr.Button('Change model', elem_id='change_checkpoint', visible=False)
        button_set_checkpoint.click(
            fn=lambda value, _: run_settings_single(value, key='sd_model_checkpoint'),
            _js="function(v){ var res = desiredCheckpointName; desiredCheckpointName = ''; return [res || v, null]; }",
            inputs=[component_dict['sd_model_checkpoint'], dummy_component],
            outputs=[component_dict['sd_model_checkpoint'], text_settings],
        )
        button_set_refiner = gr.Button('Change refiner', elem_id='change_refiner', visible=False)
        button_set_refiner.click(
            fn=lambda value, _: run_settings_single(value, key='sd_model_checkpoint'),
            _js="function(v){ var res = desiredCheckpointName; desiredCheckpointName = ''; return [res || v, null]; }",
            inputs=[component_dict['sd_model_refiner'], dummy_component],
            outputs=[component_dict['sd_model_refiner'], text_settings],
        )
        button_set_vae = gr.Button('Change VAE', elem_id='change_vae', visible=False)
        button_set_vae.click(
            fn=lambda value, _: run_settings_single(value, key='sd_vae'),
            _js="function(v){ var res = desiredVAEName; desiredVAEName = ''; return [res || v, null]; }",
            inputs=[component_dict['sd_vae'], dummy_component],
            outputs=[component_dict['sd_vae'], text_settings],
        )

        component_keys = [k for k in opts.data_labels.keys() if k in component_dict]

        def get_settings_values():
            return [get_value_for_setting(key) for key in component_keys]

        demo.load(
            fn=get_settings_values,
            inputs=[],
            outputs=[component_dict[k] for k in component_keys if component_dict[k] is not None],
            queue=False,
        )

    timer.startup.record("ui-defaults")
    loadsave.dump_defaults()
    demo.ui_loadsave = loadsave
    return demo


def webpath(fn):
    if fn.startswith(script_path):
        web_path = os.path.relpath(fn, script_path).replace('\\', '/')
    else:
        web_path = os.path.abspath(fn)
    return f'file={web_path}?{os.path.getmtime(fn)}'


def html_head():
    script_js = os.path.join(script_path, "javascript", "script.js")
    head = f'<script type="text/javascript" src="{webpath(script_js)}"></script>\n'
    added = []
    for script in modules.scripts.list_scripts("javascript", ".js"):
        if script.path == script_js:
            continue
        head += f'<script type="text/javascript" src="{webpath(script.path)}"></script>\n'
        added.append(script.path)
    for script in modules.scripts.list_scripts("javascript", ".mjs"):
        head += f'<script type="module" src="{webpath(script.path)}"></script>\n'
        added.append(script.path)
    added = [a.replace(script_path, '').replace('\\', '/') for a in added]
    # log.debug(f'Adding JS scripts: {added}')
    return head


def html_body():
    body = ''
    inline = ''
    if opts.theme_style != 'Auto':
        inline += f"set_theme('{opts.theme_style.lower()}');"
    body += f'<script type="text/javascript">{inline}</script>\n'
    return body


def html_css(is_builtin: bool):
    added = []

    def stylesheet(fn):
        added.append(fn)
        return f'<link rel="stylesheet" property="stylesheet" href="{webpath(fn)}">'

    css = 'sdnext.css' if is_builtin else 'base.css'
    head = stylesheet(os.path.join(script_path, 'javascript', css))
    for cssfile in modules.scripts.list_files_with_name("style.css"):
        if not os.path.isfile(cssfile):
            continue
        head += stylesheet(cssfile)
    if opts.gradio_theme in modules.theme.list_builtin_themes():
        head += stylesheet(os.path.join(script_path, "javascript", f"{opts.gradio_theme}.css"))
    if os.path.exists(os.path.join(data_path, "user.css")):
        head += stylesheet(os.path.join(data_path, "user.css"))
    added = [a.replace(script_path, '').replace('\\', '/') for a in added]
    # log.debug(f'Adding CSS stylesheets: {added}')
    return head


def reload_javascript():
    is_builtin = modules.theme.reload_gradio_theme()
    head = html_head()
    css = html_css(is_builtin)
    body = html_body()

    def template_response(*args, **kwargs):
        res = modules.shared.GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b'</head>', f'{head}</head>'.encode("utf8"))
        res.body = res.body.replace(b'</body>', f'{css}{body}</body>'.encode("utf8"))
        res.init_headers()
        return res

    gradio.routes.templates.TemplateResponse = template_response


def setup_ui_api(app):
    from pydantic import BaseModel, Field # pylint: disable=no-name-in-module
    from typing import List

    class QuicksettingsHint(BaseModel): # pylint: disable=too-few-public-methods
        name: str = Field(title="Name of the quicksettings field")
        label: str = Field(title="Label of the quicksettings field")

    def quicksettings_hint():
        return [QuicksettingsHint(name=k, label=v.label) for k, v in opts.data_labels.items()]

    app.add_api_route("/internal/quicksettings-hint", quicksettings_hint, methods=["GET"], response_model=List[QuicksettingsHint])
    app.add_api_route("/internal/ping", lambda: {}, methods=["GET"])


if not hasattr(modules.shared, 'GradioTemplateResponseOriginal'):
    modules.shared.GradioTemplateResponseOriginal = gradio.routes.templates.TemplateResponse
