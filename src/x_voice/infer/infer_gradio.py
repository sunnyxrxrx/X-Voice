import argparse
import logging
import warnings
from functools import lru_cache
from importlib.resources import files

import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from hydra.utils import get_class
from omegaconf import OmegaConf


def _silence_inference_logs():
    warnings.filterwarnings("ignore")
    logging.captureWarnings(True)
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)
    for logger_name in (
        "NeMo-text-processing",
        "tokenize_and_classify.py",
        "DF",
        "df",
        "fastlid",
        "logzero",
        "transformers",
        "huggingface_hub",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    try:
        from loguru import logger as loguru_logger

        loguru_logger.remove()
    except Exception as exc:
        print(f"[WARN] Failed to silence loguru logger: {exc}")


_silence_inference_logs()

from x_voice.infer.utils_infer import (
    detect_segment_lang,
    device,
    ensure_ref_text_punctuation,
    get_ipa_tokenizer_cache,
    infer_xvoice_droptext_process,
    infer_xvoice_process,
    load_model,
    load_model_sft,
    load_srp_model,
    load_vocoder,
    nfe_step,
    normalize_lang_code,
    normalize_text_for_lang,
    preprocess_ref_audio_text,
)


try:
    import spaces

    USING_SPACES = True
except ImportError:
    spaces = None
    USING_SPACES = False


def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    return func


STAGE1_MODEL = "X-Voice Stage1"
STAGE2_MODEL = "X-Voice Stage2"
STAGE2_REF_TEXT = "X-Voice Stage2 does not need reference text."

HF_REPO_ID = "XRXRX/X-Voice"
STAGE1_CKPT = "XVoice_Base_Stage1/model_600000.safetensors"
STAGE2_CKPT = "XVoice_Base_Stage2/model_70000.safetensors"
SRP_CKPT = "SpeedPredictor/model_28000.safetensors"
VOCAB_FILE = "XVoice_Base_Stage1/vocab.txt"

STAGE1_CFG = str(files("x_voice").joinpath("configs/XVoice_Base_Stage1.yaml"))
STAGE2_CFG = str(files("x_voice").joinpath("configs/XVoice_Base_Stage2.yaml"))
SRP_CFG = str(files("srp").joinpath("configs/SpeedPredict_Multilingual.yaml"))
EXAMPLE_REF_EN = str(files("x_voice").joinpath("infer/examples/gradio_sample/ref_en.wav"))
EXAMPLE_REF_ZH = str(files("x_voice").joinpath("infer/examples/gradio_sample/ref_zh.wav"))

VOCODER_NAME = "vocos"
TARGET_RMS = 0.1
CROSS_FADE_DURATION = 0.15
NFE_STEP = 32
CFG_STRENGTH = 2.5
CFG_STRENGTH2 = 4.0
CFG_SCHEDULE = "square"
CFG_DECAY_TIME = 0.6
SWAY_SAMPLING_COEF = -1.0
SPEED = 1.0
FIX_DURATION = None
REVERSE = False
NORMALIZE_TEXT = True
DENOISE_REF = True
LOUDNESS_NORM = True
POST_PROCESSING = True


vocoder = None
stage1_ema_model = None
stage2_ema_model = None
srp_model = None
stage1_runtime = None
stage2_runtime = None
normalizer_cache = {}


def download_default_file(filename):
    return hf_hub_download(repo_id=HF_REPO_ID, filename=filename)


def load_model_runtime(model_cfg_file, ckpt_filename, vocab_file, sft=False):
    ckpt_path = download_default_file(ckpt_filename)
    vocab_path = download_default_file(vocab_file)
    model_cfg = OmegaConf.load(model_cfg_file)
    model_cls = get_class(f"x_voice.model.{model_cfg.model.backbone}")
    model_arc = OmegaConf.to_container(model_cfg.model.arch, resolve=True)
    mel_spec_kwargs = OmegaConf.to_container(model_cfg.model.mel_spec, resolve=True)
    tokenizer = model_cfg.model.tokenizer
    tokenizer_path = model_cfg.model.get("tokenizer_path", None)
    dataset_name = model_cfg.datasets.name
    stress = bool(model_cfg.model.get("stress", True))

    if sft:
        ema_model = load_model_sft(
            model_cls,
            model_arc,
            ckpt_path,
            mel_spec_type=VOCODER_NAME,
            vocab_file=vocab_path,
            device=device,
            use_total_text=bool(model_cfg.model.get("use_total_text", False)),
            tokenizer=tokenizer,
            tokenizer_path=tokenizer_path,
            dataset_name=dataset_name,
            mel_spec_kwargs=mel_spec_kwargs,
        )
    else:
        ema_model = load_model(
            model_cls,
            model_arc,
            ckpt_path,
            mel_spec_type=VOCODER_NAME,
            vocab_file=vocab_path,
            device=device,
            tokenizer=tokenizer,
            tokenizer_path=tokenizer_path,
            dataset_name=dataset_name,
            mel_spec_kwargs=mel_spec_kwargs,
        )

    return {
        "model": ema_model,
        "tokenizer": tokenizer,
        "ipa_tokenizer_getter": get_ipa_tokenizer_cache(tokenizer, stress),
        "lang_to_id_map": getattr(ema_model.transformer, "lang_to_id", {}),
    }


def get_vocoder():
    global vocoder
    if vocoder is None:
        vocoder = load_vocoder(vocoder_name=VOCODER_NAME, is_local=False, device=device)
    return vocoder


def get_stage1_runtime(show_info=gr.Info):
    global stage1_ema_model, stage1_runtime
    if stage1_runtime is None:
        show_info("Loading X-Voice Stage1 model...")
        stage1_runtime = load_model_runtime(STAGE1_CFG, STAGE1_CKPT, VOCAB_FILE, sft=False)
        stage1_ema_model = stage1_runtime["model"]
    return stage1_runtime


def get_stage2_runtime(show_info=gr.Info):
    global stage2_ema_model, stage2_runtime
    if stage2_runtime is None:
        show_info("Loading X-Voice Stage2 model...")
        stage2_runtime = load_model_runtime(STAGE2_CFG, STAGE2_CKPT, VOCAB_FILE, sft=True)
        stage2_ema_model = stage2_runtime["model"]
    return stage2_runtime


def get_srp_model(show_info=gr.Info):
    global srp_model
    if srp_model is None:
        show_info("Loading X-Voice SRP duration model...")
        srp_model = load_srp_model(SRP_CFG, download_default_file(SRP_CKPT), device)
    return srp_model


def detect_required_lang(text, label):
    lang = normalize_lang_code(detect_segment_lang(text, None))
    if not lang:
        raise ValueError(f"Failed to detect {label} language. Please check fastlid installation and input text.")
    return lang


def normalize_required_text(text, lang):
    return normalize_text_for_lang(text, lang, normalizer_cache) if NORMALIZE_TEXT else text


def preprocess_stage1_ref(ref_audio, ref_text, show_info=gr.Info):
    processed_audio, processed_text = preprocess_ref_audio_text(ref_audio, ref_text.strip(), show_info=show_info)
    ref_lang = detect_required_lang(processed_text, "reference text")
    processed_text = normalize_required_text(processed_text, ref_lang)
    processed_text = ensure_ref_text_punctuation(processed_text)
    return processed_audio, processed_text, ref_lang


def preprocess_stage2_ref(ref_audio, show_info=gr.Info):
    processed_audio, _ = preprocess_ref_audio_text(ref_audio, STAGE2_REF_TEXT, show_info=show_info)
    return processed_audio


@lru_cache(maxsize=1000)
@gpu_decorator
def infer(ref_audio, ref_text, gen_text, model_choice, show_info=gr.Info):
    if not ref_audio:
        gr.Warning("Please provide reference audio.")
        return gr.update(), ref_text

    if not gen_text or not gen_text.strip():
        gr.Warning("Please enter text to generate.")
        return gr.update(), ref_text

    torch.manual_seed(np.random.randint(0, 2**31 - 1))
    current_vocoder = get_vocoder()

    try:
        if model_choice == STAGE1_MODEL:
            runtime = get_stage1_runtime(show_info=show_info)
            ref_audio, ref_text, ref_lang = preprocess_stage1_ref(ref_audio, ref_text, show_info=show_info)
            gen_lang = detect_required_lang(gen_text, "generated text")
            gen_text = normalize_required_text(gen_text.strip(), gen_lang)

            show_info(f"Detected languages: ref={ref_lang}, gen={gen_lang}")
            final_wave, final_sample_rate, _ = infer_xvoice_process(
                ref_audio,
                ref_text,
                gen_text,
                ref_lang,
                gen_lang,
                runtime["tokenizer"],
                runtime["ipa_tokenizer_getter"],
                runtime["model"],
                current_vocoder,
                runtime["lang_to_id_map"],
                srp_model=None,
                mel_spec_type_value=VOCODER_NAME,
                progress=gr.Progress(),
                target_rms_value=TARGET_RMS,
                cross_fade_duration_value=CROSS_FADE_DURATION,
                nfe_step_value=NFE_STEP,
                cfg_strength_value=CFG_STRENGTH,
                cfg_strength2_value=CFG_STRENGTH2,
                cfg_schedule_value=CFG_SCHEDULE,
                cfg_decay_time_value=CFG_DECAY_TIME,
                sway_sampling_coef_value=SWAY_SAMPLING_COEF,
                local_speed=SPEED,
                fix_duration_value=FIX_DURATION,
                sp_type="syllable",
                reverse=REVERSE,
                denoise_ref=DENOISE_REF,
                loudness_norm=LOUDNESS_NORM,
                post_processing=POST_PROCESSING,
                device_name=device,
            )
            return (final_sample_rate, final_wave), ref_text

        runtime = get_stage2_runtime(show_info=show_info)
        duration_model = get_srp_model(show_info=show_info)
        ref_audio = preprocess_stage2_ref(ref_audio, show_info=show_info)
        gen_lang = detect_required_lang(gen_text, "generated text")
        gen_text = normalize_required_text(gen_text.strip(), gen_lang)

        show_info(f"Detected language: gen={gen_lang}")
        final_wave, final_sample_rate, _ = infer_xvoice_droptext_process(
            ref_audio,
            gen_text,
            gen_lang,
            runtime["tokenizer"],
            runtime["ipa_tokenizer_getter"],
            runtime["model"],
            current_vocoder,
            runtime["lang_to_id_map"],
            duration_model,
            mel_spec_type_value=VOCODER_NAME,
            progress=gr.Progress(),
            target_rms_value=TARGET_RMS,
            cross_fade_duration_value=CROSS_FADE_DURATION,
            nfe_step_value=NFE_STEP,
            cfg_strength_value=CFG_STRENGTH,
            cfg_strength2_value=CFG_STRENGTH2,
            cfg_schedule_value=CFG_SCHEDULE,
            cfg_decay_time_value=CFG_DECAY_TIME,
            sway_sampling_coef_value=SWAY_SAMPLING_COEF,
            local_speed=SPEED,
            fix_duration_value=FIX_DURATION,
            reverse=REVERSE,
            denoise_ref=DENOISE_REF,
            loudness_norm=LOUDNESS_NORM,
            post_processing=POST_PROCESSING,
            device_name=device,
        )
        return (final_sample_rate, final_wave), STAGE2_REF_TEXT
    except Exception as exc:
        gr.Warning(str(exc))
        if model_choice == STAGE2_MODEL:
            return gr.update(), STAGE2_REF_TEXT
        return gr.update(), ref_text


def switch_model(model_choice, current_ref_text):
    if model_choice == STAGE2_MODEL:
        return gr.update(value=STAGE2_REF_TEXT, interactive=False)
    if current_ref_text == STAGE2_REF_TEXT:
        current_ref_text = ""
    return gr.update(
        value=current_ref_text,
        interactive=True,
        placeholder="Optional for Stage1. Leave empty to transcribe with Whisper.",
    )


with gr.Blocks() as app:
    gr.Markdown(
        """
# X-Voice Online Demo

Clone a reference voice and generate natural speech from text in any of 30 supported languages.

Stage 1 requires the reference voice to be in one of the 30 supported languages, while Stage 2 can use a reference voice in any language.
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            choose_model = gr.Radio(
                choices=[STAGE1_MODEL, STAGE2_MODEL],
                label="Choose Model",
                value=STAGE1_MODEL,
            )
            ref_audio_input = gr.Audio(label="Reference Audio", type="filepath")
            ref_text_input = gr.Textbox(
                label="Reference Text",
                lines=3,
                placeholder="Optional for Stage1. Leave empty to transcribe with Whisper.",
            )
            gen_text_input = gr.Textbox(label="Text to Generate", lines=8)
            generate_btn = gr.Button("Synthesize", variant="primary")

        with gr.Column(scale=1):
            audio_output = gr.Audio(label="Generated Audio")
            gr.Examples(
                examples=[
                    [EXAMPLE_REF_EN, "Some call me nature, others call me mother nature"],
                    [EXAMPLE_REF_ZH, "对，这就是我，万人敬仰的太乙真人"],
                ],
                inputs=[ref_audio_input, ref_text_input],
                label="Example Prompts",
            )

    choose_model.change(
        switch_model,
        inputs=[choose_model, ref_text_input],
        outputs=[ref_text_input],
    )
    generate_btn.click(
        infer,
        inputs=[ref_audio_input, ref_text_input, gen_text_input, choose_model],
        outputs=[audio_output, ref_text_input],
    )


def main():
    parser = argparse.ArgumentParser(description="X-Voice Gradio inference app.")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--root_path", type=str, default=None)
    parser.add_argument("--inbrowser", action="store_true")
    args = parser.parse_args()

    app.queue(api_open=args.api).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        root_path=args.root_path,
        inbrowser=args.inbrowser,
    )


if __name__ == "__main__":
    main()

# python -m x_voice.infer.infer_gradio --host 0.0.0.0 --port 7860
