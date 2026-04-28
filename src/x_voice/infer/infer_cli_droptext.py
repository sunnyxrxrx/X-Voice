import argparse
import codecs
import logging
import os
import re
import warnings
from datetime import datetime
from importlib.resources import files
from pathlib import Path

# import debugpy
# debugpy.listen(("localhost",567))
# debugpy.wait_for_client()

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
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    try:
        from loguru import logger as loguru_logger

        loguru_logger.remove()
    except Exception:
        pass


_silence_inference_logs()

import numpy as np
import soundfile as sf
import tomli
from hydra.utils import get_class
from omegaconf import OmegaConf
from unidecode import unidecode

from x_voice.infer.utils_infer import (
    cfg_strength,
    cross_fade_duration,
    detect_segment_lang,
    device,
    fix_duration,
    get_ipa_tokenizer_cache,
    infer_xvoice_droptext_process,
    load_model_sft,
    load_srp_model,
    load_vocoder,
    nfe_step,
    normalize_lang_code,
    normalize_text_for_lang,
    parse_voice_lang_tag,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    resolve_cached_path,
    resolve_ckpt_path,
    resolve_package_example,
    speed,
    sway_sampling_coef,
    target_rms,
    layered,
)


DEFAULT_MODEL = "XVoice_Base_Stage2"
DEFAULT_SRP_CFG = str(files("srp").joinpath("configs/SpeedPredict_Multilingual.yaml"))
DROP_TEXT_PLACEHOLDER = "Drop-text mode ignores reference text."


parser = argparse.ArgumentParser(
    prog="python3 infer_cli_droptext.py",
    description="Command line interface for X-Voice Stage2 drop-text inference.",
    epilog="Specify options above to override one or more settings from config.",
)
parser.add_argument("-c", "--config", type=str, help="Optional TOML configuration file.")
parser.add_argument("-m", "--model", type=str, help="Model config name under x_voice/configs.")
parser.add_argument("-mc", "--model_cfg", type=str, help="Path to model config .yaml.")
parser.add_argument("-p", "--ckpt_file", type=str, help="Path to model checkpoint .pt/.safetensors.")
parser.add_argument("--ckpt_step", type=int, help="Checkpoint step used when ckpt_file is not set.")
parser.add_argument("--srp_ckpt_file", type=str, help="Path to SRP checkpoint. Required.")
parser.add_argument("--srp_model_cfg", type=str, help="Path to SRP config .yaml.")
parser.add_argument("-v", "--vocab_file", type=str, help="Path to vocab file .txt.")
parser.add_argument("-r", "--ref_audio", type=str, help="Reference audio file.")
parser.add_argument("-s", "--ref_text", type=str, help="Ignored in drop-text mode; kept for config compatibility.")
parser.add_argument("-t", "--gen_text", type=str, help="Text to synthesize.")
parser.add_argument("-f", "--gen_file", type=str, help="Text file to synthesize; overrides gen_text.")
parser.add_argument("-o", "--output_dir", type=str, help="Output folder.")
parser.add_argument("-w", "--output_file", type=str, help="Output wav name.")
parser.add_argument("--save_chunk", action="store_true", help="Save each generated chunk.")
parser.add_argument("--no_legacy_text", action="store_false", help="Disable legacy ASCII chunk filenames.")
parser.add_argument("--remove_silence", action="store_true", help="Remove long silence from output wav.")
parser.add_argument("--load_vocoder_from_local", action="store_true", help="Load vocoder from local path.")
parser.add_argument("--vocoder_name", type=str, choices=["vocos", "bigvgan"], help="Vocoder name.")
parser.add_argument("--target_rms", type=float, help=f"Target RMS, default {target_rms}.")
parser.add_argument("--cross_fade_duration", type=float, help=f"Cross-fade duration, default {cross_fade_duration}.")
parser.add_argument("--nfe_step", type=int, help=f"Sampling steps, default {nfe_step}.")
parser.add_argument("--cfg_strength", type=float, help=f"CFG strength, default {cfg_strength}.")
parser.add_argument("--layered", action="store_true", help=f"Decoupled CFG.")
parser.add_argument("--cfg_strength2", type=float, help="Secondary CFG strength for mandatory layered CFG.")
parser.add_argument("--cfg_schedule", type=str, choices=["square", "cosine", "none"], help="CFG schedule.")
parser.add_argument("--cfg_decay_time", type=float, default=None, help="CFG schedule decay start time.")
parser.add_argument("--sway_sampling_coef", type=float, help=f"Sway sampling coefficient, default {sway_sampling_coef}.")
parser.add_argument("--speed", type=float, help=f"Speed multiplier after SRP prediction, default {speed}.")
parser.add_argument("--fix_duration", type=float, help=f"Fixed total duration in seconds, default {fix_duration}.")
parser.add_argument("--device", type=str, help="Device to run on.")
parser.add_argument("--gen_lang", type=str, help="Generated text language code.")
parser.add_argument("--tokenizer", type=str, help="Override tokenizer from model yaml.")
parser.add_argument("--auto_detect_lang", action="store_true", help="Auto-detect generation language when absent.")
parser.add_argument("--normalize_text", action="store_true", help="Normalize text by language.")
parser.add_argument("--denoise_ref", action="store_true", help="Denoise reference audio before inference.")
parser.add_argument("--loudness_norm", action="store_true", help="Normalize generated loudness.")
parser.add_argument("--post_processing", action="store_true", help="Apply generated mel post-processing.")
parser.add_argument("--reverse", action="store_true", help="Place reference audio at the end in sampling.")
args = parser.parse_args()


config = tomli.load(open(args.config, "rb")) if args.config else {}

model = args.model or config.get("model", DEFAULT_MODEL)
model_cfg_file = args.model_cfg or config.get("model_cfg") or str(files("x_voice").joinpath(f"configs/{model}.yaml"))
model_cfg = OmegaConf.load(model_cfg_file)
model_cls = get_class(f"x_voice.model.{model_cfg.model.backbone}")
model_arc = OmegaConf.to_container(model_cfg.model.arch, resolve=True)

ckpt_file = args.ckpt_file or config.get("ckpt_file", "")
ckpt_step = args.ckpt_step if args.ckpt_step is not None else config.get("ckpt_step")
vocab_file = args.vocab_file or config.get("vocab_file", "")
srp_ckpt_file = args.srp_ckpt_file or config.get("srp_ckpt_file", "")
srp_model_cfg_file = args.srp_model_cfg or config.get("srp_model_cfg") or DEFAULT_SRP_CFG

ref_audio = resolve_package_example(args.ref_audio or config.get("ref_audio", "infer/examples/basic/basic_ref_en.wav"))
ref_text = args.ref_text if args.ref_text is not None else config.get("ref_text", "")
gen_text = args.gen_text or config.get("gen_text", "Here we generate something just for test.")
gen_file = resolve_package_example(args.gen_file or config.get("gen_file", ""))

output_dir = args.output_dir or config.get("output_dir", "tests")
output_file = args.output_file or config.get(
    "output_file",
    f"infer_cli_droptext_{datetime.now().strftime(r'%Y%m%d_%H%M%S')}.wav",
)

save_chunk = args.save_chunk or config.get("save_chunk", False)
use_legacy_text = args.no_legacy_text or config.get("no_legacy_text", False)
if save_chunk and use_legacy_text:
    print(
        "\nWarning to --save_chunk: lossy ASCII transliterations of unicode text for legacy (.wav) file names, --no_legacy_text to disable.\n"
    )

remove_silence = args.remove_silence or config.get("remove_silence", False)
load_vocoder_from_local = args.load_vocoder_from_local or config.get("load_vocoder_from_local", False)
auto_detect_lang = args.auto_detect_lang or config.get("auto_detect_lang", False)
normalize_text = args.normalize_text or config.get("normalize_text", False)
denoise_ref = args.denoise_ref or config.get("denoise_ref", False)
loudness_norm = args.loudness_norm or config.get("loudness_norm", False)
post_processing = args.post_processing or config.get("post_processing", False)
reverse = args.reverse or config.get("reverse", False)

tokenizer = args.tokenizer or config.get("tokenizer") or model_cfg.model.tokenizer
tokenizer_path = model_cfg.model.get("tokenizer_path", None)
dataset_name = model_cfg.datasets.name
sft = bool(model_cfg.model.get("sft", False))
use_total_text = bool(model_cfg.model.get("use_total_text", False))
stress = bool(model_cfg.model.get("stress", True))
if not sft:
    raise ValueError("drop-text inference requires a Stage2/SFT model config with model.sft=true.")
if use_total_text:
    raise ValueError("drop-text inference expects model.use_total_text=false.")

vocoder_name = args.vocoder_name or config.get("vocoder_name", model_cfg.model.mel_spec.mel_spec_type)
target_rms = args.target_rms or config.get("target_rms", target_rms)
cross_fade_duration = args.cross_fade_duration or config.get("cross_fade_duration", cross_fade_duration)
nfe_step = args.nfe_step or config.get("nfe_step", nfe_step)
cfg_strength = args.cfg_strength or config.get("cfg_strength", 2.5)
layered = args.layered or config.get("layered", layered)
cfg_strength2 = args.cfg_strength2 if args.cfg_strength2 is not None else config.get("cfg_strength2", 4.0)
cfg_schedule = args.cfg_schedule if args.cfg_schedule is not None else config.get("cfg_schedule", "square")
if cfg_schedule == "none":
    cfg_schedule = None
cfg_decay_time = args.cfg_decay_time if args.cfg_decay_time is not None else config.get("cfg_decay_time", 0.6)
sway_sampling_coef = args.sway_sampling_coef or config.get("sway_sampling_coef", sway_sampling_coef)
speed = args.speed or config.get("speed", speed)
fix_duration = args.fix_duration or config.get("fix_duration", fix_duration)
device = args.device or config.get("device", device)
gen_lang = normalize_lang_code(args.gen_lang or config.get("gen_lang"))
if not auto_detect_lang and not gen_lang:
    raise ValueError("gen_lang is required in config or CLI.")

if ref_text:
    print("Drop-text mode ignores ref_text and uses SRP for duration prediction.")
if not srp_ckpt_file:
    raise ValueError("drop-text mode requires --srp_ckpt_file or srp_ckpt_file in config.")

if gen_file:
    gen_text = codecs.open(gen_file, "r", "utf-8").read()

wave_path = Path(output_dir) / output_file
if save_chunk:
    output_chunk_dir = os.path.join(output_dir, f"{Path(output_file).stem}_chunks")
    if not os.path.exists(output_chunk_dir):
        os.makedirs(output_chunk_dir)

if "voices" in config:
    for voice in config["voices"]:
        config["voices"][voice]["ref_audio"] = resolve_package_example(config["voices"][voice]["ref_audio"])


# load vocoder

vocoder_cfg = model_cfg.model.get("vocoder", {})
if vocoder_name == "vocos":
    vocoder_local_path = vocoder_cfg.get("local_path", "my_vocoder/vocos-mel-24khz")
elif vocoder_name == "bigvgan":
    vocoder_local_path = vocoder_cfg.get("local_path", "my_vocoder/bigvgan_v2_24khz_100band_256x")
else:
    raise ValueError(f"Unsupported vocoder: {vocoder_name}")

vocoder = load_vocoder(
    vocoder_name=vocoder_name,
    is_local=load_vocoder_from_local or bool(vocoder_cfg.get("is_local", False)),
    local_path=vocoder_local_path,
    device=device,
)

if vocoder_name != model_cfg.model.mel_spec.mel_spec_type:
    raise ValueError(f"Vocoder {vocoder_name} does not match model mel spec {model_cfg.model.mel_spec.mel_spec_type}.")


# load TTS model

ckpt_file = resolve_ckpt_path(ckpt_file, model_cfg, model, ckpt_step)
if vocab_file:
    vocab_file = resolve_cached_path(vocab_file)

mel_spec_kwargs = OmegaConf.to_container(model_cfg.model.mel_spec, resolve=True)
print(f"Using {model}...")
print(f"Checkpoint: {ckpt_file}")
print(f"Tokenizer: {tokenizer}")

ema_model = load_model_sft(
    model_cls,
    model_arc,
    ckpt_file,
    mel_spec_type=vocoder_name,
    vocab_file=vocab_file,
    device=device,
    use_total_text=use_total_text,
    tokenizer=tokenizer,
    tokenizer_path=tokenizer_path,
    dataset_name=dataset_name,
    mel_spec_kwargs=mel_spec_kwargs,
)

lang_to_id_map = getattr(ema_model.transformer, "lang_to_id", {})
ipa_tokenizer_getter = get_ipa_tokenizer_cache(tokenizer, stress)
normalizer_cache = {}
srp_model = load_srp_model(srp_model_cfg_file, srp_ckpt_file, device)


# inference process


def main():
    global gen_text
    main_voice = {
        "ref_audio": ref_audio,
        "ref_text": DROP_TEXT_PLACEHOLDER,
        "gen_lang": gen_lang,
    }
    if "voices" not in config:
        voices = {"main": main_voice}
    else:
        voices = config["voices"]
        voices["main"] = main_voice

    for voice in voices:
        print("Voice:", voice)
        print("ref_audio ", voices[voice]["ref_audio"])
        voices[voice]["ref_audio"], _ = preprocess_ref_audio_text(
            voices[voice]["ref_audio"],
            DROP_TEXT_PLACEHOLDER,
        )
        voices[voice]["ref_text"] = DROP_TEXT_PLACEHOLDER
        print("ref_audio_", voices[voice]["ref_audio"], "\n\n")

    generated_audio_segments = []
    reg1 = r"(?=\[[^\[\]]+\])"
    reg2 = r"^\[([^\[\]]+)\]"
    
    chunks = re.split(reg1, gen_text)

    # collect all chunks' information
    segments_info = []
    from x_voice.infer.utils_infer import auto_split_mixed_text
    
    for text in chunks:
        if not text.strip():
            continue
        match = re.match(reg2, text)
        segment_gen_lang = None
        if match:
            voice, segment_gen_lang = parse_voice_lang_tag(match[1], voice_names=voices.keys())
        else:
            print("No voice tag found, using main.")
            voice = "main"

        if voice not in voices:
            print(f"Voice {voice} not found, using main.")
            voice = "main"

        text = re.sub(reg2, "", text, count=1)
        gen_text_ = text.strip()
        if not gen_text_:
            continue

        if segment_gen_lang:
            spans = [(segment_gen_lang, gen_text_)]
        elif auto_detect_lang:
            fallback = normalize_lang_code(voices[voice].get("gen_lang", gen_lang))
            spans = auto_split_mixed_text(gen_text_, fallback)
        else:
            fallback = normalize_lang_code(voices[voice].get("gen_lang", gen_lang))
            if fallback is None:
                raise ValueError(f"gen_lang is required for voice '{voice}'.")
            spans = [(fallback, gen_text_)]

        if normalize_text:
            spans = [
                (span_lang, normalize_text_for_lang(span_text, span_lang, normalizer_cache))
                for span_lang, span_text in spans
            ]

        gen_text_ = "".join(span_text for _, span_text in spans)

        if segments_info and segments_info[-1]["voice"] == voice:
            segments_info[-1]["text"] += gen_text_
            segments_info[-1]["lang"].extend(spans)
        else:
            segments_info.append({
                "voice": voice,
                "text": gen_text_,
                "lang": spans,
            })

    # Grouped aggregation and reasoning by speaker
    voice_to_segments = {}
    for i, seg in enumerate(segments_info):
        voice = seg["voice"]
        if voice not in voice_to_segments:
            voice_to_segments[voice] = {"indices": [], "texts": [], "langs": []}
        voice_to_segments[voice]["indices"].append(i)
        voice_to_segments[voice]["texts"].append(seg["text"])
        voice_to_segments[voice]["langs"].append(seg["lang"])
    
    print(voice_to_segments)
    generated_audio_segments = [None] * len(segments_info)
    
    for voice, data in voice_to_segments.items():
        ref_audio_ = voices[voice]["ref_audio"]
        local_speed = voices[voice].get("speed", speed)
        print(f"\nProcessing batch for voice: {voice} ({len(data['texts'])} segments)")

        audio_segments, final_sample_rate, _ = infer_xvoice_droptext_process(
            ref_audio_,
            data["texts"],
            data["langs"],
            tokenizer,
            ipa_tokenizer_getter,
            ema_model,
            vocoder,
            lang_to_id_map,
            srp_model,
            mel_spec_type_value=vocoder_name,
            target_rms_value=target_rms,
            cross_fade_duration_value=cross_fade_duration,
            nfe_step_value=nfe_step,
            cfg_strength_value=cfg_strength,
            layered=layered,
            cfg_strength2_value=cfg_strength2,
            cfg_schedule_value=cfg_schedule,
            cfg_decay_time_value=cfg_decay_time,
            sway_sampling_coef_value=sway_sampling_coef,
            local_speed=local_speed,
            fix_duration_value=fix_duration,
            reverse=reverse,
            denoise_ref=denoise_ref,
            loudness_norm=loudness_norm,
            post_processing=post_processing,
            device_name=device,
            remove_silence_chunk=remove_silence,
        )
        
        # ensure it's a list even if 1 item returned
        if not isinstance(audio_segments, list):
            audio_segments = [audio_segments]

        for idx, segment_audio, segment_text in zip(data["indices"], audio_segments, data["texts"]):
            generated_audio_segments[idx] = segment_audio
            if save_chunk and segment_audio is not None:
                save_text = segment_text
                if len(save_text) > 200:
                    save_text = save_text[:200] + " ... "
                if use_legacy_text:
                    save_text = unidecode(save_text)
                save_text = re.sub(r"[\\/:\0]", "_", save_text)
                sf.write(
                    os.path.join(output_chunk_dir, f"{idx}_{save_text}.wav"),
                    segment_audio,
                    final_sample_rate,
                )

    # filter out None elements
    generated_audio_segments = [seg for seg in generated_audio_segments if seg is not None]

    if generated_audio_segments:
        final_wave = np.concatenate(generated_audio_segments)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(wave_path, "wb") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            if remove_silence:
                remove_silence_for_generated_wav(f.name)
            print(f.name)


if __name__ == "__main__":
    main()

# python -m x_voice.infer.infer_cli_droptext -c src/x_voice/infer/examples/basic/basic_stage2.toml
