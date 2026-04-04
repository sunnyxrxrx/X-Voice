import argparse
import codecs
import os
import sys
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import numpy as np
import soundfile as sf
import tomli
import torch
import torchaudio
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf

from f5_tts.infer.module_clf5 import SpeedPredictor
from f5_tts.infer.utils_infer import (
    cfg_strength,
    cross_fade_duration,
    device,
    fix_duration,
    hop_length,
    load_checkpoint,
    load_model,
    load_model_sft,
    load_vocoder,
    mel_spec_type,
    nfe_step,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    speed,
    sway_sampling_coef,
    target_rms,
    target_sample_rate,
)
from f5_tts.model.utils import convert_char_to_pinyin, get_ipa_id, str_to_list_ipa_all
from f5_tts.train.datasets.ipa_tokenizer import PhonemizeTextTokenizer
from f5_tts.train.datasets.ipa_v2_tokenizer import PhonemizeTextTokenizer as PhonemizeTextTokenizer_v2
from f5_tts.train.datasets.ipa_v3_tokenizer import PhonemizeTextTokenizer as PhonemizeTextTokenizer_v3
from f5_tts.train.datasets.ipa_v5_tokenizer import PhonemizeTextTokenizer as PhonemizeTextTokenizer_v5
from f5_tts.train.datasets.ipa_v6_tokenizer import PhonemizeTextTokenizer as PhonemizeTextTokenizer_v6

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRP_SRC_ROOT = PROJECT_ROOT / "SpeakingRatePredictor" / "src"
if str(SRP_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRP_SRC_ROOT))

from model.utils import count_syllables_


DROP_TEXT_PLACEHOLDER = "Useless here."
DEFAULT_SRP_MODEL_CFG = SRP_SRC_ROOT / "configs" / "SpeedPredict_Multilingual.yaml"
TOKENIZER_CLASS_MAP = {
    "ipa": PhonemizeTextTokenizer,
    "ipa_v2": PhonemizeTextTokenizer_v2,
    "ipa_v3": PhonemizeTextTokenizer_v3,
    "ipa_v5": PhonemizeTextTokenizer_v5,
    "ipa_v6": PhonemizeTextTokenizer_v6,
}


def count(text, lang):
    return count_syllables_(text, lang)


def load_model_sp(model_cfg, ckpt_path, mel_spec_kwargs, use_ema=True, device=device):
    print("srp model:", ckpt_path, "\n")
    model_sp = SpeedPredictor(
        mel_spec_kwargs=mel_spec_kwargs,
        arch_kwargs=model_cfg,
    ).to(device)
    dtype = torch.float32
    return load_checkpoint(model_sp, ckpt_path, device, dtype=dtype, use_ema=use_ema)


def get_pred_speed(speakingrate_model, ref_audio):
    audio, sr = ref_audio
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    audio = audio.to(device)
    pred_speed = speakingrate_model.predict_speed(audio=audio)
    return pred_speed, audio, rms


def load_segments(config, gen_text, gen_file, default_lang):
    config_segments = config.get("segments", [])
    if config_segments:
        segments = []
        for segment in config_segments:
            lang = segment.get("lang")
            text = segment.get("text", "")
            if not lang or not text.strip():
                continue
            segments.append({"lang": lang, "text": text.strip()})
        if segments:
            return segments

    if gen_file:
        gen_text = codecs.open(gen_file, "r", "utf-8").read()

    if not gen_text.strip():
        raise ValueError("No target text found. Provide [[segments]] or gen_text/gen_file.")

    return [{"lang": default_lang, "text": gen_text.strip()}]


def build_segment_tokenizers(tokenizer, segments):
    tokenizer_cache = {}
    if tokenizer not in TOKENIZER_CLASS_MAP:
        return tokenizer_cache

    tokenizer_class = TOKENIZER_CLASS_MAP[tokenizer]
    for segment in segments:
        lang = segment["lang"]
        if lang in tokenizer_cache:
            continue
        ipa_id = get_ipa_id(lang)
        tokenizer_cache[lang] = tokenizer_class(language=ipa_id, with_stress=True)
    return tokenizer_cache


def tokenize_segment_text(text, lang, tokenizer, tokenizer_cache):
    if tokenizer == "pinyin":
        return convert_char_to_pinyin([text], polyphone=True)[0]
    if tokenizer.startswith("ipa"):
        ipa_tokenizer = tokenizer_cache.get(lang)
        if ipa_tokenizer is None:
            raise ValueError(f"Missing IPA tokenizer for language: {lang}")
        text_str = ipa_tokenizer(text)
        return str_to_list_ipa_all(text_str, tokenizer, lang)
    return list(text)


def build_split_target(segments, tokenizer, tokenizer_cache, lang_to_id):
    target_tokens = []
    target_lang_ids = []
    total_units = 0

    for i, segment in enumerate(segments):
        lang = segment["lang"]
        text = segment["text"].strip()
        if lang not in lang_to_id:
            raise ValueError(f"Unsupported language code: {lang}")
        if not text:
            continue

        tokens = tokenize_segment_text(text, lang, tokenizer, tokenizer_cache)
        if not tokens:
            continue

        target_tokens.extend(tokens)
        target_lang_ids.extend([lang_to_id[lang]] * len(tokens))
        total_units += count(text, lang)
        print(f"segment {i}: {lang} -> {text}")
        print(tokens)

    if not target_tokens:
        raise ValueError("No valid target tokens generated from segments.")

    return target_tokens, target_lang_ids, total_units


def infer_process_split(
    speakingrate_model,
    ref_audio,
    segments,
    model_obj,
    vocoder,
    tokenizer,
    mel_spec_type=mel_spec_type,
    target_rms=target_rms,
    nfe_step=nfe_step,
    cfg_strength=cfg_strength,
    sway_sampling_coef=sway_sampling_coef,
    speed=speed,
    fix_duration=fix_duration,
    device=device,
    codeswitch_time_unknown_lang=True,
):
    pred_speed, audio, rms = get_pred_speed(speakingrate_model, torchaudio.load(ref_audio))
    lang_to_id = model_obj.transformer.lang_to_id
    tokenizer_cache = build_segment_tokenizers(tokenizer, segments)
    target_tokens, target_lang_ids, total_units = build_split_target(segments, tokenizer, tokenizer_cache, lang_to_id)

    local_speed = speed
    if total_units < 4:
        local_speed = 0.5

    ref_audio_len = audio.shape[-1] // hop_length
    if fix_duration is not None:
        duration = int(fix_duration * target_sample_rate / hop_length)
    else:
        pred_duration = (total_units / pred_speed.item()) / local_speed
        gen_audio_len = int((pred_duration * target_sample_rate) / hop_length)
        duration = ref_audio_len + gen_audio_len

    language_ids = torch.tensor([target_lang_ids], dtype=torch.long, device=device)

    with torch.inference_mode():
        generated, _ = model_obj.sample(
            cond=audio,
            text=[target_tokens],
            duration=duration,
            steps=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            language_ids=language_ids,
            codeswitch_time_unknown_lang=codeswitch_time_unknown_lang,
        )

        generated = generated.to(torch.float32)
        generated = generated[:, ref_audio_len:, :]
        generated_mel_spec = generated.permute(0, 2, 1)
        if mel_spec_type == "vocos":
            generated_wave = vocoder.decode(generated_mel_spec)
        elif mel_spec_type == "bigvgan":
            generated_wave = vocoder(generated_mel_spec)
        else:
            raise ValueError(f"Unsupported vocoder type: {mel_spec_type}")

        if rms < target_rms:
            generated_wave = generated_wave * rms / target_rms

    return generated_wave.squeeze().cpu().numpy(), generated_mel_spec[0].cpu().numpy()


parser = argparse.ArgumentParser(
    prog="python3 infer-cli-droptext-split.py",
    description="Commandline interface for multilingual F5-TTS drop-text code-switch inference with SRP duration prediction.",
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    default=os.path.join(files("f5_tts").joinpath("infer/examples/basic"), "basic.toml"),
)
parser.add_argument("-m", "--model", type=str)
parser.add_argument("-mc", "--model_cfg", type=str)
parser.add_argument("-p", "--ckpt_file", type=str)
parser.add_argument("-sp", "--srp_ckpt_file", type=str)
parser.add_argument("--srp_model_cfg", type=str)
parser.add_argument("-v", "--vocab_file", type=str)
parser.add_argument("-r", "--ref_audio", type=str)
parser.add_argument("-s", "--ref_text", type=str)
parser.add_argument("-t", "--gen_text", type=str)
parser.add_argument("-f", "--gen_file", type=str)
parser.add_argument("-o", "--output_dir", type=str)
parser.add_argument("-w", "--output_file", type=str)
parser.add_argument("--save_chunk", action="store_true")
parser.add_argument("--remove_silence", action="store_true")
parser.add_argument("--load_vocoder_from_local", action="store_true")
parser.add_argument("--vocoder_name", type=str, choices=["vocos", "bigvgan"])
parser.add_argument("--target_rms", type=float)
parser.add_argument("--nfe_step", type=int)
parser.add_argument("--cfg_strength", type=float)
parser.add_argument("--sway_sampling_coef", type=float)
parser.add_argument("--speed", type=float)
parser.add_argument("--fix_duration", type=float)
parser.add_argument("--device", type=str)
parser.add_argument("--lang", type=str, help="Fallback language code when [[segments]] is not provided.")
parser.add_argument(
    "--codeswitch_time_unknown_lang",
    action="store_true",
    help="Use unknown language id for time conditioning when token-wise language_ids are provided.",
)
args = parser.parse_args()


config = tomli.load(open(args.config, "rb"))

model = args.model or config.get("model", "F5TTS_v1_Base")
ckpt_file = args.ckpt_file or config.get("ckpt_file", "")
srp_ckpt_file = args.srp_ckpt_file or config.get("srp_ckpt_file", "")
srp_model_cfg_file = args.srp_model_cfg or config.get("srp_model_cfg", str(DEFAULT_SRP_MODEL_CFG))
vocab_file = args.vocab_file or config.get("vocab_file", "")

ref_audio = args.ref_audio or config.get("ref_audio", "infer/examples/basic/basic_ref_en.wav")
_ignored_ref_text = args.ref_text if args.ref_text is not None else config.get("ref_text", "")
gen_text = args.gen_text or config.get("gen_text", "")
gen_file = args.gen_file or config.get("gen_file", "")

output_dir = args.output_dir or config.get("output_dir", "tests")
output_file = args.output_file or config.get(
    "output_file", f"infer_cli_droptext_split_{datetime.now().strftime(r'%Y%m%d_%H%M%S')}.wav"
)

save_chunk = args.save_chunk or config.get("save_chunk", False)
remove_silence = args.remove_silence or config.get("remove_silence", False)
load_vocoder_from_local = args.load_vocoder_from_local or config.get("load_vocoder_from_local", False)

vocoder_name = args.vocoder_name or config.get("vocoder_name", mel_spec_type)
target_rms = args.target_rms or config.get("target_rms", target_rms)
nfe_step = args.nfe_step or config.get("nfe_step", nfe_step)
cfg_strength = args.cfg_strength or config.get("cfg_strength", cfg_strength)
sway_sampling_coef = args.sway_sampling_coef or config.get("sway_sampling_coef", sway_sampling_coef)
speed = args.speed or config.get("speed", speed)
fix_duration = args.fix_duration or config.get("fix_duration", fix_duration)
device = args.device or config.get("device", device)
fallback_lang = args.lang or config.get("lang", "en")
codeswitch_time_unknown_lang = args.codeswitch_time_unknown_lang or config.get("codeswitch_time_unknown_lang", True)
if fix_duration is not None and fix_duration <= 0:
    fix_duration = None

if "infer/examples/" in ref_audio:
    ref_audio = str(files("f5_tts").joinpath(f"{ref_audio}"))
if "infer/examples/" in gen_file:
    gen_file = str(files("f5_tts").joinpath(f"{gen_file}"))

segments = load_segments(config, gen_text, gen_file, fallback_lang)

wave_path = Path(output_dir) / output_file
if save_chunk and not os.path.exists(output_dir):
    os.makedirs(output_dir)

if vocoder_name == "vocos":
    vocoder_local_path = "my_vocoder/vocos-mel-24khz"
elif vocoder_name == "bigvgan":
    vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"

vocoder = load_vocoder(
    vocoder_name=vocoder_name,
    is_local=load_vocoder_from_local,
    local_path=vocoder_local_path,
    device=device,
)

model_cfg = OmegaConf.load(
    args.model_cfg or config.get("model_cfg", str(files("f5_tts").joinpath(f"configs/{model}.yaml")))
)
model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
model_arc = model_cfg.model.arch
sft = bool(model_cfg.model.get("sft", False))
use_total_text = bool(model_cfg.model.get("use_total_text", False))
tokenizer = model_cfg.model.tokenizer
tokenizer_path = model_cfg.model.get("tokenizer_path", None)
dataset_name = model_cfg.datasets.name
tts_mel_spec_kwargs = OmegaConf.to_container(model_cfg.model.mel_spec, resolve=True)
srp_cfg = OmegaConf.load(srp_model_cfg_file)
srp_arch = OmegaConf.to_container(srp_cfg.model.arch, resolve=True)
srp_mel_spec_kwargs = OmegaConf.to_container(srp_cfg.model.mel_spec, resolve=True)

repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"
if model != "F5TTS_Base":
    assert vocoder_name == model_cfg.model.mel_spec.mel_spec_type
if model == "F5TTS_Base":
    if vocoder_name == "vocos":
        ckpt_step = 1200000
    elif vocoder_name == "bigvgan":
        model = "F5TTS_Base_bigvgan"
        ckpt_type = "pt"
elif model == "E2TTS_Base":
    repo_name = "E2-TTS"
    ckpt_step = 1200000

if not ckpt_file:
    ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{model}/model_{ckpt_step}.{ckpt_type}"))
if not srp_ckpt_file:
    srp_ckpt_file = str(cached_path("hf://QingyuLiu1/Cross-Lingual_F5-TTS/syllables_gce_20000.safetensors"))

print(f"Using {model}...")
if sft:
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
        mel_spec_kwargs=tts_mel_spec_kwargs,
    )
else:
    ema_model = load_model(
        model_cls,
        model_arc,
        ckpt_file,
        mel_spec_type=vocoder_name,
        vocab_file=vocab_file,
        device=device,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        dataset_name=dataset_name,
        mel_spec_kwargs=tts_mel_spec_kwargs,
    )
speakingrate_model = load_model_sp(
    srp_arch,
    srp_ckpt_file,
    srp_mel_spec_kwargs,
    device=device,
)


def main():
    if _ignored_ref_text:
        print("Drop-text split mode ignores ref_text and uses SRP for duration prediction.")
    if not sft:
        raise ValueError("infer_cli_droptext_split.py is intended for SFT checkpoints.")
    if not model_cfg.model.tokenizer.startswith("ipa"):
        raise ValueError("Code-switch split inference currently requires an IPA tokenizer.")

    print("segments:")
    for i, segment in enumerate(segments):
        print(f"  {i}: {segment['lang']} -> {segment['text']}")

    ref_audio_preprocessed, _ = preprocess_ref_audio_text(ref_audio, DROP_TEXT_PLACEHOLDER)

    audio_segment, _ = infer_process_split(
        speakingrate_model,
        ref_audio_preprocessed,
        segments,
        ema_model,
        vocoder,
        tokenizer=tokenizer,
        mel_spec_type=vocoder_name,
        target_rms=target_rms,
        nfe_step=nfe_step,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef,
        speed=speed,
        fix_duration=fix_duration,
        device=device,
        codeswitch_time_unknown_lang=codeswitch_time_unknown_lang,
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(wave_path, "wb") as f:
        sf.write(f.name, audio_segment, target_sample_rate)
        if remove_silence:
            remove_silence_for_generated_wav(f.name)
        print(f.name)

    if save_chunk:
        chunk_path = Path(output_dir) / f"{Path(output_file).stem}_chunk0.wav"
        sf.write(chunk_path, audio_segment, target_sample_rate)


if __name__ == "__main__":
    main()

# cd /inspire/hdd/project/embodied-multimodality/chenxie-25019/qingyuliu/Multilingual_F5-TTS/F5-TTS
# python -m f5_tts.infer.infer_cli_droptext_split -c src/f5_tts/infer/examples/multitest_codeswitch/droptext_multilingual_codeswitch.toml

