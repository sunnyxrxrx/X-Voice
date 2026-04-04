import argparse
import codecs
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import numpy as np
import soundfile as sf
import tomli
import torch
import torchaudio
import tqdm
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf
from unidecode import unidecode

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
    n_fft,
    n_mel_channels,
    nfe_step,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    speed,
    sway_sampling_coef,
    target_rms,
    target_sample_rate,
    win_length,
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


def count(text, lang):
    return count_syllables_(text, lang)


def prepare_infer_text(text, tokenizer, lang, ipa_tokenizer=None):
    if tokenizer == "pinyin":
        return convert_char_to_pinyin([text], polyphone=True)
    if tokenizer.startswith("ipa"):
        assert ipa_tokenizer is not None, f"{tokenizer} requires an IPA tokenizer."
        text_str = ipa_tokenizer(text)
        text_tokenized = str_to_list_ipa_all(text_str, tokenizer, lang)
        return [text_tokenized]
    return [list(text)]


def load_model_sp(model_cfg, ckpt_path, mel_spec_kwargs, use_ema=True, device=device):
    print("srp model:", ckpt_path, "\n")

    model_sp = SpeedPredictor(
        mel_spec_kwargs=mel_spec_kwargs,
        arch_kwargs=model_cfg,
    ).to(device)

    dtype = torch.float32
    model_sp = load_checkpoint(model_sp, ckpt_path, device, dtype=dtype, use_ema=use_ema)
    return model_sp


def get_max_syllables(speakingrate_model, ref_audio):
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
    max_syllables = pred_speed * (22 - audio.shape[-1] / sr)
    return float(max_syllables.item()), pred_speed


def chunk_text_clf5(text, lang, max_syllables=65):
    chunks = []
    current_chunk = ""
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if count(current_chunk, lang) + count(sentence, lang) <= max_syllables:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def infer_process_clf5(
    speakingrate_model,
    ref_audio,
    gen_text,
    lang,
    tokenizer,
    ipa_tokenizer,
    model_obj,
    vocoder,
    mel_spec_type=mel_spec_type,
    show_info=print,
    progress=tqdm,
    target_rms=target_rms,
    cross_fade_duration=cross_fade_duration,
    nfe_step=nfe_step,
    cfg_strength=cfg_strength,
    sway_sampling_coef=sway_sampling_coef,
    speed=speed,
    fix_duration=fix_duration,
    device=device,
):
    audio, sr = torchaudio.load(ref_audio)
    max_syllables, pred_speed = get_max_syllables(speakingrate_model, (audio, sr))
    gen_text_batches = chunk_text_clf5(gen_text, lang, max_syllables)
    for i, batch_text in enumerate(gen_text_batches):
        print(f"gen_text {i}", batch_text)
    print("\n")

    show_info(f"Generating audio in {len(gen_text_batches)} batches...")
    return next(
        infer_batch_process_clf5(
            (audio, sr),
            pred_speed,
            gen_text_batches,
            lang,
            tokenizer,
            ipa_tokenizer,
            model_obj,
            vocoder,
            mel_spec_type=mel_spec_type,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=device,
        )
    )


def infer_batch_process_clf5(
    ref_audio,
    pred_speed,
    gen_text_batches,
    lang,
    tokenizer,
    ipa_tokenizer,
    model_obj,
    vocoder,
    mel_spec_type="vocos",
    progress=tqdm,
    target_rms=0.1,
    cross_fade_duration=0.15,
    nfe_step=32,
    cfg_strength=2.0,
    sway_sampling_coef=-1,
    speed=1,
    fix_duration=None,
    device=None,
    streaming=False,
    chunk_size=2048,
):
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

    generated_waves = []
    spectrograms = []

    def process_batch(gen_text):
        local_speed = speed
        if count(gen_text, lang) < 4:
            local_speed = 0.5

        final_text_list = prepare_infer_text(gen_text, tokenizer, lang, ipa_tokenizer)

        ref_audio_len = audio.shape[-1] // hop_length
        if fix_duration is not None:
            duration = int(fix_duration * target_sample_rate / hop_length)
        else:
            gt_num_unit = count(gen_text, lang)
            pred_duration = (gt_num_unit / pred_speed.item()) / local_speed
            gen_audio_len = int((pred_duration * target_sample_rate) / hop_length)
            duration = ref_audio_len + gen_audio_len

        with torch.inference_mode():
            generated, _ = model_obj.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                language_ids=[lang],
            )

            generated = generated.to(torch.float32)
            generated = generated[:, ref_audio_len:, :]
            generated_mel_spec = generated.permute(0, 2, 1)
            if mel_spec_type == "vocos":
                generated_wave = vocoder.decode(generated_mel_spec)
            elif mel_spec_type == "bigvgan":
                generated_wave = vocoder(generated_mel_spec)
            if rms < target_rms:
                generated_wave = generated_wave * rms / target_rms

            generated_wave = generated_wave.squeeze().cpu().numpy()

            if streaming:
                for i in range(0, len(generated_wave), chunk_size):
                    yield generated_wave[i : i + chunk_size], target_sample_rate
            else:
                yield generated_wave, generated_mel_spec[0].cpu().numpy()

    if streaming:
        for gen_text in progress.tqdm(gen_text_batches) if progress is not None else gen_text_batches:
            for chunk in process_batch(gen_text):
                yield chunk
    else:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_batch, gen_text) for gen_text in gen_text_batches]
            for future in progress.tqdm(futures) if progress is not None else futures:
                result = future.result()
                if result:
                    generated_wave, generated_mel_spec = next(result)
                    generated_waves.append(generated_wave)
                    spectrograms.append(generated_mel_spec)

        if generated_waves:
            if cross_fade_duration <= 0:
                final_wave = np.concatenate(generated_waves)
            else:
                final_wave = generated_waves[0]
                for i in range(1, len(generated_waves)):
                    prev_wave = final_wave
                    next_wave = generated_waves[i]

                    cross_fade_samples = int(cross_fade_duration * target_sample_rate)
                    cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

                    if cross_fade_samples <= 0:
                        final_wave = np.concatenate([prev_wave, next_wave])
                        continue

                    prev_overlap = prev_wave[-cross_fade_samples:]
                    next_overlap = next_wave[:cross_fade_samples]

                    fade_out = np.linspace(1, 0, cross_fade_samples)
                    fade_in = np.linspace(0, 1, cross_fade_samples)
                    cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

                    final_wave = np.concatenate(
                        [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
                    )

            combined_spectrogram = np.concatenate(spectrograms, axis=1)
            yield final_wave, target_sample_rate, combined_spectrogram
        else:
            yield None, target_sample_rate, None


parser = argparse.ArgumentParser(
    prog="python3 infer-cli-droptext.py",
    description="Commandline interface for multilingual F5-TTS drop-text inference with SRP duration prediction.",
    epilog="Specify options above to override one or more settings from config.",
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    default=os.path.join(files("f5_tts").joinpath("infer/examples/basic"), "basic.toml"),
    help="The configuration file, default see infer/examples/basic/basic.toml",
)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    help="The model name: F5TTS_v1_Base | F5TTS_Base | E2TTS_Base | etc.",
)
parser.add_argument(
    "-mc",
    "--model_cfg",
    type=str,
    help="The path to F5-TTS model config file .yaml",
)
parser.add_argument(
    "-p",
    "--ckpt_file",
    type=str,
    help="The path to model checkpoint .pt/.safetensors, leave blank to use default",
)
parser.add_argument(
    "-sp",
    "--srp_ckpt_file",
    type=str,
    help="The path to SRP checkpoint .pt/.safetensors, leave blank to use the Cross-Lingual_F5-TTS default",
)
parser.add_argument(
    "--srp_model_cfg",
    type=str,
    help="The path to SRP config file .yaml",
)
parser.add_argument(
    "-v",
    "--vocab_file",
    type=str,
    help="The path to vocab file .txt, leave blank to use default",
)
parser.add_argument(
    "-r",
    "--ref_audio",
    type=str,
    help="The reference audio file.",
)
parser.add_argument(
    "-s",
    "--ref_text",
    type=str,
    help="Ignored in drop-text mode. Kept only for CLI/config compatibility.",
)
parser.add_argument(
    "-t",
    "--gen_text",
    type=str,
    help="The text to make model synthesize a speech",
)
parser.add_argument(
    "-f",
    "--gen_file",
    type=str,
    help="The file with text to generate, will ignore --gen_text",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="The path to output folder",
)
parser.add_argument(
    "-w",
    "--output_file",
    type=str,
    help="The name of output file",
)
parser.add_argument(
    "--save_chunk",
    action="store_true",
    help="To save each audio chunk during inference",
)
parser.add_argument(
    "--no_legacy_text",
    action="store_false",
    help="Not to use lossy ASCII transliterations of unicode text in saved file names.",
)
parser.add_argument(
    "--remove_silence",
    action="store_true",
    help="To remove long silence found in output",
)
parser.add_argument(
    "--load_vocoder_from_local",
    action="store_true",
    help="To load vocoder from local dir, default to ../checkpoints/vocos-mel-24khz",
)
parser.add_argument(
    "--vocoder_name",
    type=str,
    choices=["vocos", "bigvgan"],
    help=f"Used vocoder name: vocos | bigvgan, default {mel_spec_type}",
)
parser.add_argument(
    "--target_rms",
    type=float,
    help=f"Target output speech loudness normalization value, default {target_rms}",
)
parser.add_argument(
    "--cross_fade_duration",
    type=float,
    help=f"Duration of cross-fade between audio segments in seconds, default {cross_fade_duration}",
)
parser.add_argument(
    "--nfe_step",
    type=int,
    help=f"The number of function evaluation (denoising steps), default {nfe_step}",
)
parser.add_argument(
    "--cfg_strength",
    type=float,
    help=f"Classifier-free guidance strength, default {cfg_strength}",
)
parser.add_argument(
    "--sway_sampling_coef",
    type=float,
    help=f"Sway Sampling coefficient, default {sway_sampling_coef}",
)
parser.add_argument(
    "--speed",
    type=float,
    help=f"Global speed multiplier applied after SRP duration prediction, default {speed}",
)
parser.add_argument(
    "--fix_duration",
    type=float,
    help=f"Fix the total duration (ref and gen audios) in seconds, default {fix_duration}",
)
parser.add_argument(
    "--device",
    type=str,
    help="Specify the device to run on",
)
parser.add_argument(
    "--lang",
    type=str,
    help="Language code for syllable counting, e.g. en/zh/ja/th/vi",
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
gen_text = args.gen_text or config.get("gen_text", "Here we generate something just for test.")
gen_file = args.gen_file or config.get("gen_file", "")

output_dir = args.output_dir or config.get("output_dir", "tests")
output_file = args.output_file or config.get(
    "output_file", f"infer_cli_droptext_{datetime.now().strftime(r'%Y%m%d_%H%M%S')}.wav"
)

save_chunk = args.save_chunk or config.get("save_chunk", False)
use_legacy_text = args.no_legacy_text or config.get("no_legacy_text", False)
if save_chunk and use_legacy_text:
    print(
        "\nWarning to --save_chunk: lossy ASCII transliterations of unicode text for legacy (.wav) file names, --no_legacy_text to disable.\n"
    )

remove_silence = args.remove_silence or config.get("remove_silence", False)
load_vocoder_from_local = args.load_vocoder_from_local or config.get("load_vocoder_from_local", False)

vocoder_name = args.vocoder_name or config.get("vocoder_name", mel_spec_type)
target_rms = args.target_rms or config.get("target_rms", target_rms)
cross_fade_duration = args.cross_fade_duration or config.get("cross_fade_duration", cross_fade_duration)
nfe_step = args.nfe_step or config.get("nfe_step", nfe_step)
cfg_strength = args.cfg_strength or config.get("cfg_strength", cfg_strength)
sway_sampling_coef = args.sway_sampling_coef or config.get("sway_sampling_coef", sway_sampling_coef)
speed = args.speed or config.get("speed", speed)
fix_duration = args.fix_duration or config.get("fix_duration", fix_duration)
device = args.device or config.get("device", device)
lang = args.lang or config.get("lang", "en")

if "infer/examples/" in ref_audio:
    ref_audio = str(files("f5_tts").joinpath(f"{ref_audio}"))
if "infer/examples/" in gen_file:
    gen_file = str(files("f5_tts").joinpath(f"{gen_file}"))
if "voices" in config:
    for voice in config["voices"]:
        voice_ref_audio = config["voices"][voice]["ref_audio"]
        if "infer/examples/" in voice_ref_audio:
            config["voices"][voice]["ref_audio"] = str(files("f5_tts").joinpath(f"{voice_ref_audio}"))

if gen_file:
    gen_text = codecs.open(gen_file, "r", "utf-8").read()

wave_path = Path(output_dir) / output_file
if save_chunk:
    output_chunk_dir = os.path.join(output_dir, f"{Path(output_file).stem}_chunks")
    if not os.path.exists(output_chunk_dir):
        os.makedirs(output_chunk_dir)

if vocoder_name == "vocos":
    vocoder_local_path = "my_vocoder/vocos-mel-24khz"
elif vocoder_name == "bigvgan":
    vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"

vocoder = load_vocoder(
    vocoder_name=vocoder_name, is_local=load_vocoder_from_local, local_path=vocoder_local_path, device=device
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
tokenizer_class_map = {
    "ipa": PhonemizeTextTokenizer,
    "ipa_v2": PhonemizeTextTokenizer_v2,
    "ipa_v3": PhonemizeTextTokenizer_v3,
    "ipa_v5": PhonemizeTextTokenizer_v5,
    "ipa_v6": PhonemizeTextTokenizer_v6,
}
ipa_tokenizer = None
if tokenizer in tokenizer_class_map:
    ipa_id = get_ipa_id(lang)
    tokenizer_class = tokenizer_class_map[tokenizer]
    ipa_tokenizer = tokenizer_class(language=ipa_id, with_stress=True)

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
        print("Drop-text mode ignores ref_text and uses SRP for duration prediction.")

    main_voice = {"ref_audio": ref_audio, "ref_text": DROP_TEXT_PLACEHOLDER}
    if "voices" not in config:
        voices = {"main": main_voice}
    else:
        voices = config["voices"]
        voices["main"] = main_voice

    for voice in voices:
        print("Voice:", voice)
        print("ref_audio", voices[voice]["ref_audio"])
        voices[voice]["ref_audio"], _ = preprocess_ref_audio_text(
            voices[voice]["ref_audio"],
            DROP_TEXT_PLACEHOLDER,
        )
        voices[voice]["ref_text"] = DROP_TEXT_PLACEHOLDER
        print("ref_audio_", voices[voice]["ref_audio"], "\n\n")

    generated_audio_segments = []
    reg1 = r"(?=\[\w+\])"
    chunks = re.split(reg1, gen_text)
    reg2 = r"\[(\w+)\]"
    for text in chunks:
        if not text.strip():
            continue
        match = re.match(reg2, text)
        if match:
            voice = match[1]
        else:
            print("No voice tag found, using main.")
            voice = "main"
        if voice not in voices:
            print(f"Voice {voice} not found, using main.")
            voice = "main"
        text = re.sub(reg2, "", text)
        ref_audio_ = voices[voice]["ref_audio"]
        local_speed = voices[voice].get("speed", speed)
        gen_text_ = text.strip()
        print(f"Voice: {voice}")
        audio_segment, final_sample_rate, _ = infer_process_clf5(
            speakingrate_model,
            ref_audio_,
            gen_text_,
            lang,
            tokenizer,
            ipa_tokenizer,
            ema_model,
            vocoder,
            mel_spec_type=vocoder_name,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=local_speed,
            fix_duration=fix_duration,
            device=device,
        )
        generated_audio_segments.append(audio_segment)

        if save_chunk:
            save_text = gen_text_
            if len(save_text) > 200:
                save_text = save_text[:200] + " ... "
            if use_legacy_text:
                save_text = unidecode(save_text)
            sf.write(
                os.path.join(output_chunk_dir, f"{len(generated_audio_segments) - 1}_{save_text}.wav"),
                audio_segment,
                final_sample_rate,
            )

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

# cd /inspire/hdd/project/embodied-multimodality/chenxie-25019/qingyuliu/Multilingual_F5-TTS/F5-TTS
# python -m f5_tts.infer.infer_cli_droptext -c src/f5_tts/infer/examples/multitest/droptext_multilingual.toml
