# A unified script for inference process
# Make adjustments inside functions, and consider both gradio and cli scripts if need to change func output format
import contextlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # for MPS device compatibility
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../../third_party/BigVGAN/")

import hashlib
import re
import tempfile
from importlib.resources import files
from pathlib import Path

import matplotlib


matplotlib.use("Agg")

import matplotlib.pylab as plt
import numpy as np
import torch
import torchaudio
import tqdm
from cached_path import cached_path
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from pydub import AudioSegment, silence
from transformers import pipeline
from vocos import Vocos
from fastlid import fastlid


@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, "w") as devnull:
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)


from x_voice.model.cfm import CFM
from x_voice.model.cfm_sft import CFM_SFT
from rate_pred.model.speed_predictor import SpeedPredictor
from rate_pred.model.utils import count_syllables
from x_voice.model.utils import convert_char_to_pinyin, get_ipa_id, get_tokenizer, str_to_list_ipa_all
from x_voice.train.datasets.ipa_v3_tokenizer import PhonemizeTextTokenizer as PhonemizeTextTokenizerV3
from x_voice.train.datasets.ipa_v6_tokenizer import PhonemizeTextTokenizer as PhonemizeTextTokenizerV6

_ref_audio_cache = {}
_ref_text_cache = {}

device = (
    "cuda"
    if torch.cuda.is_available()
    else "xpu"
    if torch.xpu.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

tempfile_kwargs = {"delete_on_close": False} if sys.version_info >= (3, 12) else {"delete": False}

# -----------------------------------------

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"
target_rms = 0.1
cross_fade_duration = 0.15
ode_method = "euler"
nfe_step = 32  # 16, 32
cfg_strength = 2.0
sway_sampling_coef = -1.0
speed = 1.0
fix_duration = None
layered = False
cfg_strength2 = 4.0
cfg_schedule = "square"
cfg_decay_time = 0.0
NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"
XVOICE_TO_NLLB = {
    "bg": "bul_Cyrl",
    "cs": "ces_Latn",
    "da": "dan_Latn",
    "de": "deu_Latn",
    "el": "ell_Grek",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "et": "est_Latn",
    "fi": "fin_Latn",
    "fr": "fra_Latn",
    "hr": "hrv_Latn",
    "hu": "hun_Latn",
    "id": "ind_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "lt": "lit_Latn",
    "lv": "lvs_Latn",
    "mt": "mlt_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "ro": "ron_Latn",
    "ru": "rus_Cyrl",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "sv": "swe_Latn",
    "th": "tha_Thai",
    "vi": "vie_Latn",
    "zh": "zho_Hans",
}

_nllb_tokenizer = None
_nllb_model = None

# -----------------------------------------


# load vocoder
def load_vocoder(vocoder_name="vocos", is_local=False, local_path="", device=device, hf_cache_dir=None):
    if vocoder_name == "vocos":
        # vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
        if is_local:
            print(f"Load vocos from local path {local_path}")
            config_path = f"{local_path}/config.yaml"
            model_path = f"{local_path}/pytorch_model.bin"
        else:
            print("Download Vocos from huggingface charactr/vocos-mel-24khz")
            repo_id = "charactr/vocos-mel-24khz"
            config_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="config.yaml")
            model_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="pytorch_model.bin")
        vocoder = Vocos.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        from vocos.feature_extractors import EncodecFeatures

        if isinstance(vocoder.feature_extractor, EncodecFeatures):
            encodec_parameters = {
                "feature_extractor.encodec." + key: value
                for key, value in vocoder.feature_extractor.encodec.state_dict().items()
            }
            state_dict.update(encodec_parameters)
        vocoder.load_state_dict(state_dict)
        vocoder = vocoder.eval().to(device)
    elif vocoder_name == "bigvgan":
        try:
            from third_party.BigVGAN import bigvgan
        except ImportError:
            print("You need to follow the README to init submodule and change the BigVGAN source code.")
        if is_local:
            # download generator from https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x/tree/main
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)
        else:
            vocoder = bigvgan.BigVGAN.from_pretrained(
                "nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False, cache_dir=hf_cache_dir
            )

        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(device)
    return vocoder


# load asr pipeline

asr_pipe = None


def initialize_asr_pipeline(device: str = device, dtype=None):
    if dtype is None:
        dtype = (
            torch.float16
            if "cuda" in device
            and torch.cuda.get_device_properties(device).major >= 7
            and not torch.cuda.get_device_name().endswith("[ZLUDA]")
            else torch.float32
        )
    global asr_pipe
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        torch_dtype=dtype,
        device=device,
    )


# transcribe


def transcribe(ref_audio, language=None):
    global asr_pipe
    if asr_pipe is None:
        initialize_asr_pipeline(device=device)
    return asr_pipe(
        ref_audio,
        chunk_length_s=30,
        batch_size=128,
        generate_kwargs={"task": "transcribe", "language": language} if language else {"task": "transcribe"},
        return_timestamps=False,
    )["text"].strip()


# load model checkpoint for inference


def load_checkpoint(model, ckpt_path, device: str, dtype=None, use_ema=True):
    if dtype is None:
        dtype = (
            torch.float16  # torch.float32
            if "cuda" in device
            and torch.cuda.get_device_properties(device).major >= 7
            and not torch.cuda.get_device_name().endswith("[ZLUDA]")
            else torch.float32
        )
    model = model.to(dtype)

    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(ckpt_path, device=device)
    else:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    if use_ema:
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }

        # patch for backward compatibility, 305e3ea
        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["model_state_dict"]:
                del checkpoint["model_state_dict"][key]

        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        if ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}
        model.load_state_dict(checkpoint["model_state_dict"])

    del checkpoint
    torch.cuda.empty_cache()

    return model.to(device)


# load model for inference


def load_model(
    model_cls,
    model_cfg,
    ckpt_path,
    mel_spec_type=mel_spec_type,
    vocab_file="",
    ode_method=ode_method,
    use_ema=True,
    device=device,
    tokenizer="custom",
    tokenizer_path="",
    dataset_name="",
    mel_spec_kwargs=None,
):
    if mel_spec_kwargs is None:
        mel_spec_kwargs = dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        )

    if vocab_file:
        tokenizer_source = vocab_file
        tokenizer_name = "custom"
    elif tokenizer == "custom":
        tokenizer_source = tokenizer_path
        tokenizer_name = tokenizer
    else:
        tokenizer_source = dataset_name
        tokenizer_name = tokenizer

    if not tokenizer_source:
        tokenizer_source = str(files("x_voice").joinpath("infer/examples/vocab.txt"))
        tokenizer_name = "custom"

    print("\nvocab : ", tokenizer_source)
    print("token : ", tokenizer_name)
    print("model : ", ckpt_path, "\n")

    vocab_char_map, vocab_size = get_tokenizer(tokenizer_source, tokenizer_name)
    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        tokenizer=tokenizer_name,
        mel_spec_kwargs=dict(
            mel_spec_kwargs
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    dtype = torch.float32 if mel_spec_type == "bigvgan" else None
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

    return model


def load_model_sft(
    model_cls,
    model_cfg,
    ckpt_path,
    mel_spec_type=mel_spec_type,
    vocab_file="",
    ode_method=ode_method,
    use_ema=True,
    device=device,
    use_total_text=False,
    tokenizer="custom",
    tokenizer_path="",
    dataset_name="",
    mel_spec_kwargs=None,
):
    if mel_spec_kwargs is None:
        mel_spec_kwargs = dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        )

    if vocab_file:
        tokenizer_source = vocab_file
        tokenizer_name = "custom"
    elif tokenizer == "custom":
        tokenizer_source = tokenizer_path
        tokenizer_name = tokenizer
    else:
        tokenizer_source = dataset_name
        tokenizer_name = tokenizer

    if not tokenizer_source:
        tokenizer_source = str(files("x_voice").joinpath("infer/examples/vocab.txt"))
        tokenizer_name = "custom"

    print("\nvocab : ", tokenizer_source)
    print("token : ", tokenizer_name)
    print("model : ", ckpt_path, "\n")

    vocab_char_map, vocab_size = get_tokenizer(tokenizer_source, tokenizer_name, sft=True)
    model = CFM_SFT(
        transformer=model_cls(
            **model_cfg,
            sft=True,
            text_num_embeds=vocab_size + 1,
            mel_dim=n_mel_channels,
        ),
        tokenizer=tokenizer_name,
        mel_spec_kwargs=dict(
            mel_spec_kwargs
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
        use_total_text=use_total_text,
    ).to(device)

    dtype = torch.float32 if mel_spec_type == "bigvgan" else None
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

    return model


def remove_silence_edges(audio, silence_threshold=-42):
    # Remove silence from the start
    non_silent_start_idx = silence.detect_leading_silence(audio, silence_threshold=silence_threshold)
    audio = audio[non_silent_start_idx:]

    # Remove silence from the end
    non_silent_end_duration = audio.duration_seconds
    for ms in reversed(audio):
        if ms.dBFS > silence_threshold:
            break
        non_silent_end_duration -= 0.001
    trimmed_audio = audio[: int(non_silent_end_duration * 1000)]

    return trimmed_audio


# preprocess reference audio and text


def preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=print):
    show_info("Converting audio...")

    # Compute a hash of the reference audio file
    with open(ref_audio_orig, "rb") as audio_file:
        audio_data = audio_file.read()
        audio_hash = hashlib.md5(audio_data).hexdigest()

    global _ref_audio_cache

    if audio_hash in _ref_audio_cache:
        show_info("Using cached preprocessed reference audio...")
        ref_audio = _ref_audio_cache[audio_hash]

    else:  # first pass, do preprocess
        with tempfile.NamedTemporaryFile(suffix=".wav", **tempfile_kwargs) as f:
            temp_path = f.name

        aseg = AudioSegment.from_file(ref_audio_orig)

        # 1. try to find long silence for clipping
        non_silent_segs = silence.split_on_silence(
            aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000, seek_step=10
        )
        non_silent_wave = AudioSegment.silent(duration=0)
        for non_silent_seg in non_silent_segs:
            if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 12000:
                show_info("Audio is over 12s, clipping short. (1)")
                break
            non_silent_wave += non_silent_seg

        # 2. try to find short silence for clipping if 1. failed
        if len(non_silent_wave) > 12000:
            non_silent_segs = silence.split_on_silence(
                aseg, min_silence_len=100, silence_thresh=-40, keep_silence=1000, seek_step=10
            )
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 12000:
                    show_info("Audio is over 12s, clipping short. (2)")
                    break
                non_silent_wave += non_silent_seg

        aseg = non_silent_wave

        # 3. if no proper silence found for clipping
        if len(aseg) > 12000:
            aseg = aseg[:12000]
            show_info("Audio is over 12s, clipping short. (3)")

        aseg = remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
        aseg.export(temp_path, format="wav")
        ref_audio = temp_path

        # Cache the processed reference audio
        _ref_audio_cache[audio_hash] = ref_audio

    if not ref_text.strip():
        global _ref_text_cache
        if audio_hash in _ref_text_cache:
            # Use cached asr transcription
            show_info("Using cached reference text...")
            ref_text = _ref_text_cache[audio_hash]
        else:
            show_info("No reference text provided, transcribing reference audio...")
            ref_text = transcribe(ref_audio)
            # Cache the transcribed text (not caching custom ref_text, enabling users to do manual tweak)
            _ref_text_cache[audio_hash] = ref_text
    else:
        show_info("Using custom reference text...")

    # Ensure ref_text ends with a proper sentence-ending punctuation
    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "

    print("\nref_text  ", ref_text)

    return ref_audio, ref_text


# remove silence from generated wav


def remove_silence_for_generated_wav(filename):
    aseg = AudioSegment.from_file(filename)
    non_silent_segs = silence.split_on_silence(
        aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500, seek_step=10
    )
    non_silent_wave = AudioSegment.silent(duration=0)
    for non_silent_seg in non_silent_segs:
        non_silent_wave += non_silent_seg
    aseg = non_silent_wave
    aseg.export(filename, format="wav")


def remove_silence_for_generated_wav_numpy(wave_np, sample_rate, keep_silence=100):
    """
    Remove silence from a numpy array representing audio wave directly, 
    useful for chunk-level trimming before concatenation.
    """
    # Convert numpy array to AudioSegment
    # pydub requires audio to be in 16-bit integer format
    wave_np_int16 = np.int16(wave_np * 32767)
    aseg = AudioSegment(
        wave_np_int16.tobytes(), 
        frame_rate=sample_rate,
        sample_width=2, 
        channels=1
    )

    # For internal chunks, we don't want long silences.
    non_silent_segs = silence.split_on_silence(
        aseg, min_silence_len=100, silence_thresh=-50, keep_silence=keep_silence, seek_step=10
    )
    
    if not non_silent_segs:
        # If everything is stripped (rare), return original
        return wave_np

    non_silent_wave = AudioSegment.silent(duration=0)
    for non_silent_seg in non_silent_segs:
        non_silent_wave += non_silent_seg
        
    # Convert back to float32 numpy array
    samples = np.array(non_silent_wave.get_array_of_samples())
    return samples.astype(np.float32) / 32767.0



# save spectrogram


def save_spectrogram(spectrogram, path):
    plt.figure(figsize=(12, 4))
    plt.imshow(spectrogram, origin="lower", aspect="auto")
    plt.colorbar()
    plt.savefig(path)
    plt.close()


_LOUDNESS_NORMALIZER = {
    "initialized": False,
    "available": False,
    "pyln": None,
    "np": None,
    "meters": {},
}

def _init_loudness_normalizer() -> bool:
    if _LOUDNESS_NORMALIZER["initialized"]:
        return _LOUDNESS_NORMALIZER["available"]

    _LOUDNESS_NORMALIZER["initialized"] = True
    try:
        import numpy as np
        import pyloudnorm as pyln

        _LOUDNESS_NORMALIZER.update(
            {
                "available": True,
                "pyln": pyln,
                "np": np,
            }
        )
    except Exception as e:
        print(f"[WARN] pyloudnorm unavailable, fallback to torchaudio loudness. Detail: {e}")

    return _LOUDNESS_NORMALIZER["available"]


def normalize_audio_loudness(
    audio: torch.Tensor,
    sample_rate: int,
    target_lufs: float = -23.0,
):
    audio = audio.to(torch.float32)
    if _init_loudness_normalizer():
        np = _LOUDNESS_NORMALIZER["np"]
        pyln = _LOUDNESS_NORMALIZER["pyln"]
        meter = _LOUDNESS_NORMALIZER["meters"].get(sample_rate)
        if meter is None:
            meter = pyln.Meter(sample_rate)
            _LOUDNESS_NORMALIZER["meters"][sample_rate] = meter

        try:
            wav_np = audio.detach().cpu().transpose(0, 1).numpy()
            if wav_np.ndim == 1:
                wav_np = wav_np.reshape(-1, 1)
            current_loudness = meter.integrated_loudness(wav_np)
            if np.isfinite(current_loudness):
                normalized_np = pyln.normalize.loudness(wav_np, current_loudness, target_lufs)
                max_val = float(np.abs(normalized_np).max()) if normalized_np.size > 0 else 0.0
                if max_val >= 1.0:
                    normalized_np = normalized_np / max_val * 0.99
                audio = torch.from_numpy(normalized_np).transpose(0, 1).to(torch.float32)
        except Exception as e:
            print(f"[WARN] pyloudnorm normalization failed, fallback to torchaudio loudness. Detail: {e}")
    return audio

_REF_WAV_DENOISER = {
    "initialized": False,
    "available": False,
    "model": None,
    "state": None,
    "enhance": None,
    "sample_rate": None,
}


def _init_ref_wav_denoiser() -> bool:
    if _REF_WAV_DENOISER["initialized"]:
        return _REF_WAV_DENOISER["available"]

    _REF_WAV_DENOISER["initialized"] = True
    try:
        # DeepFilterNet2 is a strong speech denoiser and supports real-world noise.
        with suppress_stdout_stderr():
            from df.enhance import enhance, init_df

            model, state, _ = init_df()
        _REF_WAV_DENOISER.update(
            {
                "available": True,
                "model": model,
                "state": state,
                "enhance": enhance,
                "sample_rate": state.sr(),
            }
        )
    except Exception as e:
        print(f"[WARN] Ref wav denoiser unavailable, skip denoise. Install deepfilternet to enable it. Detail: {e}")

    return _REF_WAV_DENOISER["available"]


def denoise_ref_audio(
    ref_audio: torch.Tensor,
    ref_sr: int,
):
    if not _init_ref_wav_denoiser():
        return ref_audio, ref_sr

    try:
        denoise_sr = _REF_WAV_DENOISER["sample_rate"]
        # Denoise in mono for stability, then expand back to original channel count.
        mono_audio = ref_audio.mean(dim=0, keepdim=True).to(torch.float32).cpu()
        if ref_sr != denoise_sr:
            mono_audio = torchaudio.functional.resample(mono_audio, ref_sr, denoise_sr)

        enhanced = _REF_WAV_DENOISER["enhance"](
            _REF_WAV_DENOISER["model"], _REF_WAV_DENOISER["state"], mono_audio
        )
        if enhanced.dim() == 1:
            enhanced = enhanced.unsqueeze(0)
        elif enhanced.dim() > 2:
            enhanced = enhanced.reshape(1, -1)
        enhanced = torch.clamp(enhanced, -1.0, 1.0)

        if ref_sr != denoise_sr:
            enhanced = torchaudio.functional.resample(enhanced, denoise_sr, ref_sr)

        if enhanced.shape[0] != 1:
            enhanced = enhanced.mean(dim=0, keepdim=True)
        enhanced = enhanced.repeat(ref_audio.shape[0], 1)
        return enhanced.to(dtype=ref_audio.dtype), ref_sr
    except Exception as e:
        print(f"[WARN] Failed to denoise ref wav, use original waveform. Detail: {e}")
        return ref_audio, ref_sr
    
def audio_post_processing(mel, threshold=2.8, limit=3.5, start_bin=60):
    mel_high = mel[:, :, start_bin:]
    def apply_limit(x, t, m):
        # 只有超过threshold的部分才进入tanh
        margin = m - t
        return torch.where(
            x < t,
            x,
            t + margin * torch.tanh((x - t) / margin)
        )
    mel[:, :, start_bin:] = apply_limit(mel_high, threshold, limit)
    return mel


_LANGDETECT_WARNED = False


def normalize_lang_code(lang_code):
    if not lang_code:
        return None
    return lang_code.strip().lower().replace("_", "-").split("-", 1)[0]


def parse_voice_lang_tag(tag_content, voice_names=None, default_voice="main"):
    tag_content = tag_content.strip()
    if not tag_content:
        return default_voice, None

    if "|" in tag_content:
        voice_name, segment_lang = tag_content.split("|", 1)
        return voice_name.strip() or default_voice, normalize_lang_code(segment_lang)

    lower_tag = tag_content.lower()
    if lower_tag.startswith("lang:"):
        return default_voice, normalize_lang_code(tag_content.split(":", 1)[1])
    if lower_tag.startswith("lang="):
        return default_voice, normalize_lang_code(tag_content.split("=", 1)[1])

    if voice_names is not None and tag_content in voice_names:
        return tag_content, None

    normalized_lang = normalize_lang_code(tag_content)
    if normalized_lang and re.fullmatch(r"[a-z]{2,3}", normalized_lang):
        return default_voice, normalized_lang

    return tag_content, None


def detect_segment_lang(text, fallback_lang):
    global _LANGDETECT_WARNED

    if fastlid is None:
        if not _LANGDETECT_WARNED:
            print("Warning: fastlid is not installed, automatic language detection is disabled.")
            _LANGDETECT_WARNED = True
        return fallback_lang

    text = text.strip()
    if len(text) < 3:
        return fallback_lang

    try:
        return normalize_lang_code(fastlid(text)[0]) or fallback_lang
    except Exception as exc:
        print(f"Warning: failed to detect language for text '{text}': {exc}")
        return fallback_lang

def auto_split_mixed_text(text: str, fallback_lang: str) -> list[tuple[str, str]]:
    """
    Automatically splits mixed language text into segments.
    Returns a list of tuples: [(lang, text), ...]
    """
    # Regex patterns for different script blocks
    # Japanese kana & Chinese kanji (CJK)
    cjk_pattern = r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3000-\u303F\uFF00-\uFFEF]'
    # Korean Hangul
    ko_pattern = r'[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]'
    # Thai script
    th_pattern = r'[\u0E00-\u0E7F]'
    
    segments = []
    current_segment = ""
    current_type = None

    for char in text:
        if re.match(cjk_pattern, char):
            char_type = 'cjk'
        elif re.match(ko_pattern, char):
            char_type = 'ko'
        elif re.match(th_pattern, char):
            char_type = 'th'
        # \w matches all Unicode word characters (letters from any language, plus digits and underscore).
        # So \W matches everything else (spaces, punctuation, symbols).
        elif re.match(r'[\W\d_]', char):
            char_type = 'neutral'
        else:
            # Everything else (Latin, Cyrillic, Greek, Arabic, Devanagari, etc.)
            char_type = 'other_lang'

        if current_type is None:
            if char_type != 'neutral':
                current_type = char_type
            current_segment += char
        elif char_type == 'neutral':
            current_segment += char
        elif char_type == current_type:
            current_segment += char
        else:
            segments.append((current_segment, current_type))
            current_segment = char
            current_type = char_type

    if current_segment:
        segments.append((current_segment, current_type if current_type else 'cjk'))

    result = []
    for seg_text, seg_type in segments:
        if not seg_text.strip():
            # If it's just spaces or punctuation, append to the last segment if possible
            if result:
                result[-1] = (result[-1][0], result[-1][1] + seg_text)
            else:
                result.append((fallback_lang, seg_text))
            continue
            
        if seg_type == 'cjk':
            lang = "zh" #detect_segment_lang(seg_text, "zh")
        elif seg_type == 'ko':
            lang = 'ko'
        elif seg_type == 'th':
            lang = 'th'
        elif seg_type == 'other_lang':
            lang = detect_segment_lang(seg_text, "en") # Default text to English, fallback to 'en'
        else:
            lang = fallback_lang
            
        result.append((lang, seg_text))
        
    return result


def spans_to_text(spans):
    if isinstance(spans, str):
        return spans
    return "".join(text for _, text in spans)


def normalize_text_lang_spans(text, lang_spec, fallback_lang):
    if isinstance(lang_spec, list):
        return [(normalize_lang_code(lang) or fallback_lang, span_text) for lang, span_text in lang_spec if span_text]
    return [(normalize_lang_code(lang_spec) or fallback_lang, text)]


def slice_lang_spans(spans, start, end):
    sliced = []
    cursor = 0
    for lang, span_text in spans:
        span_start = cursor
        span_end = cursor + len(span_text)
        cursor = span_end
        overlap_start = max(start, span_start)
        overlap_end = min(end, span_end)
        if overlap_start < overlap_end:
            sliced.append((lang, span_text[overlap_start - span_start : overlap_end - span_start]))
    return sliced


def split_lang_spans_by_chunks(spans, chunks):
    full_text = spans_to_text(spans)
    chunk_spans = []
    cursor = 0
    for chunk in chunks:
        start = full_text.find(chunk, cursor)
        if start < 0:
            start = cursor
        end = start + len(chunk)
        chunk_spans.append(slice_lang_spans(spans, start, end))
        cursor = end
    return chunk_spans


def prepare_codeswitch_text_tokens_and_lang_ids(spans, tokenizer_name, ipa_tokenizer_getter, lang_to_id_map):
    tokens = []
    lang_ids = []
    for lang, span_text in spans:
        if not span_text:
            continue
        span_tokens = prepare_text_tokens(span_text, tokenizer_name, lang, ipa_tokenizer_getter)
        tokens.extend(span_tokens)
        lang_ids.extend([lang_to_id(lang, lang_to_id_map)] * len(span_tokens))
    return tokens, lang_ids


def resolve_package_example(path):
    if path and "infer/examples/" in path:
        return str(files("x_voice").joinpath(path))
    return path


def resolve_cached_path(path):
    if path and path.startswith("hf://"):
        return str(cached_path(path))
    return path


def resolve_ckpt_path(ckpt_file, model_cfg, model, ckpt_step):
    if ckpt_file:
        return resolve_cached_path(ckpt_file)

    rel_root = Path(str(files("x_voice").joinpath("../.."))).resolve()
    candidates = []
    if ckpt_step is not None:
        candidates.extend(
            [
                rel_root / "ckpts" / model / f"model_{ckpt_step}.pt",
                rel_root / "ckpts" / model / f"model_{ckpt_step}.safetensors",
            ]
        )
        save_dir = OmegaConf.select(model_cfg, "ckpts.save_dir", default=None)
        if save_dir:
            candidates.extend(
                [
                    rel_root / save_dir / f"model_{ckpt_step}.pt",
                    rel_root / save_dir / f"model_{ckpt_step}.safetensors",
                ]
            )

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise ValueError("ckpt_file is required when no local checkpoint can be resolved.")


def get_ipa_tokenizer_cache(tokenizer_name, with_stress):
    tokenizer_class_map = {
        "ipa_v3": PhonemizeTextTokenizerV3,
        "ipa_v6": PhonemizeTextTokenizerV6,
    }
    cache = {}

    def get_tokenizer_for_lang(lang):
        if not tokenizer_name.startswith("ipa"):
            return None
        if tokenizer_name not in tokenizer_class_map:
            raise ValueError(f"Unsupported IPA tokenizer: {tokenizer_name}")
        if lang not in cache:
            cache[lang] = tokenizer_class_map[tokenizer_name](
                language=get_ipa_id(lang),
                with_stress=with_stress,
            )
        return cache[lang]

    return get_tokenizer_for_lang


def normalize_text_for_lang(text, lang, normalizer_cache):
    try:
        from x_voice.eval.text_normalizer import TextNormalizer
    except ImportError:
        print("Warning: TextNormalizer is unavailable, skip text normalization.")
        return text

    if lang not in normalizer_cache:
        with suppress_stdout_stderr():
            normalizer_cache[lang] = TextNormalizer(language=lang)
    with suppress_stdout_stderr():
        return normalizer_cache[lang].normalize(text)


def validate_nllb_lang(lang):
    if lang not in XVOICE_TO_NLLB:
        raise ValueError(f"NLLB language mapping is missing for '{lang}'.")
    return XVOICE_TO_NLLB[lang]


def get_nllb_translator(device_name=device, show_info=None):
    global _nllb_tokenizer, _nllb_model
    if _nllb_tokenizer is None or _nllb_model is None:
        if show_info is not None:
            show_info("Loading NLLB translation model...")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        _nllb_tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_ID)
        _nllb_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_ID).eval().to(device_name)
    return _nllb_tokenizer, _nllb_model


def translate_text_nllb(text, src_lang, tgt_lang, device_name=device, show_info=None):
    tokenizer, model = get_nllb_translator(device_name=device_name, show_info=show_info)
    src_nllb = validate_nllb_lang(src_lang)
    tgt_nllb = validate_nllb_lang(tgt_lang)
    tokenizer.src_lang = src_nllb
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {key: value.to(device_name) for key, value in inputs.items()}
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_nllb)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=256,
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()


def prepare_text_tokens(text, tokenizer_name, lang, ipa_tokenizer_getter):
    if tokenizer_name == "pinyin":
        return convert_char_to_pinyin([text], polyphone=True)[0]
    if tokenizer_name.startswith("ipa"):
        ipa_tokenizer = ipa_tokenizer_getter(lang)
        ipa_text = ipa_tokenizer(text)
        return str_to_list_ipa_all(ipa_text, tokenizer_name, lang)
    return list(text)


def ensure_ref_text_punctuation(ref_text):
    if not ref_text:
        return ref_text
    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "
    return ref_text


def count_units(text, lang):
    units = count_syllables(text, lang)
    return max(units, 1)


def count_lang_spans_units(spans):
    return max(
        sum(count_units(span_text, lang) for lang, span_text in spans if span_text),
        1,
    )


def chunk_text_by_units(text, lang, max_units):
    chunks = []
    current_chunk = ""
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if not sentence:
            continue
        if count_units(current_chunk, lang) + count_units(sentence, lang) <= max_units:
            current_chunk += sentence + " " if len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if len(sentence[-1].encode("utf-8")) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks or [text.strip()]


def chunk_text_by_chars(text, max_chars=135):
    chunks = []
    current_chunk = ""
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if not sentence:
            continue
        if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current_chunk += sentence + " " if len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if len(sentence[-1].encode("utf-8")) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks or [text.strip()]


def prepare_ref_audio_tensor(ref_audio, target_rms_value, denoise_ref, device_name):
    audio, sr = torchaudio.load(ref_audio)
    if denoise_ref:
        audio, sr = denoise_ref_audio(audio, sr)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms_value:
        audio = audio * target_rms_value / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    return audio.to(device_name), rms


def predict_ref_speed(srp_model, audio):
    if srp_model is None:
        return None
    with torch.inference_mode():
        return srp_model.predict_speed(audio=audio).item()


def estimate_duration(
    ref_audio_len,
    ref_text,
    gen_text,
    ref_lang,
    gen_lang,
    sp_type,
    local_speed,
    fix_duration_value,
    predicted_speed,
):
    if fix_duration_value is not None:
        return int(fix_duration_value * target_sample_rate / hop_length)

    if sp_type == "pretrained":
        if predicted_speed is None:
            raise ValueError("sp_type='pretrained' requires srp_ckpt_file and a loaded SRP model.")
        gen_seconds = count_units(gen_text, gen_lang) / max(predicted_speed, 0.1) / local_speed
        return ref_audio_len + int(gen_seconds * target_sample_rate / hop_length)

    if sp_type == "syllable":
        ref_units = count_units(ref_text, ref_lang)
        gen_units = count_units(gen_text, gen_lang)
        return ref_audio_len + int(ref_audio_len / ref_units * gen_units / local_speed)

    ref_text_len = max(len(ref_text.encode("utf-8")), 1)
    gen_text_len = max(len(gen_text.encode("utf-8")), 1)
    return ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / local_speed)


def lang_to_id(lang, lang_to_id_map):
    unknown_id = len(lang_to_id_map)
    return lang_to_id_map.get(lang, unknown_id)


def load_srp_model(srp_model_cfg_file, srp_ckpt_file, device_name):
    srp_ckpt_file = resolve_cached_path(srp_ckpt_file)
    srp_cfg = OmegaConf.load(srp_model_cfg_file)
    srp_arch = OmegaConf.to_container(srp_cfg.model.arch, resolve=True)
    srp_mel_spec_kwargs = OmegaConf.to_container(srp_cfg.model.mel_spec, resolve=True)
    srp_model = SpeedPredictor(
        mel_spec_kwargs=srp_mel_spec_kwargs,
        loss_type=srp_cfg.model.get("loss", "CE"),
        arch_kwargs=srp_arch,
        sigma_factor=srp_cfg.model.get("gce_sigma", 2),
        silence_prob=srp_cfg.model.get("silence_prob", 0.0),
        silence_ratio_min=srp_cfg.model.get("silence_ratio_min", 0.2),
        silence_ratio_max=srp_cfg.model.get("silence_ratio_max", 0.8),
    ).to(device_name)
    return load_checkpoint(srp_model, srp_ckpt_file, device_name, dtype=torch.float32, use_ema=True)


def infer_xvoice_process(
    ref_audio,
    ref_text,
    gen_text,
    ref_lang,
    gen_lang,
    tokenizer_name,
    ipa_tokenizer_getter,
    model_obj,
    vocoder,
    lang_to_id_map,
    dominant_lang=None,
    srp_model=None,
    mel_spec_type_value=mel_spec_type,
    progress=tqdm,
    target_rms_value=target_rms,
    cross_fade_duration_value=cross_fade_duration,
    nfe_step_value=nfe_step,
    cfg_strength_value=cfg_strength,
    layered=layered,
    cfg_strength2_value=4.0,
    cfg_schedule_value=None,
    cfg_decay_time_value=0.6,
    sway_sampling_coef_value=sway_sampling_coef,
    local_speed=speed,
    fix_duration_value=fix_duration,
    sp_type="syllable",
    reverse=False,
    denoise_ref=False,
    loudness_norm=False,
    post_processing=False,
    remove_silence_chunk=False,
    device_name=device,
):
    if isinstance(gen_text, str):
        gen_text = [gen_text]
    if isinstance(gen_lang, str):
        gen_lang = [gen_lang]
    if dominant_lang is None:
        dominant_lang = [None] * len(gen_text)
    elif isinstance(dominant_lang, str):
        dominant_lang = [dominant_lang]

    assert len(gen_text) == len(gen_lang), "gen_text and gen_lang lists must have the same length"
    assert len(gen_text) == len(dominant_lang), "gen_text and dominant_lang lists must have the same length"

    audio, rms = prepare_ref_audio_tensor(ref_audio, target_rms_value, denoise_ref, device_name)
    ref_audio_len = audio.shape[-1] // hop_length
    ref_seconds = audio.shape[-1] / target_sample_rate
    remaining_seconds = 26 - ref_seconds
    predicted_speed = predict_ref_speed(srp_model, audio)

    all_batches = []
    item_chunk_counts = []
    for text, lang, item_dominant_lang in zip(gen_text, gen_lang, dominant_lang):
        spans = normalize_text_lang_spans(text, lang, normalize_lang_code(lang[0][0]) if isinstance(lang, list) and lang else None)
        full_text = spans_to_text(spans)
        dominant_lang = normalize_lang_code(item_dominant_lang) or detect_segment_lang(full_text, spans[0][0] if spans else None)
        if sp_type in {"syllable", "pretrained"}:
            if sp_type == "pretrained" and predicted_speed:
                max_units = max(int(predicted_speed * remaining_seconds * local_speed), 1)
            else:
                max_units = max(
                    int(count_units(ref_text, ref_lang) / ref_seconds * remaining_seconds * local_speed),
                    1,
                )
            gen_text_batches = chunk_text_by_units(full_text, dominant_lang, max_units)
        else:
            max_chars = int(len(ref_text.encode("utf-8")) / ref_seconds * remaining_seconds * local_speed)
            gen_text_batches = chunk_text_by_chars(full_text, max_chars=max(max_chars, 1))

        chunk_spans = split_lang_spans_by_chunks(spans, gen_text_batches)
        item_chunk_counts.append(len(chunk_spans))
        for batch_text, batch_spans in zip(gen_text_batches, chunk_spans):
            batch_dominant_lang = dominant_lang if item_dominant_lang else detect_segment_lang(batch_text, dominant_lang)
            all_batches.append((batch_text, batch_spans, batch_dominant_lang))

    print(f"\nGenerating audio in {len(all_batches)} chunks...")
    
    if len(all_batches) == 0:
        return None, target_sample_rate, None

    # Construct batch inputs
    final_text_list = []
    language_ids_list = []
    time_language_ids_list = []
    durations = []
    
    ref_tokens = prepare_text_tokens(ref_text, tokenizer_name, ref_lang, ipa_tokenizer_getter)
    ref_lang_id = lang_to_id(ref_lang, lang_to_id_map)

    for batch_text, batch_spans, dominant_lang in all_batches:
        batch_units = count_lang_spans_units(batch_spans)
        local_batch_speed = local_speed
        if batch_units < 4:
            local_batch_speed = min(local_batch_speed, 0.5)

        gen_tokens, gen_lang_ids = prepare_codeswitch_text_tokens_and_lang_ids(
            batch_spans,
            tokenizer_name,
            ipa_tokenizer_getter,
            lang_to_id_map,
        )
        final_text_list.append(ref_tokens + gen_tokens)
        
        dominant_lang_id = lang_to_id(dominant_lang, lang_to_id_map)
        language_ids_list.append([ref_lang_id] * len(ref_tokens) + gen_lang_ids)
        time_language_ids_list.append(dominant_lang_id)

        if fix_duration_value is not None:
            duration = int(fix_duration_value * target_sample_rate / hop_length)
        elif sp_type == "pretrained":
            if predicted_speed is None:
                raise ValueError("sp_type='pretrained' requires srp_ckpt_file and a loaded SRP model.")
            gen_seconds = batch_units / max(predicted_speed, 0.1) / local_batch_speed
            duration = ref_audio_len + int(gen_seconds * target_sample_rate / hop_length)
        elif sp_type == "syllable":
            ref_units = count_units(ref_text, ref_lang)
            duration = ref_audio_len + int(ref_audio_len / ref_units * batch_units / local_batch_speed)
        else:
            ref_text_len = max(len(ref_text.encode("utf-8")), 1)
            gen_text_len = max(len(batch_text.encode("utf-8")), 1)
            duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / local_batch_speed)
        durations.append(duration)

    B = len(all_batches)
    # CFM.sample expects raw waveform as [B, T] and computes mel internally.
    cond_batch = audio.expand(B, -1)

    duration_tensor = torch.tensor(durations, dtype=torch.long, device=device_name)
    
    # language_ids_list is a list of lists of varying lengths.
    # In CFM sample, language_ids can be [b, nt]. We need to pad it to max_seq_len.
    # We can use torch.nn.utils.rnn.pad_sequence
    max_len = max(len(ids) for ids in language_ids_list)
    padded_lang_ids = [ids + [-1] * (max_len - len(ids)) for ids in language_ids_list]
    language_ids_tensor = torch.tensor(padded_lang_ids, dtype=torch.long, device=device_name)
    time_language_ids_tensor = torch.tensor(time_language_ids_list, dtype=torch.long, device=device_name)

    with torch.inference_mode():
        print("[DEBUG] infer_xvoice_process before model.sample", flush=True)
        generated, _ = model_obj.sample(
            cond=cond_batch,
            text=final_text_list,
            duration=duration_tensor,
            steps=nfe_step_value,
            cfg_strength=cfg_strength_value,
            sway_sampling_coef=sway_sampling_coef_value,
            language_ids=language_ids_tensor,
            time_language_ids=time_language_ids_tensor,
            cfg_schedule=cfg_schedule_value,
            cfg_decay_time=cfg_decay_time_value,
            reverse=reverse,
            layered=layered,
            cfg_strength2=cfg_strength2_value,
            infer_mode=True,
        )
        print("[DEBUG] infer_xvoice_process after model.sample", flush=True)

        if post_processing:
            print("[DEBUG] infer_xvoice_process before post_processing", flush=True)
            generated = audio_post_processing(generated, threshold=2.5, limit=3.5)
            print("[DEBUG] infer_xvoice_process after post_processing", flush=True)

        generated = generated.to(torch.float32)
        
        generated_waves = []
        spectrograms = []
        
        for i in range(B):
            duration_i = durations[i]
            gen_i = generated[i:i+1] # [1, max_duration, num_channels]
            
            if reverse:
                gen_i = gen_i[:, : duration_i - ref_audio_len, :]
            else:
                gen_i = gen_i[:, ref_audio_len:duration_i, :]
                
            generated_mel_spec = gen_i.permute(0, 2, 1)
            
            if mel_spec_type_value == "vocos":
                print("[DEBUG] infer_xvoice_process before vocoder.decode", flush=True)
                generated_wave = vocoder.decode(generated_mel_spec).cpu()
                print("[DEBUG] infer_xvoice_process after vocoder.decode", flush=True)
            elif mel_spec_type_value == "bigvgan":
                print("[DEBUG] infer_xvoice_process before bigvgan", flush=True)
                generated_wave = vocoder(generated_mel_spec).cpu()
                print("[DEBUG] infer_xvoice_process after bigvgan", flush=True)
            else:
                raise ValueError(f"Unsupported vocoder: {mel_spec_type_value}")

            if rms < target_rms_value:
                generated_wave = generated_wave * rms / target_rms_value
            if loudness_norm:
                print("[DEBUG] infer_xvoice_process before loudness_norm", flush=True)
                generated_wave = normalize_audio_loudness(generated_wave, target_sample_rate, target_lufs=-23.0)
                print("[DEBUG] infer_xvoice_process after loudness_norm", flush=True)

            generated_waves.append(generated_wave.squeeze().numpy())
            spectrograms.append(generated_mel_spec[0].cpu().numpy())

    # Group generated_waves back to original gen_text items
    waves_by_original = [[] for _ in range(len(gen_text))]
    chunk_idx = 0
    for text_idx, chunk_count in enumerate(item_chunk_counts):
        for _ in range(chunk_count):
            waves_by_original[text_idx].append(generated_waves[chunk_idx])
            chunk_idx += 1

    final_waves_per_text = []
    for chunks_for_text in waves_by_original:
        if not chunks_for_text:
            final_waves_per_text.append(None)
            continue
            
        if cross_fade_duration_value <= 0:
            final_wave = np.concatenate(chunks_for_text)
        else:
            final_wave = chunks_for_text[0]
            for next_wave in chunks_for_text[1:]:
                cross_fade_samples = int(cross_fade_duration_value * target_sample_rate)
                cross_fade_samples = min(cross_fade_samples, len(final_wave), len(next_wave))
                if cross_fade_samples <= 0:
                    final_wave = np.concatenate([final_wave, next_wave])
                    continue
                prev_overlap = final_wave[-cross_fade_samples:]
                next_overlap = next_wave[:cross_fade_samples]
                fade_out = np.linspace(1, 0, cross_fade_samples)
                fade_in = np.linspace(0, 1, cross_fade_samples)
                cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in
                final_wave = np.concatenate(
                    [final_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
                )
        final_waves_per_text.append(final_wave)

    combined_spectrogram = np.concatenate(spectrograms, axis=1) if spectrograms else None
    
    if len(gen_text) == 1:
        print("[DEBUG] infer_xvoice_process before single return", flush=True)
        return final_waves_per_text[0], target_sample_rate, combined_spectrogram
    print("[DEBUG] infer_xvoice_process before multi return", flush=True)
    return final_waves_per_text, target_sample_rate, combined_spectrogram


def infer_xvoice_droptext_process(
    ref_audio,
    gen_text,
    gen_lang,
    tokenizer_name,
    ipa_tokenizer_getter,
    model_obj,
    vocoder,
    lang_to_id_map,
    srp_model,
    dominant_lang=None,
    mel_spec_type_value=mel_spec_type,
    progress=tqdm,
    target_rms_value=target_rms,
    cross_fade_duration_value=cross_fade_duration,
    nfe_step_value=nfe_step,
    cfg_strength_value=cfg_strength,
    layered=layered,
    cfg_strength2_value=4.0,
    cfg_schedule_value="square",
    cfg_decay_time_value=0.6,
    sway_sampling_coef_value=sway_sampling_coef,
    local_speed=speed,
    fix_duration_value=fix_duration,
    reverse=False,
    denoise_ref=False,
    loudness_norm=False,
    post_processing=False,
    remove_silence_chunk=False,
    device_name=device,
):
    if srp_model is None:
        raise ValueError("drop-text inference requires a loaded SRP model.")

    if isinstance(gen_text, str):
        gen_text = [gen_text]
    if isinstance(gen_lang, str):
        gen_lang = [gen_lang]
    if dominant_lang is None:
        dominant_lang = [None] * len(gen_text)
    elif isinstance(dominant_lang, str):
        dominant_lang = [dominant_lang]

    assert len(gen_text) == len(gen_lang), "gen_text and gen_lang lists must have the same length"
    assert len(gen_text) == len(dominant_lang), "gen_text and dominant_lang lists must have the same length"

    audio, rms = prepare_ref_audio_tensor(ref_audio, target_rms_value, denoise_ref, device_name)
    ref_audio_len = audio.shape[-1] // hop_length
    prompt_seconds = audio.shape[-1] / target_sample_rate
    remaining_seconds = 22 - prompt_seconds
    predicted_speed = predict_ref_speed(srp_model, audio)
    if predicted_speed is None:
        raise ValueError("drop-text inference requires SRP speed prediction.")

    all_batches = []
    item_chunk_counts = []
    for text, lang, item_dominant_lang in zip(gen_text, gen_lang, dominant_lang):
        spans = normalize_text_lang_spans(text, lang, normalize_lang_code(lang[0][0]) if isinstance(lang, list) and lang else None)
        full_text = spans_to_text(spans)
        dominant_lang = normalize_lang_code(item_dominant_lang) or detect_segment_lang(full_text, spans[0][0] if spans else None)
        max_units = max(int(predicted_speed * remaining_seconds * local_speed), 1)
        gen_text_batches = chunk_text_by_units(full_text, dominant_lang, max_units)
        chunk_spans = split_lang_spans_by_chunks(spans, gen_text_batches)
        item_chunk_counts.append(len(chunk_spans))
        for batch_text, batch_spans in zip(gen_text_batches, chunk_spans):
            batch_dominant_lang = dominant_lang if item_dominant_lang else detect_segment_lang(batch_text, dominant_lang)
            all_batches.append((batch_text, batch_spans, batch_dominant_lang))

    print(f"\nGenerating audio in {len(all_batches)} chunks...")
    
    if len(all_batches) == 0:
        return None, target_sample_rate, None

    # Construct batch inputs
    final_text_list = []
    language_ids_list = []
    time_language_ids_list = []
    durations = []
    
    for batch_text, batch_spans, dominant_lang in all_batches:
        batch_units = count_lang_spans_units(batch_spans)
        local_batch_speed = local_speed
        if batch_units < 4:
            local_batch_speed = min(local_batch_speed, 0.5)

        gen_tokens, gen_lang_ids = prepare_codeswitch_text_tokens_and_lang_ids(
            batch_spans,
            tokenizer_name,
            ipa_tokenizer_getter,
            lang_to_id_map,
        )
        final_text_list.append(gen_tokens)
        
        dominant_lang_id = lang_to_id(dominant_lang, lang_to_id_map)
        language_ids_list.append(gen_lang_ids)
        time_language_ids_list.append(dominant_lang_id)

        if fix_duration_value is not None:
            duration = int(fix_duration_value * target_sample_rate / hop_length)
        else:
            gen_seconds = max(1.0, batch_units / max(predicted_speed, 0.1) / local_batch_speed)
            duration = ref_audio_len + int(gen_seconds * target_sample_rate / hop_length)
        durations.append(duration)

    B = len(all_batches)
    # Expand cond to match batch size [b, nw], transform to mel in cfm.py
    cond_batch = audio.expand(B, -1)

    duration_tensor = torch.tensor(durations, dtype=torch.long, device=device_name)
    max_len = max(len(ids) for ids in language_ids_list)
    padded_lang_ids = [ids + [-1] * (max_len - len(ids)) for ids in language_ids_list]
    language_ids_tensor = torch.tensor(padded_lang_ids, dtype=torch.long, device=device_name)
    time_language_ids_tensor = torch.tensor(time_language_ids_list, dtype=torch.long, device=device_name)

    with torch.inference_mode():
        generated, _ = model_obj.sample(
            cond=cond_batch,
            text=final_text_list,
            duration=duration_tensor,
            steps=nfe_step_value,
            cfg_strength=cfg_strength_value,
            sway_sampling_coef=sway_sampling_coef_value,
            language_ids=language_ids_tensor,
            time_language_ids=time_language_ids_tensor,
            cfg_schedule=cfg_schedule_value,
            cfg_decay_time=cfg_decay_time_value,
            reverse=reverse,
            layered=layered,
            cfg_strength2=cfg_strength2_value,
            infer_mode=True,
        )

        if post_processing:
            generated = audio_post_processing(generated, threshold=2.5, limit=3.5)

        generated = generated.to(torch.float32)
        
        generated_waves = []
        spectrograms = []
        
        # Process each item in the batch
        for i in range(B):
            duration_i = durations[i]
            gen_i = generated[i:i+1] # [1, max_duration, num_channels]
            
            if reverse:
                gen_i = gen_i[:, : duration_i - ref_audio_len, :]
            else:
                gen_i = gen_i[:, ref_audio_len:duration_i, :]
                
            generated_mel_spec = gen_i.permute(0, 2, 1)
            
            if mel_spec_type_value == "vocos":
                generated_wave = vocoder.decode(generated_mel_spec).cpu()
            elif mel_spec_type_value == "bigvgan":
                generated_wave = vocoder(generated_mel_spec).cpu()
            else:
                raise ValueError(f"Unsupported vocoder: {mel_spec_type_value}")

            if rms < target_rms_value:
                generated_wave = generated_wave * rms / target_rms_value
            if loudness_norm:
                generated_wave = normalize_audio_loudness(generated_wave, target_sample_rate, target_lufs=-23.0)
            
            wave_np = generated_wave.squeeze().numpy()
            
            if remove_silence_chunk:
                batch_text = all_batches[i][0].strip()
                # If the chunk doesn't end with punctuation, trim trailing silence aggressively
                # If it doesn't start with punctuation, trim leading silence aggressively
                is_start_punct = re.match(r'^[\W_]', batch_text) is not None
                is_end_punct = re.search(r'[\W_]$', batch_text) is not None
                
                if not (is_start_punct or is_end_punct):
                    # print("removing silence")
                    # We trim the chunk to remove extra silence generated by the model
                    wave_np = remove_silence_for_generated_wav_numpy(wave_np, target_sample_rate, keep_silence=50)

            generated_waves.append(wave_np)
            spectrograms.append(generated_mel_spec[0].cpu().numpy())

    # Group generated_waves back to original gen_text items
    waves_by_original = [[] for _ in range(len(gen_text))]
    chunk_idx = 0
    for text_idx, chunk_count in enumerate(item_chunk_counts):
        for _ in range(chunk_count):
            waves_by_original[text_idx].append(generated_waves[chunk_idx])
            chunk_idx += 1

    final_waves_per_text = []
    for chunks_for_text in waves_by_original:
        if not chunks_for_text:
            final_waves_per_text.append(None)
            continue
            
        if cross_fade_duration_value <= 0:
            final_wave = np.concatenate(chunks_for_text)
        else:
            final_wave = chunks_for_text[0]
            for next_wave in chunks_for_text[1:]:
                cross_fade_samples = int(cross_fade_duration_value * target_sample_rate)
                cross_fade_samples = min(cross_fade_samples, len(final_wave), len(next_wave))
                if cross_fade_samples <= 0:
                    final_wave = np.concatenate([final_wave, next_wave])
                    continue
                prev_overlap = final_wave[-cross_fade_samples:]
                next_overlap = next_wave[:cross_fade_samples]
                fade_out = np.linspace(1, 0, cross_fade_samples)
                fade_in = np.linspace(0, 1, cross_fade_samples)
                cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in
                final_wave = np.concatenate(
                    [final_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
                )
        final_waves_per_text.append(final_wave)

    combined_spectrogram = np.concatenate(spectrograms, axis=1) if spectrograms else None
    
    if len(gen_text) == 1:
        return final_waves_per_text[0], target_sample_rate, combined_spectrogram
    return final_waves_per_text, target_sample_rate, combined_spectrogram
