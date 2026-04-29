# X-Voice Inference

This folder contains three inference entry points:

- `infer_cli_stage1.py`: X-Voice Stage1 zero-shot voice cloning. Requires reference audio and reference text.
- `infer_cli_stage2.py`: X-Voice Stage2 drop-text voice cloning. Requires reference audio, does not need reference text, and uses SRP to predict duration.
- `infer_gradio.py`: unified web demo for zero-shot cloning and translate-and-clone.

All commands below should be run from the repository root.

Pretrained X-Voice checkpoints are available on [Hugging Face](https://huggingface.co/XRXRX/X-Voice). The Gradio demo downloads the default checkpoints automatically. For CLI usage, download the needed files from the same repository and set the paths in the TOML file or pass them with command-line flags.

## X-Voice Stage1 CLI

X-Voice Stage1 clones a reference voice with reference transcript conditioning.

Run with the example TOML:

```bash
python -m x_voice.infer.infer_cli_stage1 \
  -c src/x_voice/infer/examples/basic/basic_stage1.toml
```

The matching config is:

```text
src/x_voice/infer/examples/basic/basic_stage1.toml
```

Important fields:

| Field | Meaning |
| --- | --- |
| `model_cfg` | X-Voice Stage1 yaml, usually `src/x_voice/configs/XVoice_Base_Stage1.yaml` |
| `ckpt_file` | X-Voice Stage1 checkpoint path |
| `vocab_file` | X-Voice Stage1 vocab path |
| `ref_audio` | reference voice audio |
| `ref_text` | transcript for `ref_audio`; required for X-Voice Stage1 |
| `gen_text` / `gen_file` | text to synthesize |
| `ref_lang` / `gen_lang` | optional language codes |
| `auto_detect_lang` | if true, detect missing ref/gen language automatically |
| `normalize_text` | normalize text by language before tokenization |
| `sp_type` | duration estimator: `syllable`, `pretrained`, or `utf` |

For normal X-Voice Stage1 use, keep:

```toml
sp_type = "syllable"
auto_detect_lang = true
normalize_text = true
```

## X-Voice Stage2 CLI

X-Voice Stage2 is drop-text inference. It uses reference audio only and does not need reference text.

Run with the example TOML:

```bash
python -m x_voice.infer.infer_cli_stage2 \
  -c src/x_voice/infer/examples/basic/basic_stage2.toml
```

The matching config is:

```text
src/x_voice/infer/examples/basic/basic_stage2.toml
```

Important fields:

| Field | Meaning |
| --- | --- |
| `model_cfg` | X-Voice Stage2 yaml, usually `src/x_voice/configs/XVoice_Base_Stage2.yaml` |
| `ckpt_file` | X-Voice Stage2 checkpoint path |
| `srp_model_cfg` | SRP config, usually `src/rate_pred/configs/SpeedPredict_Multilingual.yaml` |
| `srp_ckpt_file` | SRP checkpoint path; required for X-Voice Stage2 |
| `vocab_file` | X-Voice Stage1 vocab path used by X-Voice Stage2 tokenizer |
| `ref_audio` | reference voice audio |
| `gen_text` / `gen_file` | text to synthesize |
| `gen_lang` | optional generated text language |
| `auto_detect_lang` | if true, detect missing generated text language automatically |

X-Voice Stage2 config must point to a model yaml with:

```yaml
model:
  sft: true
  use_total_text: false
```

## Code-Switch Text

Both CLI scripts support code-switch text.

Automatic mode:

```toml
auto_detect_lang = true
gen_lang = ""
gen_text = "I was planning to go out, but 外面好像快下雨了. Maybe I’ll stay home."
```

Manual language tags:

```toml
gen_text = """
[main|en]I was planning to go out for dinner, but
[main|zh]外面好像快下雨了.
[main|en]Maybe I’ll just stay home and order something.
"""
```

Tag format:

| Format | Meaning |
| --- | --- |
| `[en]text` | language tag, default voice `main` |
| `[main]text` | voice tag, language auto-detected or from config |
| `[main\|zh]text` | voice and language tag |

Language spans are used for tokenizer, token-wise LID, and duration unit counting. The dominant language is only used for time embedding.

## Multi-Voice TOML

Both CLI scripts support multiple voices through `[voices.<name>]`.

X-Voice Stage1 voice entries can include `ref_audio`, `ref_text`, `ref_lang`, `gen_lang`, and `speed`.

Example:

```toml
ref_audio = "infer/examples/basic/basic_ref_en.wav"
ref_text = "Some call me nature, others call me mother nature."
auto_detect_lang = true

gen_text = """
[main|en]This is the main voice.
[alice|zh]这是 Alice 的声音。
"""

[voices.alice]
ref_audio = "path/to/alice.wav"
ref_text = "Alice reference transcript."
ref_lang = "en"
speed = 1.0
```

X-Voice Stage2 voice entries only need `ref_audio` and optional `gen_lang` / `speed`.

## Common Parameters

| Parameter | Default in examples | Meaning |
| --- | --- | --- |
| `output_dir` | `tests` | output folder relative to current working directory |
| `output_file` | example-specific `.wav` | output filename |
| `vocoder_name` | `vocos` | vocoder type |
| `load_vocoder_from_local` | `true` | load vocoder from local path in model yaml |
| `save_chunk` | `false` | save generated chunks |
| `remove_silence` | `false` | remove long silence from final wav |
| `target_rms` | `0.1` | reference RMS normalization target |
| `cross_fade_duration` | `0.15` | cross-fade seconds between chunks |
| `nfe_step` | `32` | ODE sampling steps |
| `cfg_strength` | `2.5` | first CFG strength |
| `layered` | `true` | enable layered CFG |
| `cfg_strength2` | `4.0` | second CFG strength |
| `cfg_schedule` | `square` | `square`, `cosine`, or `none` |
| `cfg_decay_time` | `0.6` | CFG schedule decay time |
| `sway_sampling_coef` | `-1.0` | sway sampling coefficient |
| `speed` | `1.0` | speaking speed multiplier |
| `denoise_ref` | `true` | denoise reference audio before inference |
| `loudness_norm` | `true` | normalize generated loudness |
| `post_processing` | `true` | apply generated mel post-processing |

## Gradio Demo

Run:

```bash
python -m x_voice.infer.infer_gradio \
  --host 0.0.0.0 \
  --port 7860
```

For proxied environments, pass `--root_path`:

```bash
python -m x_voice.infer.infer_gradio \
  --host 0.0.0.0 \
  --port 7860 \
  --root_path /your/proxy/path/7860
```

The web demo has two modes.

### Zero-Shot Voice Cloning

This mode preserves the normal cloning workflow:

- choose X-Voice Stage1 or X-Voice Stage2
- upload reference audio
- provide reference text for X-Voice Stage1, or leave it empty to transcribe with Whisper
- enter text manually or use manual code-switch segments
- synthesize one output audio

X-Voice Stage1 requires the reference voice language to be one of the supported 30 languages. X-Voice Stage2 can use a reference voice in any language.

### Translate & Clone

This mode uses X-Voice Stage1 only.

Workflow:

1. Upload reference audio.
2. Provide reference text, or leave it empty for Whisper transcription.
3. Choose reference language, or use auto detection.
4. Choose target languages with the multi-select dropdown.
5. Click `Generate All Languages` to select all supported targets except the reference language.
6. Click `Translate & Clone`.

The demo translates the reference text with NLLB 600M, then synthesizes each translated text with the reference voice.

Outputs:

- translated text list
- preview language dropdown
- translated text preview
- generated audio preview

## Checkpoints

X-Voice checkpoints can be downloaded from:

```text
https://huggingface.co/XRXRX/X-Voice
```

The Gradio demo downloads the default files automatically from this Hugging Face repository:

- X-Voice Stage1: `XVoice_Base_Stage1/model_600000.safetensors`
- X-Voice Stage2: `XVoice_Base_Stage2/model_70000.safetensors`
- SRP: `SpeedPredictor/model_28000.safetensors`
- Vocab: `XVoice_Base_Stage1/vocab.txt`

For CLI usage, download the checkpoints manually and update the TOML paths. The example TOML files use local paths such as:

```toml
ckpt_file = "ckpts/stage1_model_600000.safetensors"
srp_ckpt_file = "ckpts/srp_model_80000.safetensors"
```

You can also override paths directly:

```bash
python -m x_voice.infer.infer_cli_stage1 \
  -c src/x_voice/infer/examples/basic/basic_stage1.toml \
  --ckpt_file /path/to/model_600000.safetensors \
  --vocab_file /path/to/vocab.txt
```

```bash
python -m x_voice.infer.infer_cli_stage2 \
  -c src/x_voice/infer/examples/basic/basic_stage2.toml \
  --ckpt_file /path/to/model_70000.safetensors \
  --srp_ckpt_file /path/to/model_28000.safetensors \
  --vocab_file /path/to/vocab.txt
```
