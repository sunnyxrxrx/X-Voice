# CLI Inference Instructions 

There are two usage methods:
1. Pass parameters directly in the command line
2. Using external TOML (command line + `-c your.toml`)

Parameter Priority:
1. Command line arguments
2. TOML configuration (only when `-c` is passed)
3. Code default values

## Pure Command Line

Using only command line arguments + code default values, suitable for fast single-speaker inference.

It is recommended to explicitly provide at least:
1. `--ref_audio`: Speaker reference audio
2. `--gen_text` or `--gen_file`: Target text (choose one)
3. `--lang`: target language (or enable `--auto_detect_lang` then detect the language of the target text automatically)

>[!NOTE]
> - ``--gen_text`` format
>   ```
>   # Provide the string directly
>   gen_text = "a sentence you want to speak"
>   # Use language tags, e.g. [en], [zh]
>   gen_text = "[lang1]sentence1. [lang2]sentence2"
>   ```
>  - Language reading order:
>     1. First, the model will check if the language is explicitly specified in the segment tag of `gen_text`.
>     2. If not specified and `--auto_detect_lang` is enabled, automatically detect it based on the text segment.
>     3. If still undetermined, fall back to the global `--lang`. If this is also not provided, it defaults to `en`.
>  - Automatic recognition and switching of multiple languages within the same sentence is currently not supported.


### Other Options and Default Values

<details>
<summary>Models and Checkpoints</summary>

| Parameter | Description |
| --- | --- |
| `--model` | Model name, default `XVoice_v1_Base_Stage2` |
| `--model_cfg` | Model configuration path, default `src/x_voice/configs/{model}.yaml` |
| `--ckpt_file` | Main model weights path, default empty (download automatically) | 
| `--srp_model_cfg` | Speaking rate predictor configuration path, default `src/srp/configs/SpeedPredict_Multilingual.yaml` |
| `--srp_ckpt_file` | Speaking rate predictor weights path, default empty (download automatically) |
| `--vocab_file` | Custom vocabulary path, default reads according to the ``--model_cfg`` configuration | 

</details>


<details>
<summary>Output Configuration</summary>

| Parameter | Description |
| --- | --- |
| `--output_dir` | Output directory, default `tests/`  |
| `--output_file` | Output filename, default `infer_cli_droptext_timestamp.wav` | 
| `--vocoder_name` | Vocoder type, default `vocos` |
| `--save_chunk` | Whether to save segmented audios, default `false` | 
| `--remove_silence` | Whether to remove long silences from the audio, default `false` | 

</details>

<details>
<summary>Sampling and Control</summary>

| Parameter | Description & Default Value |
| --- | --- |
| `--cross_fade_duration` | Cross-fade duration for segment concatenation (seconds), default `0.15` | 
| `--nfe_step` | Number of sampling steps, default `32` | 
| `--cfg_strength` |  CFG strength, default `2.0`  |
| `--sway_sampling_coef` | Sway Sampling coefficient, default `-1.0` | 
| `--speed` | Global speaking rate multiplier (relative to reference audio), default `1.0` |
| `--fix_duration` | Fixed total duration (seconds), default `None` |
| `--layered` | Whether to enable layered CFG, default `false` | 
| `--cfg_strength2` | Second strength for layered CFG, default `4.0` |
| `--cfg_schedule` | CFG schedule (square/cosine/none), default `square` | 
| `--cfg_decay_time` | CFG decay start time, default `0.6` | 

</details>

### Examples
```bash
python -m x_voice.infer.infer_cli_droptext \
  --ref_audio "path/to/main.wav" \
  --gen_text "今日はいい天気です。" \
  --auto_detect_lang \
  --output_dir tests \
  --output_file cli_single_lang.wav
```
```bash
python -m x_voice.infer.infer_cli_droptext \
  --ref_audio "path/to/main.wav" \
  --gen_text "[zh] 你好。[en] Hello there. [ja] 今日はいい天気です。" \
  --output_dir tests \
  --output_file cli_mix_lang.wav
```

## Using External TOML Configuration File

Invocation method:

```bash
python -m x_voice.infer.infer_cli_droptext -c path/to/your_config.toml
```

Suitable for complex parameter invocations, such as custom models, multi-speaker synthesis, etc.

### Multi-speaker Inference

Multi-speaker inference additionally requires:
1. Configure the speaker-to-audio-path mapping (`ref_audio`) under `[voices.<name>]`
2. Use the corresponding speaker tag in the text (e.g., `[alice]`)

> [!NOTE] 
>- Speaker priority:
>   1. Explicitly specified by segment tags first (e.g., `[alice]`, `[alice|en]`)
>   2. If not explicitly specified or the tag is invalid, fall back to `main`, i.e., the one specified in ``ref_audio``
>- Language priority:
>   1. Language explicitly stated in the segment tag (e.g., `[alice|en]`, `[en]`)
>   2. Automatic detection (if `auto_detect_lang=true`)
>   3. Specified in the global `lang`

### TOML Example

```toml
model = "XVoice_v1_Base_test"
srp_model_cfg = "src/configs/SpeedPredict_Multilingual.yaml"
ckpt_file = "ckpts/XVoice_v1_Base_test_ipa/model_70000.pt"
srp_ckpt_file = "ckpts/SpeedPredictor_multilingual_ipa/model_28000.pt"

ref_audio = "path/to/main.wav"
auto_detect_lang = true

gen_text = "[main] 这里是主说话人。 [alice|en] Hello, I am Alice. [bob] 今天天气真的好啊。 [ja] 今日はいい天気です。"
gen_file = ""

output_dir = "tests"
output_file = "toml_multi_voice.wav"

[voices.alice]
ref_audio = "path/to/alice.wav"
speed = 1.00

[voices.bob]
ref_audio = "path/to/bob.wav"
speed = 0.95
```