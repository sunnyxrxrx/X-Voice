# Speaking Rate Predictor

Speaking Rate Predictor (SRP) is the duration helper used by X-Voice Stage2, following [Cross-Lingual F5-TTS](https://arxiv.org/abs/2509.14579).

X-Voice Stage1 has reference text, so inference can estimate generation length from the reference text/audio ratio. X-Voice Stage2 drops reference text, so it uses SRP to predict the prompt speaking rate from the reference audio and then estimates the target duration from the generated text.

## What SRP Does

SRP takes reference audio and predicts a speaking-rate bucket. During X-Voice Stage2 inference, this predicted speed is used to:

- estimate each generated chunk duration,
- keep drop-text inference stable without requiring reference text.

## Files

```text
src/rate_pred/
├── configs/
│   └── SpeedPredict_Multilingual.yaml
├── model/
│   ├── speed_predictor.py
│   ├── dataset.py
│   └── utils.py
└── train/
    ├── README.md
    ├── train.py
    └── datasets/
        └── prepare_multilingual_speed.py
```

Key files:

| File | Purpose |
| --- | --- |
| `configs/SpeedPredict_Multilingual.yaml` | Default SRP training config |
| `model/speed_predictor.py` | SpeedPredictor model and `predict_speed()` |
| `model/dataset.py` | Loads prepared SRP Arrow data |
| `model/utils.py` | Language and syllable-count helpers |
| `train/datasets/prepare_multilingual_speed.py` | Builds SRP training data from the X-Voice dataset |
| `train/train.py` | SRP training entry point |

## Dataset

SRP uses the X-Voice training dataset: [XRXRX/X-Voice-Dataset-Train](https://huggingface.co/datasets/XRXRX/X-Voice-Dataset-Train)

The expected input layout is the same X-Voice dataset layout used by [src/x_voice/train/README.md](../../src/x_voice/train/README.md):

```text
x_voice/
├── wavs/
└── csvs_stage2/
    ├── metadata_bg_voxpopuli.csv
    ├── ...
    └── metadata_zh_emilia.csv
```

Prepare SRP data:

```bash
python src/rate_pred/train/datasets/prepare_multilingual_speed.py \
  --inp_dir /path/to/x_voice \
  --dataset_name multilingual_250_100
```

This writes:

```text
data/multilingual_250_100_srp/
```

## Training

Train with the default config:

```bash
accelerate launch src/rate_pred/train/train.py \
  --config-name SpeedPredict_Multilingual.yaml
```

If your prepared dataset name is different:

```bash
accelerate launch src/rate_pred/train/train.py \
  --config-name SpeedPredict_Multilingual.yaml \
  ++datasets.name=your_dataset_name
```

For full preprocessing and training details, see:

```text
src/rate_pred/train/README.md
```

## Inference

X-Voice Stage2 inference loads an SRP checkpoint through:

```text
src/x_voice/infer/infer_cli_stage2.py
src/x_voice/infer/infer_gradio.py
```

For CLI inference, set `srp_ckpt_file` in the Stage2 TOML file or pass it with a command-line flag.

For Gradio inference, the default SRP checkpoint is downloaded automatically from [XRXRX/X-Voice](https://huggingface.co/XRXRX/X-Voice)

Default SRP checkpoint path in the Hugging Face repo:

```text
SpeedPredictor/model_28000.safetensors
```

See the X-Voice inference [README](../../src/x_voice/infer/README.md) for Stage2 usage.
