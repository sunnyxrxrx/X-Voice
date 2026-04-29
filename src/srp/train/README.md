# SRP Training

This folder trains the multilingual Speed Rate Predictor (SRP) used by X-Voice Stage2 duration prediction.

## Prepare Dataset

Download the X-Voice training dataset from [Hugging Face](https://huggingface.co/datasets/XRXRX/X-Voice-Dataset-Train).

The SRP preprocessing script follows the same X-Voice dataset layout used by `src/x_voice/train/README.md`:

```text
x_voice/
├── wavs/
│   ├── bg/
│   ├── ...
│   └── zh/
└── csvs_stage2/
    ├── metadata_bg_voxpopuli.csv
    ├── ...
    └── metadata_zh_emilia.csv
```

Prepare SRP data:

```bash
python src/srp/train/datasets/prepare_multilingual_speed.py \
  --inp_dir /path/to/x_voice \
  --dataset_name multilingual_250_100
```

The script reads:

```text
/path/to/x_voice/csvs_stage2/metadata_*.csv
/path/to/x_voice/wavs/...
```

It writes:

```text
data/multilingual_250_100_srp/
├── raw.arrow
├── raw_val.arrow
├── duration.json
├── duration_val.json
├── speed_syllables_counts_train.json
└── speed_syllables_hist_train.png
```

`--dataset_name` should match the dataset name used in the SRP config. The prepare script appends `_srp` to the output folder name.

## Configure Training

The default config is:

```text
src/srp/configs/SpeedPredict_Multilingual.yaml
```

Important fields:

| Field | Meaning |
| --- | --- |
| `datasets.name` | SRP dataset name, for example `multilingual_250_100` |
| `datasets.batch_size_per_gpu` | frame/sample budget per GPU |
| `model.loss` | `GCE` or `CE` |
| `model.mel_spec` | mel configuration; should match X-Voice inference |
| `ckpts.save_dir` | checkpoint output directory |
| `ckpts.logger` | `wandb`, `tensorboard`, or `null` |

If your prepared dataset name is different, override it when launching:

```bash
++datasets.name=your_dataset_name
```

## Training

Set up accelerate first:

```bash
accelerate config
```

Launch training:

```bash
accelerate launch src/srp/train/train.py \
  --config-name SpeedPredict_Multilingual.yaml
```

Override config values if needed:

```bash
accelerate launch --mixed_precision=fp16 src/srp/train/train.py \
  --config-name SpeedPredict_Multilingual.yaml \
  ++datasets.name=multilingual_250_100 \
  ++datasets.batch_size_per_gpu=19200
```

Checkpoints are saved under:

```text
ckpts/${model.name}_${datasets.name}_${model.loss}/...
```

## W&B Logging

The `wandb/` directory will be created under the path where you run the training script.

By default, logging depends on `ckpts.logger` in the config. To use W&B, log in manually:

```bash
wandb login
```

Or set an API key:

```bash
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
```

For offline logging:

```bash
export WANDB_MODE=offline
```
