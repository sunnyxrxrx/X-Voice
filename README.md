# X-Voice: Enabling Everyone to Speak 30 Languages via Zero-Shot Cross-Lingual Voice Cloning

[![python](https://img.shields.io/badge/Python-3.11-brightgreen)](https://github.com/SWivid/F5-TTS)
[![arXiv](https://img.shields.io/badge/arXiv-Coming_soon-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/unknown)
[![demo](https://img.shields.io/badge/GitHub-Demo-orange.svg)](https://sunnyxrxrx.github.io/X-Voice-Demo/)
[![hfspace](https://img.shields.io/badge/🤗-HF%20Space-yellow)](unknown)
[![dataset](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/XRXRX/X-Voice-Dataset-Train)
[![dataset](https://img.shields.io/badge/🤗-Benchmark-yellow)](https://huggingface.co/datasets/XRXRX/X-Voice-Testset)
[![lab](https://img.shields.io/badge/🏫-X--LANCE-grey?labelColor=lightgrey)](https://x-lance.sjtu.edu.cn/)
[![lab](https://img.shields.io/badge/🏫-SII-grey?labelColor=lightgrey)](https://www.sii.edu.cn/)
[![lab](https://img.shields.io/badge/🏫-CLSP-grey?labelColor=lightgrey)](https://www.clsp.jhu.edu)
[![company](https://img.shields.io/badge/🏢-Geely-grey?labelColor=lightgrey)](https://www.geely.com)
<!-- <img src="https://github.com/user-attachments/assets/12d7749c-071a-427c-81bf-b87b91def670" alt="Watermark" style="width: 40px; height: auto"> -->

**X-Voice** is a flow-matching based multilingual zero-shot voice cloning system that enables one speaker to speak 30+ languages without requiring reference text at inference time.

Built on top of F5-TTS, X-Voice extends the framework with a large-scale multilingual training recipe, unified multilingual phonetic representations, and transcript-free fine-tuning for stronger cross-lingual voice cloning.

We also construct and open-source a curated 420,000-hour multilingual corpus and a dedicated multilingual zero-shot speech synthesis benchmark.

## News

- **2026/04/30**: X-Voice codebase, model, demo, dataset, and benchmark are released.

## Installation

### Create a separate environment if needed

```bash
# Create a conda env with python_version>=3.10
conda create -n x-voice python=3.11
conda activate x-voice

# Install FFmpeg if you haven't yet
conda install ffmpeg
```

### Install PyTorch with matched device

<details>
<summary>NVIDIA GPU</summary>

> ```bash
> # Install pytorch with your CUDA version, e.g.
> pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
> ```

</details>

<details>
<summary>AMD GPU</summary>

> ```bash
> # Install pytorch with your ROCm version (Linux only), e.g.
> pip install torch==2.5.1+rocm6.2 torchaudio==2.5.1+rocm6.2 --extra-index-url https://download.pytorch.org/whl/rocm6.2
> ```

</details>

<details>
<summary>Intel GPU</summary>

> ```bash
> # Install pytorch with your XPU version, e.g.
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/test/xpu
> ```

</details>

<details>
<summary>Apple Silicon</summary>

> ```bash
> # Install the stable pytorch, e.g.
> pip install torch torchaudio
> ```

</details>

### Install X-Voice

```bash
git clone https://github.com/QingyuLiu0521/X-Voice.git
cd X-Voice
git submodule update --init src/MAVL
pip install -e .
```

## Inference

- In order to achieve desired performance, take a moment to read [detailed guidance](src/x_voice/infer).

### 1. Gradio App

Currently supported features:

- Basic multilingual zero-shot TTS
- Web-based interactive inference

```bash
# Launch a Gradio app (web interface)
x-voice_infer-gradio

# Specify the port/host
x-voice_infer-gradio --port 7860 --host 0.0.0.0
```

### 2. CLI Inference

## Training

Refer to [training & finetuning guidance](src/x_voice/train/README.md) for best practice.

## Speaking Rate Predictor

Refer to [SRP guidance](src/srp) for the multilingual speaking rate predictor used in X-Voice.

## [Evaluation](src/x_voice/eval)

## Repo Structure

```text
X-Voice/
├── ckpts/                  # checkpoints
├── data/                   # datasets and processed data
├── src/
│   ├── MAVL/               # multilingual text / phonetic processing utilities
│   ├── srp/                # speaking rate predictor
│   └── x_voice/            # main X-Voice package
└── pyproject.toml          # package definition and dependencies
```

## Development

Use pre-commit to ensure code quality:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Acknowledgements

## Citation

## License

Our code is released under MIT License. The pre-trained models are licensed under the CC-BY-NC license due to the training data Emilia, which is an in-the-wild dataset. Sorry for any inconvenience this may cause.
