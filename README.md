# X-Voice: Enabling Everyone to Speak 30 Languages via Zero-Shot Cross-Lingual Voice Cloning

[![python](https://img.shields.io/badge/Python-3.11-brightgreen)](https://github.com/SWivid/F5-TTS)
[![arXiv](https://img.shields.io/badge/arXiv-Coming_soon-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/unknown)
[![demo](https://img.shields.io/badge/GitHub-Demo-orange.svg)](https://sunnyxrxrx.github.io/X-Voice-Demo/)
[![hfspace](https://img.shields.io/badge/🤗-HF%20Space-yellow)](unknown)
[![dataset](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/XRXRX/X-Voice-Dataset-Train)
[![dataset](https://img.shields.io/badge/🤗-Benchmark-yellow)](https://huggingface.co/datasets/XRXRX/X-Voice-Testset)
[![lab](https://img.shields.io/badge/🏫-X--LANCE-grey?labelColor=lightgrey)](https://x-lance.sjtu.edu.cn/)
[![lab](https://img.shields.io/badge/🏫-SII-grey?labelColor=lightgrey)](https://www.sii.edu.cn/)
[![company](https://img.shields.io/badge/🏢-Geely-grey?labelColor=lightgrey)](https://www.geely.com)
[![lab](https://img.shields.io/badge/🏫-CLSP-grey?labelColor=lightgrey)](https://www.clsp.jhu.edu)
<!-- <img src="https://github.com/user-attachments/assets/12d7749c-071a-427c-81bf-b87b91def670" alt="Watermark" style="width: 40px; height: auto"> -->

**X-Voice** is a flow-matching based multilingual zero-shot voice cloning system that enables one speaker to speak 30 languages.

## News

- **2026/04/30**: X-Voice codebase, model, demo, dataset, and benchmark are released.

## Installation

### Create a separate environment if needed

```bash
# Create a conda env with python_version>=3.11
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
git clone https://github.com/sunnyxrxrx/X-Voice.git
cd X-Voice
pip install -e .
```

## Inference

- In order to achieve desired performance, take a moment to read [detailed guidance](src/x_voice/infer).

### 1. Gradio App

```bash
x-voice_infer-gradio --host 0.0.0.0 --port 7860
```

### 2. CLI Inference

```bash
# X-Voice Stage1
python -m x_voice.infer.infer_cli_stage1 -c src/x_voice/infer/examples/basic/basic_stage1.toml

# X-Voice Stage2
python -m x_voice.infer.infer_cli_stage2 -c src/x_voice/infer/examples/basic/basic_stage2.toml
```

## Training

Refer to [training guidance](src/x_voice/train/README.md) for best practice.

## Speaking Rate Predictor

Refer to [SRP guidance](src/srp) for the multilingual speaking rate predictor used in X-Voice.

## Evaluation

Refer to [evaluation guidance](src/x_voice/eval/README.md) for benchmark and metric scripts.

## Repo Structure

```text
X-Voice/
├── ckpts/                  # checkpoints
├── data/                   # datasets and processed data
├── src/
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

- [F5-TTS](https://arxiv.org/abs/2410.06885) brilliant work and the foundation of this codebase
- Cross-Lingual F5-TTS 2 for its supervised fine-tuning strategy with synthetic audio prompts
- [Cross-Lingual F5-TTS](https://arxiv.org/abs/2509.14579) for its speaking rate predictor
- [NLLB](https://huggingface.co/facebook/nllb-200-distilled-600M) for translation in the Gradio demo
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq) as ODE solver, [Vocos](https://huggingface.co/charactr/vocos-mel-24khz) and [BigVGAN](https://github.com/NVIDIA/BigVGAN) as vocoder
- [FunASR](https://github.com/modelscope/FunASR), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [UniSpeech](https://github.com/microsoft/UniSpeech), [SpeechMOS](https://github.com/tarepan/SpeechMOS) for evaluation tools
- [MAVL](https://github.com/k1064190/MAVL/tree/main) for Japanese syllable counting

## License

Our code is released under MIT License. The pre-trained models are licensed under the CC-BY-NC license due to the training data Emilia, which is an in-the-wild dataset. Sorry for any inconvenience this may cause.
