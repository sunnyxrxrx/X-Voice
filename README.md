# X-Voice: Enabling Everyone to Speak 30 Languages via Zero-Shot Cross-Lingual Voice Cloning

<a href="https://arxiv.org/abs/unknown" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Paper-Coming%20Soon-b31b1b.svg?logo=arXiv&style=for-the-badge" alt="Paper"></a>
<a href="https://sunnyxrxrx.github.io/X-Voice-Demo/" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Demo-Samples-orange.svg?logo=github&style=for-the-badge" alt="Demo"></a>
<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
<a href="https://huggingface.co/spaces/chenxie95/X-Voice" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Interactive%20Demo-HF%20Space-yellow?labelColor=grey&logo=huggingface&style=for-the-badge" alt="HF Space"></a>
<a href="https://huggingface.co/datasets/XRXRX/X-Voice-Dataset-Train" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Dataset-Train%20Set-yellow?labelColor=grey&logo=huggingface&style=for-the-badge" alt="HF Dataset"></a>
<a href="https://huggingface.co/datasets/XRXRX/X-Voice-Testset" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Benchmark-Test%20Set-lightgrey?labelColor=grey&logo=huggingface&style=for-the-badge" alt="HF Benchmark"></a>
<a href="https://modelscope.cn/datasets/sunnyxrxrx/X-Voice-Dataset-Train" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Dataset-Train%20Set-blue?logo=alibabacloud&style=for-the-badge" alt="ModelScope Dataset"></a>
<a href="https://x-lance.sjtu.edu.cn/" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/X--LANCE-grey?labelColor=lightgrey&logo=leanpub&style=for-the-badge" alt="X-LANCE"></a>
<a href="https://www.sii.edu.cn/" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/SII-grey?labelColor=lightgrey&logo=leanpub&style=for-the-badge" alt="SII"></a>
<a href="https://www.geely.com" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Geely-grey?labelColor=lightgrey&logo=accenture&style=for-the-badge" alt="Geely"></a>
<a href="https://www.clsp.jhu.edu" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/CLSP-grey?labelColor=lightgrey&logo=leanpub&style=for-the-badge" alt="CLSP"></a>
<!-- <img src="https://github.com/user-attachments/assets/12d7749c-071a-427c-81bf-b87b91def670" alt="Watermark" style="width: 40px; height: auto"> -->

**X-Voice** is a flow-matching-based multilingual zero-shot voice cloning system that enables one speaker to speak 30 languages.

## News

- **2026/04/30**: X-Voice <a href="https://github.com/sunnyxrxrx/X-Voice" target="_blank" rel="noopener noreferrer">codebase</a>, <a href="https://huggingface.co/XRXRX/X-Voice" target="_blank" rel="noopener noreferrer">model</a>, <a href="https://sunnyxrxrx.github.io/X-Voice-Demo/" target="_blank" rel="noopener noreferrer">demo</a>, <a href="https://huggingface.co/spaces/chenxie95/X-Voice" target="_blank" rel="noopener noreferrer">Hugging Face Space</a>, <a href="https://huggingface.co/datasets/XRXRX/X-Voice-Dataset-Train" target="_blank" rel="noopener noreferrer">dataset</a>, and <a href="https://huggingface.co/datasets/XRXRX/X-Voice-Testset" target="_blank" rel="noopener noreferrer">benchmark</a> are released.

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
git clone https://github.com/sunnyxrxrx/X-Voice.git
cd X-Voice
pip install -e .
```

Check your ESpeak-ng installation:

```bash
espeak-ng --version
```

If not found, run `src/x_voice/prepare_ipa.sh` first.

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

### TTS Model Training

Refer to [training guidance](src/x_voice/train/README.md) for best practice.

### Speaking Rate Predictor Training

Refer to [speaking rate predictor guidance](src/rate_pred) for the multilingual speaking rate predictor used in X-Voice.

## Evaluation

Refer to [evaluation guidance](src/x_voice/eval/README.md) for benchmark and metric scripts.

## Repo Structure

```text
X-Voice/
├── ckpts/                  # checkpoints
├── data/                   # datasets and processed data
├── src/
│   ├── rate_pred/          # speaking rate predictor
│   ├── third_party/
│   │   └── BigVGAN/        # BigVGAN submodule
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

Our code is released under MIT License. The pre-trained models are licensed under the CC-BY-NC license due to the training data X-Voice Dataset.
