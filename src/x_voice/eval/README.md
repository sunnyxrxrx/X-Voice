# Evaluation
## Preparation
Install packages for evaluation:

```bash
pip install -e .[eval]
```

> [!IMPORTANT]
> For [faster-whisper](https://github.com/SYSTRAN/faster-whisper), for various compatibilities:   
> `pip install ctranslate2==4.5.0` if CUDA 12 and cuDNN 9;  
> `pip install ctranslate2==4.4.0` if CUDA 12 and cuDNN 8;  
> `pip install ctranslate2==3.24.0` if CUDA 11 and cuDNN 8.

### Prepare Test Datasets

1. *Seed-TTS testset*: Download from [seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval).
2. *LEMAS-TTS testset*: Download from [LEMAS-Dataset-eval](https://huggingface.co/datasets/LEMAS-Project/LEMAS-Dataset-eval).
3. **X-Voice testset**: Download from [X-Voice-Testset](https://huggingface.co/datasets/XRXRX/X-Voice-Testset).
4. Unzip the downloaded datasets and place them in the `./data/` directory.

### Prepare Evaluation Models

1. Chinese ASR Model: [Paraformer-zh](https://huggingface.co/funasr/paraformer-zh)
2. English ASR Model: [Faster-Whisper](https://huggingface.co/Systran/faster-whisper-large-v3)
3. WavLM Model: Download from [Google Drive](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view).

## On LEMAS / X-Voice testset
> [!NOTE]  
> ASR model will be automatically downloaded.
> Otherwise, you should update the `asr_ckpt_dir` path values in [utils/run_wer.py](https://github.com/sunnyxrxrx/X-Voice/blob/main/src/x_voice/eval/utils/run_wer.py#L96-L99).
> 
> WavLM model must be downloaded and your `wavlm_ckpt_dir` path updated in [eval_multilingual.sh](https://github.com/sunnyxrxrx/X-Voice/blob/main/src/x_voice/eval/eval_multilingual.sh#L94).

```
bash src/x_voice/eval/eval_multilingual.sh
```

### Argument Reference

| Argument | Description |
| --- | --- |
| ``num_gpus`` | Number of GPUs used for inference and WER/SIM evaluation. |
| ``dataset`` | Dataset to evaluate: ``x_voice_eval`` or ``lemas_eval``. |
| ``exp_name``, ``ckpt`` | The experiment name and checkpoint to evaluate. |
| ``drop_text`` | Whether to remove reference text input. Use **True** for stage-2 models and **False** for stage-1 models. |
| ``test_set`` | Space-separated target languages to evaluate. |
| ``ref_set`` | Space-separated reference languages. Required for cross-lingual evaluation and must match ``test_set`` in length. |
| ``sp_type`` | Duration prediction mode: ``pretrained``, ``utf``, or ``syllable``. |
| ``srp_exp_name``, ``srp_ckpt`` | Required when ``sp_type=pretrained``. |
| ``cfg_schedule``, ``cfg_decay_time`` | Options for CFG strength scheduling. |
| ``decoupled``, ``cfg_strength2`` | Options for decoupled CFG. |
| ``reverse`` | If ``True``, swap generated and reference audio positions. |

> [!NOTE]
> - Cross-lingual evaluation: provide both ``test_set`` and ``ref_set`` with equal length. Pairs are matched by position.
>   - Example: ``test_set="zh en"`` and ``ref_set="it zh"`` means *Italian->Chinese* and *Chinese->English* synthesis.
> - Intra-lingual evaluation: provide only ``test_set``.
> - If ``drop_text=True``, you can only use ``sp_type=pretrained``. Without reference text, the script cannot estimate target duration from text-length ratio, so ``utf`` and ``syllable`` are not applicable.

## On SeedTTS testset
To ensure a fair comparison, we kept the original evaluation code used in F5-TTS for SeedTTS-testset.

> [!NOTE]  
> ASR model will be automatically downloaded if `--local` not set for evaluation scripts.  
> Otherwise, you should update the `asr_ckpt_dir` path values in [eval_seedtts_testset.py](https://github.com/sunnyxrxrx/X-Voice/blob/main/src/x_voice/eval/eval_seedtts_testset.py#L45-L48).
> 
> WavLM model must be downloaded and your `wavlm_ckpt_dir` path updated in [eval_seedtts_testset.py](https://github.com/sunnyxrxrx/X-Voice/blob/main/src/x_voice/eval/eval_seedtts_testset.py#L52).

```
bash src/x_voice/eval/eval_multilingual_seedtts.sh
```
Arguments can refer to [Evaluation on LEMAS / X-Voice](#argument-reference).
