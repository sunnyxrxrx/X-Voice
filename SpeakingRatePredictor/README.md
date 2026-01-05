# Speaking Rate Predictor
Speaking Rate Predictor for [Cross-Lingual F5-TTS](https://arxiv.org/abs/2509.14579)

## Installation
This module is built upon the **F5-TTS** environment. 

1. Ensure you have the F5-TTS environment set up (refer to the main repository [F5-TTS repo](https://github.com/SWivid/F5-TTS/)).
2. Install the additional dependency:

```bash
pip install pyphen
```

## Data Augmentation: Silence Injection
During training, silence is randomly inserted into audio samples to improve generalization. The target speed label remains unchanged.

- **Silence Duration**: 30% ~ 70% of the original sample length.
- **Mode Probabilities**: (0 ≤ $p$% ≤ 1)
  - `None`($1-p\%$): No silence added.
  - `Front`($\frac{p}{3}\%$): Silence added to the start.
  - `Back`($\frac{p}{3}\%$): Silence added to the end.
  - `Both`($\frac{p}{3}\%$): Silence added to both start and end.

Here is the revised version of your text in English:

---

## Using the Speaking Rate Predictor to Evaluate F5-TTS

The script for predicting duration using the Speaking Rate Predictor to assist F5-TTS inference has been uploaded to my forked repository: [QingyuLiu0521/F5-TTSsrc/f5_tts/eval/eval_infer_batch_droptext_sp.py](https://github.com/QingyuLiu0521/F5-TTS/blob/c11cb40706d90f713dc93b297cc72f8d73edfa16/src/f5_tts/eval/eval_infer_batch_droptext_sp.py).

### Example Usage:
```bash
accelerate launch src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_v1_Base" -c 1250000 -t "ls_pc_test_clean" -nfe 32 -ns "SpeedPredict_Base" -cs 20000 --local
```