# 修改这里的配置
# bash src/f5_tts/eval/eval_multilingual_infer_only.sh
asr_gpu=1
task=zero_shot
dataset=lemas_eval # cv3_eval
ckpt=390000
exp_name=F5TTS_v1_Base_multilingual_tkncat500m # M3TTS_Small_multilingual_v1
test_set="th" # fr de vi es" #zh en da el es et fi fr hr hu id it lt mt nl pl pt sk sl sv th de" #cs da el es et fi fr ko" 


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" # to eval/
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../../" && pwd )" # to F5-TTS/
cv3_dir="${PROJECT_ROOT}/data/${dataset}"
decode_dir="${PROJECT_ROOT}/results/${exp_name}_${ckpt}/${dataset}/seed0_euler_nfe16_vocos_ss-1_cfg2.0_speed1.0zero_shot"

# Inference
accelerate launch src/f5_tts/eval/eval_infer_batch.py -s 0 -n ${exp_name} -c ${ckpt} -t "${dataset}" -nfe 16 -l "${test_set// /,}"   --force_rescan 
