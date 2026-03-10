# 修改这里的配置
# bash src/f5_tts/eval/eval_cross_lingual.sh
asr_gpu=2
task=zero_shot
dataset=lemas_eval_new # lemas_eval, cv3_eval, lemas_eval_new
ckpt=600000
exp_name=F5TTS_v1_Base_multilingual_full_catada # M3TTS_Small_multilingual_v1
test_set="en zh fr" # da el es et fi fr hr hu id it lt mt nl pl pt sk sl sv th de" #cs da el es et fi fr ko" 
ref_set="fr en zh"

concat_method=2
seed=0
nfe=16
cfg_strength=5.0
layered=False
cfg_strength2=4.0

test_langs=($test_set)
ref_langs=($ref_set)
if [ ${#test_langs[@]} -ne ${#ref_langs[@]} ]; then
    echo "[ERROR] test_set and ref_set must have the same number of languages for cross-lingual eval."
    exit 1
fi
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" # to eval/
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../../" && pwd )" # to F5-TTS/
cv3_dir="${PROJECT_ROOT}/data/${dataset}"
# decode_dir="${PROJECT_ROOT}/results/${exp_name}_${ckpt}/${dataset}/seed0_euler_nfe16_vocos_ss-1_cfg2.0_speed1.0zero_shot"

if [ "$layered" = "True" ]; then
    decode_dir="${PROJECT_ROOT}/results/${exp_name}_${ckpt}/${dataset}/seed${seed}_concat${concat_method}_euler_nfe${nfe}_vocos_ss-1_cfgI${cfg_strength}_cfgII${cfg_strength2}_speed1.0zero_shot"
    accelerate launch --main_process_port 29507 src/f5_tts/eval/eval_infer_batch.py \
        -s ${seed} -n "${exp_name}" -c ${ckpt} -t "${dataset}" -nfe ${nfe} -l "${test_set// /,}" \
        --cfg_strength ${cfg_strength} --layered --cfg_strength2 ${cfg_strength2} \
        --normalize_text --sp_type "pretrained" --concat_method ${concat_method} \
        --cross_lingual -rl "${ref_set// /,}" \
        --drop_text --decode_dir "${decode_dir}" -ns "SpeedPredict_Base" -cs 20000 # --reverse

else
    decode_dir="${PROJECT_ROOT}/results/${exp_name}_${ckpt}/${dataset}/seed${seed}_concat${concat_method}_euler_nfe${nfe}_vocos_ss-1_cfg${cfg_strength}_speed1.0zero_shot"
    # Inference
    accelerate launch --main_process_port 29508 src/f5_tts/eval/eval_infer_batch.py \
        -s ${seed} -n "${exp_name}" -c ${ckpt} -t "${dataset}" -nfe ${nfe} -l "${test_set// /,}" \
        --cfg_strength ${cfg_strength} \
        --normalize_text --sp_type "pretrained" --concat_method ${concat_method} \
        --cross_lingual -rl "${ref_set// /,}" \
        --drop_text --decode_dir "${decode_dir}" -ns "SpeedPredict_Base" -cs 20000 # --reverse 
fi


# Evaluation
apt-get install -y sox libsox-dev 
cd "$SCRIPT_DIR"
pip install -r requirements.txt
export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH


DNSMOS_LAB=utils/DNSMOS
. utils/parse_options.sh || exit 1;
dumpdir=${cv3_dir}/${task}
test_gt="false"

for ((i=0; i<${#test_langs[@]}; i++)); do
    lang=${test_langs[$i]}     # 当前目标语种
    ref_lang=${ref_langs[$i]} # 当前参考语种
    # WER
    echo "[INFO] Scoring WER for ${decode_dir}/${lang}"
    bash utils/cal_wer.sh ${dumpdir}/${lang}/text  ${decode_dir}/${ref_lang}_${lang} ${lang} ${asr_gpu} "${test_gt}"
    find ${decode_dir}/${ref_lang}_${lang}/wavs -name *.wav | awk -F '/' '{print $NF, $0}' | sed "s@\.wav @ @g" > ${decode_dir}/${ref_lang}_${lang}/wav.scp
    
    # SIM
    echo "[INFO] Scoring SIM: Comparing ${lang} output with ${ref_lang} prompt" 
    python eval_similarity.py \
    --ckpt_path ${PROJECT_ROOT}/wavlm_large_finetune.pth \
    --prompt_wavs ${dumpdir}/${ref_lang}/prompt_wav.scp \
    --hyp_wavs ${decode_dir}/${ref_lang}_${lang}/wav.scp \
    --log_file ${decode_dir}/${ref_lang}_${lang}/spk_simi_scores_cross${ref_lang}.txt \
    --devices "0" \
    --decode_dir ${decode_dir}\
    --dump_dir ${cv3_dir}

    # UTMOS
    echo "[INFO] Scoring UTSMOS  for ${decode_dir}/${ref_lang}_${lang}"  
    python eval_utmos.py --audio_dir ${decode_dir}/${ref_lang}_${lang} --ext "wav"

    # # DNSMOS
    # echo "[INFO] Scoring DNSMOS  for ${decode_dir}/${ref_lang}_${lang}"  
    # python ${DNSMOS_LAB}/dnsmos_local_wavscp.py -t ${decode_dir}/${ref_lang}_${lang}/wav.scp -e ${DNSMOS_LAB} -o ${decode_dir}/${ref_lang}_${lang}/mos.csv
    # cat ${decode_dir}/${ref_lang}_${lang}/mos.csv | sed '1d' |awk -F ',' '{ sum += $NF; count++ } END { if (count > 0) print sum / count }'  > ${decode_dir}/${ref_lang}_${lang}/dnsmos_mean.txt
done
