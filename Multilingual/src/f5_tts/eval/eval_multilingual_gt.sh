# bash src/f5_tts/eval/eval_multilingual_gt.sh
# 修改下面的配置
asr_gpu=2
task=zero_shot
test_set="bg da de el en es et fi fr hu id it ja ko lt lv mt nl pl pt ro ru sk sl sv th vi zh" #cs da el es et fi fr "zh en hard_zh hard_en ja ko ..." 
dataset=mixed_eval # cv3_eval


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../../" && pwd )"
if [ -z "$PROJECT_ROOT" ]; then
    echo "[ERROR] Failed to locate PROJECT_ROOT. Check your directory structure."
    exit 1
fi
cv3_dir="${PROJECT_ROOT}/data/${dataset}"
decode_dir="${cv3_dir}/${task}"

apt-get install -y sox libsox-dev 
cd "$SCRIPT_DIR"
pip install -r requirements.txt
export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH


DNSMOS_LAB=utils/DNSMOS
. utils/parse_options.sh || exit 1;
test_gt="true"


for lang in ${test_set}; do
    # WER
    echo "[INFO] Scoring WER for ${decode_dir}/${lang}"
    bash utils/cal_wer.sh ${decode_dir}/${lang}/prompt_text  ${decode_dir}/${lang} ${lang} ${asr_gpu} "${test_gt}"
    find ${decode_dir}/${lang}/waveform -name *.wav | awk -F '/' '{print $NF, $0}' | sed "s@\.wav @ @g" > ${decode_dir}/${lang}/wav.scp
    
    # UTMOS
    echo "[INFO] Scoring UTSMOS  for ${decode_dir}/${lang}"  
    python eval_utmos.py --audio_dir ${decode_dir}/${lang} --ext "wav"

    # # OPT: DNSMOS
    # echo "[INFO] Scoring DNSMOS  for ${decode_dir}/${lang}"  
    # python ${DNSMOS_LAB}/dnsmos_local_wavscp.py -t ${decode_dir}/${lang}/wav.scp -e ${DNSMOS_LAB} -o ${decode_dir}/${lang}/mos.csv
    # cat ${decode_dir}/${lang}/mos.csv | sed '1d' |awk -F ',' '{ sum += $NF; count++ } END { if (count > 0) print sum / count }'  > ${decode_dir}/${lang}/dnsmos_mean.txt
done