#set -x

meta_lst=$1
output_dir=$2
lang=$3
num_job=$4
test_gt=$5

wav_wav_text=$output_dir/wav_res_ref_text
score_file=$output_dir/wav_res_ref_text.wer

workdir=$(cd $(dirname $0); cd ../; pwd)

if [ "$test_gt" == "true" ]; then
    target_wav_dir="$output_dir/ground_truth"
    echo "DEBUG: Testing GT Mode. Wav Dir: $target_wav_dir"
    python3 utils/get_wav_res_ref_text_gt.py "$meta_lst" "$target_wav_dir" "$wav_wav_text"
else
    target_wav_dir="$output_dir/wavs"
    echo "DEBUG: Testing Inference Mode. Wav Dir: $target_wav_dir"
    python3 utils/get_wav_res_ref_text.py "$meta_lst" "$target_wav_dir" "$wav_wav_text"
fi
# python3 prepare_ckpt.py

timestamp=$(date +%s)
thread_dir=${output_dir}/tmp/thread_metas_$timestamp/
out_dir=${thread_dir}/results/

echo "DEBUG: output_dir is: [${output_dir}]"
echo "DEBUG: lang is:       [${lang}]"

mkdir -p $out_dir

num=`wc -l $wav_wav_text | awk -F' ' '{print $1}'`
num_per_thread=`expr $num / $num_job + 1`

split -l $num_per_thread --additional-suffix=.lst -d $wav_wav_text $thread_dir/thread-


num_job_minus_1=`expr $num_job - 1`
if [ ${num_job_minus_1} -ge 0 ];then
	for rank in $(seq 0 $((num_job - 1))); do
		sub_score_file=$out_dir/thread-0$rank.wer.out
		CUDA_VISIBLE_DEVICES=${rank} python3 utils/run_wer.py $thread_dir/thread-0$rank.lst $sub_score_file $lang &
	done
fi
wait

# rm $wav_wav_text
# rm -f $out_dir/merge.out

cat $out_dir/thread-0*.wer.out >>  $out_dir/merge.out
python3 utils/average_wer.py $out_dir/merge.out $score_file
