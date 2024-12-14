export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0,1

#for task in cola sst2 mrpc stsb qqp mnli qnli rte wnli; do
#    python run_glue_gas.py \
#        --model_name_or_path /home/nlpgpu8/hdd2/suyun/gas_cola/models/bert_gas_cola \
#        --task_name $task \
#        --do_eval \
#        --max_seq_length 128 \
#        --output_dir /home/nlpgpu8/hdd2/suyun/gas_cola/glue_cola/bert_gas/$task/
#done

python run_glue_gas.py \
    --model_name_or_path /home/nlpgpu8/hdd2/suyun/gas_cola/bert_gas/output_dir_4 \
    --task_name cola \
    --do_eval \
    --max_seq_length 128 \
    --output_dir /home/nlpgpu8/hdd2/suyun/gas_cola/glue_cola/bert_gas/cola/