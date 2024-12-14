export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0,1

python run_glue.py \
    --model_name_or_path /home/nlpgpu8/hdd2/suyun/gas_cola/bert_cased/output_dir_4 \
    --task_name cola \
    --do_eval \
    --max_seq_length 128 \
    --output_dir /home/nlpgpu8/hdd2/suyun/gas_cola/glue_cola/bert/cola/