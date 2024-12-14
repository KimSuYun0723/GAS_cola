export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0,1

# 주요 파라미터 설정
MODEL_NAME="bert-base-cased"
OUTPUT_DIR="/home/nlpgpu8/hdd2/suyun/gas_cola/bert_cased/output_dir_4"
BATCH_SIZE=8
GRAD_ACCUM_STEPS=2
LEARNING_RATE=2e-5
EPOCHS=5
MAX_SEQ_LENGTH=128
PROJECT_NAME="bert_cola"

# 파인튜닝 실행
python bert_cola.py \
    --model_name_or_path ${MODEL_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${EPOCHS} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --project_name ${PROJECT_NAME} \
    --warmup_ratio 0.1 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 5 \
    --save_strategy steps \
    --evaluation_strategy steps \
    --load_best_model_at_end \