import argparse
import wandb
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import numpy as np
from sklearn.metrics import matthews_corrcoef
import pandas as pd

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="bert-base-cased", help="Pretrained model name or path")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model and outputs")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs")
parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length")
parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio of total steps for warmup")
parser.add_argument("--eval_steps", type=int, default=500, help="Steps interval for evaluation")
parser.add_argument("--save_steps", type=int, default=500, help="Steps interval for saving model checkpoints")
parser.add_argument("--save_total_limit", type=int, default=5, help="Maximum number of checkpoints to keep")
parser.add_argument("--save_strategy", type=str, default="steps", choices=["steps", "epoch"], help="Checkpoint save strategy")
parser.add_argument("--evaluation_strategy", type=str, default="steps", choices=["steps", "epoch"], help="Evaluation strategy")
parser.add_argument("--load_best_model_at_end", action="store_true", help="Load the best model at the end of training")
parser.add_argument("--project_name", type=str, default="bert_cola", help="Wandb project name")
args = parser.parse_args()

# 데이터셋 로드
model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

train_data_path = "/home/nlpgpu8/hdd2/suyun/gas_cola/dataset/gas_cola_train_option2_7000.csv"
valid_data_path = "/home/nlpgpu8/hdd2/suyun/gas_cola/dataset/gas_cola_val_option2_1000.csv"

train_df = pd.read_csv(train_data_path)
valid_df = pd.read_csv(valid_data_path)

# 토크나이저 및 모델 설정
model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 데이터셋 전처리
def preprocess(df, label_column="original_label"):
    # 토큰화 수행
    tokenized_inputs = tokenizer(
        df["sentence"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=128
    )
    # Dataset 객체로 직접 생성
    dataset = Dataset.from_dict({
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": df[label_column].tolist()
    })
    return dataset

train_dataset = preprocess(train_df)
valid_dataset = preprocess(valid_df)

# 평가 지표
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    mcc = matthews_corrcoef(labels, predictions)  # MCC 계산
    return {"matthews_correlation": mcc}

# TrainingArguments 설정
training_args = TrainingArguments(
    output_dir=args.output_dir,
    evaluation_strategy="epoch",
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.num_train_epochs,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir=f"{args.output_dir}/logs",
    report_to="wandb",  # wandb 활성화
)

# Trainer 설정 및 실행 (test 데이터셋 사용 X)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# 모델 및 토크나이저 저장
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

# wandb 종료
wandb.finish()