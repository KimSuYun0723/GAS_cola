import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
from datasets import Dataset
import torch

# 데이터 로드
train_data_path = "/home/nlpgpu8/hdd2/suyun/gas_cola/dataset/gas_cola_train_option2_7000.csv"
valid_data_path = "/home/nlpgpu8/hdd2/suyun/gas_cola/dataset/gas_cola_val_option2_1000.csv"

train_df = pd.read_csv(train_data_path)
valid_df = pd.read_csv(valid_data_path)

# 토크나이저 및 모델 설정
model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 데이터셋 전처리
def preprocess(df, label_column="GAS"):
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

# 모델: 회귀로 설정 (num_labels=1)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

# 평가 지표
def compute_metrics(eval_pred, threshold=0.5):
    predictions, labels = eval_pred
    predictions = predictions.squeeze(-1)

    # Threshold 적용 (회귀 → 이진 분류)
    binary_preds = (predictions >= threshold).astype(int)
    binary_labels = (labels >= 0.5).astype(int)

    # 평가 지표
    mcc = matthews_corrcoef(binary_labels, binary_preds)
    acc = accuracy_score(binary_labels, binary_preds)
    f1 = f1_score(binary_labels, binary_preds)
    return {"MCC": mcc, "Accuracy": acc, "F1": f1}

# TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="/home/nlpgpu8/hdd2/suyun/gas_cola/bert_gas/output_dir_4",
    evaluation_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=5,
    weight_decay=0.01,
    save_strategy="steps",
    load_best_model_at_end=True,
    report_to="wandb",
    eval_steps=50,
    save_steps=50,
    save_total_limit=5,
    logging_dir="/home/nlpgpu8/hdd2/suyun/gas_cola/bert_gas/logs",
)

# Trainer 설정
trainer_gas = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=lambda eval_pred: compute_metrics(eval_pred, threshold=0.5)
)

# 학습 수행
trainer_gas.train()

model.save_pretrained("/home/nlpgpu8/hdd2/suyun/gas_cola/bert_gas/output_dir_4")
tokenizer.save_pretrained("/home/nlpgpu8/hdd2/suyun/gas_cola/bert_gas/output_dir_4")

# 최적 Threshold 찾기
def find_best_threshold(trainer, valid_dataset, thresholds):
    predictions = trainer.predict(valid_dataset).predictions.squeeze(-1)
    labels = np.array(valid_dataset['labels'])

    best_mcc = -1
    best_threshold = 0.5
    for t in thresholds:
        binary_preds = (predictions >= t).astype(int)
        mcc = matthews_corrcoef((labels >= 0.5).astype(int), binary_preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = t
    return best_threshold, best_mcc

# 최적 Threshold 계산
thresholds = np.linspace(0.3, 0.7, 50)  # 예: 0.3부터 0.7까지 50개 점검
best_threshold, best_mcc = find_best_threshold(trainer_gas, valid_dataset, thresholds)
print(f"Best Threshold: {best_threshold:.2f}, Best MCC: {best_mcc:.4f}")