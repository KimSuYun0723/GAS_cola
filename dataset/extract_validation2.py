from sklearn.model_selection import train_test_split
import pandas as pd

# 데이터 로드
train_data_path = "/home/nlpgpu8/hdd2/suyun/gas_cola/dataset/gas_cola_train_option2.csv"
valid_data_path = "/home/nlpgpu8/hdd2/suyun/gas_cola/dataset/gas_cola_val_option2.csv"

train_df = pd.read_csv(train_data_path)
valid_df = pd.read_csv(valid_data_path)

# 클래스별로 데이터 추출
gas_1_df = train_df[train_df['GAS'] == 1]  # GAS가 1인 데이터
gas_0_df = train_df[train_df['GAS'] != 1]  # GAS가 1이 아닌 데이터

# 각각 500개씩 샘플링 (랜덤)
gas_1_sample = gas_1_df.sample(n=500, random_state=42)
gas_0_sample = gas_0_df.sample(n=500, random_state=42)

# 두 클래스 결합 (1000개)
train_subset_df = pd.concat([gas_1_sample, gas_0_sample]).sample(frac=1, random_state=42)  # 섞기
train_subset_df.to_csv("/home/nlpgpu8/hdd2/suyun/gas_cola/dataset/gas_cola_val_option2_1000.csv", index=False)

# 3. 새로운 Train 데이터 생성 (추출한 1000개 제외)
new_train_df = train_df.drop(train_subset_df.index)  # subset에 속하지 않은 나머지 데이터
new_train_df.to_csv("/home/nlpgpu8/hdd2/suyun/gas_cola/dataset/gas_cola_train_option2_7000.csv", index=False)

print("1000개 추출 및 새로운 Train 데이터 저장 완료!")