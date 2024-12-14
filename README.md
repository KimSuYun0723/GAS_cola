# GAS_CoLA
This is a repository for experiments of GAS(Gradient Grammatical Acceptability Score) to improve CoLA benchmark.

## File Tree
```
📦 GAS_cola     
 ┣ 📂 bert_cased             // CoLA로 bert-base-cased tuing 폴더     
 ┃ ┣ 📜 bert_cola.py         // original CoLA로 fine-tuning한 학습 코드     
 ┃ ┣ 📜 bert.sh              // bert_cola.py 실행을 위한 Shell Script     
 ┃     
 ┣ 📂 bert_gas               // GAS-CoLA로 bert-base-cased tuning 폴더     
 ┃ ┣ 📜 bert_gas.py          // GAS-CoLA로 fine-tuning한 학습 코드     
 ┃     
 ┣ 📂 dataset                // 학습 및 검증 데이터셋과 데이터 추출 스크립트     
 ┃ ┣ 📜 extract_validation2.py             // 검증 데이터를 추출하는 스크립트     
 ┃ ┣ 📜 gas_cola_train_option2_7000.csv    // 학습 데이터셋 (옵션 2, train 7000 샘플)     
 ┃ ┣ 📜 gas_cola_train_option2.csv         // 학습 데이터셋 (옵션 2, train 전체)     
 ┃ ┣ 📜 gas_cola_val_option2_1000.csv      // 검증 데이터셋 (옵션 2, trian_subset 1000개)     
 ┃ ┣ 📜 gas_cola_val_option2.csv           // 테스트 데이터셋 (옵션 2, original validation 전체 )     
 ┃     
 ┣ 📂 glue_cola               // GLUE 중 CoLA 관련 코드 및 결과 폴더     
 ┃ ┣ 📂 bert/cola             // bert_cased(baseline) 결과     
 ┃ ┃ ┣ 📜 all_results.json    // 평가 결과     
 ┃ ┃ ┣ 📜 eval_results.json        
 ┃ ┃ ┣ 📜 README.md           // bert/cola 결과 설명 문서     
 ┃ ┃     
 ┃ ┣ 📂 bert_gas/cola         // bert_cased(GAS labeled) 결과     
 ┃ ┃ ┣ 📜 all_results.json    // 평가 결과     
 ┃ ┃ ┣ 📜 eval_results.json       
 ┃ ┃ ┣ 📜 README.md           // bert_gas/cola 결과 설명 문서     
 ┃ ┃
 ┃ ┣ 📜 glue_bert_cola.sh     // Baseline 평가 실행 Shell Script     
 ┃ ┣ 📜 glue_bert_gas.sh      // GAS labeled CoLA로 학습된 모델 평가 실행 Shell Script     
 ┃ ┣ 📜 run_glue_gas.py       // GAS labeled CoLA로 학습된 모델 평가     
 ┃ ┣ 📜 run_glue.py           // Baseline 평가     
 ┣ 📜 .gitignore                 
 ┣ 📜 README.md
``` 
