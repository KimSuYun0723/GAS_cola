---
library_name: transformers
language:
- en
base_model: /home/nlpgpu8/hdd2/suyun/gas_cola/bert_gas/output_dir_4
tags:
- generated_from_trainer
datasets:
- glue
model-index:
- name: cola
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# cola

This model is a fine-tuned version of [/home/nlpgpu8/hdd2/suyun/gas_cola/bert_gas/output_dir_4](https://huggingface.co//home/nlpgpu8/hdd2/suyun/gas_cola/bert_gas/output_dir_4) on the GLUE COLA dataset.
It achieves the following results on the evaluation set:
- eval_loss: 0.1401
- eval_MCC: 0.5664
- eval_runtime: 7.4338
- eval_samples_per_second: 140.304
- eval_steps_per_second: 8.878
- step: 0

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 3.0

### Framework versions

- Transformers 4.48.0.dev0
- Pytorch 2.5.1+cu124
- Datasets 3.1.0
- Tokenizers 0.21.0
