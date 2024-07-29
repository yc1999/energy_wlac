# Introduction

This code repository for [An Energy-based Model for Word-level AutoCompletion in Computer-aided Translation](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00637/119542/An-Energy-based-Model-for-Word-level).

# Project Structure

```
.
├── criterions  # Directory for custom Fairseq loss functions
├── data        # Directory for custom Fairseq data classes
├── fairseq_generate.py  # Modified file based on fairseq.generate
├── generate_ar.py       # Modified file based on fairseq.generate
├── generate.py          # Modified file based on fairseq.generate
├── __init__.py
├── interactive.py       # Modified file based on fairseq.interactive
├── models       # Directory for custom Fairseq model classes
├── modules      # Directory for custom modules
├── preprocess.py # Modified file based on fairseq.preprocess
├── README.md
├── requirements.txt
├── scripts      # Directory for various script files
├── tasks        # Directory for custom Fairseq tasks
├── train.py     # Modified file based on fairseq.train
├── validate.py  # Modified file based on fairseq.validate
```

# Local Installation Environment

```
pip install -r requirements.txt
```

**Fairseq version used**

```
https://github.com/yc1999/fairseq.git
```

This version of Fairseq is based on the official version `'1.0.0a0+c8d6fb1'`. The only modification is in the `fairseq/models/transformer/transformer_config.py` file, with no other changes.

Other packages, such as torch, should be installed separately. You can refer to the `requirements.txt` file in this directory for installation.

# Quick Start

1. Set the following three environment variables in the shell command line:

   ```
   export save_root=${YOUR_PROJECT_PATH}/save
   export project_root=${YOUR_PROJECT_PATH}
   export dataset_root=${YOUR_PROJECT_PATH}/dataset  
   ```
2. Train the model using the following command:

   ```
   # zh-en
   bash ${project_root}/scripts/rerank/zh-en/start_train.sh bi_context
   # bash ${project_root}/scripts/rerank/zh-en/start_train.sh prefix
   # bash ${project_root}/scripts/rerank/zh-en/start_train.sh suffix
   # bash ${project_root}/scripts/rerank/zh-en/start_train.sh zero_context

   # en-zh
   bash ${project_root}/scripts/rerank/en-zh/train_pretrained.sh bi_context

   # de-en
   bash ${project_root}/scripts/rerank/de-en/train_pretrained.sh bi_context

   # en-de
   bash ${project_root}/scripts/rerank/en-de/train_pretrained.sh bi_context
   ```
3. After training is completed, run the following command to infer the model:

   ```
   bash ${project_root}/scripts/rerank/zh-en/start_valid.sh
   ```

    Note: Please modify the ckpt directory in `valid.sh` as needed.
>>>>>>> ae6b684 (init)
