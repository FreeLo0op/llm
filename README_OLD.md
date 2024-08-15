## 基本环境准备
参考安装链接：https://github.com/modelscope/swift/blob/main/docs/source/LLM/Qwen1.5%E5%85%A8%E6%B5%81%E7%A8%8B%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5.md

或者直接以下命令安装：
```bash
pip install 'ms-swift[llm]' -U

# autoawq和cuda版本有对应关系，请按照`https://github.com/casper-hansen/AutoAWQ`选择版本
# 因为先安装的'ms-swift[llm]'cud是12.1，autoawq官网说可以直接用下面命令安装
pip install autoawq
# vllm与cuda版本有对应关系，请按照`https://docs.vllm.ai/en/latest/getting_started/installation.html`选择版本
# 因为先安装的'ms-swift[llm]'cud是12.1，autoawq官网说可以直接用下面命令安装

""" 一直卡在 Downloading vllm_nccl_cu12-2.18.1.0.4.0.tar.gz (6.2 kB)
参考 https://github.com/vllm-project/vllm/issues/4360 : You can download it from https://github.com/vllm-project/vllm-nccl/releases/tag/v0.1.0. Then place it in '/home/username/.config/vllm/nccl/cu12' and rename it as "libnccl.so.2.18.1" .
然后执行pip install vllm 命令。
"""
pip install vllm

pip install openai
```

## 第三方库安装
参考安装链接（modelscope 和 transformers不用重复安装）：https://github.com/datawhalechina/self-llm/blob/master/Qwen1.5/04-Qwen1.5-7B-chat%20Lora%20%E5%BE%AE%E8%B0%83.md

或者直接以下命令安装：
```bash
# python -m pip install --upgrade pip
# 更换 pypi 源加速库的安装
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install streamlit==1.24.0
pip install sentencepiece==0.1.99
# 0.24.1 会出现 TypeError: __init__() got an unexpected keyword argument 'use_seedable_sampler'
pip install accelerate==0.27.2
pip install transformers_stream_generator==0.0.4
pip install datasets==2.18.0
pip install peft==0.10.0

# 需要 cuda环境变量 添加到 .bashrc 或者 .zshrc
export PATH=/mnt/cfs/SPEECH/zhangxinke1/tools/cuda/cuda-12.1/bin/:$PATH
export LD_LIBRARY_PATH=/mnt/cfs/SPEECH/zhangxinke1/tools/cuda/cuda-12.1/lib64/:$LD_LIBRARY_PATH

# 可能清华源没资源？？手动下载安装
# https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
MAX_JOBS=8 pip install flash-attn --no-build-isolation
# MAX_JOBS=8 pip install --use-pep517 flash-attn --no-build-isolation
```

## 生成llm训练数据json格式
1. 根据 ```/mnt/cfs/SPEECH/liuhuang1/workspace/tts_frontend/tts-frontend-dataset/README_lh.md``` 生成火山标注文件
2. 将步骤1生成的文件复制到本项目目录下，运行：
```bash
# 会在output目录下生成 "g2p_train_prompt_{data_version}.json" "g2p_dev_prompt_{data_version}.json" 
python label2g2p_json_v10.py --text_file data2.txt --output_dir data/ --data_version v10
```

## 运行train命令
训练命令保存在train_scripts文件夹下，其中参数：
--model_cache_dir 选择对应的qwen模型版本
--output_dir 模型和log等保存路径
--custom_train_dataset_path 模型训练数据路径
--custom_val_dataset_path 模型测试数据路径
```bash
conda activate llm_pp
cd train_scripts
# qwen1.5-7b prompt_v2(change prompt and answer style)
sh train_pp_qwen7b_prompt_v2.sh
```

## 统计指标
vscode 正则表达式查找中文： #3[\u4e00-\u9fa5]
1. 先用开源的lstm韵律预测模型对test文本进行韵律预测。
```bash
conda activate lstm_pp
cd /mnt/cfs/SPEECH/liuhuang1/workspace/tts_frontend/Prosody_Prediction
# (1) 该工具会先用分词工具进行文本切分，然后调用韵律模型预测， 预测文本会保存在相同目录下的lstm_pp_dev.txt
python lstm_prosody_predict.py --text_file /mnt/cfs/SPEECH/liuhuang1/workspace/llm_pp/data/truth_prosody_dev.txt
```

2. 用llm模型进行韵律和多音字预测

```bash
# 修改其中模型路径，非传参手动代码文件里修改 lora_path 
# 固定选择标贝后三百句用来预测，需要选择其他句子，修改这样代码 data_inf = data[9700:]
python infer_pp_7b_prompt_v10.py --text_file 000001-010000.txt --output_dir data/ --filename infer_pp_7b_prompt_v10_ckt220_tt.txt
```

3. 统计多音字准确率
```bash
# 修改其中 真实标注文件 和 预测文件
python polyphone_metrics.py
```

4. 统计韵律预测F1值等
```bash
# 修改其中 真实标注文件 和 预测文件
python prosody_metrics.py
```

## vllm prefix模型加速inference
```bash
# 修改其中 真实标注文件 和 预测文件
python vllm_prefix_infer_pp_7b_prompt_v5.py
```