copy from https://yach-doc-shimo.zhiyinlou.com/docs/qoB87b4ua3C8FiHa/ <韵律预测和g2p qwen大模型>

## 1 环境安装
#### qwen及vllm相关环境安装，详细看文档中的 "1、Qwen1.5全流程最佳实践"步骤
https://yach-doc-shimo.zhiyinlou.com/docs/lmrKDeKVPkX9Xdpo/ <副本 LLM资料汇总>
#### 参考 个人安装时遇到的踩坑解决方法
详情查看README_OLD.md中的 ```基本环境准备``` 和 ```第三方库安装```步骤 
/mnt/cfs/SPEECH/liuhuang1/workspace/llm_pp/README_OLD.md

## 2 生成训练json文件
#### 将生成好的标注文件转换成llm json训练格式：
https://yach-doc-shimo.zhiyinlou.com/docs/e1Az4KG52RtzJQqW/ <根据字节tts接口生成标注文件>
#### 各个实验配置的修改，主要从prompt、训练数据格式、数据量等做实验：
https://yach-doc-shimo.zhiyinlou.com/sheets/NJkbEQ6JgJi2VvqR/MODOC/ <llm_pp实验>
#### 各版本prompt文件路径：/mnt/cfs/SPEECH/liuhuang1/workspace/llm_pp/prompt
训练数据需要变换prompt，请在代码里搜索 'sub_tmp_dict1' 变量进行修改
```bash
source /mnt/cfs/SPEECH/liuhuang1/conda_init_liuhuang1.sh
conda activate llm_pp
cd /mnt/cfs/SPEECH/liuhuang1/workspace/llm_pp
## 参数说明
# text_file：标准标注文件
# polyphone_file: 中文多音字字典
# en_polyphone_file: 英文多音字字典
# output_dir: 生成的json数据保存文件夹
# data_version: 实验版本
# 代码里手动修改 prompt: 搜索 'sub_tmp_dict1' 变量字典中value值替换。
python label2g2p_json_v19.py --text_file data5.txt --polyphone_file /mnt/cfs/SPEECH/liuhuang1/workspace/tts_frontend/merge_polyphone.txt --en_polyphone_file /mnt/cfs/SPEECH/liuhuang1/workspace/tts_frontend/en_polyphone_tal.txt --output_dir data --data_version v19
# 会在output目录下生成 "g2p_train_prompt_{data_version}.json" "g2p_dev_prompt_{data_version}.json" 
```

#### 2.1 儿化音处理逻辑：
训练数据中的儿化音中的文字都不会识别成多音字，儿化音有多音字会跳过该文字。
例如输入文本：
000219	您#1等会儿#3我给您#1问问#4。
	nin2 deng3 huir4 wo2 gei3 nin2 wen4 wen5

输出处理成大模型训练数据：
{
    "from": "user",
    "value": "您等会儿我给[gei3|ji3]您问[wen5|wen4]问[wen5|wen4]"
},
{
    "from": "assistant",
    "value": "您#1等会儿#3我给[gei3]您#1问[wen4]问[wen5]"
}

训练数据中的非儿化音的‘儿’：因为‘儿’是多音字，会识别成多音字。

## 3 llm大模型训练
训练命令保存在train_scripts文件夹下，其中例如'train_pp_qwen7b_prompt_v19.sh'脚本参数：
--model_cache_dir 选择对应的qwen模型版本
--output_dir 模型和log等保存路径
--custom_train_dataset_path 模型训练数据路径
--custom_val_dataset_path 模型测试数据路径
```bash
source /mnt/cfs/SPEECH/liuhuang1/conda_init_liuhuang1.sh
conda activate llm_pp
cd /mnt/cfs/SPEECH/liuhuang1/workspace/llm_pp/train_scripts
sh train_pp_qwen7b_prompt_v19.sh
```

## 4 llm大模型预测
#### 4.1 将模型转换成vllm格式
在"/mnt/cfs/SPEECH/liuhuang1/workspace/llm_pp/vllm_prefix_infer_pp_7b_prompt_v11.py"代码里修改对应变量，其中变量：
'mode_path'：对应的qwen模型版本
'lora_path'： 训练的checkpoint路径
'merged_model_path'： 转换后的vllm模型路径
```bash
source /mnt/cfs/SPEECH/liuhuang1/conda_init_liuhuang1.sh
conda activate llm_pp
cd /mnt/cfs/SPEECH/liuhuang1/workspace/llm_pp/
# mode_path = '/mnt/cfs/NLP/hub_models/Qwen1.5-4B-Chat'
# lora_path = '/mnt/cfs/SPEECH/liuhuang1/workspace/llm_pp/pp_qwen1half_4b_chat_prompt_v16/qwen1half-4b-chat/v0-20240604-114949/checkpoint-4040/'
# merged_model_path = "/mnt/cfs/SPEECH/liuhuang1/workspace/llm_pp/merged_v16_4b_ckt4040"
python vllm_prefix_infer_pp_7b_prompt_v11.py

# 执行完成后，还需要将相应的token配置文件copy到vllm模型路径
# 代码里已自动copy完成，无需手动copy。
# 例如 mode_path = '/mnt/cfs/NLP/hub_models/Qwen1.5-4B-Chat'，merged_model_path = "/mnt/cfs/SPEECH/liuhuang1/workspace/llm_pp/merged_v16_4b_ckt4040"
# cp /mnt/cfs/NLP/hub_models/Qwen1.5-4B-Chat/token* /mnt/cfs/SPEECH/liuhuang1/workspace/llm_pp/merged_v16_4b_ckt4040
```

#### 4.2 使用vllm模型预测
在 “/mnt/cfs/SPEECH/liuhuang1/workspace/llm_pp/infer_pp_7b_prompt_v19_vllm_from_labeled.py”代码里修改vllm模型路径和prompt，在代码里搜索以下变量：
'merged_model_path': vllm模型路径，在步骤4.1中所生成的。
'text_start'： 将其换成对应的prompt。
* inference预测，输入的是标准标注文本
```bash
source /mnt/cfs/SPEECH/liuhuang1/conda_init_liuhuang1.sh
conda activate llm_pp
cd /mnt/cfs/SPEECH/liuhuang1/workspace/llm_pp/
## 参数说明
# text_file：标准标注文件
# polyphone_file: 中文多音字字典
# en_polyphone_file: 英文多音字字典
# output_dir: 生成的json数据保存文件夹
# filename: 结果文件
python infer_pp_7b_prompt_v19_vllm_from_labeled.py --text_file 000001-010000.txt --polyphone_file /mnt/cfs/SPEECH/liuhuang1/workspace/tts_frontend/merge_polyphone.txt --en_polyphone_file /mnt/cfs/SPEECH/liuhuang1/workspace/tts_frontend/en_polyphone_tal.txt --output_dir data --filename infer_pp_7b_prompt_v19_ckt1330_labeled_all.txt
# 执行完成后，会在output_dir目录下生成filename文件
```

* inference预测，输入的是纯文本
```bash
source /mnt/cfs/SPEECH/liuhuang1/conda_init_liuhuang1.sh
conda activate llm_pp
cd /mnt/cfs/SPEECH/liuhuang1/workspace/llm_pp/
## 参数说明
# text_file：纯文本文件
# polyphone_file: 中文多音字字典
# en_polyphone_file: 英文多音字字典
# output_dir: 生成的json数据保存文件夹
# filename: 结果文件
python infer_pp_7b_prompt_v19_vllm_from_text.py --text_file 000001-010000-text.txt --polyphone_file /mnt/cfs/SPEECH/liuhuang1/workspace/tts_frontend/merge_polyphone.txt --en_polyphone_file /mnt/cfs/SPEECH/liuhuang1/workspace/tts_frontend/en_polyphone_tal.txt --output_dir data --filename infer_pp_7b_prompt_v19_ckt1330_text_all.txt
# 执行完成后，会在output_dir目录下生成filename文件
```

* 部分实验(answer使用特殊符号代替的实验)，需要根据question answer恢复成正确结果
例如：
```bash
question: '广[guang3|an1]东潮州芦[lu2|lu5]山大[da4|dai4|da5]宗[zong5|zong1]后陇宗[zong5|zong1]支'
answer: '#1%#&#1%#1#2$##2#

# 如果question和answer个数不一致，会导致恢复失败
恢复后的结果：'广[guang3]东#1潮州#3芦[lu2]山大[da4]宗[zong1]后#2陇宗[zong1]支'
```

恢复的代码部分在：
```bash
# 如v20实验是answer使用特殊符号代替的实验，恢复的代码在：
/mnt/cfs/SPEECH/liuhuang1/workspace/llm_pp/label2g2p_json_v20.py 的 recover_prosody_pinyin 函数
# 会根据question、模型预测的answer和多音字字典，恢复成正确结果。
```

## 5 统计指标
#### 5.1 先用开源的lstm韵律预测模型对test文本进行韵律预测
```bash
source /mnt/cfs/SPEECH/liuhuang1/conda_init_liuhuang1.sh
conda activate llm_pp
cd /mnt/cfs/SPEECH/liuhuang1/workspace/tts_frontend/Prosody_Prediction
# (1) 该工具会先用分词工具进行文本切分，然后调用韵律模型预测， 预测文本会保存在相同目录下的lstm_pp_dev.txt
python lstm_prosody_predict.py --text_file /mnt/cfs/SPEECH/liuhuang1/workspace/llm_pp/data1/truth_prosody_dev.txt
```

#### 5.2 根据标准标注文件生成需要的-对比truth文本
输入标准标注文件，输出对比truth文本
```bash
source /mnt/cfs/SPEECH/liuhuang1/conda_init_liuhuang1.sh
conda activate llm_pp
cd /mnt/cfs/SPEECH/liuhuang1/workspace/llm_pp/
## 参数说明

# polyphone_file: 中文多音字字典
# en_polyphone_file: 英文多音字字典
# output_dir: 生成的json数据保存文件夹
# data_version: truth文件版本
python label2truth.py --polyphone_file /mnt/cfs/SPEECH/liuhuang1/workspace/tts_frontend/merge_polyphone.txt --en_polyphone_file /mnt/cfs/SPEECH/liuhuang1/workspace/tts_frontend/en_polyphone_tal.txt --output_dir data --data_version v18
# 执行完成后，会在output_dir目录下生成 "truth_dev_" + data_version + ".txt"文件
```

#### 5.3 统计多音字准确率、韵律F1值
根据步骤4.2生成的大模型结果、5.1生成的lstm韵律结果和5.2生成的对比truth文件，统计相应指标
```bash
source /mnt/cfs/SPEECH/liuhuang1/conda_init_liuhuang1.sh
conda activate llm_pp
cd /mnt/cfs/SPEECH/liuhuang1/workspace/llm_pp/
# 统计多音字准确率
# 修改其中 真实标注文件 和 预测文件
python polyphone_metrics.py --truth_text_file the_truth_5.2_file --predict_text_file llm_4.2_or_lstm_5.1_file

# 统计韵律预测F1值等
# 修改其中 真实标注文件 和 预测文件
python prosody_metrics.py --truth_text_file the_truth_5.2_file --predict_text_file llm_4.2_or_lstm_5.1_file
```
