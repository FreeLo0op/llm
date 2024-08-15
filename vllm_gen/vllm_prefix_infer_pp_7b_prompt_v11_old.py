# chinese prosody prediction
import re
import os
import time
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, AutoPeftModelForCausalLM
from vllm import LLM, SamplingParams

# step = 1    # 从step几开始
mode_path = '/mnt/cfs/NLP/hub_models/Qwen1.5-4B-Chat'
lora_path = '/mnt/cfs/SPEECH/liuhuang1/workspace/llm_pp/pp_qwen1half_4b_chat_prompt_v16/qwen1half-4b-chat/v0-20240604-114949/checkpoint-4040/'
merged_model_path = "/mnt/cfs/SPEECH/liuhuang1/workspace/llm_pp/merged_v16_4b_ckt4040"

exit()
################################################## step1 merge model ##################################################
if 1 >= step:
    config = LoraConfig.from_pretrained(lora_path)
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(mode_path)
    # load model
    model = AutoModelForCausalLM.from_pretrained(mode_path,device_map="auto", torch_dtype = torch.float32)
    # load lora weights
    model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

    # merge model
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_model_path, max_shard_size="2048MB", safe_serialization=True)
exit()
################################################## step2 load mergeed model ##################################################
if 2 >= step:
    # if appear token error, copy the token_file to merged path.
    llm = LLM(model=merged_model_path, 
          enable_prefix_caching=True
         )
    print(f"load merged model sucess!")

################################################## step3 model inference ##################################################
if 3 >= step:
    sampling_params = SamplingParams(stop=["<|endoftext|>","<|im_end|>","<|im_start|>"],
                                 temperature=0,
                                 max_tokens=512
                                )
    text_start = """<|im_start|>system
对提供的中文文本根据下面规则进行处理，来完成韵律标记和多音字选择的任务，文字标点符号保持不变，只选择多音字括号内的拼音。韵律标记：为句子中的每个字添加韵律标签，只有3类标签'#1,#2,#3'。规则如下：词：成词的两个或多个字的最后一个字后面标记为#1；短语：短语中的最后一个字后面标记为#2；如果字后面跟随标点符号（且该字不是句子的最后一个字），或者单句无标点符号文本并且文本过长，需要根据语意进行切分停顿，则该字的标记为#3。多音字选择：对用户提供的多音字，根据上下文选择一个最合适的拼音。
    <|im_end|>
    <|im_start|>user
    Question: """

    text_end = """ <|im_end|>
    <|im_start|>assistant"""

    text_mid = "我们一起去上学校吧！"
    text = text_start + text_mid + text_end
    text1 = text_start + "我们一起去上家里吧！" + text_end

    time1 = time.time()
    response = llm.generate(text, sampling_params)
    time2 = time.time()
    print(time2-time1, response[0].outputs[0].text.strip("Answer:"))

    time1 = time.time()
    response = llm.generate(text, sampling_params)
    time2 = time.time()
    print(time2-time1, response[0].outputs[0].text.strip("Answer:"))

    time1 = time.time()
    response = llm.generate(text, sampling_params)
    time2 = time.time()
    print(time2-time1, response[0].outputs[0].text.strip("Answer:"))

    time1 = time.time()
    response = llm.generate(text1, sampling_params)
    time2 = time.time()
    print(time2-time1, response[0].outputs[0].text.strip("Answer:"))

    time1 = time.time()
    response = llm.generate(text1, sampling_params)
    time2 = time.time()
    print(time2-time1, response[0].outputs[0].text.strip("Answer:"))
    
### vllm load model

#%%
# text_start = """<|im_start|>system
# 对提供的中文文本根据下面规则进行处理，来完成韵律标记和多音字选择的任务。韵律标记：为句子中的每个字添加韵律标签，只有4类标签'#1,#2,#3,#4'。规则如下：词：成词的两个或多个字的最后一个字后面标记为#1；短语：短语中的最后一个字后面标记为#2；如果字后面跟随标点符号（且该字不是句子的最后一个字），或者单句无标点符号文本并且文本过长，需要根据语意进行切分停顿，则该字的标记为#3。句尾：句子中的最后一个字后面标记为#4。多音字选择：对用户提供的多音字，根据上下文选择一个最合适的拼音，例如这个例子，问：“很重[zhong4|chong2]要”，答：“很#1重[zhong4]要”。
# <|im_end|>
# <|im_start|>user
# Question: """

# text_end = """ <|im_end|>
# <|im_start|>assistant"""

# text_mid = "我们一起去上学校吧！"

# text = text_start + text_mid + text_end

# model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

# generated_ids = model.generate(input_ids=model_inputs.input_ids, max_new_tokens=512)

# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]
# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True,temperature=0)[0]

# print(response)



############### vllm prefix
exit()
from vllm import LLM
from vllm import SamplingParams
from peft import AutoPeftModelForCausalLM
model = AutoPeftModelForCausalLM.from_pretrained(
    "/mnt/cfs/NLP/ljk/code/Qwen/lora_ckpts/checkpoint-7650", # path to the output directory
    device_map="auto",
    trust_remote_code=True
).eval()

# Create an LLM.
llm = LLM(model=lora_path, tokenizer_mode='auto', trust_remote_code=True)
# get prompts
prompts = ["这是一个 Prefix 功能使用的示例，因为 Prefix 的存储以物理块为单位，所以 Prompt 的长度需要至少大于等于一个物理块，这是第一句话",
           "这是一个 Prefix 功能使用的示例，因为 Prefix 的存储以物理块为单位，所以 Prompt 的长度需要至少大于等于一个物理块，这是第二句话"]
prompt_token_ids = llm.tokenizer(prompts)
# set SamplingParams
sampling_params = SamplingParams(temperature=0,
                                 max_tokens=100,
                                 )



