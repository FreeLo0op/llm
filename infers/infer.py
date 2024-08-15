import re
import os
import time
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
from vllm import LLM, SamplingParams


merged_model_path = "/mnt/cfs/SPEECH/hupeng/github/llm_pp/vllm_models/g2ppp_qwen1half_7b_chat_prompt_v1_ckt310"

llm = LLM(model=merged_model_path, 
          enable_prefix_caching=True
         )

print(f"load merged model sucess!")

sampling_params = SamplingParams(
    stop=["<|endoftext|>","<|im_end|>","<|im_start|>"],
    temperature=0,
    max_tokens=512)

text_start = """<|im_start|>system
对提供的中文文本根据下面规则进行处理，来完成韵律标记和音节任务，文字标点符号和顺序保持不变。韵律标记：为句子中的每个字添加韵律标签，只有3类标签'#1,#2,#3'。规则如下：词：成词的两个或多个字的最后一个字后面标记为#1；短语：短语中的最后一个字后面标记为#2；如果字后面跟随标点符号（且该字不是句子的最后一个字），或者单句无标点符号文本并且文本过长，需要根据语意进行切分停顿，则该字的标记为#3。音节标记：给用户提供的文本中的中文标注音节，注意有的字是多音字，需要给出正确的读音’。例如：question1:‘卡尔普陪外孙玩滑梯’，answer1:‘ka2 er2 pu3 #2 pei2 #1 uai4 suen1 #1 uan2 hua2 ti1 #3。’；qeusion2:‘而部分网民也乐于围观起哄’，answer2:‘er2 #2 bu4 fen5 #1 uang3 min2 #2 ie3 #1 le4 v2 #1 uei2 guan1 #1 qi3 hong4 #3’；question3:‘舞池璀璨乐声欢快光影交错’，answer3:‘u3 chiii2 #1 cuei3 can4 #3 ve4 sheng1 #1 huan1 kuai4 #3 guang1 ing3 #1 jiao1 cuo4 #3’
<|im_end|>
<|im_start|>user
"""

text_end = """<|im_end|>
<|im_start|>assistant"""

def llm_pp_inf(text_mid):
    text = text_start + text_mid + text_end
    response = llm.generate(text, sampling_params)
    print(response)
    #result = response[0].outputs[0].text
    #text_prosody = result.strip("Answer: ")
    #print(text_prosody)

if __name__ == '__main__':
    text = '难兄难弟要好好切磋切磋'
    llm_pp_inf(text)