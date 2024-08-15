# chinese prosody prediction
import re
import os
import time
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig

mode_path = '/mnt/cfs/NLP/hub_models/Qwen1.5-7B-Chat'
lora_path = '/mnt/cfs/SPEECH/liuhuang1/workspace/llm_pp/pp_qwen1half_7b_chat_prompt_ori/qwen1half-7b-chat/v1-20240511-200922/checkpoint-410/'

config = LoraConfig.from_pretrained(lora_path)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path)

# load model
model = AutoModelForCausalLM.from_pretrained(mode_path,device_map="auto", torch_dtype = torch.float32)

# load lora weights
model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

#%%
text_start = """<|im_start|>system
You are a helpful assistant，你需要帮我完成一个韵律分类和汉字转拼音的任务，根据user提供的一句中文文本，你需要给出每个字的拼音，注意若遇到多音字，需要根据上下文的语境给出正确的>发音，并且在每个字的拼音后面标出改字对应的韵律标签，字后面标上#0，词语后面标上#1，短语后面标上#2，若某个字后面有标点符号，且该字不在最后一个字，则该字后面的韵律标签是#3，最后一个字后面标上#4.
<|im_end|>
<|im_start|>user
Question: """

text_end = """ <|im_end|>
<|im_start|>assistant"""


def label2text(text_path):
  """
    Args:
      text_path: the text file.
    Returns: the origin_text|pinyin|pure_text.
  """
  trans_lists = []

  with open(text_path, 'r', encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
      line = line.strip()   # convert the blank line to none
      if line != '':
        trans_lists.append(line)  
  
  data = []
  for i in range(0, len(trans_lists), 2):
    sent = trans_lists[i]
    sid = re.search(r'^(\d+_)*\d+', sent)
    if sid is None:
      raise ValueError(f"Check the sentence whether exists sid.\n{sent}")
    sid = sid.group()
    sent_cont = re.sub(r'^(\d+_)*\d+\s+', '', sent).strip()
    sent_cont = re.sub(r'\s+', ' ', sent_cont).strip()
    # strip end punct: "hello#4。" -> "hello#4"
    sent_cont_ = re.sub(r'#\d[,;:.?!，。？；：、！]$', r'#4', sent_cont)
    sent_cont = re.sub(r"#\d", r"", sent_cont_)

    # sid, 去除末尾标点符号, 拼音, llm的输入纯文本
    data.append([sid, sent_cont_, trans_lists[i+1], sent_cont])
  return data
  # random.shuffle(trans_lists)


def llm_pp_inf(text_mid):
    text = text_start + text_mid + text_end
    # print(text)
    # exit()
    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

    generated_ids = model.generate(input_ids=model_inputs.input_ids, max_new_tokens=512)

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True,temperature=0)[0]

    # print(response)
    return response


puncts = [",", ".", ";", "?", "，", "。", "；", "？", "："]
def result2label(pinyin_prosody, pure_text):
  """convert results to labeled text.
    Args:
      pinyin_prosody: llm results which contain pinyin and prosody, egs: "zai4 ci4 #1 jiao1 yi4 shi2 #3 jie4 shao4 ren2 #1 jian4 mou3 #2 he2 ta1 de5 #1 peng2 you5 #1 yang2 mou3 #3 da2 qi3 le5 #1 qi2 ta1 #1 suan4 pan5 #4".
      text: the original text, egs: '再次交易时，介绍人建某和她的朋友杨某打起了其他算盘'.
    Returns: the labeled text, egs: "再次#1交易时#3，介绍人#1建某#2和她的#1朋友#1杨某#3打起了#1其他算盘#4".
  """
  pinyin_prosody = re.sub(r'\s+', ' ', pinyin_prosody).strip()
  pure_text = pure_text.strip()
  pinyin_prosody = pinyin_prosody.split()
  text = [item for item in pure_text]
     
  print(text)
  new_results = []
  len_pinyin_prosody = len(pinyin_prosody)
  j = 0 # index of pinyin_prosody
  for i in range(len(text)):
    # pinyin_prosody is out of range
    if j >= len_pinyin_prosody:
      break

    c_text = text[i]
    new_results.append(c_text)
    # punct has no corresponding pinyin and prosody
    if c_text in puncts:
       continue
    
    c_pinyin_prosody = pinyin_prosody[j]
    if j+1 < len_pinyin_prosody:
      next_pinyin_prosody = pinyin_prosody[j+1]
    # print(c_text, c_pinyin_prosody, next_pinyin_prosody)
    # if current text has prosody, add it
    if next_pinyin_prosody in ["#1", "#2", "#3", "#4"]:
      new_results.append(next_pinyin_prosody)
      j += 1
    j += 1
  return ''.join(new_results)

def __cmd():
  description = "segment audio file"
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument(
      "--text_file",
      type=str,
      default='000001-010000.txt',
      required=False,
      help="the labeled text file")
  parser.add_argument(
      "--output_dir",
      type=str,
      default='data/',
      required=False,
      help="the output dir for the processed data.")
  args = parser.parse_args()

  text_file = args.text_file
  output_dir = args.output_dir
  data = label2text(text_file)
  data_inf = data[9700:]

  output_file = os.path.join(output_dir, "infer_pp_7b_prompt_ori.txt")
  with open(output_file, 'w', encoding='utf-8')as fw:
    for item in data_inf:
      sid = str(item[0])
      time1 = time.time()
      result = llm_pp_inf(item[3])
      result = re.sub(r'Answer:', r'', result)
      # strip #0
      result = re.sub(r"#0", r"", result)
      text_prosody = result2label(result, item[3])

      # strip prosody
      pinyin = re.sub(r"#\d", r"", result).strip()
      pinyin = re.sub(r'\s+', ' ', pinyin).strip()
      print(f"result: {result} \n {item[3]} \n {text_prosody}")
      
      time2 = time.time()
      consume_time = time2 - time1
      print(f"time: {consume_time} {consume_time/len(item[3])}")
      # write data
      fw.writelines(sid+ "\t" + text_prosody + "\n\t" + pinyin + '\n')
      # exit()

if __name__ == '__main__':
  __cmd()