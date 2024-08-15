# chinese prosody prediction
import re
import os
import time
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig

mode_path = '/mnt/cfs/NLP/hub_models/Qwen1.5-4B-Chat'
lora_path = '/mnt/cfs/SPEECH/liuhuang1/workspace/llm_pp/pp_qwen1half_4b_chat_prompt_v3/qwen1half-4b-chat/v0-20240511-073647/checkpoint-410'

config = LoraConfig.from_pretrained(lora_path)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path)

# load model
model = AutoModelForCausalLM.from_pretrained(mode_path,device_map="auto", torch_dtype = torch.float32)

# load lora weights
model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

#%%
text_start = """<|im_start|>system
对提供的中文文本根据下面规则进行处理，来完成韵律标记和汉字转拼音的任务。拼音转换：将用户提供的中文句子转换为拼音。对于多音字，根据上下文确定正确的发音。规则如下：拼音由声母、韵母和声调组成。声母有'b,c,ch,d,f,g,h,j,k,l,m,n,p,q,r,s,sh,t,w,x,y,z,zh'；韵母有'a,ai,an,ang,ao,e,ei,en,eng,er,i,ia,ian,iang,iao,ie,in,ing,iong,iu,o,ong,ou,u,ua,uai,uan,uang,ui,un,uo,v,ve,vn'；声调有'1,2,3,4,5'。韵律标记：为句子中的每个字添加韵律标签，只有5类标签'#0,#1,#2,#3,#4,#5'。规则如下：字：每个单独的字后面标记为#0。词：成词的两个或多个字的最后一个字后面标记为#1。短语：短语中的最后一个字后面标记为#2。标点符号：如果字后面跟随标点符号（且该字不是句子的最后一个字），则该字的标记为#3。句尾：句子中的最后一个字后面标记为#4。
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

  output_file = os.path.join(output_dir, "infer_pp_4b_prompt_v3.txt")
  with open(output_file, 'w', encoding='utf-8')as fw:
    i = 9701
    for item in data_inf:
      sid = str(i).zfill(6)
      time1 = time.time()
      result = llm_pp_inf(item[3])
      result = result.split("拼音：")
      text_prosody = result[0].strip("Answer:韵律标记：")
      pinyin = result[1].strip()
      # print(f"result: {result}")
      time2 = time.time()
      consume_time = time2 - time1
      print(f"time: {consume_time} {consume_time/len(item[3])}")
      # write data
      fw.writelines(sid+ "\t" + text_prosody + "\n\t" + pinyin + '\n')
      # exit()
      i += 1

if __name__ == '__main__':
  __cmd()
