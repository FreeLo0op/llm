# chinese prosody prediction
import re
import os
import time
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig

mode_path = '/mnt/cfs/NLP/hub_models/Qwen1.5-7B-Chat'
lora_path = '/mnt/cfs/SPEECH/liuhuang1/workspace/llm_pp/pp_qwen1half_7b_chat_prompt_v9_data2/qwen1half-7b-chat/v0-20240523-122944/checkpoint-40/'

config = LoraConfig.from_pretrained(lora_path)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path)

# load model
model = AutoModelForCausalLM.from_pretrained(mode_path,device_map="auto", torch_dtype = torch.float32)

# load lora weights
model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

#%%
text_start = """<|im_start|>system
对提供的中文文本根据下面规则进行处理，来完成韵律标记和多音字选择的任务，文字标点符号保持不变，只选择多音字括号内的拼音。韵律标记：为句子中的每个字添加韵律标签，只有3类标签'#1,#2,#3'。规则如下：词：成词的两个或多个字的最后一个字后面标记为#1；短语：短语中的最后一个字后面标记为#2；如果字后面跟随标点符号（且该字不是句子的最后一个字），或者单句无标点符号文本并且文本过长，需要根据语意进行切分停顿，则该字的标记为#3。多音字选择：对用户提供的多音字，根据上下文选择一个最合适的拼音，例如这个例子，问：“这时，楼房晃[huang3|huang4]得[de2|de5|dei3]更[geng4|geng1]加厉害，玻璃窗砸落[luo4|la4|lao4]下来”，答：“这时#3，楼房#1晃[huang4]得[de5]#1更[geng4]加#1厉害#3，玻璃窗#2砸落[luo4]#1下来”。
<|im_end|>
<|im_start|>user
Question: """

text_end = """ <|im_end|>
<|im_start|>assistant"""


prosody_level = ['#0', '#1', '#2', '#3', '#4']

def is_rhotic_accent(pinyin):
  erhua_list = ['ar', 'air', 'anr', 'angr', 
  'aor', 'bar', 'bair', 'banr', 'bangr', 'baor', 'beir', 'benr', 'bengr',
  'bir', 'bianr', 'biaor', 'bier', 'binr', 'bingr', 'bor', 'bur', 'car', 
  'cair', 'canr', 'cangr', 'caor', 'cer', 'cenr', 'cengr', 'char', 'chair',
  'chanr', 'changr', 'chaor', 'cher', 'chenr', 'chengr', 'chir', 'chongr',
  'chour', 'chur', 'chuar', 'chuair', 'chuanr', 'chuangr', 'chuir', 'chunr',
  'chuor', 'cir', 'congr', 'cour', 'cur', 'cuanr', 'cuir', 'cunr', 'cuor', 
  'dar', 'dair', 'danr', 'dangr', 'daor', 'der', 'deir', 'denr', 'dengr',
  'dir', 'diar', 'dianr', 'diaor', 'dier', 'dingr', 'diur', 'dongr', 'dour',
  'dur', 'duanr', 'duir', 'dunr', 'duor', 'eir', 'enr', 'engr', 'err', 'far',
  'fair', 'fanr', 'fangr', 'feir', 'fenr', 'fengr', 'fiaor', 'for', 'four', 
  'fur', 'gar', 'gair', 'ganr', 'gangr', 'gaor', 'ger', 'geir', 'genr', 
  'gengr', 'gongr', 'gour', 'gur', 'guar', 'guair', 'guanr', 'guangr',
  'guir', 'gunr', 'guor', 'har', 'hair', 'hanr', 'hangr', 'haor', 'her', 
  'heir', 'henr', 'hengr', 'hongr', 'hour', 'hur', 'huar', 'huair', 'huanr',
  'huangr', 'huir', 'hunr', 'huor', 'jir', 'jiar', 'jianr', 'jiangr', 'jiaor',
  'jier', 'jinr', 'jingr', 'jiongr', 'jiur', 'jur', 'juanr', 'juer', 'junr', 
  'kar', 'kair', 'kanr', 'kangr', 'kaor', 'ker', 'keir', 'kenr', 'kengr',
  'kiur', 'kongr', 'kour', 'kur', 'kuar', 'kuair', 'kuanr', 'kuangr', 'kuir',
  'kunr', 'kuor', 'lar', 'lair', 'lanr', 'langr', 'laor', 'ler', 'leir', 
  'lengr', 'lir', 'liar', 'lianr', 'liangr', 'liaor', 'lier', 'linr', 
  'lingr', 'liur', 'lor', 'longr', 'lour', 'lur', 'lvr', 'luanr', 
  'lver', 'lunr', 'luor', 'mar', 'mair', 'manr', 'mangr', 'maor', 
  'mer', 'meir', 'menr', 'mengr', 'mir', 'mianr', 'miaor', 'mier', 
  'minr', 'mingr', 'miur', 'mor', 'mour', 'mur', 'nar', 'nair', 
  'nanr', 'nangr', 'naor', 'ner', 'neir', 'nenr', 'nengr', 'nir', 
  'niar', 'nianr', 'niangr', 'niaor', 'nier', 'ninr', 'ningr', 
  'niur', 'nongr', 'nour', 'nur', 'nvr', 'nuanr', 'nver', 'nuor', 
  'or', 'our', 'par', 'pair', 'panr', 'pangr', 'paor', 'peir', 
  'penr', 'pengr', 'pir', 'pianr', 'piaor', 'pier', 'pinr', 
  'pingr', 'por', 'pour', 'pur', 'qir', 'qiar', 'qianr', 
  'qiangr', 'qiaor', 'qier', 'qinr', 'qingr', 'qiongr', 
  'qiur', 'qur', 'quanr', 'quer', 'qunr', 'ranr', 'rangr', 
  'raor', 'rer', 'renr', 'rengr', 'rir', 'rongr', 'rour', 'rur', 
  'ruar', 'ruanr', 'ruir', 'runr', 'ruor', 'sar', 'sair', 'sanr',
  'sangr', 'saor', 'ser', 'seir', 'senr', 'sengr', 'shar', 'shair', 
  'shanr', 'shangr', 'shaor', 'sher', 'sheir', 'shenr', 'shengr', 'shir',
  'shour', 'shur', 'shuar', 'shuair', 'shuanr', 'shuangr', 'shuir', 
  'shunr', 'shuor', 'sir', 'songr', 'sour', 'sur', 'suanr', 'suir',
  'sunr', 'suor', 'tar', 'tair', 'tanr', 'tangr', 'taor', 'ter', 'tengr',
  'tir', 'tianr', 'tiaor', 'tier', 'tingr', 'tongr', 'tour', 'tur', 'tuanr',
  'tuir', 'tunr', 'tuor', 'war', 'wair', 'wanr', 'wangr', 'weir', 'wenr', 
  'wengr', 'wor', 'wur', 'xir', 'xiar', 'xianr', 'xiangr', 'xiaor', 'xier',
  'xinr', 'xingr', 'xiongr', 'xiur', 'xur', 'xuanr', 'xuer', 'xunr', 'yar',
  'yanr', 'yangr', 'yaor', 'yer', 'yir', 'yinr', 'yingr', 'yor', 'yongr', 
  'your', 'yur', 'yuanr', 'yuer', 'yunr', 'zar', 'zair', 'zanr', 'zangr', 
  'zaor', 'zer', 'zeir', 'zenr', 'zengr', 'zhar', 'zhair', 'zhanr', 'zhangr',
  'zhaor', 'zher', 'zheir', 'zhenr', 'zhengr', 'zhir', 'zhongr', 'zhour', 
  'zhur', 'zhuar', 'zhuair', 'zhuanr', 'zhuangr', 'zhuir', 'zhunr', 'zhuor', 
  'zir', 'zongr', 'zour', 'zur', 'zuanr', 'zuir', 'zunr', 'zuor']

  if pinyin[0:-1] in erhua_list:
    return True
  else:
    return False
  
def align_text_phoneme(text_seq, phoneme_seq):
    # for text pinyin
    text_list = re.sub(r'(#\d)', r' \1 ', text_seq)   
    # text_list = re.sub(r'([a-zAZ]+)(\')(s)', r'\1\3', text_list)  # remove the \' from \'s.
    # Characters that are not in the range of the set can be matched by negation, ^ represent negation.
    # \u400e-\u9fa5 : Chinese characters range
    # text_list = re.sub(r'[^a-zA-Z\u400e-\u9fa5 ]', ' ', text_list)  
    # for not chinese and english characters
    # print(text_list)
    text_list = re.sub(r'([^a-zA-Z\u400e-\u9fa5 \'])', r' \1 ', text_list) 
    text_list = re.sub(r'\'', r'', text_list)
    # print(text_list)
    # add the blanks for words (whether is chinese characters)
    text_list = re.sub(r'([\u400e-\u9fa5])', r' \1 ', text_list)
    text_list = re.sub(r'\s+', ' ', text_list).strip()
    text_list = re.sub(r'(#)\s(\d)', r'\1\2', text_list)   # combine the split rhythm.
    text_list = text_list.split()
    text_list = [text for text in text_list if text != '']
    # for phoneme
    phoneme_list = re.sub(r'([a-z]+[1-5])', r'/ \1 /', phoneme_seq)
    phoneme_list = re.sub(r'([-])', r'/ \1 /', phoneme_list)
    phoneme_list = re.sub(r'^/ ', '', phoneme_list).strip()
    phoneme_list = re.sub(r' /$', '', phoneme_list).strip()
    phoneme_list = re.sub(r'/ /', '/', phoneme_list)
    phoneme_list = [phoneme.strip() for phoneme in phoneme_list.split('/')]
    phoneme_list = [phoneme for phoneme in phoneme_list if phoneme != '']
    # phoneme_list = phoneme_list.split('/')
    return text_list, phoneme_list

def is_mandarin(uchar):
  """判断一个unicode是否是汉字"""
  code_point = ord(uchar)
  if code_point >= 0x4e00 and code_point <= 0x9fa5:
    return True
  else:
    return False

def is_mandarin_for_spss(text_string):
  """判断一个传入字符是否是汉字，传入可能是字符串"""
  if len(text_string) > 1:
    return False
  code_point = ord(text_string)
  if code_point >= 0x4e00 and code_point <= 0x9fa5:
    return True
  else:
    return False

def is_english_for_spss(text_string):
  """判断一个传入字符是否是英文"""
  if re.search(r'^[a-zA-Z]', text_string):
    return True
  else:
	  return False

def is_cmu_for_spss(text_string):
  """判断一个传入字符是否是英文"""
  if re.search(r'^[A-Z]', text_string):
    return True
  else:
	  return False

def is_pinyin_for_spss(text_string):
  """判断一个传入字符是否是拼音"""
  if re.search(r'^[a-z]', text_string):
    return True
  else:
	  return False

def is_punct_for_spss(text_string):
	"""判断一个传入字符是否是spss接受的标点，传入可能是字符串"""
	puncts = [',', '，', ':', '：', '。', '.', '!', '！', '?', '？', ';','；','、']
	if len(text_string) > 1:
		return False
	if text_string in puncts:
		return True
	else:
		return False

def is_prosody_for_spss(text_string):
	"""判断一个传入字符是否是spss接受的标点，传入可能是字符串"""
	prosody_level = ['#1', '#2', '#3', '#4']
	if text_string in prosody_level:
		return True
	else:
		return False

def num_mandarin(text):
  """compute number of mandarin chars in text"""
  i = 0
  for t in text:
    if is_mandarin(t):
      i += 1
  return i


def check_trans_align(text_seq, phoneme_seq):
  """check the wore and pinyin align and generate the pinyin_prosody_dicts
    Args:
        text_seq: text sequence. egs: "遛弯儿#2，都得#2躲远点#4。"
        phoneme_seq: phoneme sequence. egs: "liu4 wanr1 dou1 dei3 duo2 yuan2 dian3"
    Returns: the pinyin_prosody_dicts. egs: [[['遛'], 'liu4'], [['弯', '儿'], 'wanr1'], [['#2']], [['，']], [['都'], 'dou1'], [['得'], 'dei3'], [['#2']], [['躲'], 'duo2'], [['远'], 'yuan2'], [['点'], 'dian3'], [['#4']], [['。']]]
  """
  text_list, phoneme_list = align_text_phoneme(text_seq, phoneme_seq)
  
  sequence_pinyin_prosody = []  ## for keep the current sequence's word_pinyin_prosody
  # try:
  zh_en_count = 0   # the count pinyin of cheinese and english
  len_text_list = len(text_list)
  len_phoneme_list = len(phoneme_list)
  s_index =  0
  while s_index < len_text_list:
    word_pinyin_prosody = []  # the word_pinyin_prosody
    ### the text is chinese or english
    c_index = s_index
    ### for the word is chinese or english  
    if zh_en_count < len_phoneme_list:
      c_phoneme = phoneme_list[zh_en_count]
    # the text still has word, but the pinyin no more
    elif is_mandarin_for_spss(text_list[s_index]) or is_english_for_spss(text_list[s_index]):
      raise ValueError(f"text_list and phoneme_list not align!\n{text_list}\n{phoneme_list}")
    # print(text_list[s_index])
    # print(c_phoneme)
    if is_mandarin_for_spss(text_list[s_index]):
      ### for 儿化音
      if zh_en_count < len_phoneme_list  and s_index+1 < len_text_list:
        if is_rhotic_accent(c_phoneme):              
          s_index += 1 # skip the next "儿"
      # print(text_list[c_index:s_index+1], c_phoneme, c_phoneme[:-1])

      # add tone
      # tone = c_phoneme[-1]
      # tmp_pinyin = []
      # for item in self.dict_pinyin[c_phoneme[:-1]]:
      #   tmp_pinyin.append(item+tone)
      word_pinyin_prosody.append(text_list[c_index:s_index+1])
      # origin 
      # word_pinyin_prosody.append(self.dict_pinyin[c_phoneme[:-1]])
      # add tone 
      word_pinyin_prosody.append(c_phoneme)
      zh_en_count += 1
      # print(word_pinyin_prosody)
    elif is_english_for_spss(text_list[s_index]):
      # print(text_list[c_index:s_index+1], c_phoneme)
      c_phoneme = ' ' + c_phoneme + ' '
      word_pinyin_prosody.append(text_list[c_index:s_index+1])
      # 重音变成0
      # c_phoneme = re.sub(r'(\w)\d', r'\1', c_phoneme)
      # c_phoneme = re.sub(r'(\w)\s', r'\1 0 ', c_phoneme)
      # c_phoneme = re.sub(r'(\w)\s(\d)', r'\1\2', c_phoneme)
      c_phoneme = re.sub(r'\.', r'', c_phoneme)

      word_pinyin_prosody.append(c_phoneme.split())
      zh_en_count += 1
    else:
      word_pinyin_prosody.append(text_list[c_index:s_index+1])
    ## sequence_pinyin_prosody add word_pinyin_prosody
    sequence_pinyin_prosody.append(word_pinyin_prosody)
    s_index += 1
  # the text no more, but the pinyin still has text
  if zh_en_count != len_phoneme_list:
    raise ValueError(f"text_list and phoneme_list not align!\n{text_list}\n{phoneme_list}")
  # except Exception as e:
  #   print(e)
  #   raise ValueError(f"text_list and phoneme_list not align!\n{text_list}\n{phoneme_list}")
  
  return sequence_pinyin_prosody


def extract_polyphone_dict(polyphone_file):
  """ extract polyphone dict.
    Args:
      polyphone_file: the polyphone file.
    Returns: the polyphone dict, egs: "骑,qi2,ji4" -> dict{'骑': '[qi2|ji4]'}.
  """
  with open(polyphone_file, 'r', encoding="utf-8") as f:
    lines = f.readlines()
    polyphone_dict = {}
    for line in lines:
      line = line.strip()
      if line != '':
        polyphone_list = line.split(',')
        polyphone_dict[polyphone_list[0]] = '[' + '|'.join(polyphone_list[1:]) + ']'
  return polyphone_dict


def label2text(text_path, polyphone_dict):
  """
    Args:
      text_path: the text file.
    Returns: the origin_text|pinyin|pure_text.
  """
  trans_lists = []
  polyphone_dict_keys = polyphone_dict.keys()
  
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
    # sent_cont_ = re.sub(r'#\d[,;:.?!，。？；：、！]$', r'#4', sent_cont)
    # sent_cont = re.sub(r"#\d", r"", sent_cont_)
    text_seq = re.sub(r'#\d[,;:.?!，。？；：、！]$', r'#4', sent_cont)
    phoneme_seq = trans_lists[i+1].strip()

    try:
      align_lists = check_trans_align(text_seq, phoneme_seq)
    except Exception as e:
      print(f"{e}")
      print(f"text and phoneme not align!\n{text_seq}\n{phoneme_seq}")
      continue
    
    question = ''
    answer = ''
    for item in align_lists:
      tmp_text = ''.join(item[0])
      question += tmp_text
      answer += tmp_text
      # if is polyphone
      if is_mandarin_for_spss(tmp_text) and tmp_text in polyphone_dict_keys:
        question += polyphone_dict[tmp_text]
        answer += '[' + item[1] + ']'
    # strip prosody
    question = re.sub(r'#\d', r'', question)

    # sid, 去除末尾标点符号, 拼音, llm的输入纯文本
    data.append([sid, answer, trans_lists[i+1], question])
  return data


def llm_pp_inf(text_mid):
    text = text_start + text_mid + text_end
    # print(text)
    # exit()
    # time1 = time.time()
    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
    # time2 = time.time()
    # print(f"tokenizer prompt time: {time2-time1}")

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
  parser.add_argument("--polyphone_file",type=str,
    default='/mnt/cfs/SPEECH/liuhuang1/workspace/tts_frontend/polyphone.txt',
    required=False)
  parser.add_argument(
      "--output_dir",
      type=str,
      default='data/',
      required=False,
      help="the output dir for the processed data.")
  args = parser.parse_args()

  text_file = args.text_file
  output_dir = args.output_dir
  polyphone_file = args.polyphone_file

  if not os.path.exists(text_file):
    raise ValueError(f"file not exists: {text_file}")
  if not os.path.exists(polyphone_file):
    raise ValueError(f"file not exists: {polyphone_file}")
  os.makedirs(output_dir, exist_ok=True)
  polyphone_dict = extract_polyphone_dict(polyphone_file)

  data = label2text(text_file, polyphone_dict)
  data_inf = data[9700:]

  output_file = os.path.join(output_dir, "infer_pp_7b_prompt_v9_data2.txt")
  with open(output_file, 'w', encoding='utf-8')as fw:
    i = 1
    for item in tqdm(data_inf):
      sid = str(item[0])
      time1 = time.time()
      result = llm_pp_inf(item[3])
      text_prosody = result.strip("Answer: ")
      # print(f"result: {result}")
      time2 = time.time()
      consume_time = time2 - time1
      print(f"time: {consume_time} {consume_time/len(item[3])}")
      # write data
      fw.writelines(sid+ "\t" + text_prosody + '\n')
      # exit()
      if i % 10 == 0:
        fw.flush()
      i += 1

if __name__ == '__main__':
  __cmd()
