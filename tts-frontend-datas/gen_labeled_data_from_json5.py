#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# gen text_labeled data from json file.
# Created by Huang Liu, 2024.05.20
import re
import os
import json
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

u2v_pinyin =  {'yu':'yv','yun':'yvn','yue':'yve','yuan':'yvan',
  'xu':'xv','xue':'xve','xun':'xvn','xuan':'xvan',
  'ju':'jv','jue':'jve','jun':'jvn','juan':'jvan',
  'qu':'qv','que':'qve','qun':'qvn','quan':'qvan',
  'lue':'lve','nue':'nve'}
u2v_pinyin_dict = dict((re.escape(k), v) for k, v in u2v_pinyin.items())
u2v_pattern = re.compile("|".join(u2v_pinyin_dict.keys()))

v2u_pinyin =  {'yv':'yu','yvn':'yun','yve':'yue','yvan':'yuan',
  'xv':'xu','xve':'xue','xvn':'xun','xvan':'xuan',
  'jv':'ju','jve':'jue','jvn':'jun','jvan':'juan',
  'qv':'qu','qve':'que','qvn':'qun','qvan':'quan','lue':'lve','nue':'nve'}# 'lve':'lue','nve':'nue'
v2u_pinyin_dict = dict((re.escape(k), v) for k, v in v2u_pinyin.items())
v2u_pattern = re.compile("|".join(v2u_pinyin_dict.keys()))

def transfer_v_u(pinyin, char_uv=''):
    char_uv = 'u'
    if char_uv == 'v':
      return u2v_pattern.sub(lambda m: u2v_pinyin_dict[re.escape(m.group(0))], pinyin)
    elif char_uv == 'u':
      return v2u_pattern.sub(lambda m: v2u_pinyin_dict[re.escape(m.group(0))], pinyin)
    else:
      raise ValueError(f"the char_uv parameter {char_uv} illeagel") 

prosody_sens = []
g2p_sens = []

SENTENCE_SPLITOR = re.compile(r'([：、，；。？！,;?!:][”’]?)')
SENTENCE_END_PUNCT_SPLITOR = re.compile(r'([；。？！;?!])')
punct = "：、，；。？！,;?!:"
end_punct = "；。？！;?!"

puncts = [',', '，', ':', '：', '。', '.', '!', '！', '?', '？', ';','；','、']
user_puncts = [',', '，', ':', '：', '。', '.', '!', '！', '?', '？', ';','；','、', '=', '[', ']']
end_punctuations = ['。', '.', '!', '！', '?', '？', ';','；']
user_tag = ['=', '[', ']']

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

def is_number(text_string):
  """判断一个传入字符是否是数字"""
  if re.search(r'^\d', text_string):
    return True
  else:
	  return False

def is_english_for_spss(text_string):
  """判断一个传入字符是否是英文"""
  if re.search(r'^[a-zA-Z]', text_string):
    return True
  else:
	  return False

def is_english_phoneme_for_spss(text_string):
    """判断一个传入字符是否是英文phoneme"""
    if re.search(r'^[A-Z]', text_string):
        return True
    else:
        return False

def is_punct_for_spss(text_string):
	"""判断一个传入字符是否是spss接受的标点，传入可能是字符串"""
	if len(text_string) > 1:
		return False
	if text_string in puncts:
		return True
	else:
		return False

def is_punct_for_user(text_string):
	"""判断一个传入字符是否是spss接受的标点，传入可能是字符串"""
	if len(text_string) > 1:
		return False
	if text_string in user_puncts:
		return True
	else:
		return False

def remove_illegal_punct(text):
  '''remove illegal punct: [].='''
  new_text_seq = ''
  for t in text:
    if t in [' ','	','#']:
      new_text_seq += t
    if is_number(t):
      new_text_seq += t
    if is_mandarin(t):
      new_text_seq += t
    if is_english_for_spss(t):
      new_text_seq += t
    if is_punct_for_spss(t):
      new_text_seq += t
  return new_text_seq

def split_by(text, symbol="。"):  # 这里的"下划"线是一个特殊的,罕见符号
    text = text.replace(symbol, f"{symbol}▁")
    parts = text.split("▁")
    return [x.strip() for x in parts if len(x.strip()) > 0]


def process_json_file(json_file):
    with open(json_file, "r") as f:
        result = json.load(f)
        # data = json.loads(result["addition"]["description"])
        data = json.loads(result["data"]["addition"]["description"])
        d = data[0]
        pinyin = d["pinline"]
        prosody_text = data[0]["psdline"]

        # transfer v to u
        pinyin = transfer_v_u(pinyin, 'u')
        # tone 6 to 3
        # pinyin = pinyin.replace("6", "3")
        # rm sil and sp
        pinyin = re.sub('sp', '', pinyin)
        pinyin = re.sub('sil', '', pinyin)
        pinyin = re.sub(r'\s+', ' ', pinyin)

        # print(prosody_text, pinyin)
        # exit()
        return prosody_text, pinyin


data_lists = list(Path("wenkai5_text/").glob("**/*.json"))
data_lists = sorted(data_lists)
with open("test_5.txt", "w", encoding="utf8") as wf:
  i = 1
  for item in tqdm(data_lists):
    # tmp_sid = '1' + str(i).zfill(7)
    tmp_sid = os.path.basename(item).split(".json")[0]
    # print(tmp_sid)
    try:
      prosody_text, pinyin = process_json_file(item)
      prosody_text = remove_illegal_punct(prosody_text)
      # multi continue punc
      prosody_text = re.sub(r'([，：。！？；、,:.!?;])[，：。！？；、,:.!?;]+', r'\1', prosody_text)

      prosody_text = re.sub(r'#4', r'', prosody_text)
      # Mrs.#1 dr.#1
      prosody_text = re.sub(r'[,;:.?!，。？；：、！](#\d)', r'\1', prosody_text)
      # #3:#3，   #3；#3；
      prosody_text = re.sub(r'(#\d)\s?(#\d)+', r'\1', prosody_text)

      prosody_text = re.sub(r'[,;:.?!，。？；：、！]+([,;:.?!，。？；：、！])', r'\1', prosody_text)
      prosody_text = re.sub(r'#\d[,;:.?!，。？；：、！]$', r'', prosody_text)

      pinyin = re.sub(r'/\s+/', r'/', pinyin)
      wf.write(tmp_sid + "\t" + prosody_text + "\n\t" + pinyin + "\n")
    except Exception as e:
      print(e)
      print(f"process failure: {item}")
    i += 1

    if i > 10000:
      break
print(f"process {i} json file.")
   
# with ThreadPoolExecutor(max_workers=1) as executor:
#     list(
#         tqdm(
#             executor.map(process_json_file, list(Path("output/").glob("**/*.json"))),
#             total=len(list(Path("output/").glob("**/*.json"))),
#         )
#     )

# with open("prosody.txt", "w", encoding="utf8") as wf:
#     for sen in prosody_sens:
#         wf.write(sen + "\n")


# with open("g2p.json", "w") as f:
#     json.dump(g2p_sens, f, indent=4, ensure_ascii=False)
