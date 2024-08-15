#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# text_labeled data convert to json file.
# Created by Huang Liu, 2024.05.10
import os
import sys
import json
import re
import random
import argparse
import logging
import numpy as np
sys.path.append(os.getcwd())
random.seed(1)

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
    # \u4e00-\u9fff : Chinese characters range
    # text_list = re.sub(r'[^a-zA-Z\u4e00-\u9fff ]', ' ', text_list)  
    # for not chinese and english characters
    # print(text_list)
    text_list = re.sub(r'([^a-zA-Z\u4e00-\u9fff \'])', r' \1 ', text_list) 
    text_list = re.sub(r'\'', r'', text_list)
    # print(text_list)
    # add the blanks for words (whether is chinese characters)
    text_list = re.sub(r'([\u4e00-\u9fff])', r' \1 ', text_list)
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
  if code_point >= 0x4e00 and code_point <= 0x9fff:
    return True
  else:
    return False

def is_mandarin_for_spss(text_string):
  """判断一个传入字符是否是汉字，传入可能是字符串"""
  if len(text_string) > 1:
    return False
  code_point = ord(text_string)
  if code_point >= 0x4e00 and code_point <= 0x9fff:
    return True
  else:
    return False

def is_english_for_spss(text_string):
  """判断一个传入字符是否是英文"""
  if re.search(r'^[a-zA-Z]', text_string):
    return True
  else:
	  return False

def is_exist_english_for_spss(text_string):
  """判断一个传入字符是否是英文"""
  if re.search(r'[a-zA-Z]', text_string):
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
        index_list = [ str(i) for i in range(1, len(polyphone_list[1:])+1) ]
        polyphone_dict[polyphone_list[0]] = ['[' + '|'.join(polyphone_list[1:]) + ']', polyphone_list[1:], '[' + '|'.join(index_list) + ']', index_list]
  return polyphone_dict


def convert_prosody_level2symbol(prosody_level):
  if prosody_level == '#1':
    return '%'
  elif prosody_level == '#2':
    return '$'
  elif prosody_level == '#3':
    return '&'
  else:
    return '#'
  
def convert_symbol2prosody_level(prosody_level):
  if prosody_level == '%':
    return '#1'
  elif prosody_level == '$':
    return '#2'
  elif prosody_level == '&':
    return '#3'
  else:
    return ''


def recover_prosody_pinyin(question, answer, polyphone_dict, en_polyphone_dict):
  """recover the prosody pinyin. 
    Args:
      question: the question text.
      answer: the answer text.
    Returns: the recover prosody pinyin.
  """
  # question = "包[1|2]括[1|2] Pontifex,Pontifex二之后更[1|2]名为[1|2|3] Bridge Construction Set,以及Bridge It"

  sent_cont = re.sub(r'\[[1-9|]+[1-9]\]', r' ', question)
  # print(sent_cont)
  sent_cont = re.sub(r'\s+', ' ', sent_cont).strip()
  # sent_cont = re.sub(r'[,;:.?!，。？；：、！]$', r'', sent_cont)
  sent_cont = re.sub(r'(#\d)', r' \1 ', sent_cont)   
  sent_cont = re.sub(r'([^a-zA-Z\u4e00-\u9fff \'])', r' \1 ', sent_cont) 
  sent_cont = re.sub(r'\'', r'', sent_cont)
  # add the blanks for words (whether is chinese characters)
  sent_cont = re.sub(r'([\u4e00-\u9fff])', r' \1 ', sent_cont)
  sent_cont = re.sub(r'\s+', ' ', sent_cont).strip()
  sent_cont = re.sub(r'(#)\s(\d)', r'\1\2', sent_cont)   # combine the split rhythm.
  sent_list = sent_cont.split()
  sent_list = [text for text in sent_list if text != '']
  # print(len(sent_list))
  # print(sent_list)

  # answer = "/[2]%[1]&/%&/%/[2]/%[2]%%&//%%/"
  # split first
  # print(f"answer1: {answer}")
  answer = re.sub(r'([#%$&])', r' \1 ', answer)
  # merge
  answer = re.sub(r'([#%$&])\s([1-9])', r'\1\2', answer)
  answer_list = answer.split()
  answer_list = [text for text in answer_list if text != '']
  # print(len(answer_list))
  # print(f"answer_list: {answer_list}")

  assert len(sent_list) == len(answer_list), f"sent_list and answer_list not align, {len(sent_list)}!={len(answer_list)}!\n{sent_list}\n{answer_list}"
  results = ""

  for text, prosody in zip(sent_list, answer_list):
    results += text
    prosody = re.sub(r'([#%$&])', r' \1 ', prosody)
    prosody_list = prosody.split()
    prosody_list = [text for text in prosody_list if text != '']
    for item in prosody_list:
      if item in ['#', '%', '$', '&']:
        results += convert_symbol2prosody_level(item)
      if item in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
        poly_index = int(item) - 1
        if is_mandarin_for_spss(text) and text in polyphone_dict.keys():
          results += '[' + polyphone_dict[text][1][poly_index] + ']'
        elif is_english_for_spss(text) and text in en_polyphone_dict.keys():
          results += '[' + en_polyphone_dict[text][1][poly_index] + ']'
      if is_english_for_spss(text):
        results += ' '
  
  results = re.sub(r'\s+', ' ', results).strip()
  results = re.sub(r'([^a-zA-Z0-2\]])\s+', r'\1', results)
  results = re.sub(r'\s+(#\d)', r'\1', results)
  results = re.sub(r'(#\d)\s+', r'\1', results)
  # 重#1[chong2]庆 -> 重[chong2]#1庆
  results = re.sub(r'(#\d)(\[.*?\])', r'\2\1', results)
  return results
   

def gen_json_data(trans_lists, polyphone_dict, en_polyphone_dict, split_text_numbers=80, test_dev=False):
  """Args:
    split_text_numbers: the split text numbers, for prosody level and punc is count, can set bigger.
  """
  data = []
  polyphone_dict_keys = polyphone_dict.keys()
  en_polyphone_dict_keys = en_polyphone_dict.keys()

  truth_answer_data = [] # for save the truth answer
  valid_data = 0

  english_nums = 0
  english_poly_nums = 0

  for i in range(0, len(trans_lists), 2):
    sent = trans_lists[i]
    sid = re.search(r'^(\d+_)*\d+', sent)
    if sid is None:
      raise ValueError(f"Check the sentence whether exists sid.\n{sent}")
    sid = sid.group()
    sent_cont = re.sub(r'^(\d+_)*\d+\s+', '', sent).strip()
    sent_cont = re.sub(r'\s+', ' ', sent_cont).strip()
    # sent_cont = re.sub(r'([^a-zA-Z])\s+', r'\1', sent_cont) 
    # sent_cont = re.sub(r'\s+([^a-zA-Z])', r'\1', sent_cont)
    # strip end punct: "hello#4。" -> "hello"
    sent_cont = re.sub(r'#4', r'', sent_cont)
    # Mrs.#1 dr.#1
    sent_cont = re.sub(r'[,;:.?!，。？；：、！](#\d)', r'\1', sent_cont)
    # #3:#3，   #3；#3；
    sent_cont = re.sub(r'(#\d)\s?(#\d)+', r'\1', sent_cont)

    sent_cont = re.sub(r'[,;:.?!，。？；：、！]+([,;:.?!，。？；：、！])', r'\1', sent_cont)
    text_seq = re.sub(r'#\d[,;:.?!，。？；：、！]$', r'', sent_cont)
    phoneme_seq = trans_lists[i+1].strip()
    # print(text_seq, phoneme_seq)

    # for count the english numbers
    if is_exist_english_for_spss(sent_cont):
      english_nums += 1

    try:
      align_lists = check_trans_align(text_seq, phoneme_seq)
    except Exception as e:
      # print(f"{e}")
      # print(f"text and phoneme not align!\n{text_seq}\n{phoneme_seq}")
      continue
    
    # [[['遛'], 'liu4'], [['弯', '儿'], 'wanr1'], [['#2']], [['，']], [['都'], 'dou1'], [['得'], 'dei3'], [['#2']], [['躲'], 'duo2'], [['远'], 'yuan2'], [['点'], 'dian3'], [['#4']], [['。']]]
    # print(align_lists)
    # 将过长的句子分成多段：按照标点符号所在字数分割，分段结果保存在question list中
    # 
    # if is polyphone
    question_list = []
    question_list1 = []
    answer_list = []
    question = ''
    question1 = ''  # this is for recover the data
    answer = ''
    k = 0

    # 处理成韵律在前，多音字在后的形式
    for j in range(len(align_lists)):
      prosody_symbol = '#' # #0 占位符
      item = align_lists[j]
      tmp_text = ''.join(item[0])
      question += tmp_text
      question1 += tmp_text

      # prosody skip
      if tmp_text in prosody_level:
         continue
      
      # answer += tmp_text
      if j + 1 < len(align_lists):
        next_item = align_lists[j+1]
        # print(next_item[0][0])
        prosody_symbol = convert_prosody_level2symbol(next_item[0][0])

      # add prosody symbol
      answer += prosody_symbol

      # if is chinese polyphone
      if is_mandarin_for_spss(tmp_text) and tmp_text in polyphone_dict_keys:
        question += polyphone_dict[tmp_text][0]
        question1 += polyphone_dict[tmp_text][2]
        ## in the polyphone_dict, add it
        if item[1] in polyphone_dict[tmp_text][1]:
          answer += str(polyphone_dict[tmp_text][1].index(item[1]) + 1)
        # not in the polyphone_dict, add the first
        else:
          answer += '1'

      if is_english_for_spss(tmp_text) and tmp_text in en_polyphone_dict_keys:
        english_poly_nums += 1
        question += en_polyphone_dict[tmp_text][0]
        question1 += en_polyphone_dict[tmp_text][2]

        if item[1] in en_polyphone_dict[tmp_text][1]:
          answer += str(en_polyphone_dict[tmp_text][1].index(item[1] + 1))
        # not in the polyphone_dict, add the first
        else:
          answer += '1'

      if k > split_text_numbers and is_punct_for_spss(tmp_text):
        question_list.append(question)
        question_list1.append(question1)
        answer_list.append(answer)
        question = ''
        question1 = ''
        answer = ''
        k = 0
        continue
      k += 1
      
    # add the last or the first
    question_list.append(question)
    question_list1.append(question1)
    answer_list.append(answer)
    # print(question_list)
    # print(answer_list)

    if test_dev and len(align_lists) > 1:
      continue
    
    for question, question1, answer in zip(question_list, question_list1, answer_list):
      # strip prosody, for english has space 
      question = re.sub(r'#\d', r' ', question)
      question1 = re.sub(r'#\d', r' ', question1)
      question = re.sub(r'\s+', ' ', question).strip()
      question1 = re.sub(r'\s+', ' ', question1).strip()
      # 放在最前面做
      question = re.sub(r'([^a-zA-Z0-2\]])\s+', r'\1', question) 
      question1 = re.sub(r'([^a-zA-Z0-2\]])\s+', r'\1', question1)
      question = re.sub(r'\s+([^a-zA-Z0-2])', r'\1', question)
      question1 = re.sub(r'\s+([^a-zA-Z0-2])', r'\1', question1)
      # question = re.sub(r'[,;:.?!，。？；：、！]$', r'', question)
      
      # answer = re.sub(r'[,;:.?!，。？；：、！]$', r'', answer)
      # answer = re.sub(r'#\d$', r'', answer)
      # answer = re.sub(r'#4', r'', answer)

      # print(question, '\t\t', answer)

      # 还原回去时出问题的不加入，儿化音暂时不支持
      try:
        results = recover_prosody_pinyin(question1, answer, polyphone_dict, en_polyphone_dict)
        # print(f"results: {results}")
      except Exception as e:
        print(e)
        print(question)
        print(question1)
        print(answer)
        continue

      truth_answer_data.append(sid+'\t'+answer)
      tmp_dict = {}
      tmp_dict['conversations'] = []
      
      sub_tmp_dict1 = {"from": "system", 
                      "value": "对提供的中文文本根据下面规则进行处理，来完成韵律标记和多音字选择的任务，文字标点符号和顺序保持不变。韵律标记：为句子中的每个字添加韵律标签，只有4类标签'#,%,$,&'。规则如下：没有停顿的为'#'；词：成词的两个或多个字的最后一个字后面标记为'%'；短语：短语中的最后一个字后面标记为'$'；如果字后面跟随标点符号（且该字不是句子的最后一个字），或者单句无标点符号文本并且文本过长，需要根据语意进行切分停顿，则该字的标记为'&'。多音字选择：用户提供的多音字在括号'[]'中，且用'|'分割'[]'里的各个多音字，根据上下文选择一个括号内最合适的，并用数字标识代替。例如，Question1：'卡[ka3|qia3]尔普陪外[wai5|wai4]孙玩[wan2|wan5]滑梯。'，answer1：'#1#$##2%#1###'；Question2：'假[jia3|gei1|jia4]语[yu3|yu4|yu5]村言别[bie2|bie4]再拥[yong1|yong5]抱我。'，answer2：'#1#1#$#1%#1###'；"}
      sub_tmp_dict2 = {"from": "user", 
                      "value": question}  #.encode("utf-8").decode("utf-8")
      sub_tmp_dict3 = {"from": "assistant", 
                      "value": answer}
      tmp_dict['conversations'].append(sub_tmp_dict1)
      tmp_dict['conversations'].append(sub_tmp_dict2)
      tmp_dict['conversations'].append(sub_tmp_dict3)
      data.append(tmp_dict)
      valid_data += 1
      
  print(f"process valid_data: {valid_data}")
  print(f"english_nums: {english_nums}")
  print(f"english_poly_nums: {english_poly_nums}")

  return data, truth_answer_data
  

def text2json(text_path, polyphone_dict, en_polyphone_dict, output_dir='', data_version="v1"):
  """ rename id
    Args:
      text_path: the text file.
    Returns: the rename sid file.
  """
  trans_lists = []
  train_output_file = os.path.join(output_dir, "g2p_train_prompt_" + data_version + ".json")
  dev_output_file = os.path.join(output_dir, "g2p_dev_prompt_" + data_version + ".json")
  truth_dev_file = os.path.join(output_dir, "truth_dev_" + data_version + ".txt")

  with open(text_path, 'r', encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
      line = line.strip()   # convert the blank line to none
      if line != '':
        trans_lists.append(line)  
  # random.shuffle(trans_lists)
  
  train_data, _ = gen_json_data(trans_lists[:78500] + trans_lists[83500:], polyphone_dict, en_polyphone_dict)
  dev_data, truth_answer_dev_data = gen_json_data(trans_lists[78500:83500], polyphone_dict, en_polyphone_dict)
  # dev_data, truth_answer_dev_data = gen_json_data(trans_lists, polyphone_dict, en_polyphone_dict)

  with open(truth_dev_file, 'w') as f:
     for item in truth_answer_dev_data:
        f.write(item+'\n')
     
  # # dump json
  with open(train_output_file, 'w') as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False)

  with open(dev_output_file, 'w') as f:
    json.dump(dev_data, f, indent=4, ensure_ascii=False)


def __cmd():
  description = "segment audio file"
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument(
      "--text_file",
      type=str,
      # data4.txt 000001-010000-en.txt data_test data5
      default='data5.txt',
      required=False,
      help="the labeled text file")
  parser.add_argument("--polyphone_file",type=str,
    default='/mnt/cfs/SPEECH/liuhuang1/workspace/tts_frontend/merge_polyphone.txt',
    required=False)
  parser.add_argument("--en_polyphone_file",type=str,
    default='/mnt/cfs/SPEECH/liuhuang1/workspace/tts_frontend/en_polyphone_tal.txt',
    required=False)
  parser.add_argument(
      "--output_dir",
      type=str,
      default='data/',
      required=False,
      help="the output dir for the processed data.")
  parser.add_argument(
      "--data_version",
      type=str,
      default="v20",
      required=False,
      help="data version.")
  args = parser.parse_args()

  text_file = args.text_file
  polyphone_file = args.polyphone_file
  en_polyphone_file = args.en_polyphone_file
  output_dir = args.output_dir
  data_version = args.data_version

  if not os.path.exists(text_file):
    raise ValueError(f"file not exists: {text_file}")
  if not os.path.exists(polyphone_file):
    raise ValueError(f"file not exists: {polyphone_file}")
  if not os.path.exists(en_polyphone_file):
    raise ValueError(f"file not exists: {en_polyphone_file}")
  os.makedirs(output_dir, exist_ok=True)
  polyphone_dict = extract_polyphone_dict(polyphone_file)
  en_polyphone_dict = extract_polyphone_dict(en_polyphone_file)
  # print(en_polyphone_dict)

  text2json(text_file, polyphone_dict, en_polyphone_dict, output_dir, data_version)


if __name__ == '__main__':
  __cmd()
