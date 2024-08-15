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
import scipy.io.wavfile
from pathlib import Path
from pydub import AudioSegment
sys.path.append(os.getcwd())
random.seed(1)

def gen_json_data(trans_lists):
  data = []
  for i in range(0, len(trans_lists), 2):
    sent = trans_lists[i]
    # tid = re.search(r'^(\d+_)*\d+', sent)
    # if sid is None:
    #   raise ValueError(f"Check the sentence whether exists sid.\n{sent}")
    # sid = sid.group()
    sent_cont = re.sub(r'^(\d+_)*\d+\s+', '', sent).strip()
    sent_cont = re.sub(r'\s+', ' ', sent_cont).strip()
    # strip end punct: "hello#4。" -> "hello#4"
    sent_cont_ = re.sub(r'#\d[,;:.?!，。？；：、！]$', r'#4', sent_cont)
    sent_cont = re.sub(r"#\d", r"", sent_cont_)

    tmp_dict = {}
    tmp_dict['conversations'] = []
    
    sub_tmp_dict1 = {"from": "system", 
                     "value": "prompt prompt prompt!!"}
    sub_tmp_dict2 = {"from": "user", 
                     "value": "Question: " + sent_cont.encode("utf-8").decode("utf-8")}
    sub_tmp_dict3 = {"from": "assistant", 
                     "value": "Answer: " + "韵律标记：" + sent_cont_ + "；拼音：" + trans_lists[i+1]}
    tmp_dict['conversations'].append(sub_tmp_dict1)
    tmp_dict['conversations'].append(sub_tmp_dict2)
    tmp_dict['conversations'].append(sub_tmp_dict3)
    data.append(tmp_dict)

  return data
  

def text2json(text_path, output_dir='', start_sid=0):
  """ rename id
    Args:
      text_path: the text file.
    Returns: the rename sid file.
  """
  trans_lists = []
  train_output_file = os.path.join(output_dir, "g2p_train_prompt_v2.json")
  dev_output_file = os.path.join(output_dir, "g2p_dev_prompt_v2.json")
  sid = start_sid

  with open(text_path, 'r', encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
      line = line.strip()   # convert the blank line to none
      if line != '':
        trans_lists.append(line)  
  # random.shuffle(trans_lists)
  
  
  train_data = gen_json_data(trans_lists[:19400])
  dev_data = gen_json_data(trans_lists[19400:])
  # dump json
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

  text2json(text_file, output_dir)

if __name__ == '__main__':
  __cmd()
