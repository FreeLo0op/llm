import os
import re
import argparse
from tqdm import tqdm

def caculate_prosody_metrics(truth_text_file, predict_text_file):
  # dirname = os.path.dirname(predict_text_file)
  # output_file = os.path.join(dirname, "lstm_pp_dev.txt")

  truth_text_dict = {} # {sid: text}
  predict_text_dict = {} # {sid: text}

  with open(truth_text_file, 'r', encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
      line = line.strip()   # convert the blank line to none
      if line != '': 
        sid = re.search(r'^(\d+_)*\d+', line)
        if sid is None:
          raise ValueError(f"Check the sentence whether exists sid.\n{line}")
        sid = sid.group()
        sent_cont = re.sub(r'^(\d+_)*\d+\s+', '', line).strip()
        sent_cont = re.sub(r'\s+', ' ', sent_cont).strip()
        # sent_cont strip end puncts
        sent_cont = re.sub(r'[，。？；：、！,\.;？:!]$', '', sent_cont)
        sent_cont = re.sub(r'#\d', '', sent_cont)
        sent_cont = re.sub(r'#\d$', '', sent_cont)
        # sent_cont
        truth_text_dict[sid] = sent_cont

  with open(predict_text_file, 'r', encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
      line = line.strip()   # convert the blank line to none
      if line != '': 
        sid = re.search(r'^(\d+_)*\d+', line)
        if sid is None:
          raise ValueError(f"Check the sentence whether exists sid.\n{line}")
        sid = sid.group()
        sent_cont = re.sub(r'^(\d+_)*\d+\s+', '', line).strip()
        sent_cont = re.sub(r'\s+', ' ', sent_cont).strip()
        # sent_cont = re.sub(r'\[\w+\d\]', '', sent_cont).strip()
        # sent_cont strip end puncts
        sent_cont = re.sub(r'[，。？；：、！,\.;？:!]$', '', sent_cont)
        sent_cont = re.sub(r'#\d', '', sent_cont)
        sent_cont = re.sub(r'#\d$', '', sent_cont)
        predict_text_dict[sid] = sent_cont

  # true_text = "我#1爱#1中国#1。"
  # pred_text = "我爱#1中国#1。"
  # 计算指标

  sid_lists = truth_text_dict.keys()
  acc_num, sums = 0, 0
  for item in sid_lists:
    if item not in predict_text_dict.keys():
      print(f"{item} not in predict_text_dict.")
      continue
    
    true_text = truth_text_dict[item]
    pred_text = predict_text_dict[item]

    true_text_matches = re.findall(r'\[([a-z]+\d)\]', true_text)
    pred_text_matches = re.findall(r'\[([a-z]+\d)\]', pred_text)
    if len(true_text_matches) != len(pred_text_matches):
      print(f"{item} true_text_matches != pred_text_matches.")
      continue
    else:
      sums += len(true_text_matches)
      for (item1, item2) in zip(true_text_matches, pred_text_matches):
        if item1 == item2:
          acc_num += 1
  print(f"acc_num: {acc_num}, sums: {sums}, acc_rate: {acc_num/sums}")


def __cmd():
  description = "segment audio file"
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument(
      "--truth_text_file",
      type=str,
      default='/mnt/cfs/SPEECH/liuhuang1/workspace/llm_pp/data/truth_dev_v17.txt',
      required=False,
      help="the truth prosody text file, must has sentence id.")
  parser.add_argument(
      "--predict_text_file",
      type=str,
      # lstm_pp_dev infer_pp_7b_prompt_v5 infer_pp_0.5b_prompt_v5 infer_pp_0.5b_prompt_v5_ckt160
      # infer_pp_1.8b_prompt_v5 infer_pp_1.8b_prompt_v5_ckt160
      # infer_pp_4b_prompt_v5 infer_pp_4b_prompt_v5_ckt160
      # infer_pp_7b_prompt_v12_ckt3660_all.txt infer_pp_7b_prompt_v12_ckt6350_all
      # infer_pp_7b_prompt_v14_ckt3490_all
      default='/mnt/cfs/SPEECH/liuhuang1/workspace/llm_pp/data/infer_pp_7b_prompt_v17_ckt8620_all.txt',
      required=False,
      help="the predict prosody text file")
#   parser.add_argument(
#       "--output_dir",
#       type=str,
#       default='data/',
#       required=False,
#       help="the output dir for the processed data.")
  args = parser.parse_args()

  truth_text_file = args.truth_text_file
  predict_text_file = args.predict_text_file
#   output_dir = args.output_dir

  if not os.path.exists(truth_text_file):
    raise ValueError(f"file not exists: {truth_text_file}")
  if not os.path.exists(predict_text_file):
    raise ValueError(f"file not exists: {predict_text_file}")
  
  caculate_prosody_metrics(truth_text_file, predict_text_file)


if __name__ == '__main__':
  __cmd()