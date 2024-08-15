import os
import re
import argparse
from tqdm import tqdm


def extract_boundaries(text):
    """提取韵律边界的索引位置及其类型"""
    boundaries = []
    current_index = 0
    i = 0
    while i < len(text):
        if text[i] == '#':
            # 提取数字作为边界类型
            boundary_type = ''
            i += 1
            while i < len(text) and text[i].isdigit():
                boundary_type += text[i]
                i += 1
            boundaries.append((current_index, int(boundary_type)))
        else:
            if text[i] != ' ':
                current_index += 1
            i += 1
    return boundaries

def calculate_metrics(true_text, pred_text):
    """计算各级别韵律边界的Precision, Recall和F1分数，以及块准确率"""
    try:
      true_boundaries = extract_boundaries(true_text)
      pred_boundaries = extract_boundaries(pred_text)
    except Exception as e:
      print(true_text, pred_text)
      exit()
    
    true_dict = {}
    pred_dict = {}
    
    # 将边界按类型分类
    for index, btype in true_boundaries:
        if btype not in true_dict:
            true_dict[btype] = set()
        true_dict[btype].add(index)
    
    for index, btype in pred_boundaries:
        if btype not in pred_dict:
            pred_dict[btype] = set()
        pred_dict[btype].add(index)
    
    results = {}
    
    # 计算每个类型的Precision, Recall, F1
    all_types = set(true_dict.keys()).union(set(pred_dict.keys()))
    TP, FP, FN = 0, 0, 0
    for btype in all_types:
        TP = len(true_dict.get(btype, set()) & pred_dict.get(btype, set()))
        FP = len(pred_dict.get(btype, set()) - true_dict.get(btype, set()))
        FN = len(true_dict.get(btype, set()) - pred_dict.get(btype, set()))
        
        # precision = TP / (TP + FP) if TP + FP > 0 else 0
        # recall = TP / (TP + FN) if TP + FN > 0 else 0
        # f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        
        # results[btype] = {
        #     "Precision": precision,
        #     "Recall": recall,
        #     "F1 Score": f1_score
        # }

    
    # 计算块准确率
    total_blocks = len(true_boundaries)
    correct_blocks = sum(1 for t, p in zip(true_boundaries, pred_boundaries) if t == p)
    # block_accuracy = correct_blocks / total_blocks if total_blocks > 0 else 0
    
    # results['Block Accuracy'] = block_accuracy

    # return results
    # try:
    return TP, FP, FN, total_blocks, correct_blocks
    # except:
    #   print(true_text, pred_text)
    #   exit()

# 示例数据
# true_text = "我#1爱#2中国#3。"
# pred_text = "我爱#1中国#2。"

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
        sent_cont = re.sub(r'\[\w+\d\]', '', sent_cont).strip()
        # sent_cont strip end puncts
        sent_cont = re.sub(r'[，。？；：、！,\.;？:!]$', '', sent_cont)
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
        sent_cont = re.sub(r'\[\w+\d\]', '', sent_cont).strip()
        # sent_cont strip end puncts
        sent_cont = re.sub(r'[，。？；：、！,\.;？:!]$', '', sent_cont)
        sent_cont = re.sub(r'#\d$', '', sent_cont)
        predict_text_dict[sid] = sent_cont

  # true_text = "我#1爱#1中国#1。"
  # pred_text = "我爱#1中国#1。"
  # 计算指标
  for prosody_level in ["#1", "#2", "#3"]:
    print(f" ---------------------- prosody_level: {prosody_level} ---------------------- ")
    TPs, FPs, FNs, total_blocks, correct_blocks = 0, 0, 0, 0, 0
    sid_lists = truth_text_dict.keys()
    for item in sid_lists:
      if item not in predict_text_dict.keys():
        print(f"{item} not in predict_text_dict.")
        continue
      
      true_text = truth_text_dict[item]
      pred_text = predict_text_dict[item]
      # print(true_text, pred_text)
      # exit()

      ### #3 > #2 > #1, #3是#2#1， #2是#1
      if prosody_level == "#1":
         true_text = re.sub(r'#\d', '#1', true_text)
         pred_text = re.sub(r'#\d', '#1', pred_text)
      elif prosody_level == "#2":
        true_text = re.sub(r'#1', '', true_text)
        true_text = re.sub(r'#3', '#2', true_text)
        pred_text = re.sub(r'#1', '', pred_text)
        pred_text = re.sub(r'#3', '#2', pred_text)
      elif prosody_level == "#3":
        true_text = re.sub(r'#1', '', true_text)
        true_text = re.sub(r'#2', '', true_text)
        pred_text = re.sub(r'#1', '', pred_text)
        pred_text = re.sub(r'#2', '', pred_text)

      # if prosody_level == "#1":
      #    true_text = re.sub(r'#2', '', true_text)
      #    pred_text = re.sub(r'#3', '', pred_text)
      # elif prosody_level == "#2":
      #   true_text = re.sub(r'#1', '', true_text)
      #   true_text = re.sub(r'#3', '', true_text)
      #   pred_text = re.sub(r'#1', '', pred_text)
      #   pred_text = re.sub(r'#3', '', pred_text)
      # elif prosody_level == "#3":
      #   true_text = re.sub(r'#1', '', true_text)
      #   true_text = re.sub(r'#2', '', true_text)
      #   pred_text = re.sub(r'#1', '', pred_text)
      #   pred_text = re.sub(r'#2', '', pred_text)

      TP, FP, FN, total_block, correct_block = calculate_metrics(true_text, pred_text)
      TPs += TP
      FPs += FP
      FNs += FN
      total_blocks += total_block
      correct_blocks += correct_block
    
    precision = TPs / (TPs + FPs) if TPs + FPs > 0 else 0
    recall = TPs / (TPs + FNs) if TPs + FNs > 0 else 0
    print(f"TPs: {TPs}, FPs: {FPs}, FNs: {FNs}")
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    block_accuracy = correct_blocks / total_blocks if total_blocks > 0 else 0

    print(precision, recall, f1_score, block_accuracy)
  # 打印结果
  # for key, value in results.items():
  #     if isinstance(value, dict):
  #         print(f"Type #{key}:")
  #         for metric, score in value.items():
  #             print(f"  {metric}: {score:.2f}")
  #     else:
  #         print(f"Block Accuracy: {value:.2f}")


def __cmd():
  description = "segment audio file"
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument(
      "--truth_text_file",
      type=str,
      # truth_dev_v10_poly.txt  truth_dev_v10_biaobei_poly.txt
      default='/mnt/cfs/SPEECH/liuhuang1/workspace/llm_pp/data/truth_dev_v17.txt', 
      required=False,
      help="the truth prosody text file, must has sentence id.")
  parser.add_argument(
      "--predict_text_file",
      type=str,
      # lstm_pp_dev  infer_pp_0.5b_prompt_v5 infer_pp_0.5b_prompt_v5_ckt160
      # infer_pp_1.8b_prompt_v5 infer_pp_1.8b_prompt_v5_ckt160
      # infer_pp_4b_prompt_v5 infer_pp_4b_prompt_v5_ckt160
      # infer_pp_7b_prompt_v5 infer_pp_7b_prompt_v5_ckt160
      # infer_pp_7b_prompt_v6 infer_pp_7b_prompt_v6_ckt160
      # infer_pp_7b_prompt_v7 infer_pp_7b_prompt_v7_ckt90
      # lstm_zijie_dev infer_pp_7b_prompt_v10_ckt1710.txt
      # infer_pp_7b_prompt_v11_ckt1580_all.txt lstm_biaobei_dev_all.txt
      # infer_pp_7b_prompt_v12_ckt1680_all
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