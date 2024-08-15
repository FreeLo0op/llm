import os
import sys
import json
import re
import random
import numpy as np

'''
000001	卡尔普#2陪#1外孙#1玩滑梯#3
	ka2 er2 pu3 pei2 uai4 suen1 uan2 hua2 ti1
'''


prosody_level = set(['#0', '#1', '#2', '#3'])

def is_chinese(char: chr) -> bool:
    return re.search(r'[\u4e00-\u9fff]', char)

def string_split(string:str) -> list[str]:
    return re.findall(r'[\u4e00-\u9fff]|#\d', string)

def syllable_split(string: str) -> list[str]:
    return string.split()

def label_align(texts: list[str], syllables: list[str]):
    question, answer = list(), list()
    for text in texts:
        if text in prosody_level:
            answer.append(text)
        else:
            answer.append(syllables.pop(0))
            question.append(text)
    return ''.join(question), ' '.join(answer)

def prompt_gen(question: str, answer: str):
    prompt = {}
    
    prompt['conversations'] = []
    sub_system = {
        "from": "system",
        "vale": "对提供的中文文本根据下面规则进行处理，来完成韵律标记和音节任务，文字标点符号和顺序保持不变。韵律标记：为句子中的每个字添加韵律标签，只有3类标签'#1,#2,#3'。规则如下：词：成词的两个或多个字的最后一个字后面标记为#1；短语：短语中的最后一个字后面标记为#2；如果字后面跟随标点符号（且该字不是句子的最后一个字），或者单句无标点符号文本并且文本过长，需要根据语意进行切分停顿，则该字的标记为#3。音节标记：给用户提供的文本中的中文标注音节，注意有的字是多音字，需要给出正确的读音’。例如：question1:‘卡尔普陪外孙玩滑梯’，answer1:‘ka2 er2 pu3 #2 pei2 #1 uai4 suen1 #1 uan2 hua2 ti1 #3。’；qeusion2:‘而部分网民也乐于围观起哄’，answer2:‘er2 #2 bu4 fen5 #1 uang3 min2 #2 ie3 #1 le4 v2 #1 uei2 guan1 #1 qi3 hong4 #3’；question3:‘舞池璀璨乐声欢快光影交错’，answer3:‘u3 chiii2 #1 cuei3 can4 #3 ve4 sheng1 #1 huan1 kuai4 #3 guang1 ing3 #1 jiao1 cuo4 #3’"
    }
    sub_user = {
        "from": "user",
        "value": question
    }
    sub_answer = {
        "from": "assistant",
        "value": answer
    }
    prompt['conversations'].append(sub_system)
    prompt['conversations'].append(sub_user)
    prompt['conversations'].append(sub_answer)
    return prompt

if __name__ == '__main__':
    file_in = r'/mnt/cfs/SPEECH/hupeng/github/llm_pp/datas/hp_data/biaobei.txt_syllable'
    train_fo = r'/mnt/cfs/SPEECH/hupeng/github/llm_pp/datas/hp_data/prompts/train_prompt_v1.json'
    dev_fo = r'/mnt/cfs/SPEECH/hupeng/github/llm_pp/datas/hp_data/prompts/dev_prompt_v1.json'
    
    train_data, dev_data = [], []
    count = 0
    lines = open(file_in, 'r', encoding='utf8').readlines()
    for i in range(0, len(lines), 2):
        count += 1
        content, syllable = lines[i], lines[i+1]
        
        content = content.strip().split('\t')[1]
        content = string_split(content)
        syllable = syllable_split(syllable)
        question, answer = label_align(content, syllable)
        prompt = prompt_gen(question, answer)
        
        if count <= 7000:
            train_data.append(prompt)
        else:
            dev_data.append(prompt)
    
    with open(train_fo, 'w', encoding='utf8') as fo:
        json.dump(train_data, fo, indent=4, ensure_ascii=False)
    
    with open(dev_fo, 'w', encoding='utf8') as fo:
        json.dump(dev_data, fo, indent=4, ensure_ascii=False)
    
    
    
    
        
        
