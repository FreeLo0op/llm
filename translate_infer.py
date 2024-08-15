#%%
# cn-en translation
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel,LoraConfig

mode_path = '/mnt/cfs/NLP/hub_models/Qwen1.5-7B-Chat'
lora_path = '/mnt/cfs/SPEECH/zhangxinke1/work/tal_git/latex2chinese/v1/qwen1half-7b-chat/v0-20240421-164810/checkpoint-150'

config = LoraConfig.from_pretrained(lora_path)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(mode_path,device_map="auto", torch_dtype = torch.float32)

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

#%%
text = """<|im_start|>system
You are a helpful assistant.
Answer the following questions as best you can. 
You must access to the following APIs:
    translation_lookup: Call this tool to interact with the 查翻译 API. What is the 查翻译 API useful for? 这是一个查找翻译的工具,功能有1.中译英 2.英译中
    结果返回json的格式 Parameters: [{\"name\": \"source_text\", \"description\": \"这个参数是要翻译的目标语句, 从Question中提取的内容。\", \"required\": true, \"schema\": {\"type\": \"string\"}}, {\"name\": \"translation_method\", \"description\": \"使用哪种翻译方式，to_chinese：翻译成中文，to_english：翻译成英文\", \"required\": true, \"schema\": {\"type\": \"string\", \"enum\": [\"to_chinese\", \"to_english\"]}}]\n\n    Use the following format:\n    Question: the input question you must answer\n    Thought: you should always think about what to do, 能使用工具就必须使用工具，不能自己给出答案，不可以直接作答\n    Action: the action to take, should be one of [translation_lookup]\n    Action Input: the input to the action\n    Observation: the result of the action\n    ... (this Thought/Action/Action Input/Observation can be repeated zero or more times)\n    Thought: I now know the final answer\n    Final Answer: the final answer to the original input question\n\n    Begin!
<|im_end|>
<|im_start|>user
Question: 空调的英文 <|im_end|>
<|im_start|>assistant"""

model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

generated_ids = model.generate(input_ids=model_inputs.input_ids, max_new_tokens=512)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True,temperature=0)[0]

print(response)


# %%
# translate latex to chinese text
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig

mode_path = '/mnt/cfs/NLP/hub_models/Qwen1.5-7B-Chat'
lora_path = '/mnt/cfs/SPEECH/zhangxinke1/work/tal_git/latex2chinese/latex_v1/qwen1half-7b-chat/v0-20240421-181437/checkpoint-150'

config = LoraConfig.from_pretrained(lora_path)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path)

# load model
model = AutoModelForCausalLM.from_pretrained(mode_path,device_map="auto", torch_dtype = torch.float32)

# load lora weights
model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

#%%
text = """<|im_start|>system
You are a translation assistant. You can translate the mathematical content in latex format into Chinese text.
<|im_end|>
<|im_start|>user
Question: 1\\frac{a+b}{c}-\\frac{b+c}{d} <|im_end|>
<|im_start|>assistant"""

model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

generated_ids = model.generate(input_ids=model_inputs.input_ids, max_new_tokens=512)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True,temperature=0)[0]

print(response)

# %%
