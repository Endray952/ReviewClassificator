# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
torch.set_default_device("cuda")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)

inputs = tokenizer('''

''', return_tensors="pt", return_attention_mask=False)


outputs = model.generate(**inputs, max_length=200)


text = tokenizer.batch_decode(outputs)[0]

print(text)