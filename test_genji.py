from time import time
mark0 = time()
from extern.finetuneanon_transformers.src.transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPTNeoForCausalLM,
)


model = AutoModelForCausalLM.from_pretrained("extern/genji-python-6B-split/model").half().eval().cuda()
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

mark1 = time()
print((mark1 - mark0), 'seconds loading model')

text = '''def print_customer_name'''

tokens = tokenizer(text, return_tensors="pt").input_ids
generated_tokens = model.generate(tokens.long().cuda(), use_cache=True, do_sample=True, top_k=50, temperature=0.3, top_p=0.9, repetition_penalty=1.125, min_length=1, max_length=len(tokens[0]) + 100, pad_token_id=tokenizer.eos_token_id)
last_tokens = generated_tokens[0][len(tokens[0]):]
generated_text = tokenizer.decode(last_tokens)

mark2 = time()
print((mark2 - mark1), 'seconds generating:')

print(text + generated_text)
