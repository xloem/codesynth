import transformers

# 2k, 4k, 8k, 20k, 40k .  20k is latest.
pipeline = transformers.pipeline('text-generation', 'lg/ghpy_2k')

print(pipeline('def print_customer_name')[0]['generated_text'])
