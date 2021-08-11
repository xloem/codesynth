
class genji:
    def __init__(self):
        import extern.finetuneanon_transformers.src.transformers as finetuneanon
        self.model = finetuneanon.AutoModelForCausalLM.from_pretrained('extern/genji-python-6B-split/model')
        self.model = self.model.half().eval()
        self.model = self._cuda(self.model)

        self.tokenizer = finetuneanon.AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')

    def _cuda(self, obj):
        return obj.cuda()

    def __call__(self, text, use_cache=True, do_sample=True, top_k=50, temperature=0.3, top_p=0.9, repetition_penalty=1.125, min_length=1, pad_token_id=None, **kwparams):
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        tokens = self.tokenizer(text, return_tensors='pt').input_ids
        tokens = self._cuda(tokens.long())
        generated_tokens = self.model.generate(tokens, use_cache=use_cache, do_sample=do_sample, top_k=top_k, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, min_length=min_length, pad_token_id=pad_token_id, **kwparams)
        last_tokens = [gentoks[len(toks):] for gentoks, toks in zip(generated_tokens, tokens)]
        return [{'generated_text':self.tokenizer.decode(tokens)} for tokens in last_tokens]

class ghpy:
    def __init__(self, model='lg/ghpy_20k'):
        import transformers
        self.pipeline = transformers.pipeline('text-generation', model)
        self.model = self.pipeline.model
        self.tokenizer = self.pipeline.tokenizer
    def __call__(self, *params, **kwparams):
        return self.pipeline(*params, **kwparams)

class ghpy_tiny(ghpy):
    def __init__(self, model='lg/ghpy_2k'):
        super().__init__(model)

xloems_ai21_apikey = '5ZqHci4NdEii0VYq9VFBJ8njMRqRiCaR'

class ai21:
    def __init__(self, apikey, model='j1-large'):
        import requests
        self._authorization = 'Bearer ' + apikey
        self._model = model
        self.requests = requests
    def __call__(self, text, num_return_sequences = 1, max_length = 8, top_k = 0, temperature = 0.0, top_p = 1.0, stop_sequences = []):
        result = self.requests.post('https://api.ai21.com/studio/v1/' + self._model + '/complete',
            headers={'Authorization': self._authorization},
            json={
                'prompt': text,
                'numResults': num_return_sequences,
                'maxTokens': max_length,
                'topP': top_p,
                'stopSequences': stop_sequences,
                'topKReturn': top_k,
                'temperature': temperature
            }
        )
        return [{
                'generated_text': text + completion['data']['text'],
                'tokens': completion['data']['tokens'],
                'finishReason': completion['finishReason']
            }
            for completion in result.json()['completions']
        ]
        return result


class ai21_jumbo(ai21):
    def __init__(self, apikey, model='j1-jumbo'):
        super().__init__(model, apikey)
