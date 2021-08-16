class genji:
    def __init__(self):
        import extern.finetuneanon_transformers.src.transformers as finetuneanon
        self.model = finetuneanon.AutoModelForCausalLM.from_pretrained('extern/genji-python-6B-split/model')
        self.model = self.model.half().eval()
        self.model = self._cuda(self.model)

        self.tokenizer = finetuneanon.AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')

    def _cuda(self, obj):
        return obj.cuda()

    def __call__(self, text, use_cache=True, do_sample=True, top_k=50, temperature=0.3, top_p=0.9, repetition_penalty=1.125, min_length=1, return_full_text = True, pad_token_id=None, **kwparams):
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        tokens = self.tokenizer(text, return_tensors='pt').input_ids
        tokens = self._cuda(tokens.long())
        generated_tokens = self.model.generate(tokens, use_cache=use_cache, do_sample=do_sample, top_k=top_k, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, min_length=min_length, pad_token_id=pad_token_id, **kwparams)
        if return_full_text:
            last_tokens = generated_tokens
        else:
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

class ai21:
    def __init__(self, apikey, model='j1-large'):
        import requests
        self._authorization = 'Bearer ' + apikey
        self._model = model
        self.requests = requests
    def _request(self, **json):
        result = self.requests.post('https://api.ai21.com/studio/v1/' + self._model + '/complete',
            headers={'Authorization': self._authorization},
            json=json
        ).json()
        if 'detail' in result and len(result.keys()) == 1:
            raise RuntimeError(*result['detail'])
    def tokenizer(self, text):
        #result = self._request(
        #    prompt = text,
        #    maxTokens = 0
        #)
        #return {
        #    'input_ids': [
        #        token['generatedToken']['token']
        #        # also has logprob and textRange
        #        for token in result['prompt']['tokens']
        #    ]
        #}
        return {
            'input_ids': [ text ]
        }
    def __call__(self, text, num_return_sequences = 1, max_length = 8, top_k = 0, temperature = 0.0, top_p = 1.0, return_full_text = True, eos_token_id = None):
        if eos_token_id is not None:
            stop_sequences = [eos_token_id]
        else:
            stop_sequences = []
        result = self._request(
            prompt = text,
            numResults = num_return_sequences,
            maxTokens = max_length,
            topP = top_p,
            stopSequences = stop_sequences,
            # if topKReturn is > 0.0 then alternative tokens for both the prompt and completion
            #   are returned.
            topKReturn = top_k,
            temperature = temperature
        )
        prompt = result['prompt']
        final_result = []
        for completion in result['completions']:
            if return_full_text:
                generated_text = prompt['text']
                generated_tokens = prompt['tokens']
            else:
                generated_text = ''
                generated_tokens = []
            generated_text += completion['data']['text']
            generated_tokens += completion['data']['tokens']
            if completion['finishReason']['reason'] == 'stop':
                completion_sequence = completion['finishReason']['sequence']
                generated_text += completion_sequence
                #generated_tokens += [completion_sequence]
            next_result = {
                'prompt_text': prompt['text'],
                'prompt_tokens': prompt['tokens'],
                'generated_text': generated_text,
                'tokens': generated_tokens,
                'finish_reason': completion['finishReason']
            }
            final_result.append(next_result)
        return final_result

class ai21_jumbo(ai21):
    def __init__(self, apikey, model='j1-jumbo'):
        super().__init__(model, apikey)

class openai:
    def __init__(self, apikey, engine='davinci'):
        import requests
        self._authorization = 'Bearer ' + apikey
        self._engine = engine
        self.requests = requests
    def engines(self):
        return self.requests.get('https://api.openai.com/v1/engines',
            headers={'Authorization': self._authorization}
        ).json()['data']
    def _request(self, **json):
        result = self.requests.post('https://api.openai.com/v1/engines/' + self._engine + '/completions',
            headers={'Authorization': self._authorization},
            json=json
        ).json()
        if 'error' in result:
            raise RuntimeError(*result['error'].items())
    def tokenizer(self, text):
        return { 'input_ids': [ text ] }
    def __call__(self, text, num_return_sequences = 1, max_length = 8, top_k = 0, temperature = 0.0, top_p = 1.0, return_full_text = True, eos_token_id = None):
        # supports streaming
        result = self._request(
            prompt = text,
            max_tokens = max_length,
            temperature = temperature,
            top_p = top_p,
            n = num_return_sequences,
            stop = eos_token_id,
        )
        final_result = []
        for choice in result['choices']:
            if return_full_text:
                generated_text = text
            else:
                generated_text = ''
            generated_text += choice['text']
            final_result.append({
                'generated_text': generated_text,
                **{k:v for k,v in result.items() if k != 'choices'},
                **{k:v for k,v in choice.items() if k != 'text'}
            })
        return final_result

class rpc_client:
    def __init__(self, model='genji', url='http://127.0.0.1/'):
        import requests
        self.model = model
        self.url = url
        self.requests = requests
    def _request(self, method, **kwparams):
        return self.requests.post(self.url,
            json=dict(
                jsonrpc="2.0",
                method=method,
                id=0,
                params=kwparams
            )
        ).json()['result']
    def tokenizer(self, text, **kwparams):
        return self._request('tokenizer', text=text, model=self.model, **kwparams)
    def __call__(self, text, **kwparams):
        return self._request('generate_text', text=text, model=self.model, **kwparams)

MODELS = {
    name: model
    for name, model in globals().items()
    if type(model) is type
}
