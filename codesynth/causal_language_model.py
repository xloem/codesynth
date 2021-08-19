import os

class CausalLanguageModel:
    '''
    Causal Language Models
    
    These are models similar to the well-known GPT-3.  They generate arbitrary text when provided
    up to 2048 tokens of preceding context, where a token is roughly a short word.  The generation
    is done token-by-token and can be arbitrarily long.
    
    The interface for these classes is based on the text generation pipeline interface from the
    transformers library.  An object may be called like a function to generate text, and has a
    .tokenizer member which may be called like a function to convert text into tokens.
    
    Example:
    
    import codesynth
    
    # load the genji model, takes some time
    generator = codesynth.genji()
    
    # generate text that follows a prompt
    text = generator("def print_hello_world")
    print(text)
    '''
    
    def __call__(self, text, use_cache=True, do_sample=True, top_k=50, temperature=0.3, top_p=0.9, repetition_penality=1.125, min_length=1, return_full_text=True, pad_token_id=None, **kwparams):
        raise NotImplementedError(self)

    def tokenizer(self, text):
        raise NotImplementedError(self)

    abstract = True

## split models
# split models have a file in their folder called transformers.file_utils.SPLIT_WEIGHTS_NAME='m.pt'
# SplitCheckpoint is the class used as the state_dict.  it can take the m.pt file or its
#  parent folder.  it mutates the m.pt file into its parent folder, and loads the m.pt file
#  via torch.load .  torch.load returns a dict, the values of which are tensor filenames
#  the tensor filenames are given in nonexistent subdirs, so SplitCheckpoint uses Path(val).name
#  to convert them to filenames and load them.
#  we could make a remote SplitCheckpoint by subclassing and overriding getitem to use the cache.

try:
    from collections.abc import MutableMapping
except ImportError:
    from collections import MutableMapping
class CachedSplitCheckpoint(MutableMapping):
    '''
    This mimics a transformers class for loading models, and provides for
    loading split models from the hugging face servers using the transformers cache.
    Otherwise, at time of writing, every split model needed to be included with git-lfs.
    '''
    def __init__(self, file_utils, modelnamepath, device="cpu"):
        import torch
        self._file_utils = file_utils
        self._torch =  torch
        self._modelpath = modelnamepath + '/'
        self.device = device
        self.checkpoint = self._load(self._file_utils.SPLIT_WEIGHTS_NAME)
    def _load(self, modelpath, **kwparams):
        bucket_url = self._file_utils.hf_bucket_url(self._modelpath + modelpath)
        cached_path = self._file_utils.cached_path(bucket_url)
        return torch.load(cached_path, **kwparams)
    def __getitem__(self, key):
        return self._load(str(path), map_location=self.device)
    def __copy__(self):
        return CachedSplitCheckpoint(self._file_utils, self._modelpath, self.device)
    def copy(self):
        return CachedSplitCheckpoint(self._file_utils, self._modelpath, self.device)
    # these methods are just copied from SplitCheckpoint
    def __setitem__(self, key, value):
        return
    def __delitem__(self, key, value):
        return
    def keys(self):
        return self.checkpoint.keys()
    def __iter__(self):
        for key in self.checkpoints:
            yield (key, self.__getitem__(key))
def finetuneanon_model(model):
    import finetuneanon_transformers as finetuneanon
    if os.path.exists('extern/' + model + '/model'):
        # TODO: move genji-python-6B-split into NovelAI subfolder
        self.model = finetuneanon.AutoModelForCausalLM.from_pretrained('extern/' + model + '/model')
    else:
        from finetuneanon_transformers.file_utils import cached_path, hf_bucket_url, TRANSFORMERS_CACHE

class finetuneanon_split(CausalLanguageModel):
    def __init__(self, model='NovelAI/genji-python-6B-split'):
        import finetuneanon_transformers as finetuneanon
        if os.path.exists('extern/' + model + '/model'):
            # TODO: move genji-python-6B-split into NovelAI subfolder
            self.model = finetuneanon.AutoModelForCausalLM.from_pretrained('extern/' + model + '/model')
        else:
            import finetuneanon_transformers.file_utils as file_utils
            state_dict = CachedSplitCheckpoint(finetuneanon_transformers.file_utils, model + '/model')
            config_file = file_utils.cached_path(file_utils.hf_bucket_url(model + '/model/config.json'))
            config = finetuneanon.PretrainedConfig.from_json_file(config_file)
            self.model = finetuneanon.AutoModelForCausalLM.from_pretrained(model, state_dict=state_dict, config=config)

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

class hf(CausalLanguageModel):
    def __init__(self, model=None):
        import transformers
        self.pipeline = transformers.pipeline('text-generation', model)
        self.model = self.pipeline.model
        self.tokenizer = self.pipeline.tokenizer
    def __call__(self, *params, **kwparams):
        return self.pipeline(*params, **kwparams)

class ghpy(hf):
    def __init__(self, model='lg/ghpy_20k'):
        super().__init__(model)

class ghpy_tiny(ghpy):
    def __init__(self, model='lg/ghpy_2k'):
        super().__init__(model)

class ai21(CausalLanguageModel):
    apikey = os.environ.get('AI21_API_KEY')
    def __init__(self, model='j1-large', apikey=None):
        if apikey is None:
            apikey = ai21.apikey
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
        return result
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
    def __call__(self, text, num_return_sequences = 1, max_length = None, max_new_tokens = 8, top_k = 0, temperature = 0.0, top_p = 1.0, return_full_text = True, eos_token_id = None):
        if eos_token_id is not None:
            stop_sequences = [eos_token_id]
        else:
            stop_sequences = []
        reqparams = dict(
            numResults = num_return_sequences,
            maxTokens = max_new_tokens or (max_length - 1),
            topP = top_p,
            stopSequences = stop_sequences,
            # if topKReturn is > 0.0 then alternative tokens for both the prompt and completion
            #   are returned.
            topKReturn = top_k,
            temperature = temperature
        )
        if type(text) is str:
            results = [self._request(prompt = text, **reqparams)]
        else:
            results = [self._request(prompt = prompt, **reqparams) for prompt in text]
        final_results = []
        for result in results:
            final_result = []
            prompt = result['prompt']
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
            final_results.append(final_result)
        if len(final_results) == 1:
            return final_results[0]
        else:
            return final_results

class ai21_jumbo(ai21):
    def __init__(self, model='j1-jumbo', apikey=None):
        super().__init__(model, apikey)

class openai(CausalLanguageModel):
    apikey = os.environ.get('OPENAI_API_KEY')
    def __init__(self, engine='davinci', apikey=None):
        if apikey is None:
            apikey = openai.apikey
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
        return result
    def tokenizer(self, text):
        return { 'input_ids': [ text ] }
    def __call__(self, text, num_return_sequences = 1, max_length = None, max_new_tokens = 8, top_k = 0, temperature = 0.0, top_p = 1.0, return_full_text = True, eos_token_id = None):
        if type(text) is str or text is None:
            prompts = [text]
        else:
            prompts = text
        # supports streaming
        result = self._request(
            prompt = text,
            max_tokens = max_new_tokens or (max_length - 1),
            temperature = temperature,
            top_p = top_p,
            n = num_return_sequences,
            stop = eos_token_id,
        )
        choices = result['choices']
        choice_idx = 0
        final_results = []
        for result_idx, text in enumerate(prompts):
            final_result = []
            for choice in choices[choice_idx:choice_idx + num_return_sequences]:
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
            choice_idx += len(final_result)
            final_results.append(final_result)
        if len(final_results) == 1:
            return final_results[0]
        else:
            return final_results

class rpc_client(CausalLanguageModel):
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
    if model is not CausalLanguageModel and issubclass(model, CausalLanguageModel)
}
