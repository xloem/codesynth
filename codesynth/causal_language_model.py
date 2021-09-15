# Copyright (C) 2021
#
# This file is part of codesynth.
#
# codesynth is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# codesynth is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with codesynth.  If not, see <http://www.gnu.org/licenses/>.

import os

class CausalLanguageModel:
    '''
    Causal Language Models
    
    These are models similar to the well-known GPT-3.  They generate arbitrary text when provided
    up to 2048 tokens of preceding context, where a token is roughly a short word.  The generation
    is done token-by-token, shifts out the context, and can be arbitrarily long.
    
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

class transformers_base:
    modelsdir = os.environ.get('TRANSFORMERS_MODELS', os.path.join(os.path.dirname(__file__), '..', 'extern'))
    def __init__(self, transformers, model, tokenizer=None, device=0):
        self.logger = transformers.logging.get_logger()

        if device >= 0:
            try:
                import torch
                torch.tensor([]).cuda()
            except AssertionError as e:
                e.args = (*e.args, 'pass device=-1 to force using the cpu')
                raise

        # first check if it is cached
        try:
            model_org, model_name, *subfolders = model.split('/')
            config_subpath = '/'.join((*subfolders, 'config.json'))
            hf_bucket_url = transformers.file_utils.hf_bucket_url(f'model_org/model_name', config_subpath)
            transformers.file_utils.cached_path(hf_bucket_url, local_files_only = True)
            self.logger.warning('===')
            self.logger.warning('Found %s in %s, using cloud cache.', transformers.file_utils.TRANSFORMERS_CACHE)
            self.logger.warning('===')
        except FileNotFoundError:
            # if not cached, check if local path exists
            localpath = os.path.join(transformers_base.modelsdir, model)
            if os.path.isdir(localpath):
                model = localpath
            else:
                # if not local, output warning and let transformers use its cache
                self.logger.warning('===')
                self.logger.warning('Model %s not found in %s.', model, transformers_base.modelsdir)
                self.logger.warning('Will download model files from cloud.')
                self.logger.warning('')
                self.logger.warning('For model management, do this in a folder:')
                self.logger.warning('\tgit-lfs install --skip-repo')
                self.logger.warning('\tgit clone %s/%s %s', transformers.file_utils.HUGGINGFACE_CO_RESOLVE_ENDPOINT, model, model)
                self.logger.warning('')
                self.logger.warning('Then set one of these to that folder path:')
                self.logger.warning('  the python variable %s.modelsdir, or', transformers_base.__name__)
                self.logger.warning('  the environment variable TRANSFORMERS_MODELS')
                self.logger.warning('and delete %s.', transformers.file_utils.TRANSFORMERS_CACHE)
                self.logger.warning('===')

        try:
            self.pipeline = transformers.pipeline(
                'text-generation',
                model,
                tokenizer=tokenizer,
                device=device
            )
        except OSError as e:
            e.args = (*e.args, 'This can stem from a model clone without a prior `git-lfs install`.')
            raise

        self.model = self.pipeline.model
        self.tokenizer = self.pipeline.tokenizer
    # todo? probably clearer to use where-ever transformers stores the default parameters,
    #   rather than all these defaults. some are probably appropriate model-wide, and some per-model
    def __call__(self, *params, return_full_text=True, min_length=1, max_length=16, use_cache=True, do_sample=True, top_k=50, temperature=0.3, top_p=0.9, repetition_penalty=1.125, **kwparams):
        return self.pipeline(*params, return_full_text=return_full_text, min_length=min_length, max_length=max_length, use_cache=use_cache, do_sample=do_sample, top_k=top_k, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, **kwparams)
    def cpu(self):
        self.model.device.type = 'cpu'
        self.model.device.index = None
        self.pipeline.device = self.model.device
        return self
    def cuda(self, device=0):
        if device < 0:
            return self.cpu()
        else:
            self.model.device.type = 'cuda'
            self.model.device.index = device
            self.pipeline.device = self.model.device
            return self

class finetuneanon(transformers_base, CausalLanguageModel):
    def __init__(self, model='NovelAI/genji-python-6B-split/model', tokenizer='EleutherAI/gpt-neo-2.7B', device=0):
        import finetuneanon_transformers_gn_la3_rpb as finetuneanon
        super().__init__(finetuneanon, model, tokenizer, device)

class genji(finetuneanon):
    def __init__(self, model='NovelAI/genji-python-6B-split/model', tokenizer='EleutherAI/gpt-neo-2.7B', device=0):
        super().__init__(model, tokenizer, device)

class stellaathena(transformers_base, CausalLanguageModel):
    def __init__(self, model='EleutherAI/gpt-j-6B', *params, **kwparams):
        import stellaathena_transformers as stellaathena
        super().__init__(stellaathena, model, *params, **kwparams)

class gptj6b(stellaathena):
    def __init__(self, model='EleutherAI/gpt-j-6B', *params, **kwparams):
        super().__init__(model, *params, **kwparams)

class huggingface(transformers_base, CausalLanguageModel):
    def __init__(self, model=None, *params, **kwparams):
        import transformers
        super().__init__(transformers, model, *params, **kwparams)

class ghpy(huggingface):
    def __init__(self, model='lg/ghpy_20k', *params, **kwparams):
        super().__init__(model, *params, **kwparams)

class ghpy_tiny(ghpy):
    def __init__(self, model='lg/ghpy_2k', *params, **kwparams):
        super().__init__(model, *params, **kwparams)

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

class rpc_client:
    def __init__(self, model='genji', url='http://127.0.0.1:6686'):
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

class rate_limited:
    class TemporaryChange(Exception):
        pass
    def __init__(self, duration):
        import time
        self.time = time
        self._duration = duration
        self._mark = self.time.time()
    def wait(self):
        now = self.time.time()
        diff = self.wait_needed()
        if diff > 0:
            #print('sleeping for', diff)
            self.time.sleep(diff)
        #else:
        #    print('no sleep needed')
        self._mark += self._duration
    def wait_needed(self):
        return max(0,self._mark + self._duration - self.time.time())

class eleuther_demo(rate_limited, CausalLanguageModel):
    def __init__(self, url='https://api.eleuther.ai/completion'):
        super().__init__(5)
        import requests
        self.url = url
        self.requests = requests
    def tokenizer(self, text):
        return {
            'input_ids': [ text ]
        }
    def __call__(self, texts, max_new_tokens = 128, top_p = 1, temperature = 0, return_full_text = True, eos_token_id = None):
        if type(texts) is str:
            texts = [texts]
        final_results = []
        for text in texts:
            if len(text) == 0:
                raise AssertionError('eleutherai demo needs a prompt')
            tailtrim = 3072
            json=dict(
                context=text[-tailtrim:],
                top_p=top_p,
                temp=temperature,
                remove_input=not return_full_text
            )
            mark = self._mark
            while True:
                self.wait()
                response = self.requests.post(self.url,
                    json=json
                )
                if response.status_code == 500:
                    tailtrim -=4
                elif response.status_code != 503 and response.status_code != 502:
                    break
                else:
                    print('rate limit on eleuther, delay = ', self.time.time() - mark, 'next attempt at', self._mark + self._duration - mark)
                    raise rate_limited.TemporaryChange('exceeded eleuther rate limit')
                #import time
                #time.sleep(5)
            try:
                response.raise_for_status()
                results = response.json()
            except Exception as e:
                print(json)
                e.args = (*e.args, response.text)
                raise
            if eos_token_id is not None:
                for result in results:
                    text = result['generated_text']
                    offset = text.find(eos_token_id)
                    if offset >= 0:
                        offset += len(eos_token_id)
                        text = text[:offset]
                        result['generated_text'] = text
            final_results.extend(results)
        return final_results

_global_seed = 0

class bellard_demo(rate_limited, CausalLanguageModel):
    def __init__(self, model='gptj_6B', url='https://bellard.org/textsynth/api/v1/engines'):
        super().__init__(21)
        import json, requests
        self.url = url + '/' + model + '/completions'
        self.json = json
        self.requests = requests
    def tokenizer(self, text):
        return {
            'input_ids': [ text ]
        }
    def __call__(self, texts, max_new_tokens = 128, top_p = 0.9, top_k = 40, temperature = 1, seed = None, return_full_text = True, eos_token_id = None):
        if type(texts) is str:
            texts = [texts]
        if seed is None:
            global _global_seed
            seed = _global_seed
            _global_seed += 1
        final_results = []
        for text in texts:
            #if len(text) == 0:
            #    raise AssertionError('eleutherai demo needs a prompt')
            tailtrim = 3072
            result = ''
            token_ct = 0
            done = False
            prompt = text[-tailtrim:]
            while not done:
                json=dict(
                    prompt=prompt,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    seed=seed,
                    stream=True
                )
                while True:
                    self.wait()
                    response = self.requests.post(self.url,
                        json=json,
                        stream=True
                    )
                    self._mark = self.time.time() + self._duration
                    if response.status_code == 509:
                        # rate limit exceeded
                        print('bellard rate limit hit =/')
                        self._mark += 60 *30
                        raise rate_limited.TemporaryChange('bellard rate limit exceeded')
                    elif response.status_code == 400:
                        # i'm guessing this is a server bug where it produces unicode output that it can't handle
                        print('bellard status code 400, unicode output produced?')
                        raise rate_limited.TemporaryChange('bellard status code 400, unicode output produced?')
                    else:
                        break
                try:
                    response.raise_for_status()
                except Exception as e:
                    print(response.text)
                    print(json)
                    e.args = (*e.args, response.text)
                    raise
                guesstoken_ct = 0
                for line in response.iter_lines():
                    if len(line.strip()):
                        line = self.json.loads(line)
                        if 'total_tokens' in line:
                            token_ct += line['total_tokens']
                            guesstoken_ct = 0
                        else:
                            guesstoken_ct += 3
                        #print(token_ct, token_ct + guesstoken_ct, max_new_tokens, line)
                        portion = line['text']
                        result += portion
                        if (eos_token_id is not None and eos_token_id in result) or token_ct + guesstoken_ct >= max_new_tokens:
                            done = True
                            break
                if not done:
                    prompt = (text + result)[-tailtrim:]
                
            if eos_token_id is not None:
                    offset = result.find(eos_token_id)
                    if offset >= 0:
                        offset += len(eos_token_id)
                        result = result[:offset]
            final_results.append({'generated_text': result})
        return final_results

class multi_demo(rate_limited, CausalLanguageModel):
    def __init__(self, *submodels):
        super().__init__(0)
        self.submodels = submodels
    def min_wait_model(self):
        min = None
        for model in self.submodels:
            wt = model.wait_needed()
            if min is None or wt < min:
                min = wt
                minmodel = model
        #print('min model is ', minmodel.__class__.__name__, 'all are', [model.wait_needed() for model in self.submodels])
        return min, minmodel
    def wait_needed(self):
        return self.min_wait_model()[0]
    def tokenizer(self, text):
        return self.min_wait_model()[1].tokenizer(text)
    def __call__(self, *params, **kwparams):
        while True:
            try:
                self.submodels = [*self.submodels[1:], self.submodels[0]]
                return self.min_wait_model()[1](*params, **kwparams)
            except rate_limited.TemporaryChange:
                pass

MODELS = {
    name: model
    for name, model in globals().items()
    if model is not CausalLanguageModel
        and type(model) is type
        and issubclass(model, CausalLanguageModel)
}
