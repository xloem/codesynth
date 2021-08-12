#!/usr/bin/env python3


import os

import generate

class Instructions:
    def __init__(self, pipeline, dir = 'assistant/'):
        self.dir = os.path.abspath(dir)
        self.pipeline = pipeline
        self.eos_token_id = self.pipeline.tokenizer('##')['token_ids'][-1]
        os.makedirs(self.dir)
    def _all_data(self):
        return open(os.path.join(self.dir, 'instructions.txt')).read()
    def prompt(self, prompt):
        data = self._all_data() + ' ' + prompt + '\n'
        result = self.pipeline(data, return_full_text=False, eos_token_id = self.eos_token_id, max_length=1024)
        result = [item['generated_text'] for item in result]
        return result[0]


if __name__ == '__main__':
    instructions = Instructions()
