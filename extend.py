#!/usr/bin/env python3

import argparse
import sys
import inspect

import generate

parser = argparse.ArgumentParser(description='extend code or text')
parser.add_argument('--model', default='genji', help=','.join(generate.MODELS.keys()))
parser.add_argument('--apikey', default=None, help='needed for openai or ai21')
parser.add_argument('--tokens', default=8, type=int, help='tokens to generate')
parser.add_argument('files', default=[sys.stdin], nargs='*', type=argparse.FileType('a+'), help='files to mutate if not stdio')
params = parser.parse_args()

sys.stderr.write('Loading ...\n')
model = generate.MODELS[params.model]
modparams = dict()
if params.apikey is not None and 'apikey' in inspect.signature(model).parameters:
    modparams['apikey'] = params.apikey
model = model(**modparams)

if params.files[0] is sys.stdin:
    sys.stderr.write('Reading ...\n')
    data = params.files[0].read()
    sys.stderr.write('Generating ...\n')
    # could have it start outputing during pauses
    sys.stdout.write(data)
    result = model(data, max_new_tokens=params.tokens, return_full_text=False)
    sys.stdout.write(result[0]['generated_text'])
else:
    while True:
        sys.stderr.write('Reading ...\n')
        data = []
        for f in params.files:
            f.seek(0)
            data.append(f.read())
        sys.stderr.write('Generating ...\n')
        results = model(data, max_new_tokens=params.tokens, return_full_text=False)
        if len(data) == 1:
            results = [results]
        for f, result in zip(params.files, results):
            f.write(result[0]['generated_text'])
        print('Files appended to.  Will reprocess files if a linebreak is received on stdin.')
        if not sys.stdin.readline():
            break
    
