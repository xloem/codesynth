#!/usr/bin/env python3

import sys
sys.stderr.write('Loading ...\n')
import generate
genji = generate.genji()

if len(sys.argv) <= 1:
    sys.stderr.write('Reading ...\n')
    data = sys.stdin.read()
    sys.stderr.write('Generating ...\n')
    # could have it start outputing during pauses
    result = genji(data)
    print(result[0]['generated_text'])
else:
    while True:
        sys.stderr.write('Reading ...\n')
        data = []
        for fn in sys.argv[1:]:
            with open(fn, 'rt') as f:
                data.append(f.read())
        sys.stderr.write('Generating ...\n')
        results = genji(data)
        for fn, result in zip(sys.argv[1:], results):
            with open(fn, 'at') as f:
                f.write(result['generated_text'])
        print('Files appended to.  Will reprocess files if a linebreak is received on stdin.')
        if not sys.stdin.readline():
            break
    
