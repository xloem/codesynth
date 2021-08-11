print('loading ...')
from generate import genji

genji = genji()

while True:
    line = input('input: ')
    print(genji(line, max_length=80)[0]['generated_text'])
