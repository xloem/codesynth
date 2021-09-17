FROM python:3.8

WORKDIR /discord_bot

RUN [ "pip3", "install", "discord", "pandas", "requests", "uri", "http.request", "google", "ticker", "yahoo_finance" ]

# some files are presently excluded for speed in .dockerignore
COPY codesynth ./codesynth
COPY *.py ./
RUN [ "mkdir", "-p", "ctxs" ]

CMD [ "python3", "discord_bot.py" ]
