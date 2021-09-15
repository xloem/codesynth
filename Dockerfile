FROM python:3.8

WORKDIR /discord_bot

RUN [ "pip3", "install", "discord", "requests" ]

# some files are presently excluded for speed in .dockerignore
COPY codesynth ./codesynth
COPY *.py ./

CMD [ "python3", "discord_bot.py" ]
