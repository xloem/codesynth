FROM python:3.8

WORKDIR /discord_bot

RUN [ "apt-get", "update" ]
RUN [ "apt-get", "dist-upgrade", "-y" ]
RUN [ "apt-get", "install", "-y", "libpng-dev", "libjpeg-dev", "build-essential", "cmake" ]
RUN [ "pip3", "install", "discord", "pandas", "requests", "uri", "http.request", "ps", "google", "ticker", "yahoo_finance", "sympy", "pillow", "torch", "transformers" ]

# some files are presently excluded for speed in .dockerignore
COPY codesynth ./codesynth
COPY *.py ./
RUN [ "mkdir", "-p", "ctxs" ]

CMD [ "python3", "discord_bot.py" ]
