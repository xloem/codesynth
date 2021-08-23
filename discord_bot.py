import asyncio
from collections import defaultdict
import os
import traceback

import codesynth
import discord

discord_token = os.environ['DISCORD_TOKEN']

class bot:
    def __init__(self, token, model):
        self.model = model
        self.client = discord.Client()
        self.client.event(self.on_ready)
        self.client.event(self.on_message)
        self.token = token

        self.channels_pending = defaultdict(list)
        self.channels_history = defaultdict(list)
        self.new_messages = asyncio.Event()
        self.start_replying = asyncio.Event()

    def run(self):
        loop = self.client.loop
        try:
            loop.run_until_complete(self.loop())
        finally:
            loop.close()

    async def loop(self):
        try:
            await asyncio.gather(self.client.start(self.token), self.pump())#asyncio.to_thread(self.pump))
        except:
            await self.client.close()
            raise

    async def pump(self):
        await self.start_replying.wait()
        while True:
            found = False
            for channel, msgs in self.channels_pending.items():
                if len(msgs):
                    history = self.channels_history[channel]
                    history.extend((f'{msg.author}: {msg.content}' for msg in msgs))
                    found = True
                    if msgs[-1].author != self.client.user:
                        prompt = '\n'.join(history) + f'\n{self.client.user}:'
                        try:
                            reply = self.model(
                                prompt,
                                eos_token_id='\n',
                                return_full_text=False,
                                max_new_tokens=1024,
                                #top_p=0.25
                                temperature=1.0
                            )[0]['generated_text'].strip()
                        except Exception as e:
                            reply = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                        await channel.send(reply)
                    msgs.clear()
            if not found:
                self.new_messages.clear()
                await self.new_messages.wait()

    async def on_ready(self):
        print('We have logged in as {0.user}'.format(self.client))
        for channel in self.client.get_all_channels():
            if type(channel) is discord.TextChannel:
                async for message in channel.history(limit=1024, oldest_first=True):
                    await self.on_message(message)
        self.start_replying.set()
    
    async def on_message(self, message):
        print(message.channel, message.author, message.content)
        self.channels_pending[message.channel].append(message)
        if message.author == self.client.user:
            return

        self.new_messages.set()

        #if message.content.startswith('$hello'):
        #    await message.channel.send('Hello!')

#model = codesynth.ai21_jumbo()
model = codesynth.eleuther_demo()
bot(discord_token, model).run()

