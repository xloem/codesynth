import asyncio
from collections import defaultdict
import functools
import os
import traceback

import codesynth
import discord
import random

discord_token = os.environ['DISCORD_TOKEN']

def list_randshrink(list, count):
    result = [item for item in list]
    for idx in range(count):
        result.pop(random.randint(0, len(result)-1))
    return result

def asyncify(func):
    def asynced(*params, **kwparams):
        return asyncio.get_running_loop().run_in_executor(None, functools.partial(func, **kwparams), *params)
    return asynced

class bot:
    def __init__(self, token, model):
        self.model = model
        self.client = discord.Client()
        self.client.event(self.on_ready)
        self.client.event(self.on_message)
        self.token = token

        self.nonself_end_of_line_token = '~~'

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
            await asyncio.gather(self.client.start(self.token), self.pump_out())
        except:
            await self.client.close()
            raise

    def msg2history(self, msg):
        return f'{msg.author}: {msg.content}'
    def usr2history(self, user):
        return f'{user}:'

    async def pump_in(self):
        #print('pump in loop')
        await asyncio.sleep(0)
        found = False
        for channel, msgs in self.channels_pending.items():
            if len(msgs):
                history = self.channels_history[channel]
                found = True
                while len(msgs):
                    msg = msgs.pop(0)
                    if msg.content.strip():
                        #print('adding to history:', msg.author, msg.content)
                        history.append(msg)
        if not found:
            self.new_messages.clear()
        return found
    async def pump_out(self):
        #print('pump out start')
        await self.start_replying.wait()
        while True:
            #print('pump out loop')
            found = await self.pump_in()
            for channel, history in self.channels_history.items():
                talk = False
                for msg in history:
                    if (str(self.client.user).split('#')[0] + ', you can talk') in msg.content:
                        talk = True
                        break
                #print(channel, 'talk =', talk, 'len(history) =', len(history))
                if talk and history[-1].author != self.client.user:
                    #print('responding to', history[-1].author, history[-1].content)
                    found = True
                    preprompt = '\n' + self.usr2history(self.client.user).strip()
                    try:
                        removect = 0
                        reply = "don't say that"
                        while "don't say that" in reply.lower() or "stop saying that" in reply.lower():
                            await self.pump_in()
                            prompt = '\n'.join([self.msg2history(msg) for msg in list_randshrink(history, removect)]) + preprompt
                            reply = (await asyncify(self.model)(
                                prompt.strip(),
                                eos_token_id=self.nonself_end_of_line_token,
                                return_full_text=False,
                                max_new_tokens=1024,
                                #top_p=0.25
                                temperature=1.0
                            ))[0]['generated_text'].strip()
                            print('considering:', reply)
                            # quick fix: remove items from prompt to change context
                            if removect < len(history):
                                removect += 1
                        reply = reply.split('\n')[0]
                    except Exception as e:
                        reply = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                    if len(reply) == 0:
                        reply = '[empty message??]'
                    await channel.send(reply)
            if not found:
                await self.new_messages.wait()

    async def on_ready(self):
        print('We have logged in as {0.user}'.format(self.client))
        for channel in self.client.get_all_channels():
            print('channel:', channel)
            if type(channel) is discord.TextChannel:
                async for message in channel.history(limit=1024, oldest_first=True):
                    print(channel, message.channel, message.author, message.content)
                    await self.on_message(message)
        self.nonself_end_of_line_token = self.usr2history(self.client.user)
        self.start_replying.set()
    
    async def on_message(self, message):
        print(message.channel, message.author, 'in response to =', message.reference, ':', message.content)
        self.channels_pending[message.channel].append(message)
        #if message.author == self.client.user:
        #    return

        self.new_messages.set()

        #if message.content.startswith('$hello'):
        #    await message.channel.send('Hello!')

#model = codesynth.ai21_jumbo()
model = codesynth.eleuther_demo()
#model = codesynth.openai()
bot(discord_token, model).run()

