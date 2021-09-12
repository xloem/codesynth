import asyncio
from collections import defaultdict
from dataclasses import dataclass
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

class emoji:
    thumbsup = '👍'
    thumbsdown = '👎'
    smiley = '😃'
    poop = '💩'
    plusone = thumbsup + smiley
    minusone = thumbsdown

class bot:
    class channel:
        def __init__(self):
            self.maxscore = 0
            self.pending = []
            self.history = []
            self.can_talk = False
    def __init__(self, token, model):
        self.model = model
        self.client = discord.Client()
        self.client.event(self.on_ready)
        self.client.event(self.on_message)
        self.client.event(self.on_raw_reaction_add)
        #self.client.event(self.on_raw_reaction_remove)
        self.token = token

        self.nonself_end_of_line_token = '~~'

        self.channels = defaultdict(bot.channel)
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

    def msgscore(self, msg):
        score = 0
        for reaction in msg.reactions:
            if reaction.emoji in emoji.plusone:
                score += reaction.count
            elif reaction.emoji in emoji.minusone:
                score -= reaction.count
        return score

    def scorestr(self, score):
        if score < 0:
            str = 'bad'
        elif score > 0:
            str = 'good'
        else:
            str = 'soso'
        return f'{str} {score}:'

    def msg2history(self, msg, chandata):
        return f'{msg.author}: {self.scorestr(self.msgscore(msg))} {msg.content}'
    def usr2history(self, user, chandata = None):
        score = self.scorestr(chandata.maxscore) if chandata is not None else ''
        return f'{user}: {score} '

    async def pump_in(self):
        #print('pump in loop')
        await asyncio.sleep(0)
        found = False
        for name, channel in self.channels.items():
            if len(channel.pending):
                found = True
                while len(channel.pending):
                    msg = channel.pending.pop(0)
                    if msg.content.strip():
                        #print('adding to history:', msg.author, msg.content)
                        if not channel.can_talk and (self.name + ', you can talk') in msg.content:
                            channel.can_talk = True
                        channel.history.append(msg)
        return found
    @property
    def name(self):
        return str(self.client.user).split('#')[0]
    async def pump_out(self):
        #print('pump out start')
        await self.start_replying.wait()
        while True:
            #print('pump out loop')
            found = await self.pump_in()
            for channel, chandata in self.channels.items():
                #print(channel, 'talk =', talk, 'len(history) =', len(history))
                #if chandata.can_talk:
                #    print(channel, 'score of last message =', self.msgscore(chandata.history[-1]))
                if chandata.can_talk and (
                    chandata.history[-1].author != self.client.user or
                    self.msgscore(chandata.history[-1]) < 0
                ):
                    #print('responding to', history[-1].author, history[-1].content)
                    found = True
                    try:
                        removect = 0
                        await self.pump_in()
                        prompt = '\n'.join([self.msg2history(msg, chandata) for msg in list_randshrink(chandata.history[-1024:], removect)])
                        chandata.maxscore = max(0,max((self.msgscore(msg) for msg in chandata.history[-16:])))
                        preprompt = '\n' + self.usr2history(self.client.user, chandata).strip()
                        prompt += preprompt
                        async with channel.typing():
                            reply = (await asyncify(self.model)(
                                prompt.strip(),
                                eos_token_id=self.nonself_end_of_line_token,
                                return_full_text=False,
                                max_new_tokens=1024,
                                #top_p=0.25
                                #temperature=1.0
                            ))[0]['generated_text'].strip()
                        print(prompt[-256:])
                        print('considering:', preprompt + ' ' + reply)
                        # quick fix: remove items from prompt to change context
                        if removect < len(chandata.history):
                            removect += 1

                        # for multiline: read up until another message is expected
                        lines = reply.split('\n')
                        reply = ''
                        for idx, line in enumerate(lines):
                            if '#' in line: # hacky way to identify that a line is message
                                name, bit = line.split('#', 1)
                                if ':' in bit:
                                    if bit.split(':')[0].isnumeric() and all((not char.isspace() for char in name)):
                                        break
                        reply = '\n'.join(lines[:idx])
                    except Exception as e:
                        reply = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                    if len(reply) == 0:
                        reply = '[empty message??]'
                    await channel.send(reply)
            if not found:
                self.new_messages.clear()
                await self.new_messages.wait()

    async def on_ready(self):
        print('We have logged in as {0.user}'.format(self.client))
        for channel in self.client.get_all_channels():
            print('channel:', channel)
            if type(channel) is discord.TextChannel:
                async for message in channel.history(limit=1024, oldest_first=True):
                    #print(channel, message.channel, message.author, message.content)
                    await self.on_message(message)
        self.nonself_end_of_line_token = self.usr2history(self.client.user)
        self.start_replying.set()

    async def delmsg(self, message):
        message.content = ''
        await message.delete()
    
    async def on_message(self, message):
        print(message.channel, message.author, 'in response to =', message.reference, ':', message.content)
        if message.reference is not None and (message.content.startswith(f'{self.name}, replace with: ') or message.content.lower().startswith('replace: ')):
            newcontent = message.content[len(message.content.split(': ', 1)[0]):]
            await message.reference.resolved.edit(content = newcontent)
            print('UPDATED CONTENT:', message.reference.resolved.content)
        elif message.reference is not None and (message.content == f'{self.name}, delete.' or message.content.lower().strip() == 'delete'):
            await self.delmsg(message.reference.resolved)
        else:
            self.channels[message.channel].pending.append(message)

        self.new_messages.set()

    async def on_raw_reaction_add(self, payload):
        if str(payload.emoji) == emoji.poop:
            for channel, chandata in self.channels.items():
                if channel.id == payload.channel_id:
                    for message in (*chandata.pending, *chandata.history):
                        if message.id == payload.message_id:
                            await self.delmsg(message)
                            break
        self.new_messages.set()
        print('reaction', str(payload.emoji))

#model = codesynth.ai21_jumbo()
model = codesynth.eleuther_demo()
#model = codesynth.openai()
if __name__ == '__main__':
    bot(discord_token, model).run()

