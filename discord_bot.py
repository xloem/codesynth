import asyncio
from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime
import functools
import io
import os
import pickle
import random
import sys
import traceback

import codesynth
import discord
import json

discord_token = os.environ['DISCORD_TOKEN']
allow_exec = bool(os.environ.get('DISCORD_EXEC'))

def list_randshrink(list, count):
    result = [item for item in list]
    for idx in range(count):
        result.pop(random.randint(0, len(result)-1))
    return result

def asyncify(func):
    def asynced(*params, **kwparams):
        return asyncio.get_running_loop().run_in_executor(None, functools.partial(func, **kwparams), *params)
    return asynced

def err2str(error):
    return ''.join(traceback.format_exception(type(error), error, error.__traceback__))

class emoji:
    _list = None
    def random():
        if emoji._list is None:
            import requests
            emojis = requests.get('https://unicode.org/Public/emoji/14.0/emoji-test.txt')
            emoji._list = [line for line in emojis.text.split('\n') if len(line) and line[0] in set('0123456789ABCDEFabcdef')]
        import random
        result = random.choice(emoji._list)
        result = result.split(';')[0].strip()
        result = ''.join((chr(int(code, 16)) for code in result.split(' ')))
        return result
    repeat = chr(0x1F501)
    thinking = chr(0x1F914)
    scissors = chr(0x2702)
    knife = chr(0x1F52A)
    running = chr(0x1F3C3)
    thumbsup = '👍'
    thumbsdown = '👎'
    smiley = '😃'
    poop = '💩'
    plusone = thumbsup + smiley
    minusone = thumbsdown

class Channel:
    def __init__(self, channel):
        self.maxscore = 0
        self.channel = channel
        self.pending = []
        self.history = []
        self.can_talk = False
        self.boringness = 0
        self.timemark = datetime.now()
class PromptCtx:
    dir = os.path.abspath('ctxs')
    default_model_kwparams = dict(return_full_text=False, max_new_tokens=512)
    default_local_kwparams = dict(append_delimiter=True, delimiter='\n')

    def __init__(self, ctx, prompt = '', **kwparams):
        pending_kwparams = {}
        pending_kwparams.update(self.default_model_kwparams)
        pending_kwparams.update(self.default_local_kwparams)
        pending_kwparams.update(kwparams)
        kwparams = pending_kwparams

        self.ctx = ctx
        self.path = os.path.join(PromptCtx.dir, ctx)
        self.kwparams0 = {}
        self.kwparams0.update(self.default_model_kwparams)
        self.kwparams0.update(self.default_local_kwparams)
        self.prompt0 = ''
        self.last_filename = ''
        state = {}
        self.state0 = {}
        if not os.path.exists(self.path):
            os.makedirs(self.path, exist_ok=True)
        else:
            files = os.listdir(self.path)
            files.sort()
            while len(files):
                filename = files.pop()
                filepath = os.path.join(self.path, filename)
                try:
                    kwparams, prompt, state = pickle.load(open(filepath, 'rb'))
                    self.kwparams0 = {}
                    self.kwparams0.update(kwparams)
                    self.state0 = {}
                    self.state0.update(state)
                    self.prompt0 = prompt
                    self.last_filename = filename
                    break
                except:
                    continue
        self.kwparams = kwparams
        self.prompt = prompt
        self.state = state
    @property
    def model_kwparams(self):
        result = {}
        result.update(self.kwparams)
        for key in PromptCtx.default_local_kwparams:
            if key in result:
                del result[key]
        return result
    def history(self):
        files = os.listdir(self.path)
        files.sort()
        return files
    @property
    def is_mutated(self):
        return self.kwparams0 != self.kwparams or self.prompt0 != self.prompt or self.state0 != self.state
    def load(self, filename):
        self.save()
        filepath = os.path.join(self.path, filename)
        self.kwparams, self.prompt, self.state = pickle.load(open(filepath, 'rb'))
        self.kwparams0 = {}
        self.kwparams0.update(self.kwparams)
        self.state0 = {}
        self.state0.update(self.state)
        self.prompt0 = self.prompt
        self.last_filename = filename
    def save(self):
        if self.is_mutated:
            now = datetime.now().isoformat()
            filename = str(now) + '-' + self.ctx + '.pickle'
            filepath = os.path.join(self.path, filename)
            state_for_saving = {}
            for key, value in self.state.items():
                try:
                    pickle.dump(value)
                    state_for_saving[key] = value
                except:
                    pass
            pickle.dump((self.kwparams, self.prompt, state_for_saving), open(filepath, 'wb'))
            self.kwparams0 = {}
            self.kwparams0.update(self.kwparams)
            self.state0 = {}
            self.state0.update(self.state)
            self.prompt0 = self.prompt
            self.last_filename = filename
            return filename
        else:
            return self.last_filename
    def __del__(self):
        self.save()
    def kwparams2str(self):
        return ' '.join((f'{key}={json.dumps(value)}' for key, value in self.kwparams.items() if key != 'append_delimiter' or self.kwparams.get('delimiter')))
    def str2kwparams(self, str):
        new_kwparams = {}
        del_ = lambda: None
        str = str.strip()
        if len(str):
            for part in str.split(' '):
                if '=' not in part:
                    key = part
                    val = True
                else:
                    key, val = part.split('=', 1)
                    key = key.strip()
                if val == 'del':
                    val = del_
                else:
                    val =json.loads(val)
                new_kwparams[key] = val
        for key, val in new_kwparams.items():
            if val is del_:
                del self.kwparams[key]
            else:
                self.kwparams[key] = val


class Bot:
    def __init__(self, token):
        self.client = discord.Client()
        self.client.event(self.on_ready)
        self.client.event(self.on_message)
        self.client.event(self.on_raw_reaction_add)
        self.client.event(self.on_raw_reaction_remove)
        self.token = token

        self.channels = {}
        self.new_messages = asyncio.Event()
        self.start_replying = asyncio.Event()

        self.on_reaction = {}

    @property
    def name(self):
        return str(self.client.user).split('#')[0]

    async def fill_history(self):
        await asyncio.sleep(0)
        ct = 0
        for name, channel in self.channels.items():
            if len(channel.pending):
                while len(channel.pending):
                    ct += 1
                    msg = channel.pending.pop(0)
                    if msg.content.strip():
                        #print('adding to history:', msg.author, msg.content)
                        if not channel.can_talk and (self.name + ', you can talk') in msg.content:
                            channel.can_talk = True
                        elif channel.can_talk and (self.name + ', stop talking') in msg.content:
                            channel.can_talk = False
                        channel.history.append(msg)
                if len(channel.history) > 2048:
                    channel.history = channel.history[-2048:]
        return ct

    def run(self):
        loop = self.client.loop
        async def do_loop():
            try:
                await asyncio.gather(self.client.start(self.token), self.pump())
            except:
                await self.client.close()
                raise
        try:
            loop.run_until_complete(do_loop())
        finally:
            loop.close()

    async def on_ready(self):
        print('We have logged in as {0.user}'.format(self.client))
        for channel in self.client.get_all_channels():
            print('channel:', channel)
            if type(channel) is discord.TextChannel:
                messages = []
                async for message in channel.history(limit=1024, oldest_first=False):
                    messages.insert(0, message)
                for message in messages:
                    #print(channel, message.channel, message.author, message.content)
                    await self.on_message(message)
            sys.stdout.flush()
        #self.nonself_end_of_line_token = self.usr2history(self.client.user)
        self.start_replying.set()

    async def delmsg(self, message):
        if not isinstance(message, discord.DeletedReferencedMessage):
            for channel in self.channels.values():
                if channel.channel == message.channel:
                    try:
                        channel.history.remove(message)
                    except:
                        try:
                            channel.pending.remove(message)
                        except:
                            pass
                    break
            message.content = ''
            await message.delete()

    async def preprocess_message(self, message):
        return True
    
    async def on_message(self, message):
        print(message.channel, message.author, 'in response to =', message.reference, ':', message.content)
        try:
            if await self.preprocess_message(message):
                channel = self.channels.setdefault(message.channel, Channel(message.channel))
                channel.pending.append(message)
                channel.boringness = 0
            self.new_messages.set()
        except Exception as e:
            print(err2str(e))
        sys.stdout.flush()

    async def on_raw_reaction_add(self, payload):
        self.new_messages.set()

    async def on_raw_reaction_remove(self, payload):
        self.new_messages.set()
        print('reaction', str(payload.emoji))

class bot(Bot):
    def __init__(self, token, model):
        super().__init__(token)
        self.model = model
        self.ctxs = {}

    def msgscore(self, msg):
        score = 0
        for reaction in msg.reactions:
            if str(reaction.emoji) in emoji.plusone:
                score += reaction.count
            elif str(reaction.emoji) in emoji.minusone:
                score -= reaction.count
        return score
    def scorestr(self, score):
        if score < 0:
            str = 'bad'
        elif score > 0:
            str = 'good'
        else:
            str = 'soso'
        return f'{str} {score}'

    def isscorestr(self, scorestr):
        parts = scorestr.split(' ')
        return len(parts) == 2 and parts[0] in ('bad','good','soso') and (parts[1].isnumeric() or parts[1][0] == '-' and parts[1][1:].isnumeric())

    def filtercontent(self, content):
        replacement = content.find('{replaced from:')
        if replacement >= 0:
            content = content[:replacement]
        return content

    def msg2history(self, msg, chandata):
        botstr = '(bot)' if msg.author.bot else '(human)'
        content = self.filtercontent(msg.content)
        return f'{msg.author} {botstr}: {self.scorestr(self.msgscore(msg))}: {msg.created_at.isoformat(" ", "milliseconds")} {content}'
    def usr2history(self, user, chandata = None):
        botstr = '(bot)' if user.bot else '(human)'
        score = self.scorestr(chandata.maxscore) if chandata is not None else ''
        return f'{user} {botstr}: {score}: '

    async def pump(self):
        #print('pump out start')
        await self.start_replying.wait()
        while True:
            #print('pump out loop')
            found = await self.fill_history()
            for channel, chandata in [*self.channels.items()]:
                #print(channel, 'talk =', talk, 'len(history) =', len(history))
                #if chandata.can_talk:
                #    print(channel, 'score of last message =', self.msgscore(chandata.history[-1]))
                if chandata.can_talk and (
                    chandata.history[-1].author != self.client.user or
                    self.msgscore(chandata.history[-1]) < 0
                ) and chandata.boringness < 128:
                    #print('responding to', history[-1].author, history[-1].content)
                    found = True
                    reply_datetime = datetime.now()
                    try:
                        removect = 0
                        await self.fill_history()
                        prompt = '\n'.join([self.msg2history(msg, chandata) for msg in list_randshrink(chandata.history[-1024:], removect)])
                        if '(human)' not in prompt:
                            continue
                        chandata.maxscore = max(0,max((self.msgscore(msg) for msg in chandata.history[-16:])))
                        preprompt = '\n' + self.usr2history(self.client.user, chandata).strip()
                        prompt += preprompt
                        model_kwparams = dict(
                            #eos_token_id=self.nonself_end_of_line_token,
                            return_full_text=False,
                            max_new_tokens=512,
                            #top_p=0.25
                            #temperature=1.0
                        )
                        #print(model_kwparams)
                        sys.stdout.flush()
                        if (chandata.timemark - datetime.now()).total_seconds() <= 10:
                            print('typing since, given now is', datetime.now(), 'then timemark is soon:', chandata.timemark)
                            async with channel.typing():
                                reply = await asyncify(self.model)(prompt.strip(), **model_kwparams)
                        else:
                            reply = await asyncify(self.model)(prompt.strip(), **model_kwparams)
                        reply = reply[0]['generated_text'].strip()
                        print(prompt[-256:])
                        print('considering:', preprompt + ' ' + reply)
                        date, time, reply = reply.split(' ', 2)
                        try:
                            reply_datetime = datetime.fromisoformat(date  + ' ' + time)
                        except ValueError as e:
                            print(e)
                            continue
                        print('time =', reply_datetime.isoformat())
                        #time = datetime.datetime.fromisoformat(date + ' ' + time)
                        lines = reply.split('\n')
                        # quick fix: remove items from prompt to change context
                        if removect < len(chandata.history):
                            removect += 1

                       # if '(human)' not in lines[1] and '(human)' not in lines[2]:
                       #     reply = '' #'!' + reply
                       #     lines = ['']
                        #elif '(human)' not in lines[1]:
                        #    reply = '!' + reply

                        # for multiline: read up until another message is expected
                        reply = ''
                        humanct = 0
                        botct = 0
                        mark = 0
                        for idx, line in enumerate(lines):
                            if '#' in line: # hacky way to identify that a line is message
                                name, bit = line.split('#', 1)
                                if ':' in bit:
                                    bits = bit.split(':')
                                    namebits = bits[0].split(' ')
                                    if len(namebits) == 2 and len(bits) > 2 and namebits[0].isnumeric() and namebits[1] in ('(bot)', '(human)') and self.isscorestr(bits[1].strip()):
                                        if mark == 0:
                                            mark = idx
                                        if '(human)' not in line:
                                            botct += 1
                                        else:
                                            humanct += 1
                                        if botct + humanct < 3:
                                            break
                        if humanct > 0:
                            reply = '\n'.join(lines[:mark])
                    except Exception as e:
                        print(reply)
                        reply = err2str(e)
                    if len(reply) == 0:
                        reply = '[empty message??]'
                        print(reply)
                        reply = ''
                        chandata.boringness += 1
                    sys.stdout.flush()
                    if len(reply) > 0:
                        delay = (reply_datetime - datetime.now()).total_seconds()
                        if delay > 10:
                            chandata.timemark = reply_datetime
                            print('too far in future to wait here for, moving on', delay, 'to', chandata.timemark)
                            sys.stdout.flush()
                            continue
                        elif delay > 0:
                            if delay > 1:
                                await asyncio.sleep(delay - 1)
                                delay = 1
                            async with channel.typing():
                                await asyncio.sleep(delay)
                        try:
                            await channel.send(reply)
                        except Exception as e:
                            print('TOO LONG?')
                            print(reply)
                            print(e)
                            await channel.send('SERVER ERROR, MESSAGE TOO LONG?  PLEASE REVIEW OUTPUT. oh next message might work who knows, maybe i can try again.')
            if not found:
                self.new_messages.clear()
                await self.new_messages.wait()

    async def preprocess_message(self, message):
        is_bot_reply = False
        is_reply_by_bot = False
        if message.reference is not None and message.reference.resolved is not None and not isinstance(message.reference.resolved, discord.DeletedReferencedMessage):
            message_replied = message.reference.resolved
            if message_replied.author == self.client.user:
                if any((reaction.me for reaction in message.reactions)):
                    return False
                while True:
                    try:
                        await message.add_reaction(emoji.random())
                        break
                    except discord.errors.HTTPException: # unknown emoji
                        continue
                is_bot_reply = True
                if (message.content.startswith(f'{self.name}, replace with:') or message.content.lower().startswith('replace:')):
                    newcontent = message.content[len(message.content.split(':', 1)[0]) + 2:].strip()
                    oldcontent = message.reference.resolved.content
                    while '{replaced from: ' in oldcontent:
                        oldcontent = oldcontent[oldcontent.find('{replaced from: ') + len('{replaced from: '):]
                        oldconent = oldcontent[:-1]
                    await message.reference.resolved.edit(content = newcontent + '{replaced from: ' + oldcontent + '}' )
                    print('UPDATED CONTENT:', message.reference.resolved.content)
                    sys.stdout.flush()
                    return False
                elif (message.content.lower().startswith(f'{self.name}, delete') or message.content.lower().strip() == 'delete'):
                    print('DELETE')
                    sys.stdout.flush()
                    await self.delmsg(message.reference.resolved)
                    return False
                elif message.author == self.client.user:
                    is_reply_by_bot = True
        if is_reply_by_bot:
            if message_replied.content.lower().startswith('ctx ') and message_replied.reference is not None:
                # command result, hopefully?
                return False
        if is_bot_reply: # could also check for name mention
            try:
                if message.content.lower().startswith('ctx '):
                    _, name, cmd, *params = message.content.split(' ', 3)
                    content = params[0] if len(params) else ''
                    ctx = self.ctxs.get(name)
                    if ctx is None:
                        ctx = self.ctxs.setdefault(name, PromptCtx(name))
    
                    if cmd == "dump":
                        await self.reply_msg(message, ctx.kwparams2str() + '\n`' + ctx.prompt + '`')
                    elif cmd == "guess":
                        await message.add_reaction(emoji.thinking)
                        if ctx.kwparams.get('delimiter'):
                            if len(ctx.prompt) and ctx.prompt[-1] != ctx.kwparams['delimiter']:
                                content = ctx.kwparams['delimiter'] + content
                        if ctx.kwparams['append_delimiter']:
                            content += ctx.kwparams['delimiter']
                        response = (await asyncify(self.model)(ctx.prompt + content, **ctx.model_kwparams))[0]['generated_text']
                        sent = await self.reply_msg(message, '`'+response+'`')
                        await message.remove_reaction(emoji.thinking, self.client.user)
                        await sent.add_reaction(emoji.thumbsup)
                        await sent.add_reaction(emoji.knife)
                        await sent.add_reaction(emoji.scissors)
                        await sent.add_reaction(emoji.repeat)
                        if allow_exec:
                            await sent.add_reaction(emoji.running)
                        await sent.add_reaction(emoji.poop)
                        async def handle(reaction_payload):
                            try:
                                if reaction_payload.user_id == message.author.id:
                                    if str(reaction_payload.emoji) == emoji.thumbsup:
                                        ctx.prompt += content + sent.content
                                        ctx.save()
                                        await self.reply_msg(sent, '... ' + content + sent.content)
                                    elif str(reaction_payload.emoji) == emoji.repeat:
                                        await sent.add_reaction(emoji.thinking)
                                        response = (await asyncify(self.model)(ctx.prompt + content, **ctx.model_kwparams))[0]['generated_text']
                                        await sent.edit(content='`'+response+'`')
                                        await sent.remove_reaction(emoji.thinking, self.client.user)
                                    elif str(reaction_payload.emoji) == emoji.scissors:
                                        idx = sent.content.rfind(ctx.kwparams['delimiter'])
                                        if idx == -1:
                                            idx = sent.content.rfind('\n')
                                        if idx == -1:
                                            idx = sent.content.rfind(' ')
                                        if idx == -1:
                                            idx = len(sent.content) - 1
                                        if idx > 0:
                                            await sent.edit(content=sent.content[:idx] + '`')
                                    elif str(reaction_payload.emoji) == emoji.knife:
                                        idx = sent.content.find(ctx.kwparams['delimiter'])
                                        if idx == -1:
                                            idx = sent.content.find('\n')
                                        if idx == -1:
                                            idx = sent.content.find(' ')
                                        if idx == -1 and len(sent.content) > 1:
                                            idx = 0
                                        if idx >= 0:
                                            await sent.edit(content='`'+sent.content[idx+1:])
                                    elif str(reaction_payload.emoji) == emoji.running and allow_exec:
                                        ctx.save()
                                        state0 = {}
                                        state0.update(ctx.state)
                                        stdout = io.StringIO()
                                        with redirect_stdout(stdout):
                                            exec(sent.content[1:-1], {}, ctx.state)
                                        ctx.save()
                                        result = stdout.getvalue()
                                        if not len(result):
                                            result = str(ctx.state)
                                        await self.reply_msg(sent, str(result))
                            except Exception as e:
                                await self.reply_msg(sent, err2str(e))
                        self.on_reaction[sent.id] = handle
                    #elif cmd == 'join':
                    #    ctx.prompt += content
                    #    await self.reply_msg(message, '... ' + ctx.prompt[-256:])
                    elif cmd == 'add':
                        if ctx.kwparams.get('delimiter') and len(ctx.prompt) and ctx.prompt[-1] != ctx.kwparams['delimiter']:
                            ctx.prompt += ctx.kwparams['delimiter']
                        ctx.prompt += content
                        await self.reply_msg(message, '... ' + ctx.prompt[-256:])
                    elif cmd == 'set':
                        ctx.prompt = content
                        await self.reply_msg(message, ctx.prompt)
                    elif cmd == 'params':
                        ctx.str2kwparams(content)
                        await self.reply_msg(message, ctx.kwparams2str())
                    elif cmd == 'fork':
                        ctx_src = self.ctxs.get(content)
                        if ctx_src is None:
                            ctx_src = self.ctxs.setdefault(content, PromptCtx(content))
                        if ctx.is_mutated:
                            ctx.save()
                        ctx.kwparams = {}
                        ctx.kwparams.update(ctx_src.kwparams)
                        ctx.prompt = ctx_src.prompt
                        await self.reply_msg(message, ctx.kwparams2str())
                    elif cmd == 'save':
                        await self.reply_msg(message, ctx.save())
                    elif cmd == 'list':
                        await self.reply_msg(message, '\n'.join(ctx.history()))
                    elif cmd == 'load':
                        if content == '':
                            content = ctx.last_filename
                        ctx.load(content)
                        await self.reply_msg(message, ctx.kwparams2str())
                    elif cmd == 'state':
                        await self.reply_msg(message, str(ctx.state))
                    elif cmd == 'clearstate':
                        ctx.state = {}
                        await self.reply_msg(message, str(ctx.state))
                    #elif cmd == 'addline':
                    #    if len(content) == 0:
                    #        content = message_replied.content
                    #    if len(ctx.prompt) and ctx.prompt[-1] != '\n':
                    #        ctx.prompt += 
                    #    ctx.prompt += 
                    else:
                        await self.reply_msg(message, 'cmds are: dump guess params fork add set save list load state clearstate')
            except Exception as e:
                reply = err2str(e)
                print(reply)
                await self.reply_msg(message, reply)
            return False
        return True

    async def reply_msg(self, replyto, replywith):
        #print('reply msg', replyto, replywith)
        if not len(replywith):
            replywith = '<no data>'
        return await replyto.channel.send(replywith, reference=replyto)
        #print('sent')

    async def on_raw_reaction_add(self, payload):
        if payload.user_id != self.client.user.id:
            if str(payload.emoji) == emoji.poop:
                for channel, chandata in [*self.channels.items()]:
                    if channel.id == payload.channel_id:
                        for message in (*chandata.pending, *chandata.history):
                            if message.id == payload.message_id:
                                await self.delmsg(message)
                                break
            else:
                handler = self.on_reaction.get(payload.message_id)
                if handler is not None:
                    await handler(payload)
        return await super().on_raw_reaction_add(payload)
    async def on_raw_reaction_remove(self, payload):
        if payload.user_id != self.client.user.id:
            handler = self.on_reaction.get(payload.message_id)
            if handler is not None:
                await handler(payload)
        return await super().on_raw_reaction_remove(payload)

#model = codesynth.ai21_jumbo()
model = codesynth.multi_demo(codesynth.eleuther_demo(), codesynth.bellard_demo())
#model = codesynth.openai()
if __name__ == '__main__':
    bot(discord_token, model).run()

