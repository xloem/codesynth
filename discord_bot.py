import asyncio
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import contextlib
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
    name_by_unicode = None
    def random():
        if emoji.name_by_unicode is None:
            import requests
            emojis = requests.get('https://unicode.org/Public/emoji/14.0/emoji-test.txt')
            emoji_lines = [line for line in emojis.text.split('\n') if len(line) and line[0] in set('0123456789ABCDEFabcdef')]
            emoji.name_by_unicode = {}
            for line in emoji_lines:
                parts = line.split(';', 1)
                ucode = parts[0].strip()
                ucode = ''.join((chr(int(code, 16)) for code in ucode.split(' ')))
                name = parts[1].split('#')[-1]
                name = name.split(' ', 3)[-1]
                emoji.name_by_unicode[ucode] = name
        import random
        return random.choice([*emoji.name_by_unicode.keys()])
    repeat = chr(0x1F501)
    thinking = chr(0x1F914)
    scissors = chr(0x2702)
    knife = chr(0x1F52A)
    running = chr(0x1F3C3)
    fountain_pen = chr(0x1F58B)
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
        self.timemark = datetime.now(timezone.utc)
        self.ctx = None
class PromptCtx:
    dir = os.path.abspath('ctxs')
    default_model_kwparams = dict(return_full_text=False, max_new_tokens=512)
    default_local_kwparams = dict(append_delimiter=True, delimiter='\n', prefix='', include_eos=False)

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
        self.chandata = None
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
            now = datetime.now(timezone.utc).isoformat()
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
        return ', '.join((
            f'{key}={repr(value)}'
            for key, value in self.kwparams.items()
            if (
                (key != 'append_delimiter' or self.kwparams.get('delimiter')) and
                (key != 'include_eos' or self.kwparams.get('eos_token_id'))
            )
        ))
    def str2kwparams(self, str):
        new_kwparams = {}
        rm = lambda: None
        new_kwparams.update(self.kwparams)
        new_kwparams.update(eval('dict('+str+')', {}, dict(rm=rm, del_=rm, _del=rm, delete=rm, remove=rm, drop=rm)))
        for key, val in [*new_kwparams.items()]:
            if val is rm:
                del new_kwparams[key]
        self.kwparams = new_kwparams
        return self.kwparams
    async def guess(self, model, content):
        content = self.kwparams.get('prefix','') + content
        if self.kwparams.get('delimiter'):
            if len(self.prompt) and self.prompt[-1] != self.kwparams['delimiter']:
                content = self.kwparams['delimiter'] + content
        if self.kwparams['append_delimiter']:
            content += self.kwparams['delimiter']
        response = (await asyncify(model)(self.prompt + content, **self.model_kwparams))[0]['generated_text']
        eos = self.kwparams.get('eos_token_id')
        if eos:
            if response.endswith(eos) and not self.kwparams.get('include_eos'):
                response = response[:-len(eos)]
            elif not response.endswith(eos) and self.kwparams.get('include_eos'):
                response += eos
        return response, content


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
                            if channel.ctx is not None:
                                channel.ctx.chandata = None
                                channel.ctx = None
                        elif channel.ctx is not None:
                            if channel.can_talk:
                                response = await chandata.ctx.guess(self.model, msg.content)
                                await self.reply_msg(msg, '`'+response+'`')
                            continue
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
                try:
                    async for message in channel.history(limit=1024, oldest_first=False):
                        messages.insert(0, message)
                except discord.errors.Forbidden:
                    pass
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
        print(message.channel, message.author, 'in response to =', message.reference, ':', repr(message.content))
        try:
            if await self.preprocess_message(message):
                #print('PENDING MESSAGE')
                channel = self.channels.setdefault(message.channel, Channel(message.channel))
                channel.pending.append(message)
                channel.boringness = 0
            self.new_messages.set()
        except Exception as e:
            print('284',err2str(e))
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
            str = 'ok'
        return f'{str} {score}'

    def isscorestr(self, scorestr):
        parts = scorestr.split(' ')
        return len(parts) == 2 and parts[0] in ('bad','good','ok') and (parts[1].isnumeric() or parts[1][0] == '-' and parts[1][1:].isnumeric())

    def filtercontent(self, content):
        replacement = content.find('{replaced from:')
        if replacement >= 0:
            content = content[:replacement]
        return content

    def msg2history(self, msg, chandata):
        botstr = '(bot)' if msg.author.bot else '(human)'
        content = self.filtercontent(msg.content)
        return f'{self.scorestr(self.msgscore(msg))} {botstr} {msg.author.name} {msg.created_at.isoformat(" ", "seconds")}: {content}'
    def usr2history(self, user, chandata = None, nextuser = None):
        botstr = '(bot)' if user.bot else '(human)'
        score = self.scorestr(chandata.maxscore) if chandata is not None else ''
        if nextuser is None:
            botstr2 = '(bot)'
        elif nextuser.bot:
            botstr2 = '(bot)'
        else:
            botstr2 = '(human)'
        return f'{score} {botstr} {user.name} '
    def parsehistory(self, line):
        spaceparts = line.split(' ', 6)
        if len(spaceparts) < 7:
            return None
        if not self.isscorestr(spaceparts[0] + ' ' + spaceparts[1]):
            return None
        botstr = spaceparts[2]
        if botstr not in ('(bot)','(human)'):
            return None
        user = spaceparts[3]
        date = spaceparts[4] + ' ' + spaceparts[5]
        if date[-1] != ':':
            return None
        date = date[:-1]
        content = spaceparts[6]
        return botstr, user, date, content

    async def pump(self):
        #print('pump out start')
        await self.start_replying.wait()
        while True:
            #print('pump out loop')
            found = await self.fill_history()
            for channel, chandata in [*self.channels.items()]:
              try:
                #print(channel, 'talk =', chandata.can_talk, 'len(history) =', len(chandata.history))
                #if chandata.can_talk:
                #    print(channel, 'score of last message =', self.msgscore(chandata.history[-1]))
                if chandata.can_talk and (
                    chandata.history[-1].author != self.client.user or
                    self.msgscore(chandata.history[-1]) < 0
                ) and chandata.boringness < 128:
                    #print('responding to', chandata.history[-1].author, chandata.history[-1].content)
                    found = True
                    reply_datetime = datetime.now(timezone.utc)
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
                        if (chandata.timemark - datetime.now(timezone.utc)).total_seconds() <= 10:
                            print('typing since, given now is', datetime.now(timezone.utc), 'then timemark is soon:', chandata.timemark)
                            async with channel.typing():
                                reply = await asyncify(self.model)(prompt.strip(), **model_kwparams)
                        else:
                            reply = await asyncify(self.model)(prompt.strip(), **model_kwparams)
                        reply = reply[0]['generated_text'].strip()
                        print(prompt[-256:])
                        print('considering:', preprompt + ' ' + reply)
                        try:
                            reply_datetime, reply = reply.split(': ', 1)
                            reply_datetime = datetime.fromisoformat(reply_datetime + '+00:00')
                        except ValueError as e:
                            print(err2str(e))
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
                            parsed = self.parsehistory(line)
                            if parsed is not None:
                                if mark == 0:
                                    mark = idx
                                if parsed[0] != '(human)':
                                    botct += 1
                                    if botct >= 3:
                                        break
                                else:
                                    humanct += 1
                                    break
                        if humanct > 0:
                            reply = '\n'.join(lines[:mark])
                    except Exception as e:
                        reply = err2str(e)
                        print(reply)
                    if len(reply) == 0:
                        reply = '[empty message??]'
                        print(reply)
                        reply = ''
                        chandata.boringness += 1
                    sys.stdout.flush()
                    if len(reply) > 0:
                        delay = (reply_datetime - datetime.now(timezone.utc)).total_seconds()
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
              except Exception as e:
                  # this is usually not hit, rather the exception farther inside is hit.  haven't reviewed if it would be good to merge them.
                  print(err2str(e))
                  if chandata.can_talk:
                      await channel.send(err2str(e)[:2000])
            if not found:
                self.new_messages.clear()
                await self.new_messages.wait()
                #print('done waiting for messages')

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
                    chandata = self.channels.setdefault(message.channel, Channel(message.channel))
                    for histmessage in (*chandata.history, *chandata.pending):
                        if histmessage.id == message.reference.resolved.id:
                            histmessage.content = newcontent
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
                        response, content = await ctx.guess(self.model, content)
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
                                        ctx.prompt += content + sent.content[1:-1]
                                        ctx.save()
                                        await self.reply_msg(sent, '... `' + content + sent.content[1:])
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
                                        captured_output = io.StringIO()
                                        with contextlib.redirect_stdout(captured_output):
                                            with contextlib.redirect_stderr(captured_output):
                                                exec(sent.content[1:-1], {}, ctx.state)
                                        ctx.save()
                                        result = captured_output.getvalue()
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
                        ctx.prompt += ctx.kwparams.get('prefix','') + content
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
                    elif cmd == 'channel':
                        chandata = self.channels.setdefault(message.channel, Channel(message.channel))
                        chandata.ctx = ctx
                        chandata.can_talk = True
                        ctx.chandata = chandata
                        await self.reply_msg(message, str(message.channel))
                    #elif cmd == 'addline':
                    #    if len(content) == 0:
                    #        content = message_replied.content
                    #    if len(ctx.prompt) and ctx.prompt[-1] != '\n':
                    #        ctx.prompt += 
                    #    ctx.prompt += 
                    else:
                        await self.reply_msg(message, 'cmds are: dump guess params fork add set save list load state clearstate channel')
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

