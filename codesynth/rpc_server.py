#!/usr/bin/env python3
# Copyright (C) 2021
#
# This file is part of codesynth.
#
# codesynth is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# codesynth is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with codesynth.  If not, see <http://www.gnu.org/licenses/>.

from aiohttp import web
import pjrpc.server
from pjrpc.server.integration import aiohttp

methods = pjrpc.server.MethodRegistry()

from . import causal_language_model as generate

def get_model(request: web.Request, model):
    models = request.app['models']
    if model not in models:
        print('Trying to load ' + model + ' ...')
        models[model] = getattr(generate, model)()
        print('Loaded ' + model)
    return models[model]

@methods.add(context='request')
async def generate_text(request: web.Request, text, model = 'genji', **params):
    if 'params' in params:
        params = params['params']
    print(text, params)
    return get_model(request, model)(text, **params)

@methods.add(context='request')
async def tokenizer(request: web.Request, text, model = 'genji', **params):
    if 'params' in params:
        params = params['params']
    return dict(get_model(request, model).tokenizer(text, **params))

def prefixmethods(model):
    methods = pjrpc.server.MethodRegistry(model)
    @methods.add(context='request')
    async def generate_text(request: web.Request, text, **params):
        if 'params' in params:
            params = params['params']
        return get_model(request, model)(text, **params)
    @methods.add(context='request')
    async def tokenizer(request: web.Request, **params):
        if 'params' in params:
            params = params['params']
        return dict(get_model(request, model).tokenizer(text, **params))
    jsonrpc_app.dispatcher.add_methods(methods)

jsonrpc_app = aiohttp.Application('/')
for model in generate.__dict__.values():
    if type(model) is type:
        prefixmethods(model.__name__)

jsonrpc_app.dispatcher.add_methods(methods)
jsonrpc_app.app['models'] = {}

def server():
    web.run_app(jsonrpc_app.app, port=6686)

__all__ = ['server']

if __name__ == '__main__':
    server()
