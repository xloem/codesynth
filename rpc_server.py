#!/usr/bin/env python3

from aiohttp import web
import pjrpc.server
from pjrpc.server.integration import aiohttp

methods = pjrpc.server.MethodRegistry()

import generate

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

if __name__ == '__main__':
    web.run_app(jsonrpc_app.app, port=80)

