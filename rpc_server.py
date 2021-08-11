#!/usr/bin/env python3

from aiohttp import web
import pjrpc.server
from pjrpc.server.integration import aiohttp

methods = pjrpc.server.MethodRegistry()

import generate

@methods.add(context='request')
async def generate_text(request: web.Request, text, model = 'genji', **params):
    models = request.app['models']
    if model not in models:
        models[model] = getattr(generate, model)()
    return models[model](**params)

jsonrpc_app = aiohttp.Application('/')
jsonrpc_app.dispatcher.add_methods(methods)
jsonrpc_app.app['models'] = {}

if __name__ == '__main__':
    web.run_app(jsonrpc_app.app, port=80)

