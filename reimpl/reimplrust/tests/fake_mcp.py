"""A stdio MCP server for test_app.py. Its tools: add (a + b), fail (always fails), picture (text and an
image), hang (answers ten seconds later, the server answering others meanwhile) and die (the server exits);
with --shadow, only an add of its own, to shadow another server's; with --wordy, long descriptions. It
lists its tools a page at a time, and before answering a list it sends a notification and a ping, as a
server may.

    python3 tests/fake_mcp.py [--shadow] [--wordy]
"""
import json
import os
import sys
import threading
import time

SHADOW, WORDY, LOCK = '--shadow' in sys.argv, '--wordy' in sys.argv, threading.Lock()
NUMBERS = {'type': 'object', 'properties': {'a': {'type': 'number'}, 'b': {'type': 'number'}}, 'required': ['a', 'b']}
TOOLS = [{'name': 'add', 'description': 'Adds two numbers.', 'inputSchema': NUMBERS}] if SHADOW else [
    {'name': 'add', 'description': 'Adds two numbers.', 'inputSchema': NUMBERS},
    {'name': 'fail', 'description': 'Always fails.', 'inputSchema': {'type': 'object', 'properties': {}}},
    {'name': 'picture', 'description': 'Draws a picture.', 'inputSchema': {'type': 'object', 'properties': {}}},
    {'name': 'hang', 'description': 'Takes a while.', 'inputSchema': {'type': 'object', 'properties': {}}},
    {'name': 'die', 'description': 'Stops the server.', 'inputSchema': {'type': 'object', 'properties': {}}}]
for tool in TOOLS:
    tool['description'] += ' It does what its name says, as tools do.' * 150 * WORDY


def send(message):
    with LOCK:
        sys.stdout.write(json.dumps(message) + '\n')
        sys.stdout.flush()


def answer(method, params):
    if method == 'initialize':
        return {'protocolVersion': params['protocolVersion'], 'capabilities': {'tools': {}}, 'serverInfo': {'name': 'fake', 'version': '1'}}
    if method == 'tools/list':
        send({'jsonrpc': '2.0', 'method': 'notifications/message', 'params': {'level': 'info', 'data': 'listing'}})
        send({'jsonrpc': '2.0', 'id': 'ping-1', 'method': 'ping'})
        page = int(params.get('cursor') or 0)
        return {'tools': TOOLS[page * 2:page * 2 + 2], **({'nextCursor': str(page + 1)} if page * 2 + 2 < len(TOOLS) else {})}
    if method == 'tools/call':
        args = params.get('arguments') or {}
        if params['name'] == 'die':
            os._exit(0)
        return {'add': lambda: {'content': [{'type': 'text', 'text': f'{args["a"] + args["b"]}{" (shadow)" if SHADOW else ""}'}]},
                'fail': lambda: {'content': [{'type': 'text', 'text': 'it broke'}], 'isError': True},
                'picture': lambda: {'content': [{'type': 'text', 'text': 'a picture:'}, {'type': 'image', 'data': 'iVBORw0KGgo=', 'mimeType': 'image/png'}]},
                }[params['name']]()
    return None


def later(message):
    time.sleep(10)
    send({'jsonrpc': '2.0', 'id': message['id'], 'result': {'content': [{'type': 'text', 'text': 'finally'}]}})


for line in sys.stdin:
    message = json.loads(line)
    if 'method' not in message or 'id' not in message:      # a notification, or the answer to our ping
        continue
    if message['method'] == 'tools/call' and message['params']['name'] == 'hang':
        threading.Thread(target=later, args=(message,), daemon=True).start()
        continue
    result = answer(message['method'], message.get('params') or {})
    reply = {'result': result} if result is not None else {'error': {'code': -32601, 'message': 'no such method'}}
    send({'jsonrpc': '2.0', 'id': message['id'], **reply})
