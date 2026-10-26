"""End-to-end test of connectome-web against a mock of the Anthropic Messages API, through the app's
own HTTP API: a long chat in a small window, then a fork and more chat in the fork. Replies stream:
the mock sends server-sent events, chunked or until the close, and the page gets each piece of text
and reasoning as it comes, the same as the log then holds. The mock serves
HTTPS with a certificate from a CA made for the test, which the app trusts through --ca, and plain
HTTP on a second port. The mock rejects
any request the API would (a body not sent in chunks, roles that don't alternate or don't start and end
with the user, empty content or blank text, more than four cache breakpoints) and answers chat turns
and memory requests. Checked: every chat turn gets its reply; memories form and the context comes
to show them; the context stays within the window; the fork shares its parent's messages through the
fork point, and its settings are its parent's; a provider saved from the page, and a model a session
picks, reach the API, the key never the page or a session's file, in a file only its user can read,
and the models it lists (each once) stay as they are when it's saved without them; a provider is
deleted, unless new sessions start on it;
a session without settings says so, and one whose provider has gone shows its settings and takes
another; a reply whose stream is cut off is an error, not a logged reply; a turn on a log the library
can't read fails, and the session goes on once it's mended; a fork onto a session there is is refused;
a session's settings refuse a provider there isn't, a workspace that isn't a directory and a new
merge_threshold once memories have merged; a session that calls bash after losing its workspace
gets a failed call, the request still declaring bash; a turn continues without a message, a reply
cut short going on from its text and a turn whose request failed answering the message it had;
another site's page can't change the settings; a server whose certificate names another
host gets nothing; the model's tool calls, through MCP servers, get their results and form
memories; the app's peak memory doesn't grow with the contexts it streams; and in a window too
small for a memory no merge has taken on each layer and the tail, the oldest of them merge across
layers (lifted), and every turn still gets its reply. In a channel, each post, the user's or a
session's (message_channel), goes to the sessions there with inputs on but its poster. On a board,
sessions work on tasks as files, and a task's owner hears when it's handed to them or commented on.
Through the gateway, a request goes upstream as it came, but for this connection's headers, and its
answer streams back as it comes, the exchange recorded without credentials. A Claude Code session
(a fake one, fake_claude.py) takes its turns in Claude Code, in its sandbox, through the gateway,
the page asking before its calls.

The nine parts (the chat, the tool calls, the large contexts, the companion, the lifted memories,
the channels, the boards, the gateway, Claude Code) run at once, in processes of their own, each with its own mock; the first to fail stops the rest.

    python3 tests/test_app.py [turns]
"""
import datetime
import itertools
import json
import multiprocessing
import os
import re
import signal
import ssl
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = Path(os.environ.get('CONNECTOME_WEB') or HERE.parent / 'target' / 'release' / 'connectome-web')   # CONNECTOME_WEB: test that build instead
WINDOW = 8000
LIFTED = 2300   # lifted's window: without lifting, its context stops fitting by the 31st turn
seen = {'folds': {}, 'costs': {}, 'prompts': {}, 'calls': [], 'forming': False, 'prefills': 0, 'failed': [], 'model': None, 'chat': 0, 'memory': 0, 'bytes': 0, 'largest': 0, 'problems': [], 'key': None, 'thought': 0, 'results': [], 'tool memories': 0, 'events': [], 'asked': [], 'limit': None, 'requests': [], 'retries': [], 'running': [], 'cut': [], 'planned': {}, 'passbacks': 0, 'forgetful': False, 'long-winded': False, 'memory max': []}
PONDERED = 'I weighed what this stretch should keep. '   # what a memory's summarizer thinks first: never sent again


class Mock(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        """A SearXNG server's JSON search: two pages for a query, 25 for 'many' (a query of 'no json'
        as a server whose settings.yml doesn't allow JSON answers it). The path is the request line's:
        http.server collapses a leading // in self.path, which a real server may not."""
        url = urllib.parse.urlparse(self.requestline.split()[1])
        asked = urllib.parse.parse_qs(url.query)
        if url.path != '/search' or asked.get('format') != ['json'] or asked['q'] == ['no json']:
            return self.answer(403, {'error': 'forbidden'})
        seen['searched'] = asked['q'][0]
        many = [{'title': f'Page {n}', 'url': f'https://example.com/{n}', 'content': '', 'publishedDate': None} for n in range(25)]
        self.answer(200, {'query': asked['q'][0], 'results': many if asked['q'] == ['many'] else [
            {'title': 'Bees', 'url': 'https://example.com/bees', 'content': f'All about {asked["q"][0]}. ', 'publishedDate': '2026-07-27T00:00:00'},
            {'title': 'Wasps', 'url': 'https://example.com/wasps', 'content': 'Less about bees.', 'publishedDate': None}]})

    def do_POST(self):
        if self.path != '/v1/messages':
            return self.problem(f'no endpoint {self.path}')
        if self.headers.get('transfer-encoding') != 'chunked':
            return self.problem('the body was not sent in chunks')
        body = b''
        while (size := int(self.rfile.readline().split(b';')[0], 16)):
            body += self.rfile.read(size)
            self.rfile.readline()
        self.rfile.readline()
        seen['bytes'] += len(body)
        seen['key'] = self.headers.get('x-api-key')
        seen['model'] = json.loads(body).get('model')
        seen['largest'] = max(seen['largest'], len(body))
        request = json.loads(body)
        if problem := invalid(request):
            return self.problem(problem)
        seen['prefills'] += request['messages'][-1]['role'] == 'assistant'
        if request.get('stream') is not True:
            return self.problem('the request did not ask for a stream')
        prompt = tokens(body, request)
        if seen['limit'] and not request.get('tool_choice') and prompt + request['max_tokens'] > seen['limit']:
            return self.problem(f'prompt is too long: {prompt} tokens and {request["max_tokens"]} to answer in a window of {seen["limit"]}')
        messages = request['messages']
        seen['requests'].append((request.get('system') or '', [t['name'] for t in request.get('tools') or []]))
        last = ' '.join(b.get('text', '') for b in messages[-1]['content'])
        last = re.sub(r'^\s*\S+ (?:ended its turn|sends you a message while it works|asks you, and waits for your answer):\n\n', '', last)   # from a session's partner: what it says
        last = re.sub(r'^\s*(?:\S+|The user) posted in #\S+:\n\n([\s\S]*)\n\n\(To reply in #\S+, use message_channel [^\n]*\)(?:\n\nAbout #\S+: [\s\S]*)?$', r'\1', last)   # a post in a channel: what it says
        results = [b for b in messages[-1]['content'] if b['type'] == 'tool_result']
        asked = next(' '.join(b.get('text', '') for b in m['content']) for m in reversed(messages) if m['role'] == 'user' and any(b['type'] == 'text' for b in m['content']))
        stop = 'end_turn'
        if messages[-1]['role'] == 'assistant':                    # a reply cut short, which this one goes on from
            seen['chat'] += 1
            blocks = [{'type': 'text', 'text': ' and so the story ends.'}]
        elif 'fail once' in last and last not in seen['failed']:          # a failure that may pass: tried again
            seen['failed'].append(last)
            return self.answer(529, {'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'Overloaded'}})
        elif 'refuse once' in last and last not in seen['failed']:        # one that won't
            seen['failed'].append(last)
            return self.answer(400, {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'Refused'}})
        elif last.startswith('echo '):                             # the message back, as the reply
            seen['chat'] += 1
            blocks = [{'type': 'text', 'text': last[len('echo '):]}]
        elif last.startswith('think '):                            # the message back as reasoning, then an answer
            seen['chat'] += 1
            blocks = [{'type': 'thinking', 'thinking': last[len('think '):], 'signature': 'c2ln'}, {'type': 'text', 'text': 'Thought it through.'}]
        elif 'a long story' in last:
            seen['chat'] += 1
            blocks, stop = [{'type': 'text', 'text': 'Once upon a time'}], 'max_tokens'
        elif 'memory' in last and ('Write the memory' in last or 'consolidate' in last) and seen['forgetful']:   # memories down, at once
            return self.answer(400, {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'No memories today'}})
        elif 'memory' in last and ('Write the memory' in last or 'consolidate' in last):
            if request.get('tools') and request.get('tool_choice') != {'type': 'none'}:
                return self.problem(f'a memory request may call tools: tool_choice {request.get("tool_choice")}')
            seen['memory'] += 1
            seen['memory max'].append(request.get('max_tokens'))
            seen['tool memories'] += any(b['type'] == 'tool_use' for m in messages for b in m['content'])
            blocks = [{'type': 'thinking', 'thinking': PONDERED * 40, 'signature': 'bWVt'}, {'type': 'text', 'text': f'I remember stretch {seen["memory"]}: the user talked about ' + ' '.join(['things'] * 60)}]
            if seen['long-winded']:                                # its tokens run out while it thinks
                blocks, stop = blocks[:1], 'max_tokens'
        elif 'use tools' in last or last.startswith('call ') or results and 'forever' in asked:
            seen['chat'] += 1
            seen['results'].append(results)
            calls = json.loads(last[5:]) if last.startswith('call ') else [['add', {'a': 2, 'b': 3}]] if results else \
                [['add', {'a': 2, 'b': 3}], ['fail', {}], ['picture', {}], ['nope', {}]]
            blocks = [{'type': 'text', 'text': 'Calling tools.'}] + [{'type': 'tool_use', 'id': f'call{seen["chat"]}-{k}', 'name': name, 'input': args}
                                                                   for k, (name, args) in enumerate(calls)]
            if 'reasoned' in last:                                 # reasoning first, which must come back with the calls (thinking mode)
                planned = {'type': 'thinking', 'thinking': f'Planning the calls of reply {seen["chat"]}.', 'signature': f'c2ln{seen["chat"]}'}
                seen['planned'][f'call{seen["chat"]}-0'] = planned
                blocks.insert(0, planned)
            stop = 'tool_use'
        else:
            seen['chat'] += 1
            if results:
                seen['results'].append(results)
            blocks = [{'type': 'thinking', 'thinking': 'Let me think about it.', 'signature': 'c2ln'}] * (seen['chat'] % 4 == 0) \
                + [{'type': 'text', 'text': f'Reply {seen["chat"]}: ' + ' '.join(['words'] * 80)}]
        memory = 'memory' in last and ('Write the memory' in last or 'consolidate' in last)
        cut = 'cut off once' in last and last not in seen['cut']
        seen['cut'] += [last] * cut
        self.stream(blocks, prompt, 0.25 if 'slowly' in last else 0.01 if memory else 0, stop, cut)   # a memory takes a moment, as a page may see

    def stream(self, blocks, input_tokens, pause, stop, cut=False):
        """The reply as server-sent events, its text in five pieces `pause` seconds apart and a tool
        call's input in three, the same, the stream ending there if `cut`; chunked or until the close,
        by turns."""
        chunked = (seen['chat'] + seen['memory']) % 2 == 0
        self.protocol_version = 'HTTP/1.1' if chunked else 'HTTP/1.0'
        self.send_response(200)
        self.send_header('content-type', 'text/event-stream')
        if chunked:
            self.send_header('transfer-encoding', 'chunked')
        self.end_headers()

        def event(kind, **fields):
            data = f'event: {kind}\ndata: {json.dumps({"type": kind, **fields})}\n\n'.encode()
            self.wfile.write(f'{len(data):x}\r\n'.encode() + data + b'\r\n' if chunked else data)
            self.wfile.flush()
        prompt = {'input_tokens': input_tokens - input_tokens // 3, 'cache_read_input_tokens': input_tokens // 3}   # a third of it cached
        event('message_start', message={'type': 'message', 'role': 'assistant', 'content': [], 'usage': {'input_tokens': 0, 'output_tokens': 0} if chunked else {**prompt, 'output_tokens': 1}})
        for i, block in enumerate(blocks):
            if block['type'] == 'tool_use':
                event('content_block_start', index=i, content_block={**block, 'input': {}})
                text = json.dumps(block['input'])
                for k in range(3):
                    time.sleep(pause)
                    event('content_block_delta', index=i, delta={'type': 'input_json_delta', 'partial_json': text[k * len(text) // 3:(k + 1) * len(text) // 3]})
                event('content_block_stop', index=i)
                continue
            key = 'text' if block['type'] == 'text' else 'thinking'
            event('content_block_start', index=i, content_block={'type': block['type'], key: '', **({'signature': ''} if key == 'thinking' else {})})
            words, n = block[key], len(block[key])
            for k in range(5):
                time.sleep(pause)
                event('content_block_delta', index=i, delta={'type': f'{key}_delta', key: words[k * n // 5:(k + 1) * n // 5]})
            if key == 'thinking':
                event('content_block_delta', index=i, delta={'type': 'signature_delta', 'signature': block['signature']})
            event('content_block_stop', index=i)
        if cut:
            return self.wfile.write(b'0\r\n\r\n' if chunked else b'')
        out = sum(len(b.get('text', '')) for b in blocks) // 4
        priced = {} if seen['model'] == 'mock-2' else {'cost': (input_tokens * 3 + out * 15) / 1e6}   # as OpenRouter reports it; mock-2's provider doesn't
        event('message_delta', delta={'stop_reason': stop}, usage={'output_tokens': out, **priced, **(prompt if chunked else {})})   # chunked: the prompt's usage at the end, as OpenRouter has it
        event('message_stop')
        if seen['chat'] % 3 == 0:                          # as OpenRouter ends its streams
            data = b'event: data\ndata: [DONE]\n\n'
            self.wfile.write(f'{len(data):x}\r\n'.encode() + data + b'\r\n' if chunked else data)
        if chunked:
            self.wfile.write(b'0\r\n\r\n')

    def problem(self, text):
        seen['problems'].append(text)
        self.answer(400, {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': text}})

    def answer(self, code, value):
        data = json.dumps(value).encode()
        self.send_response(code)
        self.send_header('content-type', 'application/json')
        self.send_header('content-length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def tokens(body, request) -> int:
    """A request's prompt tokens, as its usage reports them: a quarter of its bytes, but 1600 for each
    image, as a screenshot costs (the tool's pictures are a few bytes)."""
    images = [c for m in request['messages'] for b in m['content'] for c in [b, *(b['content'] if isinstance(b.get('content'), list) else [])] if c['type'] == 'image']
    return (len(body) - sum(len(c['source']['data']) for c in images)) // 4 + 1600 * len(images)


def invalid(request) -> str | None:
    """Why the Messages API would reject the request, if it would."""
    messages = request.get('messages') or []
    roles = [m['role'] for m in messages]
    if not messages or roles[0] != 'user' or any(a == b for a, b in zip(roles, roles[1:])):   # one that ends with the assistant goes on from it
        return f'roles {roles}'
    blocks = [b for m in messages for b in m['content']]
    if any(not m['content'] for m in messages) or any(b['type'] == 'text' and not b['text'].strip() for b in blocks):
        return 'empty content or blank text'
    if sum('cache_control' in b for b in blocks) > 4:
        return 'more than four cache breakpoints'
    if not isinstance(request.get('max_tokens'), int) or not request.get('model'):
        return 'no model or max_tokens'
    tools = request.get('tools')
    if any(b['type'] in ('tool_use', 'tool_result') for b in blocks) and not tools:
        return 'tool blocks without tools'
    if tools and not all(t.get('name') and isinstance(t.get('input_schema'), dict) for t in tools):
        return f'a tool without a name or input_schema: {tools}'
    if (request.get('tool_choice') or {'type': 'auto'}).get('type') not in ('auto', 'any', 'none', 'tool'):
        return f'tool_choice {request["tool_choice"]}'
    if PONDERED in json.dumps(messages):
        return 'a memory\'s reasoning went into a request'
    chat = (request.get('tool_choice') or {}).get('type') != 'none'   # a memory's request replays turns without their reasoning
    for m in messages if chat else []:
        planned = [seen['planned'][b['id']] for b in m['content'] if b['type'] == 'tool_use' and b['id'] in seen['planned']]
        if planned and m['content'][0] != planned[0]:
            return f'the reasoning before tool calls didn\'t come back first, as it was: {m["content"][:2]}'
        seen['passbacks'] += bool(planned)
    for m, after in zip(messages, messages[1:] + [{'content': []}]):
        uses, answered = [b['id'] for b in m['content'] if b['type'] == 'tool_use'], [b['tool_use_id'] for b in after['content'] if b['type'] == 'tool_result']
        if sorted(uses) != sorted(answered):
            return f'tool calls {uses} answered by {answered}'
        first = [b['type'] == 'tool_result' for b in m['content']]
        if first != sorted(first, reverse=True):
            return 'tool results after other content'
    return None


def certificates(folder: str) -> tuple[str, str, str]:
    """A CA, and a certificate it issues for localhost: (the CA's PEM, the certificate's, its key's)."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID
    now, name = datetime.datetime.now(datetime.timezone.utc), lambda cn: x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])
    ca_key, key = ec.generate_private_key(ec.SECP384R1()), ec.generate_private_key(ec.SECP256R1())
    ca = (x509.CertificateBuilder().subject_name(name('test CA')).issuer_name(name('test CA')).public_key(ca_key.public_key())
          .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1)).not_valid_after(now + datetime.timedelta(days=1))
          .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True).sign(ca_key, hashes.SHA384()))
    cert = (x509.CertificateBuilder().subject_name(name('localhost')).issuer_name(ca.subject).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1)).not_valid_after(now + datetime.timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName('localhost')]), critical=False).sign(ca_key, hashes.SHA384()))
    paths = [os.path.join(folder, f) for f in ('ca.pem', 'cert.pem', 'key.pem')]
    for path, data in zip(paths, [ca.public_bytes(serialization.Encoding.PEM), cert.public_bytes(serialization.Encoding.PEM),
                                  key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())]):
        Path(path).write_bytes(data)
    return paths[0], paths[1], paths[2]


def serve(port: int, tls: tuple[str, str] | None = None, handler=Mock) -> ThreadingHTTPServer:
    """The mock on `port`, over TLS 1.3 with (certificate, key) if given."""
    server = ThreadingHTTPServer(('127.0.0.1', port), handler)
    if tls:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        context.load_cert_chain(*tls)
        server.socket = context.wrap_socket(server.socket, server_side=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def call(port, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    with urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=data, headers={'x-connectome': '1'}), timeout=60) as r:
        return json.loads(r.read())


def launch(root, port, ca, *flags) -> subprocess.Popen:
    """The app on `port`, its sessions in `root`, trusting `ca`, with `flags`, once it answers. What it
    writes goes to ROOT.log beside `root` rather than to a pipe nobody reads, which would stop the app
    once full; an app that stops, or doesn't answer in 10 s, fails the part with what it wrote."""
    log = f'{root}.log'
    with open(log, 'a') as output:   # the app has its own copy of the file
        app = subprocess.Popen([str(APP), '--root', root, '--addr', f'127.0.0.1:{port}', '--ca', ca, '--no-open', *flags], stderr=output)
    for _ in range(200):
        try:
            call(port, '/api/sessions')
            return app
        except OSError:
            if app.poll() is not None:
                break
            time.sleep(0.05)
    app.kill()
    raise SystemExit(f'the app on port {port} never answered; it wrote: {Path(log).read_text()[-2000:]}')


def configure(root, base, window, recent, chunk):
    """settings.json as the app's first run wrote it, edited as a user would: the default provider at
    `base`, and new sessions on the mock's model in `window` tokens, 1000 of them the reply's, folding
    in the band the window gives (fold_above the window less the reply, fold_to config.slack, 10%,
    under that), with `chunk`s and a recent window of `recent` sized for it."""
    path = Path(root, 'settings.json')
    everything = json.loads(path.read_text())
    if everything['defaults']['sandbox'] != sys.platform.startswith('linux') or everything['providers']['default']['models'] != [everything['defaults']['model']] \
            or everything['defaults']['config']['target_chunk_tokens'] != 3000 or (everything['defaults']['fold_above'], everything['defaults']['fold_to']) != (90000, 65000):
        raise SystemExit(f'a first run wrote {everything["defaults"]["sandbox"]=}, {everything["providers"]["default"]["models"]=} and chunks of {everything["defaults"]["config"]["target_chunk_tokens"]}')
    everything['providers']['default']['base'] = base
    everything['defaults'].update(model='mock', window=window, reserve=1000, fold_above=window - 1000, fold_to=(window - 1000) * 0.9)   # the window's own band
    everything['defaults']['config'].update(recent_window_tokens=recent, target_chunk_tokens=chunk)
    path.write_text(json.dumps(everything))


def words(message, kind='text'):
    """A logged message's text, or with `kind` thinking its reasoning, from the blocks the page gets."""
    return ''.join(b[kind] for b in message['content'] if b['type'] == kind)


def following(port, name, start=2 ** 62, timeout=10):
    """A page following a session from message `start` (by default the log's end; None, the stream's
    default): the response, and its events as they come, failing once `timeout` seconds pass with none
    (heartbeats aside)."""
    r = urllib.request.urlopen(f'http://127.0.0.1:{port}/api/s/{name}/stream' + ('' if start is None else f'?from={start}'), timeout=timeout)

    def events():
        heard = time.monotonic()
        for line in r:
            if line.strip():
                heard = time.monotonic()
                yield json.loads(line)
            elif time.monotonic() - heard > timeout:
                raise TimeoutError(f'{name}: {timeout} s of heartbeats alone')
    return r, events()


def say(port, name, text, answers=None) -> tuple[str, str, list[float]]:
    """A chat turn as a page following the session has it, or without text one that continues the log,
    until its memories have formed: the reply's text and reasoning, and when each piece of text arrived.
    A call that asks to run gets the answer `answers` has for its tool, else allow. The calls as the
    replies stream them in, each its tool and its input's JSON, go to seen['calls']."""
    pieces, thoughts, times, last, end, logged, calls, heard, after = [], [], [], {}, {}, 0, [], [], {}
    r, events = following(port, name)                                   # no turn here goes 10 s without an event
    with r:
        start, mine = next(events).get('state', {}).get('messages'), False   # the session and its log's length (none if it can't be read)
        job = call(port, f'/api/s/{name}/{"continue" if text is None else "say"}', {'text': text})['job']
        for last in events:
            heard += [last['message']['id']] if 'message' in last else []
            mine |= last.get('turn') == job
            if not mine:                                                # the end of a turn this one waits for
                continue
            if last.get('done') == job:
                after = next(events)                                    # the session as the job leaves it
                break
            logged = last.get('logged', logged)
            if 'text' in last:
                pieces.append(last['text'])
                times.append(time.monotonic())
            if 'result' in last:
                seen['events'].append(last)
            if 'retry' in last:
                seen['retries'].append(last['retry'])
            if 'running' in last:
                seen['running'].append(last['running'])
            if 'spent' in last:
                seen['costs'][name] = last['spent']
            if 'prompt' in last:
                seen['prompts'][name] = last['prompt']
            if 'fold' in last:                                              # before the reply to come: the log's next message
                seen['folds'].setdefault(name, {})[logged] = last['fold']
            if 'calling' in last:
                calls.append([last['calling'], ''])
            if 'input' in last:
                calls[-1][1] += last['input']
            seen['forming'] |= last.get('forming', False)
            end = last if 'end' in last else end
            if 'approve' in last:                                           # a call of the reply just logged, or a subagent's, which the log doesn't have
                asked = last['approve']
                if not asked.get('agent'):
                    reply = call(port, f'/api/s/{name}/messages?from={logged - 1}')[0]
                    if not any(b.get('id') == asked['id'] and b['name'] == asked['tool'] for b in reply['content']):
                        raise SystemExit(f'{name}: asked about {asked}, which the reply just logged doesn\'t have: {reply}')
                seen['asked'].append(asked['tool'])
                seen.setdefault('approvals', []).append(asked)
                choice = (answers or {}).get(asked['tool'], 'allow')   # an answer, or for questions what the user gives: {answers, annotations}
                call(port, f'/api/s/{name}/approve', {'id': asked['id'], 'answer': 'answer', **choice} if isinstance(choice, dict) else {'id': asked['id'], 'answer': choice})
            thoughts.append(last.get('thinking', ''))
    seen['calls'] = calls
    if end.get('end', 'never ended') is not None:
        raise SystemExit(f'{name}: "{(text or "(continue)")[:40]}" got {end or last}')
    # the session is read after the messages the page has, so a message that joins in between comes
    # after it (api::follow): the first heard may be below start
    first = heard[0] if heard else start
    if start is not None and (heard != list(range(first, first + len(heard))) or first > start) or 'state' not in after \
            or not after['state']['running'] and first + len(heard) != after['state']['messages']:
        raise SystemExit(f'{name}: a page following the session heard the messages {heard} after {start}, then {str(after)[:80]}')   # each once, as they joined the log
    return ''.join(pieces), ''.join(thoughts), times


def settle(port, name):
    """The session's state once its job (a turn, then its memories forming) is done, in seconds."""
    for _ in range(400):
        if not (state := call(port, f'/api/s/{name}'))['running']:
            return state
        time.sleep(0.05)
    raise SystemExit(f'{name}: a job still runs after 20 s')


def peak_kb(pid, of='VmHWM') -> int:
    return int(next(line.split()[1] for line in open(f'/proc/{pid}/status') if line.startswith(of)))


def peak_from_now(pid) -> int:
    """The process's size now, from which its peak counts again (Linux resets it on clear_refs 5)."""
    Path(f'/proc/{pid}/clear_refs').write_text('5')
    return peak_kb(pid, 'VmRSS')


def chat(port, name, turns, start, window=WINDOW):
    for t in range(start, start + turns):
        text = f'Message {t}: ' + ' '.join(f'word{t % 7}' for _ in range(150))
        reply, thinking, _ = say(port, name, text)
        state = settle(port, name)
        logged = call(port, f'/api/s/{name}/messages?from={state["messages"] - 1}')[0]
        # what streamed is the logged reply's start: a page a moment behind when it's logged reads the rest from the log;
        # the page gets the reasoning without its signature
        if not words(logged).startswith('Reply ') or not words(logged).startswith(reply) or not words(logged, 'thinking').startswith(thinking) \
                or any('signature' in b for b in logged['content']):
            raise SystemExit(f'turn {t}: the page got {reply[:60]!r} after {thinking[:40]!r}, the log holds {logged}')
        seen['thought'] += bool(thinking)
        if state['armed'] and state['armed'] > window * 1.02:
            raise SystemExit(f'turn {t}: a context of {state["armed"]} tokens in a window of {window}')
        if state['prompt'] != seen['prompts'][name]:                         # the last reply's, though memories' requests come after it
            raise SystemExit(f'turn {t}: the last prompt {state["prompt"]}, the last reply\'s {seen["prompts"][name]}')
    return state


def misgrouped(state, layers) -> str | None:
    """Why a session's merges aren't as memory formation makes them, if they aren't: each the fewest
    memories, at least min_merge_memories (or two, when folding needs them), that hold a chunk's tokens, of the layer below, or, lifted, of runs of
    several layers below, the highest first, side by side; each layer's merged memories its oldest,
    each merged once; each memory covering the chunks its members do."""
    target, covers, taken = state['settings']['config']['target_chunk_tokens'], state['covers'], set()
    least = state['settings']['config']['min_merge_memories']
    chunks = lambda level, j: (j, j + 1) if level == 1 else tuple(covers[level - 2][j])
    for n in range(2, len(layers)):
        for p, m in enumerate(layers[n]):
            runs = m['merges'] if isinstance(m['merges'][0], list) else [[n - 1, *m['merges']]]
            members = [(level, j) for level, a, b in runs for j in range(a, b + 1)]
            tokens, spans = [layers[level][j]['tokens'] for level, j in members], [chunks(*node) for node in members]
            if runs[0][0] != n - 1 or [r[0] for r in runs] != sorted({r[0] for r in runs}, reverse=True) or len(members) < 2 or sum(tokens) < target \
                    or len(members) > least and sum(tokens[:-1]) >= target or any(a[1] != b[0] for a, b in zip(spans, spans[1:])) or tuple(covers[n - 2][p]) != (spans[0][0], spans[-1][1]) or taken & set(members):
                return f'L{n} {p} merged {runs}, memories of {tokens} tokens (a chunk {target}) covering {spans}, as L{n} covers {covers[n - 2][p]}'
            taken |= set(members)
    for level in range(1, len(layers)):
        if (merged := sorted(j for k, j in taken if k == level)) != list(range(len(merged))):
            return f'L{level}\'s merged memories {merged[:12]} are not its oldest'
    return None


def lifted(tls, ca) -> str:
    """A session whose memories are large beside its chunks, so they merge two at a time, a layer
    forms every few turns and each keeps a memory no merge has taken: in a window too small for those
    and the tail, the oldest of them merge across layers (lifted), and every turn still gets its
    reply."""
    port, root, mock_port = free_port(), tempfile.mkdtemp(), free_port()
    mock = serve(mock_port, tls)
    app = launch(root, port, ca, '--no-bash')
    try:
        configure(root, f'https://localhost:{mock_port}', LIFTED, 400, 200)
        everything = json.loads(Path(root, 'settings.json').read_text())
        everything['defaults']['questions'] = False   # its windows are too small for ask_user's definition
        Path(root, 'settings.json').write_text(json.dumps(everything))
        call(port, '/api/sessions', {'name': 'lifted'})
        state = chat(port, 'lifted', 60, 0, LIFTED)
        layers = [call(port, f'/api/s/lifted/layer/{n}?from=0') for n in range(len(state['layers']))]
        lifts = [(n, m['merges']) for n in range(2, len(layers)) for m in layers[n] if isinstance(m['merges'][0], list)]
        if (why := misgrouped(state, layers)) or not lifts:
            raise SystemExit(why or f'no memories lifted in a window of {LIFTED}: layers {state["layers"]}, frontier {state["frontier"]}')
        call(port, '/api/sessions', {'name': 'backlog'})                 # memories down while a session grows in a roomy window,
        call(port, '/api/s/backlog/settings', {'window': 20000})         # then back as the window shrinks: the next turn waits
        seen['forgetful'] = True                                         # for every memory it takes to fit
        chat(port, 'backlog', 12, 0, 20000)
        seen['forgetful'], before = False, call(port, '/api/s/backlog')['layers']
        call(port, '/api/s/backlog/settings', {'window': 3000})
        after = chat(port, 'backlog', 1, 12, 3000)['layers']
        if sum(before[1:]) or sum(after[1:]) <= 3:
            raise SystemExit(f'a turn that waited for its memories found layers {before} and left {after}')
        call(port, '/api/sessions', {'name': 'pondering'})               # a summarizer whose tokens run out while it thinks: the turn
        call(port, '/api/s/pondering/settings', {'window': LIFTED})     # that waits for its memories says why they don't form
        seen['long-winded'] = True
        try:
            for t in range(40):
                say(port, 'pondering', f'Message {t}: ' + ' '.join(f'word{t % 7}' for _ in range(150)))
                settle(port, 'pondering')
            raise SystemExit('a session whose memories never formed fit its window for 40 turns')
        except SystemExit as stopped:
            if "The context doesn't fit" not in str(stopped) or 'A memory it waits for doesn\'t form: the summarizer\'s reply ran out of its' not in str(stopped):
                raise
        seen['long-winded'] = False
        call(port, '/api/sessions', {'name': 'banded'})                  # a band in a big window: past fold_above it folds, down to fold_to
        for wrong in ({'fold_above': 19500}, {'fold_above': 4000, 'fold_to': 4500}, {'fold_above': None}, {'fold_to': None}, {'config': {'target_chunk_tokens': 0}}):   # never blank
            try:
                call(port, '/api/s/banded/settings', {'window': 20000, **wrong})
                raise SystemExit(f'the settings took {wrong} in a window of 20000, 1000 of it the reply\'s')
            except urllib.error.HTTPError as error:
                if error.code != 400 or not re.search('fold_|a chunk', json.loads(error.read())['error']):
                    raise SystemExit(f'{wrong} got {error.code}')
        band, asked = {'window': 20000, 'fold_above': 4000, 'fold_to': 2000, 'config': {'compression_max_tokens': 24000}}, len(seen['memory max'])   # memories of up to 24000 tokens, past the library's 16000
        overhead, contexts = call(port, '/api/s/banded/settings', band)['overhead'], []   # the tools and system prompt
        for t in range(40):
            contexts.append(chat(port, 'banded', 1, t, 20000)['armed'])
        folds = [i for i in range(1, len(contexts)) if contexts[i] < contexts[i - 1]]
        if max(contexts) > 4000 - overhead or not folds or any(contexts[i] > 2000 - overhead for i in folds) or len(folds) > len(contexts) // 4:
            raise SystemExit(f'contexts in a band of 2000 to 4000 tokens of prompt: {contexts}, folding at {folds}')
        if set(seen['memory max'][:asked]) != {20000} or set(seen['memory max'][asked:]) != {24000}:   # a first run's 20000, then the session's own
            raise SystemExit(f'memory requests asked for {seen["memory max"][:asked][:4]}, then {seen["memory max"][asked:][:4]} tokens')
        return f'{len(lifts)} lifted merges (the first L{lifts[0][0]} of {lifts[0][1]}), layers {state["layers"]}, frontier {state["frontier"]}; ' \
               f'a turn waited for {sum(after[1:])} memories; contexts in a band of 2000 to 4000 folded {len(folds)} times in 40 turns'
    finally:
        app.terminate()
        mock.shutdown()


def streamed_large(tls, ca) -> str:
    """A session of large messages in a huge window, each context streaming all of them: from the
    second turn to the last the contexts grow sixfold, and the app's peak memory may grow by less
    than three messages, and no more as the page lists them all; a message pasted past a megabyte joins
    the log whole, and a body that isn't JSON is refused. The app runs without bash (--no-bash), which a
    session with a workspace then lacks."""
    port, root, size, mock_port = free_port(), tempfile.mkdtemp(), 900_000, free_port()
    mock = serve(mock_port, tls)
    app = launch(root, port, ca, '--no-bash')
    try:
        configure(root, f'https://localhost:{mock_port}', 20000000, 30000, 3000)
        call(port, '/api/sessions', {'name': 'large', 'workspace': tempfile.mkdtemp()})
        r, _ = following(port, 'large')                                  # an idle page hears a blank line every few seconds
        with r:
            waited = (r.readline(), time.monotonic())[1]
            if r.readline() != b'\n' or time.monotonic() - waited > 6:
                raise SystemExit('a page following a session no turn runs in heard no heartbeat')
        if (tools := call(port, '/api/s/large')['tools']) != ['edit']:
            raise SystemExit(f'without bash, a session with a workspace has tools {tools}, not only edit')
        peaks, contexts = [], []
        for t in range(12):
            words = ' '.join(f'w{t}x{i}' for i in range(size // 8))[:size]
            say(port, 'large', words)
            settle(port, 'large')
            peaks.append(peak_kb(app.pid))
            contexts.append(seen['largest'] // 1024)
        if peaks[-1] - peaks[1] > 3 * size // 1024:
            raise SystemExit(f'peak memory grew from {peaks[1]} KB to {peaks[-1]} KB as contexts grew from {contexts[1]} KB to {contexts[-1]} KB')
        before = peak_from_now(app.pid)                                  # the listing's own peak, however high the turns' went
        if len(call(port, '/api/s/large/messages')) != 24 or (listed := peak_kb(app.pid)) - before > 3 * size // 1024:
            raise SystemExit(f'listing the messages took the app from {before} KB to a peak of {listed} KB')
        say(port, 'large', 'p' * 1_500_000)                              # a paste past a megabyte
        if len(call(port, '/api/s/large/messages?from=24')[0]['content'][0]['text']) != 1_500_000:
            raise SystemExit('a pasted message past a megabyte did not join the log whole')
        try:
            urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/api/s/large/say', data=b'{not JSON', headers={'x-connectome': '1'}), timeout=10)
            raise SystemExit('the app took a body that is not JSON')
        except urllib.error.HTTPError as error:
            if error.code != 400 or 'body' not in json.loads(error.read())['error']:
                raise SystemExit(f'a body that is not JSON got {error.code}')
        return f'contexts of {contexts[1]} KB to {contexts[-1]} KB streamed at peaks of {peaks[1]} KB to {peaks[-1]} KB'
    finally:
        app.terminate()
        mock.shutdown()


def tooled(tls, ca, baseline) -> str:
    """A session whose model calls bash in its workspace and the tools of two MCP servers (fake_mcp.py,
    the second, whose tools are wordy, shadowing the first's add), and of none; a session without a
    workspace has no bash. Each call runs once the user allows it, or always allows its tool for the
    session, and fails if they deny it, the tools always allowed kept across a restart; a command past
    its timeout is killed with the processes it started, and it or an MCP call past its timeout fails
    without holding up the next call; a server that stops starts again. In the sandbox, bash sees its
    workspace, read-write, and the paths settings.json names, as an edit by hand leaves them, and a
    timeout still kills all it started. A page that comes in while a turn waits for an answer gets
    the turn so far, then the rest as it happens, the same as the page that started it, and answers;
    a message sent meanwhile waits for the turn; a page following no turn gets nothing. Every call gets its result,
    in the log and on the page; the requests fit the window with the tool definitions counted, which
    leave the price multiplier near a session's without them (from `baseline`); a turn that never stops
    calling tools runs on past 64 steps until the user stops it, memories forming between its steps;
    and memories of the calls form with the tools declared but not callable. ask_user's questions wait
    for the user's answers, which are their result, or their decline, never the allow list; with the
    session's questions off, it isn't declared."""
    port, root, work, fake, window, mock_port = free_port(), tempfile.mkdtemp(), tempfile.mkdtemp(), str(HERE / 'fake_mcp.py'), 12000, free_port()
    mock = serve(mock_port, tls)
    servers = {'first': {'command': sys.executable, 'args': [fake]}, 'second': {'command': sys.executable, 'args': [fake, '--shadow', '--wordy']},
               'broken': {'command': '/nonexistent/server'}}
    Path(root, 'mcp.json').write_text(json.dumps({'mcpServers': servers}))

    def start():
        return launch(root, port, ca, '--tool-timeout', '1')
    app, seen['limit'] = start(), window
    configure(root, f'https://localhost:{mock_port}', window, 2381, 595.25)
    everything = json.loads(Path(root, 'settings.json').read_text())
    everything['defaults']['questions'] = False   # its window's budget is set without ask_user's definition; questions go on where they're tried
    Path(root, 'settings.json').write_text(json.dumps(everything))

    def results(text, answers=None, asked=None):
        """The results the turn's tool calls got, by the call's place in the reply."""
        before = len(seen['asked'])
        say(port, 'tools', text, answers)
        settle(port, 'tools')
        if asked is not None and seen['asked'][before:] != asked:
            raise SystemExit(f'"{text}" asked to run {seen["asked"][before:]}, not {asked}')
        return {r['tool_use_id'].split('-')[1]: r for r in seen['results'][-1]}
    try:
        call(port, '/api/sessions', {'name': 'tools', 'workspace': work + '/'})   # kept without the slash
        if (names := call(port, '/api/sessions', {'name': 'bare'})['tools']) != ['fail', 'picture', 'hang', 'die', 'add']:
            raise SystemExit(f'a session without a workspace has tools {names}')
        if call(port, '/api/s/bare')['spent'] is not None or call(port, '/api/s/bare')['prompt'] is not None:
            raise SystemExit('a session no request has priced has a cost')
        if (spaces := {s['name']: s['workspace'] for s in call(port, '/api/sessions')}) != {'tools': work, 'bare': None}:
            raise SystemExit(f'the sessions\' workspaces: {spaces}')
        state = call(port, '/api/s/tools')
        if (names := state['tools']) != ['bash', 'edit', 'fail', 'picture', 'hang', 'die', 'add'] or state['overhead'] < 2000:
            raise SystemExit(f'tools {names} ({state["overhead"]} tokens): expected bash, edit, the first server\'s but its add, then the second\'s wordy add')
        ran = len(seen['running'])
        got = results('please use tools now', {'fail': 'deny', 'picture': 'always'}, ['add', 'fail', 'picture'])
        if [(r['tool'], r['id'].split('-')[1]) for r in seen['running'][ran:]] != [('add', '0'), ('picture', '2'), ('nope', '3')]:   # the denied one never runs
            raise SystemExit(f'the page heard calls running: {seen["running"][ran:]}')
        expected = {'0': [{'type': 'text', 'text': '5 (shadow)'}], '1': "the user didn't allow this call",
                    '2': [{'type': 'text', 'text': 'a picture:'}, {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'iVBORw0KGgo='}}],
                    '3': 'no tool nope'}
        if {k: r['content'] for k, r in got.items()} != expected or [k for k, r in sorted(got.items()) if r.get('is_error')] != ['1', '3']:
            raise SystemExit(f'tool results: {got}')
        if [e['result']['toolUseId'][-1] for e in seen['events']] != ['0', '1', '2', '3'] or 'iVBORw0KGgo' in json.dumps(seen['events']):
            raise SystemExit(f'the page got {seen["events"]}')
        call(port, '/api/sessions', {'name': 'asker'})                   # its own session, so the tools session's budget stays as it was
        call(port, '/api/s/asker/settings', {'questions': True})
        colors, pets = 'What is your favorite color?', 'Which pets do you like?'
        questions = {'questions': [{'question': colors, 'header': 'Color', 'options': [{'label': 'Red', 'description': 'Warm'}, {'label': 'Blue', 'description': 'Cool'}]},
                                   {'question': pets, 'header': 'Pets', 'options': [{'label': 'Cats', 'description': 'Cats'}, {'label': 'Dogs', 'description': 'Dogs'}], 'multiSelect': True}]}
        given = {'answers': {colors: 'Blue', pets: 'Cats, Dogs'}, 'annotations': {colors: {'notes': 'navy if possible'}}}
        asking = 'call ' + json.dumps([['ask_user', questions]])
        before = len(seen['asked'])
        say(port, 'asker', asking, {'ask_user': given})
        got = seen['results'][-1][0]
        if got.get('content') != f'The user answered: "{colors}"="Blue" notes: navy if possible, "{pets}"="Cats, Dogs".' or got.get('is_error') \
                or seen['approvals'][-1]['input'] != questions or seen['asked'][before:] != ['ask_user']:
            raise SystemExit(f'ask_user, answered, gave {got}, asked as {seen["approvals"][-1]}')
        say(port, 'asker', asking, {'ask_user': 'deny'})
        got = seen['results'][-1][0]
        if (got.get('content'), got.get('is_error')) != ('The user declined to answer the questions.', True) or 'ask_user' in call(port, '/api/s/asker')['tools']:
            raise SystemExit(f'ask_user, declined, gave {got}; the tools listed to always allow are {call(port, "/api/s/asker")["tools"]}')
        call(port, '/api/s/asker/settings', {'questions': False})
        say(port, 'asker', 'hello')
        if 'ask_user' in seen['requests'][-1][1] or 'ask_user' not in seen['requests'][-3][1]:
            raise SystemExit(f'with questions off the session declares {seen["requests"][-1][1]}')
        allowed = json.loads(Path(root, 'tools', 'session_settings.json').read_text())['allow']
        if allowed != ['picture'] or call(port, '/api/s/tools')['settings']['allow'] != ['picture'] or call(port, '/api/s/bare')['settings']['allow']:
            raise SystemExit(f'always allowed: {allowed}, in the session only')
        blocks = [b for m in call(port, '/api/s/tools/messages') for b in m['content']]          # the page gets an image's size, not its data
        if not any(b['type'] == 'tool_use' and (b['name'], b['input']) == ('add', {'a': 2, 'b': 3}) for b in blocks) or 'iVBORw0KGgo' in json.dumps(blocks) \
                or not any(b['type'] == 'tool_result' and b.get('isError') and b['content'] == "the user didn't allow this call" for b in blocks) \
                or {'type': 'image', 'kb': 0} not in [c for b in blocks if b['type'] == 'tool_result' and isinstance(b['content'], list) for c in b['content']]:
            raise SystemExit(f'the log shows {json.dumps(blocks)[:400]}')
        got = results('call [["bash", {"command": "pwd; echo out; echo err >&2; exit 3"}]]', {'bash': 'always'}, ['bash'])
        if got['0']['content'] != f'{work}\nout\nerr\n\n[exit code 3]' or got['0'].get('is_error'):
            raise SystemExit(f'bash: {got}')
        started = time.monotonic()
        got = results('call [["bash", {"command": "(sleep 0.8; touch survived) & sleep 5; echo never", "timeout": 0.4}]]', asked=[])
        if got['0']['content'] != '\n[killed after 0.4 s]' or not got['0'].get('is_error') or time.monotonic() - started > 4:
            raise SystemExit(f'bash past its timeout: {got} after {time.monotonic() - started:.1f} s')
        time.sleep(max(0.0, started + 1.4 - time.monotonic()))
        if Path(work, 'survived').exists():
            raise SystemExit('a process that a command past its timeout started outlived it')
        shared, shown, secret = Path(root, 'shared'), Path(root, 'shown'), Path(root, 'secret')
        shared.mkdir()
        shown.mkdir()
        Path(shown, 'seen').write_text('seen\n')
        secret.write_text('hidden')
        everything = json.loads(Path(root, 'settings.json').read_text())
        everything['sandbox'].update(ro=everything['sandbox']['ro'] + [str(shown)], rw=[str(shared)], network=False)   # the rest of /tmp, where root is, out of sight
        Path(root, 'settings.json').write_text(json.dumps(everything))
        call(port, '/api/s/tools/settings', {'sandbox': True})
        command = f'pwd; echo made > inside; cat inside; echo noted > {shared}/note; cat {secret} 2>&1; cat {shown}/seen; touch {shown}/x 2>&1; touch /stray 2>&1; echo $$; tail -n +3 /proc/net/dev | cut -d: -f1 | tr -d " "'
        got = results('call ' + json.dumps([['bash', {'command': command}]]), asked=[])
        sandboxed = f"{work}\nmade\ncat: {secret}: No such file or directory\nseen\ntouch: cannot touch '{shown}/x': Read-only file system\ntouch: cannot touch '/stray': Read-only file system\n2\nlo\n\n[exit code 0]"
        if got['0']['content'] != sandboxed or Path(work, 'inside').read_text() != 'made\n' or Path(shared, 'note').read_text() != 'noted\n':
            raise SystemExit(f'bash in the sandbox: {got}')
        started = time.monotonic()
        got = results('call ' + json.dumps([['bash', {'command': '(sleep 0.8; touch survived) & sleep 5', 'timeout': 0.4}]]), asked=[])
        time.sleep(max(0.0, started + 1.4 - time.monotonic()))
        if got['0']['content'] != '\n[killed after 0.4 s]' or Path(work, 'survived').exists():
            raise SystemExit(f'a sandboxed command past its timeout: {got}, or what it started outlived it')
        os.symlink(root, Path(work, 'out'))                             # edits, in the sandbox and out of it: a file made, its text replaced; refusals; ways out
        for sandboxed in (True, False):
            call(port, '/api/s/tools/settings', {'sandbox': sandboxed})
            shutil.rmtree(Path(work, 'notes'), ignore_errors=True)
            edits = [{'path': 'notes/a.txt', 'old': '', 'new': 'one\ntwo\ntwo\n'}, {'path': 'notes/a.txt', 'old': 'one', 'new': 'uno'},
                     {'path': 'notes/a.txt', 'old': 'two', 'new': 'dos'}, {'path': 'notes/a.txt', 'old': 'two', 'new': 'dos', 'all': True},
                     {'path': 'notes/a.txt', 'old': 'three', 'new': 'x'}, {'path': 'notes/a.txt', 'old': '', 'new': 'x'},
                     {'path': '../escaped', 'old': '', 'new': 'x'}, {'path': 'new/../../escaped', 'old': '', 'new': 'x'}, {'path': f'{root}/escaped', 'old': '', 'new': 'x'},
                     {'path': 'out/escaped', 'old': '', 'new': 'x'}]
            got = results('call ' + json.dumps([['edit', e] for e in edits]), asked=['edit'] * len(edits))
            outcomes = [(got[str(k)]['content'], got[str(k)].get('is_error', False)) for k in range(len(edits))]
            outside = lambda path: (f'{path} is outside the workspace, {work}', True)
            expected = [('made notes/a.txt, 3 lines', False), ('edited notes/a.txt at line 1', False),
                        ('the text to replace is in notes/a.txt 2 times: give more of it, around the change, or all', True), ('edited notes/a.txt in 2 places, the first at line 2', False),
                        ("the text to replace isn't in notes/a.txt", True), ('notes/a.txt exists: give the text to replace in it', True), outside('../escaped'), outside('new/../../escaped'), outside(f'{root}/escaped'), outside('out/escaped')]
            if outcomes != expected or Path(work, 'notes/a.txt').read_text() != 'uno\ndos\ndos\n' or Path(root, 'escaped').exists() or Path(work).parent.joinpath('escaped').exists():
                raise SystemExit(f'edits{" in the sandbox" if sandboxed else ""}: {outcomes}')
            if [[tool, json.loads(given)] for tool, given in seen['calls']] != [['edit', e] for e in edits]:
                raise SystemExit(f'the calls as the reply streamed them: {seen["calls"][:2]}')
        os.remove(Path(work, 'out'))
        call(port, '/api/sessions', {'name': 'reading', 'workspace': work})   # in the sandbox, a command reads the file, which says so when there's none
        call(port, '/api/s/reading/settings', {'allow': ['edit']})
        for sandboxed in (True, False):
            call(port, '/api/s/reading/settings', {'sandbox': sandboxed})
            say(port, 'reading', 'call ' + json.dumps([['edit', {'path': 'notes/none.txt', 'old': 'x', 'new': 'y'}]]))
            missing = [b for m in call(port, '/api/s/reading/messages') for b in m['content'] if b['type'] == 'tool_result'][-1]['content']
            if missing != (f'notes/none.txt: cat: {work}/notes/none.txt: No such file or directory' if sandboxed else 'notes/none.txt: No such file or directory (os error 2)'):
                raise SystemExit(f'an edit of a file there isn\'t{", in the sandbox," if sandboxed else ""} said {missing!r}')
            say(port, 'reading', 'echo next')                           # the mock answers results and a message together with a reply
        got = results('call [["hang", {}], ["picture", {}]]', asked=['hang'])
        if got['0']['content'] != 'MCP server first: tools/call timed out after 1 s' or got['1'].get('is_error'):
            raise SystemExit(f'a call past its timeout, then another of the same server: {got}')
        got = results('call [["die", {}], ["picture", {}]]', asked=['die'])
        if got['0']['content'] != 'MCP server first: the server stopped' or got['1'].get('is_error'):
            raise SystemExit(f'a server that stops, then a call that starts it again: {got}')
        first, job, asking, state = [], [], 'call ' + json.dumps([['add', {'a': 1, 'b': 2}]]), call(port, '/api/s/tools')

        def started():                                                  # a page that sends a message, following the session until its turn is done
            r, events = following(port, 'tools', timeout=60)
            with r:
                first.append(next(events))
                job.append(call(port, '/api/s/tools/say', {'text': asking})['job'])
                for e in events:
                    first.append(e)
                    if e.get('done') == job[0]:
                        break
        starter = threading.Thread(target=started)
        starter.start()
        for _ in range(500):                                            # until the turn waits for an answer, which it does until one comes
            if any('approve' in e for e in first) and call(port, '/api/s/tools')['running']:
                break
            time.sleep(0.02)
        else:
            raise SystemExit('a turn waiting for an answer did not show as running')
        try:
            call(port, '/api/s/tools/approve', {'id': 'call-none', 'answer': 'allow'})
            raise SystemExit('a turn waiting for an answer took one for a call that is not waiting')
        except urllib.error.HTTPError as error:
            if error.code != 400:
                raise
        queued = 'Message queued while a turn waits'
        waiter = threading.Thread(target=lambda: say(port, 'tools', queued))
        waiter.start()
        time.sleep(0.2)                                                 # it reaches the session's thread before the answer
        late = []
        r, events = following(port, 'tools')
        with r:
            for e in events:
                late.append(e)
                if 'approve' in e:
                    call(port, '/api/s/tools/approve', {'id': e['approve']['id'], 'answer': 'allow'})
                if e.get('done') == job[0]:
                    break
        starter.join()
        waiter.join()
        # it hears the session (the turn running, and the message queued for after it), then the turn's start and where
        # the log stands (the message and the reply that calls logged, none of the reply's pieces), then what the first
        # page hears from there on, but the messages, which each page reads from the log from where it started, and the
        # queue, which a page hears in the session as it comes in and then as it changes
        outline = lambda events: [e for e in events if not {'text', 'thinking', 'message', 'state', 'queue'} & set(e)]
        kinds = [next(iter(e)) for e in outline(late) if not {'logged', 'forming', 'spent', 'prompt'} & set(e)]
        if outline(first)[-len(outline(late)) + 1:] != outline(late)[1:] or not late[0]['state']['running'] or [q['text'] for q in late[0]['state']['queued']] != [queued] \
                or outline(late)[:2] != [{'turn': job[0]}, {'logged': state['messages'] + 2}] \
                or 'approve' not in outline(late)[2] or kinds != ['turn', 'approve', 'answered', 'running', 'result', 'end', 'done'] or call(port, '/api/s/tools')['running']:
            raise SystemExit(f'a page that came in during the turn got {kinds} from {late[:3]}; the page that started it {len(first)} events, it {len(late)}')
        told = call(port, f'/api/s/tools/messages?from={state["messages"]}')
        if [m['participant'] for m in told] != ['user', 'assistant', 'user', 'assistant', 'user', 'assistant'] or words(told[4]) != queued:
            raise SystemExit(f'a message sent while a turn ran joined the log as {[(m["participant"], words(m)[:20]) for m in told]}')
        r, events = following(port, 'tools', timeout=1)                 # a page following a session no turn runs in hears the session, and no more
        with r:
            heard = [next(events)]
            try:
                heard.append(next(events))
            except TimeoutError:
                pass
        if len(heard) != 1 or heard[0]['state']['running']:
            raise SystemExit(f'a page following a session no turn runs in heard {heard}')

        call(port, '/api/sessions', {'name': 'halt', 'workspace': tempfile.mkdtemp()})   # a session to stop turns in, bash always allowed
        call(port, '/api/s/halt/settings', {'allow': ['bash', 'hang']})

        def stopped(text, at):
            """The turn `text` starts, stopped at the first event `at` takes: its events, and the seconds
            from the stop to the turn's end."""
            r, events = following(port, 'halt')
            with r:
                next(events)
                job, heard, asked = call(port, '/api/s/halt/say', {'text': text})['job'], [], None
                for e in events:
                    heard.append(e)
                    if asked is None and at(e):
                        asked = time.monotonic()
                        call(port, '/api/s/halt/stop', {})
                    if 'end' in e:
                        took = time.monotonic() - asked
                    if e.get('done') == job:
                        return heard, took
        last = lambda: call(port, '/api/s/halt/messages')[-1]
        story = 'stop me ' * 40
        heard, took = stopped(f'echo {story}slowly', lambda e: 'text' in e)   # a reply streaming in ends where it is, and joins the log so
        if {'end': 'Stopped by the user.'} not in heard or took > 1 or not 0 < len(words(last())) < len(story) or any('forming' in e for e in heard):
            raise SystemExit(f'a reply stopped as it streamed ended {[e for e in heard if "end" in e]} {took:.1f} s on, the log has {words(last())[:40]!r}')
        def running(tool):                                              # a call running a tenth of a second: once the reply that makes it has joined the log
            logged = []
            return lambda e: (logged.append(e) if 'logged' in e else None, len(logged) == 2 and not time.sleep(0.1))[1]
        heard, took = stopped('call ' + json.dumps([['bash', {'command': 'sleep 30; echo late'}]]), running('bash'))   # a command running is killed
        if not last()['content'][0]['content'].endswith('[stopped by the user]') or 'late' in last()['content'][0]['content'] or took > 0.6:   # --tool-timeout is 1 s
            raise SystemExit(f'a command stopped as it ran gave {last()["content"]}, the turn ending {took:.1f} s on')
        say(port, 'halt', 'echo next')                                  # the mock answers results and a message together with a reply
        passbacks = seen['passbacks']                                   # a reply's reasoning before its calls goes back with them, as it was
        say(port, 'halt', 'call ' + json.dumps([['bash', {'command': 'echo reasoned'}]]))
        if seen['passbacks'] <= passbacks:
            raise SystemExit('the reasoning before a reply\'s calls never went back with them')
        heard, took = stopped('call ' + json.dumps([['hang', {}]]), running('hang'))   # an MCP call running is left
        if 'stopped by the user' not in last()['content'][0]['content'] or took > 0.6:
            raise SystemExit(f'an MCP call stopped as it ran gave {last()["content"]}, the turn ending {took:.1f} s on')
        say(port, 'halt', 'echo next')
        heard, took = stopped('call ' + json.dumps([['add', {'a': 1, 'b': 1, 'pace': 'slowly'}]]), lambda e: 'calling' in e)   # a call the reply is writing goes
        if [b['type'] for b in last()['content']] != ['text'] or {'end': 'Stopped by the user.'} not in heard:
            raise SystemExit(f'a reply stopped as it wrote a call joined the log as {last()}')
        heard, took = stopped('call ' + json.dumps([['add', {'a': 1, 'b': 1}], ['add', {'a': 2, 'b': 2}]]), lambda e: 'approve' in e)   # calls waiting or yet to run don't
        if [(b['content'], b.get('isError')) for b in last()['content']] != [('Stopped by the user.', True)] * 2 or {'end': 'Stopped by the user.'} not in heard:
            raise SystemExit(f'calls stopped while one waited for an answer: {last()["content"]}')
        try:
            call(port, '/api/s/halt/stop', {})
            raise SystemExit('a session no turn runs in took a stop')
        except urllib.error.HTTPError as error:
            if error.code != 400:
                raise
        everything = json.loads(Path(root, 'settings.json').read_text())   # a SearXNG server named in settings.json: sessions get web_search
        Path(root, 'settings.json').write_text(json.dumps({**everything, 'search': everything['providers']['default']['base'] + '/'}))
        if 'web_search' not in call(port, '/api/s/halt')['tools']:
            raise SystemExit(f'with a SearXNG server, a session has the tools {call(port, "/api/s/halt")["tools"]}')
        say(port, 'halt', 'echo next')
        say(port, 'halt', 'call ' + json.dumps([['web_search', {'query': 'bees and wasps', 'count': 1}], ['web_search', {'query': 'bees'}], ['web_search', {'query': 'no json'}],
                                                ['web_search', {'query': 'many', 'count': 99}]]))
        found = [(r['content'], r.get('is_error', False)) for r in seen['results'][-1]]
        Path(root, 'settings.json').write_text(json.dumps(everything))
        if found[:2] != [('1. Bees\nhttps://example.com/bees\n2026-07-27 · All about bees and wasps.', False),
                         ('1. Bees\nhttps://example.com/bees\n2026-07-27 · All about bees.\n\n2. Wasps\nhttps://example.com/wasps\nLess about bees.', False)] \
                or 'search.formats' not in found[2][0] or not found[2][1] or found[3][0].count('https://example.com/') != 20 or seen['asked'][-4:] != ['web_search'] * 4:
            raise SystemExit(f'web_search gave {found}, the server asked {seen.get("searched")!r}')
        r, events = following(port, 'tools')                            # a turn that never stops calling tools runs on until the user stops it
        with r:
            next(events)
            job, results, ends = call(port, '/api/s/tools/say', {'text': 'use tools forever'})['job'], 0, []
            for e in events:
                if 'approve' in e:
                    call(port, '/api/s/tools/approve', {'id': e['approve']['id'], 'answer': 'always'})
                results += 'result' in e
                if 'result' in e and results == 70:                     # its first step calls four tools, each after one: 67 steps
                    call(port, '/api/s/tools/stop', {})
                ends += [e['end']] if 'end' in e else []
                if e.get('done') == job:
                    break
        if results < 70 or ends != ['Stopped by the user.']:
            raise SystemExit(f'a turn that never stops calling tools ended {ends} after {results} results')
        settle(port, 'tools')
        compiles = [json.loads(line) for line in Path(root, 'tools', 'compiles.jsonl').read_text().splitlines()]
        if (escalated := sum(c['escalated'] for c in compiles)) > 2:
            raise SystemExit(f'{escalated} of {len(compiles)} compiles ran out of room: memories did not form between steps')
        state = chat(port, 'tools', 24, 1000, window)
        if seen['problems']:
            raise SystemExit(f'the API would reject a request: {seen["problems"][0]}')
        if not seen['tool memories']:
            raise SystemExit('no memory request carried tool calls')
        if state['multiplier'] > (multiplier := baseline.get(timeout=120)) + 0.1:
            raise SystemExit(f'the tool definitions skewed the price multiplier: {state["multiplier"]:.3f}, without them {multiplier:.3f}')
        call(port, '/api/s/tools/settings', {'allow': ['bash']})
        app.terminate()
        app.wait()
        app = start()
        if (allowed := call(port, '/api/s/tools')['settings']['allow']) != ['bash']:
            raise SystemExit(f'after a restart the tools always allowed are {allowed}, not bash, as the page set them')
        if (ids := [m['id'] for m in call(port, '/api/s/tools/messages?to=100')]) != list(range(100)):   # the newest 200 up to 100
            raise SystemExit(f'the messages up to 100 listed as {ids[:3]}…{ids[-3:]}')
        r, events = following(port, 'tools', start=None)                # by default a page hears the log's newest 200 messages, then the session
        with r:
            told = [e['message']['id'] for e in itertools.takewhile(lambda e: 'state' not in e, events)]
        if told != list(range(state['messages'] - 200, state['messages'])):
            raise SystemExit(f'a page following a session of {state["messages"]} messages heard {told[:3]}…{told[-3:]} first')
        r, events = following(port, 'tools')                            # the server stops while a call waits for its answer: Continue goes on
        with r:
            next(events)
            call(port, '/api/s/tools/say', {'text': 'call [["add", {"a": 1, "b": 2}]]'})
            next(e for e in events if 'approve' in e)
        app.kill()
        app.wait()
        app = start()
        say(port, 'tools', None)
        mended = call(port, f'/api/s/tools/messages?from={call(port, "/api/s/tools")["messages"] - 3}')
        if [m['participant'] for m in mended] != ['assistant', 'user', 'assistant'] or not mended[1]['content'][0].get('isError') \
                or 'the server stopped before this call finished' not in mended[1]['content'][0]['content']:
            raise SystemExit(f'a call the server stopped in, then Continue: {mended}')
        return f'{len(seen["results"])} tool steps, {len(seen["asked"])} calls asked to run, {seen["tool memories"]} memory requests with tool calls, {state["messages"]} messages, multiplier {state["multiplier"]:.3f}'
    finally:
        app.terminate()
        mock.shutdown()


def calm(port, *names):
    """Once no turn has run in the sessions for six polls in a row, a tenth of a second apart."""
    still = 0
    for _ in range(300):
        still = 0 if any(call(port, f'/api/s/{n}')['running'] for n in names) else still + 1
        if still == 6:
            return
        time.sleep(0.1)
    raise SystemExit(f'turns still run in {names} after 30 s')


def accompanied(tls, ca) -> str:
    """A session and its companion, made with a goal: it starts on the session's provider and model,
    without a workspace, the goal in its system prompt, Auto on; each declares message, the companion
    finish too. The session's last turn goes to the companion, and each one's turn's end to the other,
    each heading what it hears from the other with how it came, until the companion calls finish, after
    which its answer is the last the session hears. With Auto off, a message waited for comes back as
    the call's result, and one sent while working gets its answer once; nothing more runs after; one
    that comes during a turn's last step, the turn goes on to answer. Auto
    turned on from the page sends the session's last turn, with what it ran, to the companion; a turn
    that fails turns Auto off for both. A session paired already, one paired by the other side, a
    name taken and a missing goal are refused, as is Auto without a companion; a fork has none. Messages
    sent during a turn wait, queued, and can be changed or taken back until they go."""
    port, root, mock_port, work = free_port(), tempfile.mkdtemp(), free_port(), tempfile.mkdtemp()
    mock = serve(mock_port, tls)
    app = launch(root, port, ca)
    configure(root, f'https://localhost:{mock_port}', 12000, 2381, 595.25)
    quiet = lambda *names: calm(port, *names)
    log = lambda name: [(m['participant'], words(m)) for m in call(port, f'/api/s/{name}/messages')]
    try:
        call(port, '/api/sessions', {'name': 'pair', 'workspace': work})
        say(port, 'pair', 'echo echo echo call [["finish", {}]]')
        settle(port, 'pair')
        before = len(seen['requests'])
        call(port, '/api/s/pair/companion', {'name': 'pal', 'goal': 'Count the bees.'})
        quiet('pair', 'pal')
        finish = 'call [["finish", {}]]'
        reply = lambda text: text.startswith('Reply ')
        pair, pal = log('pair'), log('pal')
        if pair[:4] != [('user', 'echo echo echo ' + finish), ('assistant', 'echo echo ' + finish), ('pal', 'pal ended its turn:\n\necho ' + finish), ('assistant', finish)] \
                or [p for p, _ in pair[4:]] != ['pal', 'assistant'] or not pair[4][1].startswith('pal ended its turn:\n\nReply ') or not reply(pair[5][1]):
            raise SystemExit(f'the session heard its companion as {pair}')
        if pal[:3] != [('pair', 'pair ended its turn:\n\necho echo ' + finish), ('assistant', 'echo ' + finish), ('pair', 'pair ended its turn:\n\n' + finish)] \
                or [p for p, _ in pal[3:]] != ['assistant', 'user', 'assistant'] or not reply(pal[5][1]):
            raise SystemExit(f'the companion heard the session as {pal}')
        own, its = call(port, '/api/s/pair')['settings'], call(port, '/api/s/pal')['settings']
        if (own.get('companion'), own.get('auto'), its.get('accompanies'), its.get('auto'), its['workspace']) != ('pal', None, 'pair', None, None) \
                or (its['provider'], its['model']) != (own['provider'], own['model']) or 'Count the bees.' not in its['system']:
            raise SystemExit(f'after finish the session has {own}, its companion {its}')
        declared = {('pal' if 'Count the bees.' in system else 'pair', tuple(t for t in tools if t in ('message', 'finish'))) for system, tools in seen['requests'][before:]}
        if declared != {('pal', ('message', 'finish')), ('pair', ('message',))}:
            raise SystemExit(f'the requests declared {declared}: {len(seen["requests"])} requests, {before} before the companion, the last {seen["requests"][-3:]}')
        if {s['name']: (s['companion'], s['accompanies']) for s in call(port, '/api/sessions')} != {'pair': ('pal', None), 'pal': (None, 'pair')}:
            raise SystemExit(f'the sessions listed as {call(port, "/api/sessions")}')
        say(port, 'pair', 'call ' + json.dumps([['message', {'text': 'echo the answer', 'wait': True}]]))   # Auto off
        quiet('pair', 'pal')
        answer = [b['content'] for m in call(port, '/api/s/pair/messages')[-2:] for b in m['content'] if b['type'] == 'tool_result']
        if answer != ['pal ended its turn:\n\nthe answer'] or log('pal')[-2:] != [('pair', 'pair asks you, and waits for your answer:\n\necho the answer'), ('assistant', 'the answer')]:
            raise SystemExit(f'a message waited for got {answer}, the companion heard {log("pal")[-2:]}')
        counts = len(log('pair')), len(log('pal'))
        say(port, 'pair', 'call ' + json.dumps([['message', {'text': 'echo noted'}]]))
        quiet('pair', 'pal')
        pair, pal = log('pair')[counts[0]:], log('pal')[counts[1]:]
        if pal != [('pair', 'pair sends you a message while it works:\n\necho noted'), ('assistant', 'noted')] or pair.count(('pal', 'pal ended its turn:\n\nnoted')) != 1 \
                or pair[-1][0] != 'assistant' or len(pair) > 6:
            raise SystemExit(f'a message sent while working: the companion heard {pal}, the session {pair}')
        counts = len(log('pal'))                                        # a message that comes during a turn's last step: the turn goes on to answer it
        call(port, '/api/s/pal/say', {'text': 'echo wait for it, slowly'})
        time.sleep(0.3)
        say(port, 'pair', 'call ' + json.dumps([['message', {'text': 'echo also noted'}]]))
        quiet('pair', 'pal')
        if (pal := log('pal')[counts:]) != [('user', 'echo wait for it, slowly'), ('assistant', 'wait for it, slowly'), ('pair', 'pair sends you a message while it works:\n\necho also noted'), ('assistant', 'also noted')]:
            raise SystemExit(f'a message during a turn\'s last step: the companion\'s log went on {pal}')
        call(port, '/api/s/pair/settings', {'allow': ['bash']})
        say(port, 'pair', 'call ' + json.dumps([['bash', {'command': 'echo refuse once please'}]]).replace(' once', '\\u0020once'))   # the mock refuses only the companion's request
        settle(port, 'pair')
        counts = len(log('pair'))
        state = call(port, '/api/s/pair/auto', {'on': True})
        quiet('pair', 'pal')
        last, ran = log('pal')[-1], 'What pair ran this turn: $ echo refuse once please'
        if last[0] != 'pair' or not last[1].startswith('pair ended its turn:\n\nReply ') or not last[1].endswith(ran) or not state['settings'].get('auto'):
            raise SystemExit(f'Auto on sent the companion {last}')
        if call(port, '/api/s/pair')['settings'].get('auto') or call(port, '/api/s/pal')['settings'].get('auto') or len(log('pair')) != counts:
            raise SystemExit('a companion\'s failed turn left Auto on, or the session went on')
        counts = len(log('pal'))                                        # Auto on while a turn runs: its end goes to the companion once
        call(port, '/api/s/pair/say', {'text': 'echo call [["finish", {"pace": "slowly"}]]'})
        time.sleep(0.3)
        if call(port, '/api/s/pair/auto', {'on': True})['settings'].get('auto') is not True:
            raise SystemExit('Auto did not go on during a turn')
        quiet('pair', 'pal')
        heard = [text for who, text in log('pal')[counts:] if who == 'pair']
        if heard != ['pair ended its turn:\n\ncall [["finish", {"pace": "slowly"}]]'] or call(port, '/api/s/pal')['settings'].get('auto'):
            raise SystemExit(f'Auto on during a turn: the companion heard {heard}')
        call(port, '/api/sessions', {'name': 'lone'})
        for path, wrong in [('pair/companion', {'name': 'pal2', 'goal': 'g'}), ('pal/companion', {'name': 'pal3', 'goal': 'g'}), ('lone/companion', {'name': 'pair', 'goal': 'g'}),
                            ('lone/companion', {'name': 'pal4'}), ('lone/auto', {'on': True})]:
            try:
                call(port, f'/api/s/{path}', wrong)
                raise SystemExit(f'{path} took {wrong}')
            except urllib.error.HTTPError as error:
                if error.code != 400:
                    raise
        if (made := sorted(s['name'] for s in call(port, '/api/sessions'))) != ['lone', 'pair', 'pal']:
            raise SystemExit(f'refused companions made sessions: {made}')
        call(port, '/api/s/pair/fork', {'at': 1, 'name': 'pair-fork'})
        if (forked := call(port, '/api/s/pair-fork')['settings']).get('companion') or forked.get('auto'):
            raise SystemExit(f'a fork of a session with a companion has {forked}')
        call(port, '/api/s/lone/say', {'text': 'echo first, slowly'})   # messages sent during a turn wait, queued: one changed, one taken back
        time.sleep(0.3)
        second, third = (call(port, '/api/s/lone/say', {'text': f'echo {n}'})['job'] for n in ('second', 'third'))
        if call(port, '/api/s/lone')['queued'] != [{'id': second, 'text': 'echo second'}, {'id': third, 'text': 'echo third'}]:
            raise SystemExit(f'queued during a turn: {call(port, "/api/s/lone")["queued"]}')
        call(port, '/api/s/lone/queued', {'id': second, 'text': 'echo second, edited'})
        call(port, '/api/s/lone/queued', {'id': third, 'cancel': True})
        quiet('lone')
        if log('lone') != [('user', 'echo first, slowly'), ('assistant', 'first, slowly'), ('user', 'echo second, edited'), ('assistant', 'second, edited')] or call(port, '/api/s/lone')['queued']:
            raise SystemExit(f'the queued messages went on as {log("lone")}, {call(port, "/api/s/lone")["queued"]} left queued')
        for wrong in ({'id': third, 'cancel': True}, {'id': second, 'text': 'too late'}):   # taken back, or gone to the model
            try:
                call(port, '/api/s/lone/queued', wrong)
                raise SystemExit(f'a queued message took {wrong} once it had gone')
            except urllib.error.HTTPError as error:
                if error.code != 400:
                    raise
        return f'a session and its companion: {len(log("pair"))} and {len(log("pal"))} messages'
    finally:
        app.terminate()
        mock.shutdown()


def chatted(turns, tls, ca, baseline) -> str:
    """The long chat, the fork and the settings; the main session's price multiplier goes to `baseline`."""
    mock_port, plain_port, port, root = free_port(), free_port(), free_port(), tempfile.mkdtemp()
    mock, plain = serve(mock_port, tls), serve(plain_port)
    app = launch(root, port, ca)
    try:
        configure(root, f'https://localhost:{mock_port}', WINDOW, 1948, 487)
        call(port, '/api/sessions', {'name': 'main'})
        state = chat(port, 'main', turns, 0)
        if abs(state['multiplier'] - 1) < 0.01:
            raise SystemExit(f'the price multiplier ({state["multiplier"]}) never calibrated to the prompt sizes the API reports')
        baseline.put(state['multiplier'])
        costs = [json.loads(line) for line in Path(root, 'main', 'costs.jsonl').read_text().splitlines()]   # each step's, each memory round's
        sums = {what: {k: sum(c['used'][k] for c in costs if c['for'] == what) for k in ('cost', 'input', 'cached', 'output')} for what in ('reply', 'memories')}
        sums['total'] = {k: sums['reply'][k] + sums['memories'][k] for k in sums['reply']}
        spent = state['spent']
        if {c['for'] for c in costs} != {'reply', 'memories'} or any(abs(sums[w][k] - spent[w][k]) > 1e-9 for w in sums for k in sums[w]) or seen['costs']['main'] != spent \
                or any(c['used']['input'] <= 0 or c['used']['cached'] <= 0 or c['used']['output'] <= 0 for c in costs) or costs[-1]['total'] != {w: spent[w] for w in ('reply', 'memories')}:
            raise SystemExit(f'main used {spent}, its log {sums} in {len(costs)} records, the page last heard {seen["costs"].get("main")}')
        listed = call(port, '/api/s/main/messages?from=0')              # folds: before replies, as streamed, each the same messages in fewer tokens
        folds, layers = [m['fold'] for m in listed if 'fold' in m], [call(port, f'/api/s/main/layer/{n}?from=0') for n in range(len(state['layers']))]
        reach = lambda nodes: (min(layers[l][p]['first'] for l, p in nodes), max(layers[l][p]['last'] for l, p in nodes))
        if not folds or {m['id']: m['fold'] for m in listed if 'fold' in m} != seen['folds']['main'] or any(m['participant'] != 'assistant' for m in listed if 'fold' in m) or any(reach(f['from']) != reach(f['into']) for f in folds) \
                or any(f['saved'] <= 0 for f in folds if max(l for l, _ in f['into']) > max(l for l, _ in f['from'])) or {c['message'] for c in costs if c['for'] == 'reply'} - {m['id'] for m in listed if m['participant'] == 'assistant'}:
            raise SystemExit(f'main folded {len(folds)} times ({folds[:2]}); the page heard {len(seen["folds"].get("main", []))}')
        start = [m['id'] for m in listed if 'fold' in m][1]                # a view starting at a reply with a fold still shows it
        if [m.get('fold') for m in call(port, f'/api/s/main/messages?from={start}')] != [m.get('fold') for m in listed if m['id'] >= start]:
            raise SystemExit(f'the messages from {start} show other folds than all of them do')
        if state['prompt'] != next(c['used']['input'] for c in reversed(costs) if c['for'] == 'reply') or state['prompt'] != seen['prompts']['main']:
            raise SystemExit(f'main\'s last prompt {state["prompt"]}, the page last heard {seen["prompts"].get("main")}')
        if seen['problems']:
            raise SystemExit(f'the API would reject a request: {seen["problems"][0]}')
        if not seen['thought']:
            raise SystemExit('no reply streamed its reasoning')
        if not seen['forming']:
            raise SystemExit('the page never saw memories forming')
        levels = {level for level, _ in state['frontier'] or []}
        if len(state['layers']) < 3 or not levels - {0}:
            raise SystemExit(f'memories did not form or show: layers {state["layers"]}, frontier {state["frontier"]}')
        if why := misgrouped(state, layers):
            raise SystemExit(why)
        call(port, '/api/s/main/settings', {'system': 'You are being tested.'})
        if (made := call(port, '/api/s/main/fork', {'at': 20, 'name': 'branch'})).get('messages') != 21:   # the fork as the page then shows it
            raise SystemExit(f'the fork answered {str(made)[:80]}, not its state with 21 messages')
        logged = lambda name, query: call(port, f'/api/s/{name}/messages?{query}')
        parent, fork = logged('main', 'from=0&to=21'), logged('branch', 'from=0')
        if fork != parent:
            raise SystemExit('the fork does not share its parent\'s messages')
        if call(port, '/api/s/branch')['settings'] != call(port, '/api/s/main')['settings']:
            raise SystemExit('the fork\'s settings are not its parent\'s')
        forked = chat(port, 'branch', turns // 3, turns)
        if seen['problems']:
            raise SystemExit(f'the API would reject a request: {seen["problems"][0]}')
        for headers in ({'Host': 'evil.example:80', 'x-connectome': '1'}, {}):     # another site's page, by DNS or by form
            request = urllib.request.Request(f'http://127.0.0.1:{port}/api/settings', data=b'{"base": "http://evil.example"}', headers=headers)
            try:
                urllib.request.urlopen(request, timeout=10)
                raise SystemExit(f'the app took settings from another site ({headers})')
            except urllib.error.HTTPError as error:
                if error.code != 403:
                    raise
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/libs.js', timeout=10) as r:        # the page's markdown, and what cleans it
            if b'marked v16' not in (libs := r.read()) or b'DOMPurify' not in libs or b'katex' not in libs:
                raise SystemExit('the app does not serve the page its libraries')
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/katex.css', timeout=10) as css, urllib.request.urlopen(f'http://127.0.0.1:{port}/fonts/KaTeX_Main-Regular.woff2', timeout=10) as font:
            if b'KaTeX_Main' not in css.read() or font.read(4) != b'wOF2':
                raise SystemExit('the app does not serve the math its stylesheet and fonts')
        call(port, '/api/settings', {'name': 'plain', 'key': 'k-123', 'base': f'http://127.0.0.1:{plain_port}/v1'})   # a base with /v1, as OpenRouter's
        call(port, '/api/s/branch/settings', {'provider': 'plain', 'model': 'mock-2'})
        before = call(port, '/api/s/branch')['spent']
        say(port, 'branch', 'with the saved key, over plain HTTP')
        after = settle(port, 'branch')['spent']                          # mock-2's provider reports no cost, but tokens
        if after['total']['cost'] != before['total']['cost'] or after['total']['input'] <= before['total']['input']:
            raise SystemExit(f'a turn whose provider reports no cost: {before} then {after}')
        try:
            say(port, 'branch', 'refuse once, then go on')
            raise SystemExit('a turn whose request failed went through')
        except SystemExit as failed:
            if 'Refused' not in str(failed):
                raise
        say(port, 'branch', None)
        learnt = [json.loads(line)['multiplier'] for line in Path(root, 'branch', 'compiles.jsonl').read_text().splitlines()][-3:]
        if learnt[0] == learnt[1] or learnt[1] != learnt[2]:            # the failed request's compile learnt from the reply before it, and nothing since has
            raise SystemExit(f'around a failed request the price multiplier went {learnt}')
        shown, mode = call(port, '/api/settings')['providers'], os.stat(os.path.join(root, 'settings.json')).st_mode & 0o777
        listed = next((p for p in shown if p['name'] == 'plain'), None)
        kept, own = Path(root, 'settings.json').read_text(), Path(root, 'branch', 'session_settings.json').read_text()
        if (seen['key'], seen['model']) != ('k-123', 'mock-2') or listed != {'name': 'plain', 'base': f'http://127.0.0.1:{plain_port}/v1', 'key': True, 'models': []} \
                or 'k-123' in json.dumps(shown) + own or 'k-123' not in kept or mode != 0o600:
            raise SystemExit(f'settings: the API saw key {seen["key"]!r} and model {seen["model"]!r}, the page sees {shown}, settings.json has mode {mode:o}')
        call(port, '/api/settings', {'name': 'plain', 'models': [' mock-2', 'mock-3', 'mock-2', '']})   # each once, trimmed, blanks left out
        if not (resaved := next(p for p in call(port, '/api/settings', {'name': 'plain', 'base': f'http://127.0.0.1:{plain_port}/v1', 'key': ' '})['providers'] if p['name'] == 'plain'))['key'] \
                or resaved['models'] != ['mock-2', 'mock-3']:
            raise SystemExit(f'a provider saved without a key or models lost them: {resaved}')
        everything = json.loads(Path(root, 'settings.json').read_text())
        del everything['providers']['plain']['models']                  # a provider written by hand lists none
        Path(root, 'settings.json').write_text(json.dumps(everything))
        try:
            call(port, '/api/settings', {'name': 'plain', 'models': 'mock-2'})
            raise SystemExit('a provider took models that are not a list')
        except urllib.error.HTTPError as error:
            if error.code != 400:
                raise
        if next(p for p in call(port, '/api/settings')['providers'] if p['name'] == 'plain')['models'] != []:
            raise SystemExit('a provider without models in settings.json lists some')
        call(port, '/api/settings', {'name': 'spare', 'base': 'https://spare.example'})
        if 'spare' in [p['name'] for p in call(port, '/api/settings', {'name': 'spare', 'delete': True})['providers']]:
            raise SystemExit('a provider deleted is still there')
        for wrong in [{'name': 'spare', 'delete': True}, {'name': 'default', 'delete': True}]:   # one there isn't, and the one new sessions start on
            try:
                call(port, '/api/settings', wrong)
                raise SystemExit(f'the settings took {wrong}')
            except urllib.error.HTTPError as error:
                if error.code != 400:
                    raise
        Path(root, 'old').mkdir()                                       # a session without settings has none to run with
        if 'session_settings.json' not in (call(port, '/api/s/old')['problem'] or '') or call(port, '/api/s/old/messages') != []:
            raise SystemExit('a session without settings did not say so, or show its log')
        try:
            call(port, '/api/s/old/say', {'text': 'hello'})
            raise SystemExit('a session without settings took a message')
        except urllib.error.HTTPError as error:
            if error.code != 400:
                raise
        own = json.loads(Path(root, 'branch', 'session_settings.json').read_text())
        Path(root, 'branch', 'session_settings.json').write_text(json.dumps({**own, 'provider': 'gone'}))    # its provider removed by hand
        if (lost := call(port, '/api/s/branch'))['settings'] != {**own, 'provider': 'gone'} or 'gone' not in (lost['problem'] or '') \
                or call(port, '/api/s/branch/settings', {'provider': 'plain'})['problem']:
            raise SystemExit(f'a session whose provider has gone shows {lost["settings"]}, {lost["problem"]}')
        for path, wrong in [('settings', {'provider': 'nowhere'}), ('settings', {'workspace': '/nonexistent/dir'}), ('settings', {'config': {'merge_threshold': 3}}),
                            ('fork', {'at': 5, 'name': 'branch'}), ('approve', {'id': 'call-none', 'answer': 'allow'})]:   # a fork onto a session there is, an answer no call waits for
            try:
                call(port, f'/api/s/main/{path}', wrong)
                raise SystemExit(f'a session took the {path} {wrong}')
            except urllib.error.HTTPError as error:
                if error.code != 400:
                    raise
        call(port, '/api/sessions', {'name': 'shell', 'workspace': tempfile.mkdtemp()})
        say(port, 'shell', 'call ' + json.dumps([['bash', {'command': 'echo hi'}]]))
        call(port, '/api/s/shell/settings', {'workspace': None})
        asked = len(seen['asked'])
        say(port, 'shell', 'call ' + json.dumps([['bash', {'command': 'echo again'}]]))
        if 'no workspace' not in (lost := seen['results'][-1][0]).get('content', '') or not lost.get('is_error') or seen['problems'] or seen['asked'][asked:]:
            raise SystemExit(f'bash after the workspace went: {lost}, asked {seen["asked"]}, problems {seen["problems"][:1]}')
        call(port, '/api/sessions', {'name': 'story'})
        try:
            call(port, '/api/s/story/continue', {})
            raise SystemExit('a session without messages continued')
        except urllib.error.HTTPError as error:
            if error.code != 400:
                raise
        say(port, 'story', 'tell me a long story')
        say(port, 'story', None)                                        # the reply was cut short: it goes on
        try:
            say(port, 'story', 'refuse once please')
            raise SystemExit('a turn whose request failed went through')
        except SystemExit as failed:
            if 'Refused' not in str(failed) or 'tried' in str(failed):
                raise
        say(port, 'story', None)                                        # the request failed: the message gets its answer
        call(port, '/api/sessions', {'name': 'unpriced'})                # a model whose provider reports no cost: tokens, but no cost
        call(port, '/api/s/unpriced/settings', {'model': 'mock-2'})
        say(port, 'unpriced', 'hello')
        if (spent := settle(port, 'unpriced')['spent']['total'])['cost'] is not None or spent['input'] <= 0:
            raise SystemExit(f'a session no provider priced has used {spent}')
        costs = Path(root, 'unpriced', 'costs.jsonl')                    # a record edited into one the app can't read is passed over
        costs.write_text(costs.read_text() + json.dumps({'id': len(costs.read_text().splitlines()), 'for': 'lunch'}, separators=(',', ':')) + '\n')
        if (edited := call(port, '/api/s/unpriced'))['spent']['total'] != spent or edited['prompt'] is None:
            raise SystemExit(f'after a hand edit to its costs the session shows {edited["spent"]}, its prompt {edited["prompt"]}')
        call(port, '/api/sessions', {'name': 'tight'})                   # a message bigger than the window leaves the context
        call(port, '/api/s/tight/settings', {'window': 3000, 'reserve': 1000, 'fold_above': 2000, 'fold_to': 1800})   # the band the window gives
        try:
            say(port, 'tight', 'too long ' * 2000)
            raise SystemExit('a context that does not fit was sent')
        except SystemExit as refused:
            took, told = re.search(r'it takes (\d+) tokens', str(refused)), re.search(r'It holds 0 memories \(0 tokens\), 0 chunks raw that have no memory yet \(0\), (\d+) tokens of messages not yet in a chunk .* and the 0 most recent', str(refused))
            if "The context doesn't fit" not in str(refused) or 'less 1000 for the reply' not in str(refused) or 'Raise the window' not in str(refused) \
                    or not took or not told or took[1] != told[1]:   # all of it the message: past the recent window, the tail has none, and alone it closes no chunk
                raise
        call(port, '/api/sessions', {'name': 'cut'})                     # a reply whose stream is cut off, then one whose request fails: each tried again
        before, started = len(seen['retries']), time.monotonic()
        say(port, 'cut', 'cut off once')
        say(port, 'cut', 'fail once, then answer')
        retries, logged = seen['retries'][before:], call(port, '/api/s/cut/messages')
        if [(r['n'], r['of']) for r in retries] != [(1, 5), (1, 5)] or 'ended before' not in retries[0]['why'] or 'Overloaded' not in retries[1]['why'] \
                or [m['participant'] for m in logged] != ['user', 'assistant'] * 2 or not all(words(m).startswith('Reply ') for m in logged[1::2]) or time.monotonic() - started < 1:
            raise SystemExit(f'tried again: the page heard {retries}, the log holds {[(m["participant"], words(m)[:20]) for m in logged]}')
        call(port, '/api/sessions', {'name': 'mended'})
        say(port, 'mended', 'a first message')
        log = Path(root, 'mended', 'messages.jsonl')
        kept = log.read_bytes()
        log.write_bytes(kept + b'not a record\n')                      # a hand edit the library can't read
        try:
            say(port, 'mended', 'after a bad edit')
            raise SystemExit('a turn on a log the library can\'t read went through')
        except SystemExit as failed:
            if 'failed' not in str(failed):
                raise
        log.write_bytes(kept)                                           # mended, the session's thread goes on
        say(port, 'mended', 'after the mend')
        told = [(m['participant'], words(m)[:16]) for m in call(port, '/api/s/story/messages')]
        if told[:4] != [('user', 'tell me a long s'), ('assistant', 'Once upon a time'), ('assistant', ' and so the stor'), ('user', 'refuse once plea')] \
                or told[4:] != [('assistant', told[4][1])] or not told[4][1].startswith('Reply') or seen['prefills'] != 1:
            raise SystemExit(f'continuing: the log holds {told}, {seen["prefills"]} requests went on from a reply')
        call(port, '/api/s/branch/settings', {'provider': 'default'})
        call(port, '/api/settings', {'name': 'default', 'base': f'https://127.0.0.1:{mock_port}'})
        try:
            say(port, 'branch', 'to a name the certificate does not hold')
            raise SystemExit('the app talked to a server whose certificate names another host')
        except SystemExit as refused:
            if 'certificate' not in str(refused).lower():
                raise
        call(port, '/api/settings', {'name': 'default', 'base': f'https://localhost:{mock_port}'})
        settle(port, 'branch')
        _, _, times = say(port, 'branch', 'answer slowly please')
        if len(times) < 5 or times[-1] - times[0] < 0.75:
            raise SystemExit(f'the reply reached the page all at once: {len(times)} pieces over {times[-1] - times[0]:.2f} s')
        return (f'main: {state["messages"]} messages, layers {state["layers"]}, context {state["armed"]:.0f}/{WINDOW} tokens, multiplier {state["multiplier"]:.3f}; '
                f'branch: {forked["messages"]} messages, layers {forked["layers"]}; app peak memory {peak_kb(app.pid)} KB')
    finally:
        app.terminate()
        mock.shutdown()
        plain.shutdown()


def channeled(tls, ca) -> str:
    """Channels: one made through the API (a name taken, or not a name, refused) and sessions put in
    it, two with inputs on and one without. A post, the user's or one a session makes with
    message_channel, goes at once to each session there with inputs on but the one that posted it,
    headed by the channel and who posted it and ending with how to reply there, then the channel's
    description once it has one, and an idle session takes a turn on it; the call's result names who
    heard it, and, under a limit of posts a day, how many the session has left, a post past it refused
    (0 or less: no limit). A page following the channel gets each
    post once, as it's logged. Posting in a channel the session isn't in fails; joining or posting to a
    channel there isn't is refused, leaving takes a session out, the sessions listed leave out the
    channels' directory, and a fork is in no channel."""
    port, root, mock_port = free_port(), tempfile.mkdtemp(), free_port()
    mock = serve(mock_port, tls)
    app = launch(root, port, ca)
    configure(root, f'https://localhost:{mock_port}', 12000, 2381, 595.25)
    log = lambda name: [(m['participant'], words(m)) for m in call(port, f'/api/s/{name}/messages')]
    results = lambda name, since=0: [(b['content'], b.get('isError', False)) for m in call(port, f'/api/s/{name}/messages')[since:] for b in m['content'] if b['type'] == 'tool_result']
    posts = lambda: [(p['from'], p['text']) for p in call(port, '/api/c/lounge/posts')]
    members = lambda: {m['name']: m['inputs'] for c in call(port, '/api/channels') if c['name'] == 'lounge' for m in c['members']}
    heard = lambda who, text, about='': f'{who} posted in #lounge:\n\n{text}\n\n(To reply in #lounge, use message_channel with channel "lounge"; what you say otherwise stays in your session.)' \
        + (f'\n\nAbout #lounge: {about}' if about else '')
    described = lambda: next(c for c in call(port, '/api/channels') if c['name'] == 'lounge').get('description')
    post = lambda text: 'call ' + json.dumps([['message_channel', {'channel': 'lounge', 'text': text}]])

    def refused(path, body):
        try:
            call(port, path, body)
            raise SystemExit(f'{path} took {body}')
        except urllib.error.HTTPError as error:
            if error.code != 400:
                raise
    try:
        if [c['name'] for c in call(port, '/api/channels', {'name': 'lounge'})] != ['lounge']:
            raise SystemExit(f'the channels listed as {call(port, "/api/channels")}')
        refused('/api/channels', {'name': 'lounge'})
        refused('/api/channels', {'name': 'no good'})
        for name in ('ann', 'bob', 'cy'):
            call(port, '/api/sessions', {'name': name})
        call(port, '/api/s/ann/channel', {'channel': 'lounge', 'inputs': True})
        call(port, '/api/s/bob/channel', {'channel': 'lounge', 'inputs': True})
        if call(port, '/api/s/cy/channel', {'channel': 'lounge'})['settings'].get('channels') != {'lounge': {'inputs': False}} \
                or members() != {'ann': True, 'bob': True, 'cy': False}:
            raise SystemExit(f'joined, the channel has {members()}')
        page = urllib.request.urlopen(f'http://127.0.0.1:{port}/api/c/lounge/stream', timeout=10)
        call(port, '/api/c/lounge/post', {'text': post('echo hi all')})   # ann and bob each post it, and each hears the other's
        calm(port, 'ann', 'bob', 'cy')
        if posts()[0] != (None, post('echo hi all')) or sorted(posts()[1:]) != [('ann', 'echo hi all'), ('bob', 'echo hi all')] or log('cy'):
            raise SystemExit(f'the channel has {posts()}, cy heard {log("cy")}')
        for me, other in (('ann', 'bob'), ('bob', 'ann')):
            got = [text for who, text in log(me) if who == '#lounge']
            if sorted(got) != sorted([heard('The user', post('echo hi all')), heard(other, 'echo hi all')]) or log(me)[-1] != ('assistant', 'hi all') \
                    or results(me) != [(f'Posted in #lounge; {other} heard it.', False)]:
                raise SystemExit(f'{me} heard {log(me)}, its call got {results(me)}')
        streamed = []
        while len(streamed) < 3:
            if (line := page.readline()).strip():
                streamed.append(json.loads(line)['post'])
        page.close()
        if [p['id'] for p in streamed] != [0, 1, 2] or [(p['from'], p['text']) for p in streamed] != posts():
            raise SystemExit(f'a page following the channel got {streamed}')
        about = 'Post only when it matters.'
        if [c.get('description') for c in call(port, '/api/c/lounge/describe', {'description': f'  {about}\n'})] != [about] or described() != about:
            raise SystemExit(f'the channel described as {described()!r}')
        counts = len(log('ann')), len(log('bob'))                      # cy, without inputs, posts all the same
        say(port, 'cy', post('echo from cy'))
        calm(port, 'ann', 'bob', 'cy')
        if posts()[-1] != ('cy', 'echo from cy') or results('cy') != [('Posted in #lounge; ann and bob heard it.', False)] \
                or log('ann')[counts[0]:] != [('#lounge', heard('cy', 'echo from cy', about)), ('assistant', 'from cy')] \
                or log('bob')[counts[1]:] != [('#lounge', heard('cy', 'echo from cy', about)), ('assistant', 'from cy')]:
            raise SystemExit(f'cy posted {posts()[-1]}, its call got {results("cy")}, ann heard {log("ann")[counts[0]:]}')
        call(port, '/api/c/lounge/describe', {'description': ' '})
        if described() is not None:
            raise SystemExit(f'a blank description left {described()!r}')
        count = len(call(port, '/api/s/cy/messages'))
        say(port, 'cy', 'call ' + json.dumps([['message_channel', {'channel': 'elsewhere', 'text': 'hi'}]]))
        if results('cy', count) != [("you're not in #elsewhere", True)] or len(posts()) != 4:
            raise SystemExit(f'a post in a channel cy isn\'t in got {results("cy", count)}')
        call(port, '/api/s/cy/channel', {'channel': 'lounge', 'per_day': 2})   # cy, which has posted once, may post twice a day: once more, then no more
        if {m['name']: m['per_day'] for c in call(port, '/api/channels') if c['name'] == 'lounge' for m in c['members']} != {'ann': None, 'bob': None, 'cy': 2}:
            raise SystemExit(f'the channel lists {call(port, "/api/channels")}')
        count = len(call(port, '/api/s/cy/messages'))
        say(port, 'cy', post('echo two'))
        say(port, 'cy', post('echo three'))
        calm(port, 'ann', 'bob', 'cy')
        limited = results('cy', count)
        if limited[0] != ('Posted in #lounge; ann and bob heard it. You have 0 of your 2 posts a day left there.', False) or not limited[1][1] \
                or not re.fullmatch(r"Rate limit exceeded: you've posted 2 times in #lounge in the last 24 hours, the most the user allows there; your next post can go in (23 h 59|24 h 0) min\.", limited[1][0]) \
                or posts()[-1] != ('cy', 'echo two'):
            raise SystemExit(f'cy, two posts a day, got {limited}, and the channel ends {posts()[-1]}')
        for unlimited in (0, -1):                                       # 0 or less: no limit
            call(port, '/api/s/cy/channel', {'channel': 'lounge', 'per_day': unlimited})
            count = len(call(port, '/api/s/cy/messages'))
            say(port, 'cy', post(f'echo free {unlimited}'))
            calm(port, 'ann', 'bob', 'cy')
            if results('cy', count) != [('Posted in #lounge; ann and bob heard it.', False)] or call(port, '/api/s/cy')['settings']['channels'] != {'lounge': {'inputs': False}}:
                raise SystemExit(f'cy, {unlimited} posts a day, got {results("cy", count)}')
        refused('/api/s/ann/channel', {'channel': 'nowhere', 'inputs': True})
        refused('/api/c/nowhere/post', {'text': 'hi'})
        refused('/api/c/nowhere/describe', {'description': 'hi'})
        refused('/api/c/lounge/post', {'text': '  '})
        call(port, '/api/s/bob/channel', {'channel': 'lounge', 'leave': True})
        call(port, '/api/s/ann/fork', {'at': 0, 'name': 'ann-fork'})
        if members() != {'ann': True, 'cy': False} or call(port, '/api/s/ann-fork')['settings'].get('channels') \
                or sorted(s['name'] for s in call(port, '/api/sessions')) != ['ann', 'ann-fork', 'bob', 'cy']:
            raise SystemExit(f'the channel has {members()}, the fork {call(port, "/api/s/ann-fork")["settings"].get("channels")}, the sessions {call(port, "/api/sessions")}')
        return f'a channel of {len(posts())} posts among three sessions, which ann and bob heard in {len(log("ann"))} and {len(log("bob"))} messages'
    finally:
        app.terminate()
        mock.shutdown()

def boarded(tls, ca) -> str:
    """Boards: one made through the API (a name taken, or not a name, refused), with sessions put on it:
    ones with a workspace, not one without or a companion. A session on it has its system prompt say
    where it is, and reaches its folder, bound into the sandbox (which can't write the rest of /tmp
    here), with bash and edit. A task handed to a session tells it, as a message ending with how boards
    work and then the board's description, unless it took the task itself, and a comment tells the
    task's owner, unless they wrote it; changes from the page are the user's. Columns are added and taken
    out (an empty one, not done), and a task moved and deleted; a fork is on no board."""
    port, root, mock_port = free_port(), tempfile.mkdtemp(), free_port()
    mock = serve(mock_port, tls)
    app = launch(root, port, ca)
    configure(root, f'https://localhost:{mock_port}', 12000, 2381, 595.25)
    everything = json.loads(Path(root, 'settings.json').read_text())
    everything['sandbox']['rw'] = []                                   # /tmp, where root and the board are, out of reach but for the binding
    Path(root, 'settings.json').write_text(json.dumps(everything))
    folder = Path(root, '.boards', 'sprint')
    log = lambda name: [(m['participant'], words(m)) for m in call(port, f'/api/s/{name}/messages')]
    notes = lambda name: [text for who, text in log(name) if who == 'board:sprint']
    results = lambda name, since: [(b['content'], b.get('isError', False)) for m in call(port, f'/api/s/{name}/messages')[since:] for b in m['content'] if b['type'] == 'tool_result']
    view = lambda: call(port, '/api/b/sprint')
    change = lambda body: call(port, '/api/b/sprint/change', body)
    reminder = lambda me, about='': f'(Board sprint is the folder {folder}: a folder per column, and a .md file per task, its owner in its front matter and comments appended to it as lines "- **{me}**: …".)' \
        + (f'\nAbout sprint: {about}' if about else '')
    bash = lambda command: 'call ' + json.dumps([['bash', {'command': command}]])

    def refused(path, body):
        try:
            call(port, path, body)
            raise SystemExit(f'{path} took {body}')
        except urllib.error.HTTPError as error:
            if error.code != 400:
                raise
    try:
        if call(port, '/api/boards', {'name': 'sprint'}) != [{'name': 'sprint', 'open': 0, 'members': 0}]:
            raise SystemExit(f'the boards listed as {call(port, "/api/boards")}')
        refused('/api/boards', {'name': 'sprint'})
        refused('/api/boards', {'name': 'no good'})
        for name in ('ann', 'bob', 'pal'):
            call(port, '/api/sessions', {'name': name, 'workspace': tempfile.mkdtemp()})
        call(port, '/api/sessions', {'name': 'lone'})
        call(port, '/api/s/pal/settings', {'accompanies': 'ann'})       # a companion, as its settings say
        for name in ('ann', 'bob'):
            call(port, f'/api/s/{name}/settings', {'allow': ['bash', 'edit']})
            call(port, f'/api/s/{name}/board', {'board': 'sprint'})
        refused('/api/s/lone/board', {'board': 'sprint'})
        refused('/api/s/pal/board', {'board': 'sprint'})
        refused('/api/s/ann/board', {'board': 'nowhere'})
        if view()['members'] != ['ann', 'bob'] or [c['name'] for c in view()['columns']] != ['todo', 'doing', 'done']:
            raise SystemExit(f'the board has {view()}')
        task = f'{folder}/todo/1-count-the-bees.md'                    # ann makes a task for bob with bash, in the sandbox
        say(port, 'ann', bash(f"printf -- '---\\ntitle: Count the bees\\nowner: bob\\n---\\nAll of them.\\n\\n## Comments\\n' > {task}"))
        calm(port, 'ann', 'bob')
        if not any(f'You\'re on the kanban board sprint (at {folder})' in system for system, _ in seen['requests']):
            raise SystemExit('no request\'s system prompt said where the board is')
        handed = f'On board sprint, ann handed you the new task "Count the bees" (todo/1-count-the-bees.md).\n\n{reminder("bob")}'
        if notes('bob') != [handed] or notes('ann'):
            raise SystemExit(f'bob was told {notes("bob")}, ann {notes("ann")}; the board has {view()}')
        counts = len(log('ann')), len(call(port, '/api/s/ann/messages'))   # ann takes a task itself, and edits bob's task: no word to either
        say(port, 'ann', bash(f"printf -- '---\\ntitle: Sweep up\\nowner: ann\\n---\\n' > {folder}/todo/2-sweep-up.md"))
        say(port, 'ann', 'call ' + json.dumps([['edit', {'path': task, 'old': 'All of them.', 'new': 'All of them, twice.'}]]))
        calm(port, 'ann', 'bob')
        if results('ann', counts[1]) != [('\n[exit code 0]', False), (f'edited {task} at line 5', False)] or notes('ann') or len(notes('bob')) != 1:
            raise SystemExit(f'ann\'s calls got {results("ann", counts[1])}; ann was told {notes("ann")}, bob {notes("bob")[1:]}')
        about = 'Count carefully.'
        call(port, '/api/b/sprint/describe', {'description': about})
        say(port, 'ann', bash(f"printf -- '- **ann**: how many so far?\\n' >> {task}"))     # ann comments on bob's task, bob on it too
        say(port, 'bob', bash(f"printf -- '- **bob**: seven\\n' >> {task}"))
        calm(port, 'ann', 'bob')
        asked = f'On board sprint, ann commented on your task "Count the bees" (todo/1-count-the-bees.md):\n\nhow many so far?\n\n{reminder("bob", about)}'
        if notes('bob')[1:] != [asked] or notes('ann'):
            raise SystemExit(f'after the comments bob was told {notes("bob")[1:]}, ann {notes("ann")}')
        change({'op': 'update', 'file': '1-count-the-bees.md', 'column': 'doing'})   # the user's: a move tells no one, a comment and a hand-over do
        change({'op': 'comment', 'file': '1-count-the-bees.md', 'text': 'thanks!'})
        change({'op': 'update', 'file': '1-count-the-bees.md', 'owner': 'ann'})
        calm(port, 'ann', 'bob')
        thanked = f'On board sprint, the user commented on your task "Count the bees" (doing/1-count-the-bees.md):\n\nthanks!\n\n{reminder("bob", about)}'
        given = f'On board sprint, the user handed you the task "Count the bees" (doing/1-count-the-bees.md).\n\n{reminder("ann", about)}'
        if notes('bob')[2:] != [thanked] or notes('ann') != [given]:
            raise SystemExit(f'after the user\'s changes bob was told {notes("bob")[2:]}, ann {notes("ann")}')
        got = call(port, '/api/b/sprint/t/1-count-the-bees.md')
        if (got['column'], got['owner'], got['description'], [(c['who'], c['text']) for c in got['comments']]) != ('doing', 'ann', 'All of them, twice.', [('ann', 'how many so far?'), ('bob', 'seven'), ('user', 'thanks!')]):
            raise SystemExit(f'the task reads {got}')
        change({'op': 'make', 'column': 'todo', 'title': 'Write it up'})
        change({'op': 'column', 'add': 'review'})
        if [(c['name'], [t['file'] for t in c['tasks']]) for c in view()['columns']] != [('todo', ['2-sweep-up.md', '3-write-it-up.md']), ('doing', ['1-count-the-bees.md']), ('review', []), ('done', [])]:
            raise SystemExit(f'the board has {view()["columns"]}')
        for wrong in ({'op': 'column', 'remove': 'done'}, {'op': 'column', 'remove': 'todo'}, {'op': 'column', 'add': 'no good'}, {'op': 'make', 'column': 'nowhere', 'title': 't'},
                      {'op': 'update', 'file': 'nothing.md', 'column': 'done'}):
            refused('/api/b/sprint/change', wrong)
        change({'op': 'column', 'order': ['review', 'todo', 'doing']})   # columns put in order, done last whatever
        if [c['name'] for c in view()['columns']] != ['review', 'todo', 'doing', 'done']:
            raise SystemExit(f'put in order, the board has {[c["name"] for c in view()["columns"]]}')
        for wrong in ({'op': 'column', 'order': ['todo', 'doing']}, {'op': 'column', 'order': ['review', 'todo', 'doing', 'done']}, {'op': 'column', 'order': ['todo', 'todo', 'doing', 'review']}):
            refused('/api/b/sprint/change', wrong)
        change({'op': 'column', 'order': ['todo', 'doing', 'review']})
        change({'op': 'column', 'remove': 'review'})
        change({'op': 'delete', 'file': '3-write-it-up.md'})
        if [c['name'] for c in view()['columns']] != ['todo', 'doing', 'done'] or call(port, '/api/boards') != [{'name': 'sprint', 'open': 2, 'members': 2}]:
            raise SystemExit(f'the board has {view()["columns"]}, the boards {call(port, "/api/boards")}')
        call(port, '/api/s/bob/board', {'board': 'sprint', 'leave': True})
        call(port, '/api/s/ann/fork', {'at': 0, 'name': 'ann-fork'})
        if view()['members'] != ['ann'] or call(port, '/api/s/ann-fork')['settings'].get('boards') \
                or sorted(s['name'] for s in call(port, '/api/sessions')) != ['ann', 'ann-fork', 'bob', 'lone', 'pal']:
            raise SystemExit(f'the board has {view()["members"]}, the fork {call(port, "/api/s/ann-fork")["settings"].get("boards")}')
        return f'a board of {sum(len(c["tasks"]) for c in view()["columns"])} tasks, bob told of it {len(notes("bob"))} times and ann {len(notes("ann"))}'
    finally:
        app.terminate()
        mock.shutdown()


forwarded = []   # what the gateway's upstream got: (method, path, headers, body)
streamed, held = threading.Event(), []   # set once the gateway's client has the first event; whether the upstream waited in vain
EVENTS = ['event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":5}}}\n\n',
          'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"h\u00e9"}}\n\n',
          'event: message_stop\ndata: {"type":"message_stop"}\n\n']


class Upstream(BaseHTTPRequestHandler):
    """What the gateway forwards to: it keeps each request as it came (forwarded) and answers
    /v1/messages with EVENTS, chunked, the rest only once the client has the first (or after 5 s,
    which held notes); a model of 'refused' with a 401, as the API answers a bad key, and one of
    'vanish' not at all, the connection closed; and a GET with a list of models."""
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def keep(self, body):
        forwarded.append((self.command, self.path, {k.lower(): v for k, v in self.headers.items()}, body))

    def do_GET(self):
        self.keep(b'')
        self.whole(200, json.dumps({'data': [{'id': 'claude-test'}]}).encode())

    def do_POST(self):
        body = self.rfile.read(int(self.headers['content-length']))
        self.keep(body)
        if json.loads(body)['model'] == 'vanish':
            self.close_connection = True
            return
        if json.loads(body)['model'] == 'refused':
            return self.whole(401, json.dumps({'type': 'error', 'error': {'type': 'authentication_error', 'message': 'invalid x-api-key'}}).encode())
        self.send_response(200)
        for name, value in (('content-type', 'text/event-stream'), ('request-id', 'req_1'), ('transfer-encoding', 'chunked')):
            self.send_header(name, value)
        self.end_headers()
        for i, event in enumerate(EVENTS):
            data = event.encode()
            self.wfile.write(b'%x\r\n%s\r\n' % (len(data), data))
            self.wfile.flush()
            if i == 0:
                held.append(not streamed.wait(5))
        self.wfile.write(b'0\r\n\r\n')

    def whole(self, code, body):
        self.send_response(code)
        self.send_header('content-type', 'application/json')
        self.send_header('content-length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def gatewayed(tls, ca) -> str:
    """The gateway, to an upstream over TLS (--upstream, trusted through --ca): a request goes on with
    its body byte for byte and its headers, credentials and the session's id among them, but this
    connection's (host, x-connectome, accept-encoding); the answer comes back with its status, its
    headers but the connection's, and its body as the upstream sent it, each event as it comes, a 401
    as one; an upstream that never answers is a 502 that says why. Each exchange joins its session's recording, credentials left out; one without a session id,
    or with one that isn't a name, joins 'unnamed'. A POST without x-connectome goes nowhere."""
    import http.client
    port, root, mock_port = free_port(), tempfile.mkdtemp(), free_port()
    mock = serve(mock_port, tls, Upstream)
    app = launch(root, port, ca, '--upstream', f'https://localhost:{mock_port}')

    def through(method, path, body=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
        connection.request(method, path, body=body, headers=headers or {})
        return connection.getresponse()

    def recorded(session, count):
        path = Path(root) / '.gateway' / session / 'exchanges.jsonl'
        for _ in range(100):   # the hub writes it once the answer has gone
            if path.exists() and len(lines := path.read_text().splitlines()) >= count:
                return [json.loads(line) for line in lines]
            time.sleep(0.05)
        raise SystemExit(f'{session} recorded {path.read_text() if path.exists() else "nothing"}')
    try:
        body = '{"model": "claude-test",   "stream": true, "messages": [{"role": "user", "content": "h\u00e9 é"}]}'.encode()
        sent = {'authorization': 'Bearer secret-token', 'anthropic-beta': 'oauth-2025-04-20,fine-grained', 'x-claude-code-session-id': 'abc-123',
                'content-type': 'application/json', 'user-agent': 'claude-cli/test', 'accept-encoding': 'gzip, br', 'x-connectome': '1'}
        answer = through('POST', '/gateway/v1/messages?beta=true', body, sent)
        first = answer.readline() + answer.readline() + answer.readline()
        streamed.set()
        rest = answer.read()
        method, path, headers, got = forwarded[0]
        kept = {k: v for k, v in sent.items() if k not in ('accept-encoding', 'x-connectome')}
        if (method, path, got) != ('POST', '/v1/messages?beta=true', body) or any(headers.get(k) != v for k, v in kept.items()) \
                or 'x-connectome' in headers or headers.get('host') != f'localhost:{mock_port}' or 'gzip' in headers.get('accept-encoding', ''):
            raise SystemExit(f'the upstream got {forwarded[0]}')
        if answer.status != 200 or answer.getheader('content-type') != 'text/event-stream' or answer.getheader('request-id') != 'req_1' \
                or answer.getheader('transfer-encoding') or (first + rest).decode() != ''.join(EVENTS) or held != [False]:
            raise SystemExit(f'the client got {answer.status} {answer.getheaders()} {first + rest!r}, the first event held back: {held}')
        refused = through('POST', '/gateway/v1/messages', b'{"model":"refused"}', {'x-api-key': 'secret-key', 'x-claude-code-session-id': 'abc-123', 'x-connectome': '1'})
        if refused.status != 401 or json.loads(refused.read())['error']['type'] != 'authentication_error':
            raise SystemExit(f'a 401 upstream came back as {refused.status}')
        listed = through('GET', '/gateway/v1/models', headers={'x-claude-code-session-id': '../away'})
        if listed.status != 200 or json.loads(listed.read()) != {'data': [{'id': 'claude-test'}]} or forwarded[2][:2] != ('GET', '/v1/models'):
            raise SystemExit(f'a GET came back as {listed.status}, the upstream got {forwarded[2:]}')
        if through('POST', '/gateway/v1/messages', b'{}').status != 403 or len(forwarded) != 3:
            raise SystemExit(f'a POST without x-connectome went on: {forwarded[3:]}')
        gone = through('POST', '/gateway/v1/messages', b'{"model":"vanish"}', {'x-claude-code-session-id': 'abc-123', 'x-connectome': '1'})
        if gone.status != 502 or json.loads(gone.read())['error']['type'] != 'api_error':
            raise SystemExit(f'an upstream that never answered came back as {gone.status}')
        stream, failed, vanished = recorded('abc-123', 3)
        if (stream['method'], stream['path'], stream['body'], stream['status'], stream['answer'], stream['failed']) != ('POST', '/v1/messages?beta=true', json.loads(body), 200, ''.join(EVENTS), None) \
                or stream['headers'].get('authorization') != '(left out)' or stream['headers'].get('anthropic-beta') != sent['anthropic-beta'] \
                or stream['answer_headers'].get('content-type') != 'text/event-stream' or 'x-connectome' in stream['headers']:
            raise SystemExit(f'the stream was recorded as {stream}')
        if failed['status'] != 401 or failed['headers'].get('x-api-key') != '(left out)':
            raise SystemExit(f'the 401 was recorded as {failed}')
        if vanished['status'] is not None or not vanished['failed']:
            raise SystemExit(f'the upstream that never answered was recorded as {vanished}')
        if [r['path'] for r in recorded('unnamed', 1)] != ['/v1/models'] or (Path(root) / 'away').exists():
            raise SystemExit(f'the GET was recorded in {os.listdir(Path(root) / ".gateway")}')
        leaked = [str(f) for f in Path(root).rglob('*') if f.is_file() and b'secret-' in f.read_bytes()]
        if leaked:
            raise SystemExit(f'credentials written to {leaked}')
        return f'{len(forwarded)} requests through the gateway'
    finally:
        app.terminate()
        mock.shutdown()


def clauded(tls, ca) -> str:
    """A Claude Code session, on a provider whose command runs fake_claude.py, in its sandbox: a turn
    streams its reply to the page and joins the log, the first starting Claude Code's session and the
    rest resuming it, Claude Code run with the stream-json flags and the session's model, unable to
    write outside its workspace, its requests going through the gateway (on the network, though the
    sandbox has none) (the session named in
    x-connectome-session) and its cost joining the session's, with the plan's limits it said. A call Claude Code asks about waits for
    the page: allowed, it runs; denied, its result is the denial; always allowed, the next call to its
    tool doesn't ask, and the tool is the session's to toggle. A call one of Claude Code's subagents makes
    is asked about whole (its tool, input and subagent), since the log, which has only Claude Code's
    own messages, doesn't have it; the subagent's stream stays out of the page. Its questions
    (AskUserQuestion) wait for the user's answers, which the call runs with, even when the tool is
    always allowed; declined, its result says so; with the session's questions off, Claude Code runs
    without the tool (--disallowedTools AskUserQuestion). What the user says while
    a turn runs goes to Claude Code at once, shown as with Claude Code until it takes it in, after the
    call it's on, where the log has it, the turn going on with it; said during the turn's reply, it's
    a turn of its own in the same run, which the turn waits for. The context shown is the last request's
    prompt, not the sum of the turn's that Claude Code's result counts. Its memories form after its
    turns, each written by a run of Claude Code of its own, on the session's model, with no tools,
    Claude Code's least system prompt and no session kept, told the library's request as a transcript
    (calls and results written out), its reasoning left out of the memory. A stopped turn ends at once, its reply as
    far as it came in the log, and what Claude Code hadn't taken in yet is the next turn. A Claude Code session can't fork, have a companion, or go on without a
    message."""
    streamed.set()   # the upstream answers at once: this part has its own process, so the gateway part waits as before
    port, root, mock_port, work = free_port(), tempfile.mkdtemp(), free_port(), tempfile.mkdtemp()
    mock = serve(mock_port, tls, Upstream)
    app = launch(root, port, ca, '--upstream', f'https://localhost:{mock_port}')
    runs = lambda: [json.loads(line) for line in Path(work, 'claude-runs.jsonl').read_text().splitlines()]
    log = lambda: [(m['participant'], m['content']) for m in call(port, '/api/s/cc/messages')]

    def refused(path, body):
        try:
            call(port, path, body)
            raise SystemExit(f'{path} took {body}')
        except urllib.error.HTTPError as error:
            if error.code != 400:
                raise
    try:
        configure(root, f'https://localhost:{mock_port}', 12000, 2381, 595.25)
        everything = json.loads(Path(root, 'settings.json').read_text())
        everything['providers']['code'] = {'base': '', 'key': '', 'models': ['fake'], 'command': [os.path.realpath(sys.executable), str(HERE / 'fake_claude.py')]}
        everything['sandbox']['rw'] = []   # so the test's temporary directory, beside the workspace, is outside the sandbox
        everything['sandbox']['network'] = False   # which Claude Code keeps all the same, to reach the gateway
        Path(root, 'settings.json').write_text(json.dumps(everything))
        call(port, '/api/sessions', {'name': 'cc', 'workspace': work})
        call(port, '/api/s/cc/settings', {'provider': 'code', 'model': 'fake', 'sandbox': True})
        if (said := say(port, 'cc', 'hello')[0]) != 'You said: hello':
            raise SystemExit(f'Claude Code\'s reply streamed as {said!r}')
        state = call(port, '/api/s/cc')
        session, run = state['settings'].get('claude_session'), runs()[0]
        flags = ['--input-format', 'stream-json', '--output-format', 'stream-json', '--permission-prompt-tool', 'stdio', '--model', 'fake', '--session-id', session,
                 '--settings', '{"sandbox":{"enabled":false}}', '--setting-sources', '']   # Claude Code's own sandbox off: bubblewrap is the session's
        if not session or any(run['args'][run['args'].index(flag) + 1] != value for flag, value in zip(flags[::2], flags[1::2])) or not {'--print', '--replay-user-messages'} <= set(run['args']):
            raise SystemExit(f'Claude Code ran as {run["args"]}, the session keeping {session}')
        if not run['outside_failed'] or run['gateway'] != 200:
            raise SystemExit(f'in its sandbox Claude Code wrote outside its workspace ({not run["outside_failed"]}), and the gateway answered {run["gateway"]}')
        if log() != [('user', [{'type': 'text', 'text': 'hello'}]), ('assistant', [{'type': 'text', 'text': 'You said: hello'}])]:
            raise SystemExit(f'the log has {log()}')
        recording = Path(root, '.gateway', session, 'exchanges.jsonl')
        for _ in range(100):   # the hub writes it once the answer has gone
            if recording.exists():
                break
            time.sleep(0.05)
        exchange = json.loads(recording.read_text().splitlines()[0])
        limits = state.get('limits') or {}
        if (limits.get('five_hour'), limits.get('seven_day')) != ({'utilization': 0.26, 'resetsAt': 1792970400}, {'utilization': 0.07, 'resetsAt': 1793300400}) or not limits.get('at'):
            raise SystemExit(f'the plan\'s limits, as Claude Code said them, show as {limits}')
        if exchange['headers'].get('x-connectome-session') != 'cc' or state['spent']['total']['cost'] != 0.01:
            raise SystemExit(f'the gateway recorded {exchange["headers"]}, the session spent {state["spent"]}')
        asked = len(seen['asked'])
        if (said := say(port, 'cc', 'call Write')[0]) != 'Calling.It ran.' or seen['asked'][asked:] != ['Write'] or runs()[1]['args'][-2:] != ['--resume', session]:
            raise SystemExit(f'an allowed call streamed as {said!r}, asked about {seen["asked"][asked:]}, Claude Code run as {runs()[1]["args"]}')
        call_ = {'type': 'tool_use', 'id': 'toolu_1', 'name': 'Write', 'input': {'file_path': 'x.txt'}}
        if log()[2:] != [('user', [{'type': 'text', 'text': 'call Write'}]), ('assistant', [{'type': 'text', 'text': 'Calling.'}, call_]),
                         ('user', [{'type': 'tool_result', 'toolUseId': 'toolu_1', 'content': 'did it'}]), ('assistant', [{'type': 'text', 'text': 'It ran.'}])]:
            raise SystemExit(f'the allowed call was logged as {log()[2:]}')
        say(port, 'cc', 'call Write', {'Write': 'deny'})
        if log()[-2] != ('user', [{'type': 'tool_result', 'toolUseId': 'toolu_1', 'content': "the user didn't allow this call", 'isError': True}]):
            raise SystemExit(f'the denied call was logged as {log()[-2]}')
        say(port, 'cc', 'call Edit', {'Edit': 'always'})
        asked = len(seen['asked'])
        if say(port, 'cc', 'call Edit')[0] != 'Calling.It ran.' or seen['asked'][asked:] or call(port, '/api/s/cc')['tools'] != ['Edit']:
            raise SystemExit(f'a call to a tool always allowed asked about {seen["asked"][asked:]}, the tools listed are {call(port, "/api/s/cc")["tools"]}')
        asked = len(seen['asked'])
        said = say(port, 'cc', 'delegate Bash')[0]
        logged_ids = [b.get('id') or b.get('toolUseId') for _, content in log() for b in content]
        if said != 'Delegating.The subagent did it.' or seen['asked'][asked:] != ['Bash'] or seen['approvals'][-1] != {'id': 'toolu_sub', 'tool': 'Bash', 'input': {'command': 'touch sub'}, 'agent': 'agent-1'} \
                or 'toolu_sub' in logged_ids or 'toolu_agent' not in logged_ids:
            raise SystemExit(f'a subagent\'s call streamed as {said!r}, asked as {seen["approvals"][-1]}, the log holding calls {logged_ids}')
        def meanwhile(first, cue, then, until='end'):
            """A turn on `first` as a page following it hears it, `then` said when `cue` streams in: the
            turns it heard start, the queues it heard, and its last event, `until`'s."""
            r, events = following(port, 'cc')
            with r:
                next(events)
                call(port, '/api/s/cc/say', {'text': first})
                turns, queues = [], []
                for event in events:
                    turns += [event['turn']] if 'turn' in event else []
                    queues += [event['queue']] if 'queue' in event else []
                    if event.get('text') == cue or event.get('calling') == cue:
                        then()
                    if until in event:
                        return turns, queues, event
        text = lambda t: [{'type': 'text', 'text': t}]
        before, processes = len(log()), len(runs())
        turns, queues, end = meanwhile('wait', 'Bash', lambda: call(port, '/api/s/cc/say', {'text': 'and also this'}))
        handed = [q for queue in queues for q in queue if q.get('handed')]
        wait_call = {'type': 'tool_use', 'id': 'toolu_wait', 'name': 'Bash', 'input': {'command': 'sleep'}}
        if log()[before:] != [('user', text('wait')), ('assistant', text('Working.') + [wait_call]), ('user', [{'type': 'tool_result', 'toolUseId': 'toolu_wait', 'content': 'slept'}]),
                              ('user', text('and also this')), ('assistant', text('Also: and also this.'))] \
                or len(turns) != 1 or end['end'] is not None or [q['text'] for q in handed[:1]] != ['and also this'] or queues[-1] != []:
            raise SystemExit(f'a message said during a call was logged as {log()[before:]}, in turns {turns}, the queue heard as {queues}')
        if (prompt := call(port, '/api/s/cc')['prompt']) != 1200:   # the context is the last request's prompt, not the turn's sum (15)
            raise SystemExit(f'after a turn of two requests, of 1000 and 1200 tokens, the session shows a context of {prompt}')
        before = len(log())
        turns, queues, end = meanwhile('linger', 'Lingering', lambda: call(port, '/api/s/cc/say', {'text': 'after the reply'}), until='done')
        if log()[before:] != [('user', text('linger')), ('assistant', text('Lingering done.')), ('user', text('after the reply')), ('assistant', text('You said: after the reply'))] \
                or len(turns) != 1 or len(runs()) != processes + 2:
            raise SystemExit(f'a message said during a reply was logged as {log()[before:]}, in turns {turns}, Claude Code run {len(runs()) - processes} times')
        stopped = time.monotonic()

        def stop():
            nonlocal stopped
            call(port, '/api/s/cc/say', {'text': 'pending one'})
            time.sleep(0.5)   # the session's thread hands it to Claude Code, which doesn't take it in before the stop
            call(port, '/api/s/cc/stop', {})
            stopped = time.monotonic()
        turns, queues, event = meanwhile('hang', 'Working', stop)
        if event['end'] != 'Stopped by the user.' or time.monotonic() - stopped > 5:
            raise SystemExit(f'a stopped turn ended {event} after {time.monotonic() - stopped:.1f} s')
        expected = [('assistant', text('Working')), ('user', text('pending one')), ('assistant', text('You said: pending one'))]
        for _ in range(100):   # the next turn starts as the stopped one's job ends
            if log()[-3:] == expected:
                break
            time.sleep(0.05)
        if log()[-3:] != expected:
            raise SystemExit(f'after a stop with a message Claude Code hadn\'t taken in, the log ends {log()[-3:]}')
        settle(port, 'cc')   # that turn is done, so a page following the session next hears none of it
        story = ' '.join(f'word{i}' for i in range(1500))   # past a chunk's 595 tokens, so chunks close and their memories form
        for n in range(3):
            say(port, 'cc', f'part {n}: {story}')
        memories = call(port, '/api/s/cc/layer/1')
        runs_of = [json.loads(line) for line in Path(work, 'claude-memories.jsonl').read_text().splitlines()]
        flags = {flag: runs_of[0]['args'][runs_of[0]['args'].index(flag) + 1] for flag in ('--model', '--tools', '--system-prompt')}
        told = runs_of[0]['transcript']
        if not memories or not all(m['text'].startswith('I remember ') and 'recall' not in m['text'] for m in memories) or len(runs_of) != len(memories) \
                or flags != {'--model': 'fake', '--tools': '', '--system-prompt': ''} or runs_of[0]['max_tokens'] != '20000':
            raise SystemExit(f'memories formed as {[m["text"] for m in memories]}, Claude Code run for them {len(runs_of)} times as {runs_of[:1] and runs_of[0]["args"]}')
        if abs(call(port, '/api/s/cc')['spent']['memories']['cost'] - 0.002 * len(runs_of)) > 1e-9:
            raise SystemExit(f'{len(runs_of)} memories\' runs cost the session {call(port, "/api/s/cc")["spent"].get("memories")}')
        if not told.startswith('This is part of your conversation in this session') or '\n\n[user]\nhello' not in told or '[Context Manager]\nSystem: You will soon form a new memory' not in told \
                or '[called Write with {"file_path":"x.txt"}]' not in told or "[Write's result: did it]" not in told:
            raise SystemExit(f'the first memory was asked for as {told[:1500]}')
        colors, pets = 'What is your favorite color?', 'Which pets do you like?'
        given = {'answers': {colors: 'Blue', pets: 'Cats, Dogs'}, 'annotations': {colors: {'notes': 'navy if possible'}}}
        told = f'The user answered: "{colors}"="Blue" notes: navy if possible, "{pets}"="Cats, Dogs"'
        if (said := say(port, 'cc', 'question', {'AskUserQuestion': given})[0]) != f'Heard: {told}' or log()[-2] != ('user', [{'type': 'tool_result', 'toolUseId': 'toolu_q', 'content': told}]):
            raise SystemExit(f'answered questions came back as {said!r}, logged as {log()[-2]}')
        call(port, '/api/s/cc/settings', {'questions': False})
        say(port, 'cc', 'hello again')
        if runs()[-1]['args'][runs()[-1]['args'].index('--disallowedTools') + 1:][:1] != ['AskUserQuestion']:
            raise SystemExit(f'with questions off, Claude Code ran as {runs()[-1]["args"]}')
        call(port, '/api/s/cc/settings', {'questions': True})
        call(port, '/api/s/cc/settings', {'allow': ['AskUserQuestion']})   # questions only the user can answer, so they ask all the same
        asked = len(seen['asked'])
        say(port, 'cc', 'question', {'AskUserQuestion': 'deny'})
        if seen['asked'][asked:] != ['AskUserQuestion'] or log()[-2] != ('user', [{'type': 'tool_result', 'toolUseId': 'toolu_q', 'content': 'The user declined to answer the questions.', 'isError': True}]):
            raise SystemExit(f'questions always allowed asked {seen["asked"][asked:]}, declined, were logged as {log()[-2]}')
        refused('/api/s/cc/fork', {'at': 1, 'name': 'cc-fork'})
        refused('/api/s/cc/companion', {'name': 'pal', 'goal': 'help'})
        r, events = following(port, 'cc')
        with r:
            next(events)
            call(port, '/api/s/cc/continue', {})
            ended = next(e for e in events if 'end' in e)['end']
        if 'goes on from a message' not in (ended or ''):
            raise SystemExit(f'a Claude Code session went on without a message: {ended}')
        return f'{len(runs())} Claude Code turns, {len(log())} messages logged'
    finally:
        app.terminate()
        mock.shutdown()


def part(name, run, done):
    """A part of the test in a process group of its own, which its apps and their servers join, so
    that killing the group stops them all: (name, its summary or None, why it failed or None, counts)."""
    os.setpgrp()
    try:
        done.put((name, run(), None, seen))
    except SystemExit as failed:
        done.put((name, None, str(failed), seen))
    except Exception as e:
        lines = [frame.lineno for frame in traceback.extract_tb(e.__traceback__) if frame.filename == __file__]
        done.put((name, None, f'{type(e).__name__}: {e} (test_app.py lines {lines})', seen))


def main(turns=60):
    turns = int(turns)
    if not os.environ.get('CONNECTOME_WEB'):
        build = subprocess.run(['cargo', 'build', '--release', '--quiet', '-p', 'connectome-web'], cwd=HERE.parent, capture_output=True, text=True)
        if build.returncode:
            print(build.stderr)
            return 1
    tempfile.tempdir = tempfile.mkdtemp()   # every part's directories, removed at the end whatever happens
    ca, cert, key = certificates(tempfile.mkdtemp())
    forked = multiprocessing.get_context('fork')
    done, baseline = forked.Queue(), forked.Queue()
    parts = {'chat': lambda: chatted(turns, (cert, key), ca, baseline), 'tools': lambda: tooled((cert, key), ca, baseline), 'large': lambda: streamed_large((cert, key), ca),
             'companion': lambda: accompanied((cert, key), ca), 'lifted': lambda: lifted((cert, key), ca), 'channels': lambda: channeled((cert, key), ca),
             'boards': lambda: boarded((cert, key), ca), 'gateway': lambda: gatewayed((cert, key), ca),
             'claude': lambda: clauded((cert, key), ca)}
    running = [forked.Process(target=part, args=(name, run, done)) for name, run in parts.items()]
    summaries, totals, failure = {}, {'chat': 0, 'memory': 0, 'bytes': 0, 'largest': 0}, None
    try:
        for p in running:
            p.start()
        for _ in running:
            name, summary, failed, counts = done.get(timeout=300)
            if failed:
                failure = f'{name}: {failed}'
                break
            summaries[name] = summary
            totals = {k: max(v, counts[k]) if k == 'largest' else v + counts[k] for k, v in totals.items()}
    finally:
        for p in running:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):   # it ended, or never got its own group
                p.kill()
            p.join()
        shutil.rmtree(tempfile.tempdir)
    if failure:
        print(failure)
        return 1
    print(f'{totals["chat"]} chat turns and {totals["memory"]} memory requests; {summaries["chat"]}; {totals["bytes"] // 1024} KB of requests streamed '
          f'(largest {totals["largest"] // 1024} KB); {summaries["tools"]}; {summaries["large"]}; {summaries["companion"]}; {summaries["lifted"]}; {summaries["channels"]}; {summaries["boards"]}; {summaries["gateway"]}; {summaries["claude"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
