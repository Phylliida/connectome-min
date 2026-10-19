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
picks, reach the API, the key never the page or a session's file, in a file only its user can read;
a session without settings says so, and one whose provider has gone shows its settings and takes
another; a reply whose stream is cut off is an error, not a logged reply; a turn on a log the library
can't read fails, and the session goes on once it's mended; a fork onto a session there is is refused;
a session's settings refuse a provider there isn't, a workspace that isn't a directory and a new
merge_threshold once memories have merged; a session that calls bash after losing its workspace
gets a failed call, the request still declaring bash; a turn continues without a message, a reply
cut short going on from its text and a turn whose request failed answering the message it had;
another site's page can't change the settings; a server whose certificate names another
host gets nothing; the model's tool calls, through MCP servers, get their results and form
memories; and the app's peak memory doesn't grow with the contexts it streams.

The three parts (the chat, the tool calls, the large contexts) run at once, in processes of their
own, each with its own mock; the first to fail stops the rest.

    python3 tests/test_app.py [turns]
"""
import datetime
import json
import multiprocessing
import os
import signal
import ssl
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = Path(os.environ.get('CONNECTOME_WEB') or HERE.parent / 'target' / 'release' / 'connectome-web')   # CONNECTOME_WEB: test that build instead
WINDOW = 8000
seen = {'calls': [], 'forming': False, 'prefills': 0, 'failed': [], 'model': None, 'chat': 0, 'memory': 0, 'bytes': 0, 'largest': 0, 'problems': [], 'key': None, 'thought': 0, 'results': [], 'tool memories': 0, 'events': [], 'asked': [], 'limit': None}


class Mock(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

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
        last = ' '.join(b.get('text', '') for b in messages[-1]['content'])
        results = [b for b in messages[-1]['content'] if b['type'] == 'tool_result']
        asked = next(' '.join(b.get('text', '') for b in m['content']) for m in reversed(messages) if m['role'] == 'user' and any(b['type'] == 'text' for b in m['content']))
        stop = 'end_turn'
        if messages[-1]['role'] == 'assistant':                    # a reply cut short, which this one goes on from
            seen['chat'] += 1
            blocks = [{'type': 'text', 'text': ' and so the story ends.'}]
        elif 'fail once' in last and last not in seen['failed']:
            seen['failed'].append(last)
            return self.answer(529, {'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'Overloaded'}})
        elif last.startswith('echo '):                             # the message back, as the reply
            seen['chat'] += 1
            blocks = [{'type': 'text', 'text': last[len('echo '):]}]
        elif 'a long story' in last:
            seen['chat'] += 1
            blocks, stop = [{'type': 'text', 'text': 'Once upon a time'}], 'max_tokens'
        elif 'memory' in last and ('Write the memory' in last or 'consolidate' in last):
            if request.get('tools') and request.get('tool_choice') != {'type': 'none'}:
                return self.problem(f'a memory request may call tools: tool_choice {request.get("tool_choice")}')
            seen['memory'] += 1
            seen['tool memories'] += any(b['type'] == 'tool_use' for m in messages for b in m['content'])
            blocks = [{'type': 'text', 'text': f'I remember stretch {seen["memory"]}: the user talked about ' + ' '.join(['things'] * 60)}]
        elif 'use tools' in last or last.startswith('call ') or results and 'forever' in asked:
            seen['chat'] += 1
            seen['results'].append(results)
            calls = json.loads(last[5:]) if last.startswith('call ') else [['add', {'a': 2, 'b': 3}]] if results else \
                [['add', {'a': 2, 'b': 3}], ['fail', {}], ['picture', {}], ['nope', {}]]
            blocks = [{'type': 'text', 'text': 'Calling tools.'}] + [{'type': 'tool_use', 'id': f'call{seen["chat"]}-{k}', 'name': name, 'input': args}
                                                                   for k, (name, args) in enumerate(calls)]
            stop = 'tool_use'
        else:
            seen['chat'] += 1
            if results:
                seen['results'].append(results)
            blocks = [{'type': 'thinking', 'thinking': 'Let me think about it.', 'signature': 'c2ln'}] * (seen['chat'] % 4 == 0) \
                + [{'type': 'text', 'text': f'Reply {seen["chat"]}: ' + ' '.join(['words'] * 80)}]
        memory = 'memory' in last and ('Write the memory' in last or 'consolidate' in last)
        self.stream(blocks, prompt, 0.25 if 'slowly' in last else 0.01 if memory else 0, stop, 'cut off please' in last)   # a memory takes a moment, as a page may see

    def stream(self, blocks, input_tokens, pause, stop, cut=False):
        """The reply as server-sent events, its text in five pieces `pause` seconds apart and a tool
        call's input in three, the stream ending there if `cut`; chunked or until the close, by turns."""
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
        event('message_start', message={'type': 'message', 'role': 'assistant', 'content': [], 'usage': {'input_tokens': input_tokens, 'output_tokens': 1}})
        for i, block in enumerate(blocks):
            if block['type'] == 'tool_use':
                event('content_block_start', index=i, content_block={**block, 'input': {}})
                text = json.dumps(block['input'])
                for k in range(3):
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
        event('message_delta', delta={'stop_reason': stop}, usage={'output_tokens': sum(len(b.get('text', '')) for b in blocks) // 4})
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


def serve(port: int, tls: tuple[str, str] | None = None) -> ThreadingHTTPServer:
    """The mock on `port`, over TLS 1.3 with (certificate, key) if given."""
    server = ThreadingHTTPServer(('127.0.0.1', port), Mock)
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


def configure(root, base, window, recent, chunk):
    """settings.json as the app's first run wrote it, edited as a user would: the default provider at
    `base`, and new sessions on the mock's model in `window` tokens, 1000 of them the reply's, the
    library's chunks and recent window sized for it (as a first run sizes them for its window, less
    the tool definitions)."""
    path = Path(root, 'settings.json')
    everything = json.loads(path.read_text())
    everything['providers']['default']['base'] = base
    everything['defaults'].update(model='mock', window=window, reserve=1000)
    everything['defaults']['config'].update(recent_window_tokens=recent, target_chunk_tokens=chunk)
    path.write_text(json.dumps(everything))


def words(message, kind='text'):
    """A logged message's text, or with `kind` thinking its reasoning, from the blocks the page gets."""
    return ''.join(b[kind] for b in message['content'] if b['type'] == kind)


def say(port, name, text, answers=None) -> tuple[str, str, list[float]]:
    """A chat turn as the page has it, or without text one that continues the log, followed until its
    memories have formed: the reply's text and reasoning, and when each piece of text arrived. A call
    that asks to run gets the answer `answers` has for its tool, else allow. The calls as the replies
    stream them in, each its tool and its input's JSON, go to seen['calls']."""
    request = urllib.request.Request(f'http://127.0.0.1:{port}/api/s/{name}/{"continue" if text is None else "say"}', data=json.dumps({'text': text}).encode(), headers={'x-connectome': '1'})
    pieces, thoughts, times, last, end, logged, calls = [], [], [], {}, {}, 0, []
    with urllib.request.urlopen(request, timeout=15) as r:                   # no turn here goes 15 s without an event
        for line in r:
            last = json.loads(line)
            logged = last.get('logged', logged)
            if 'text' in last:
                pieces.append(last['text'])
                times.append(time.monotonic())
            if 'result' in last:
                seen['events'].append(last)
            if 'calling' in last:
                calls.append([last['calling'], ''])
            if 'input' in last:
                calls[-1][1] += last['input']
            seen['forming'] |= last.get('forming', False)
            end = last if {'ok', 'error'} & set(last) else end
            if 'approve' in last:                                           # a call of the reply just logged
                reply = call(port, f'/api/s/{name}/messages?from={logged - 1}')[0]
                tool = next(b['name'] for b in reply['content'] if b.get('id') == last['approve'])
                seen['asked'].append(tool)
                call(port, f'/api/s/{name}/approve', {'id': last['approve'], 'answer': (answers or {}).get(tool, 'allow')})
            thoughts.append(last.get('thinking', ''))
    seen['calls'] = calls
    if not end.get('ok'):
        raise SystemExit(f'{name}: "{(text or "(continue)")[:40]}" got {end or last}')
    return ''.join(pieces), ''.join(thoughts), times


def settle(port, name):
    """The session's state once its job (a turn, then its memories forming) is done, in seconds."""
    for _ in range(400):
        if not (state := call(port, f'/api/s/{name}'))['running']:
            return state
        time.sleep(0.05)
    raise SystemExit(f'{name}: a job still runs after 20 s')


def peak_kb(pid) -> int:
    return int(next(line.split()[1] for line in open(f'/proc/{pid}/status') if line.startswith('VmHWM')))


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
    return state


def streamed_large(tls, ca) -> str:
    """A session of large messages in a huge window, each context streaming all of them: from the
    second turn to the last the contexts grow sixfold, and the app's peak memory may grow by less
    than three messages, and no more as the page lists them all; a message pasted past a megabyte joins
    the log whole, and a body that isn't JSON is refused. The app runs without bash (--no-bash), which a
    session with a workspace then lacks."""
    port, root, size, mock_port = free_port(), tempfile.mkdtemp(), 900_000, free_port()
    mock = serve(mock_port, tls)
    app = subprocess.Popen([str(APP), '--root', root, '--addr', f'127.0.0.1:{port}', '--ca', ca, '--no-bash', '--no-open'], stderr=subprocess.PIPE, text=True)
    try:
        for _ in range(100):
            try:
                call(port, '/api/sessions')
                break
            except OSError:
                time.sleep(0.05)
        configure(root, f'https://localhost:{mock_port}', 20000000, 30000, 3000)
        call(port, '/api/sessions', {'name': 'large', 'workspace': tempfile.mkdtemp()})
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
        if len(call(port, '/api/s/large/messages')) != 24 or (listed := peak_kb(app.pid)) - peaks[-1] > 3 * size // 1024:
            raise SystemExit(f'listing the messages took the peak from {peaks[-1]} KB to {listed} KB')
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
    leave the price multiplier near a session's without them (from `baseline`); once a turn never stops
    calling tools; and memories of the calls form with the tools declared but not callable."""
    port, root, work, fake, window, mock_port = free_port(), tempfile.mkdtemp(), tempfile.mkdtemp(), str(HERE / 'fake_mcp.py'), 12000, free_port()
    mock = serve(mock_port, tls)
    servers = {'first': {'command': sys.executable, 'args': [fake]}, 'second': {'command': sys.executable, 'args': [fake, '--shadow', '--wordy']},
               'broken': {'command': '/nonexistent/server'}}
    Path(root, 'mcp.json').write_text(json.dumps({'mcpServers': servers}))

    def start():
        app = subprocess.Popen([str(APP), '--root', root, '--addr', f'127.0.0.1:{port}', '--ca', ca, '--tool-timeout', '1', '--no-open'], stderr=subprocess.PIPE, text=True)
        for _ in range(100):
            try:
                call(port, '/api/sessions')
                break
            except OSError:
                time.sleep(0.05)
        return app
    app, seen['limit'] = start(), window
    configure(root, f'https://localhost:{mock_port}', window, 2381, 595.25)

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
        if (spaces := {s['name']: s['workspace'] for s in call(port, '/api/sessions')}) != {'tools': work, 'bare': None}:
            raise SystemExit(f'the sessions\' workspaces: {spaces}')
        state = call(port, '/api/s/tools')
        if (names := state['tools']) != ['bash', 'edit', 'fail', 'picture', 'hang', 'die', 'add'] or state['overhead'] < 2000:
            raise SystemExit(f'tools {names} ({state["overhead"]} tokens): expected bash, edit, the first server\'s but its add, then the second\'s wordy add')
        got = results('please use tools now', {'fail': 'deny', 'picture': 'always'}, ['add', 'fail', 'picture'])
        expected = {'0': [{'type': 'text', 'text': '5 (shadow)'}], '1': "the user didn't allow this call",
                    '2': [{'type': 'text', 'text': 'a picture:'}, {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'iVBORw0KGgo='}}],
                    '3': 'no tool nope'}
        if {k: r['content'] for k, r in got.items()} != expected or [k for k, r in sorted(got.items()) if r.get('is_error')] != ['1', '3']:
            raise SystemExit(f'tool results: {got}')
        if [e['result']['toolUseId'][-1] for e in seen['events']] != ['0', '1', '2', '3'] or 'iVBORw0KGgo' in json.dumps(seen['events']):
            raise SystemExit(f'the page got {seen["events"]}')
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
        command = f'pwd; echo made > inside; cat inside; echo noted > {shared}/note; cat {secret} 2>&1; cat {shown}/seen; touch {shown}/x 2>&1; echo $$; tail -n +3 /proc/net/dev | cut -d: -f1 | tr -d " "'
        got = results('call ' + json.dumps([['bash', {'command': command}]]), asked=[])
        sandboxed = f"{work}\nmade\ncat: {secret}: No such file or directory\nseen\ntouch: cannot touch '{shown}/x': Read-only file system\n2\nlo\n\n[exit code 0]"
        if got['0']['content'] != sandboxed or Path(work, 'inside').read_text() != 'made\n' or Path(shared, 'note').read_text() != 'noted\n':
            raise SystemExit(f'bash in the sandbox: {got}')
        started = time.monotonic()
        got = results('call ' + json.dumps([['bash', {'command': '(sleep 0.8; touch survived) & sleep 5', 'timeout': 0.4}]]), asked=[])
        time.sleep(max(0.0, started + 1.4 - time.monotonic()))
        if got['0']['content'] != '\n[killed after 0.4 s]' or Path(work, 'survived').exists():
            raise SystemExit(f'a sandboxed command past its timeout: {got}, or what it started outlived it')
        call(port, '/api/s/tools/settings', {'sandbox': False})
        os.symlink(root, Path(work, 'out'))                             # edits: a file made, its text replaced; refusals; ways out
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
            raise SystemExit(f'edits: {outcomes}')
        if [[tool, json.loads(given)] for tool, given in seen['calls']] != [['edit', e] for e in edits]:
            raise SystemExit(f'the calls as the reply streamed them: {seen["calls"][:2]}')
        os.remove(Path(work, 'out'))
        got = results('call [["hang", {}], ["picture", {}]]', asked=['hang'])
        if got['0']['content'] != 'MCP server first: tools/call timed out after 1 s' or got['1'].get('is_error'):
            raise SystemExit(f'a call past its timeout, then another of the same server: {got}')
        got = results('call [["die", {}], ["picture", {}]]', asked=['die'])
        if got['0']['content'] != 'MCP server first: the server stopped' or got['1'].get('is_error'):
            raise SystemExit(f'a server that stops, then a call that starts it again: {got}')
        first, asking, state = [], 'call ' + json.dumps([['add', {'a': 1, 'b': 2}]]), call(port, '/api/s/tools')

        def started():
            request = urllib.request.Request(f'http://127.0.0.1:{port}/api/s/tools/say', data=json.dumps({'text': asking}).encode(), headers={'x-connectome': '1'})
            with urllib.request.urlopen(request, timeout=60) as r:
                first.extend(json.loads(line) for line in r)
        starter = threading.Thread(target=started)
        starter.start()
        for _ in range(500):                                            # until the turn waits for an answer, which it does until one comes
            if any('approve' in e for e in first) and call(port, '/api/s/tools')['running']:
                break
            time.sleep(0.02)
        else:
            raise SystemExit('a turn waiting for an answer did not show as running')
        queued = 'Message queued while a turn waits'
        waiter = threading.Thread(target=lambda: say(port, 'tools', queued))
        waiter.start()
        time.sleep(0.2)                                                 # it reaches the session's thread before the answer
        late = []
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/s/tools/live', timeout=15) as r:
            for line in r:
                late.append(json.loads(line))
                if 'approve' in late[-1]:
                    call(port, '/api/s/tools/approve', {'id': late[-1]['approve'], 'answer': 'allow'})
        starter.join()
        waiter.join()
        # it starts where the log stands (the message and the reply that calls logged, none of the reply's pieces), then
        # gets what the first page gets from there on, but the reply's pieces, which a page a moment behind when the log
        # grows reads from the log
        outline = lambda events: [e for e in events if not {'text', 'thinking'} & set(e)]
        kinds = [next(iter(e)) for e in outline(late) if not {'logged', 'forming'} & set(e)]
        if outline(first)[-len(outline(late)):] != outline(late) or late[0] != {'logged': state['messages'] + 2} or 'approve' not in late[1] \
                or kinds != ['approve', 'answered', 'result', 'ok'] or call(port, '/api/s/tools')['running']:
            raise SystemExit(f'a page that came in during the turn got {kinds} from {late[:2]}; the page that started it {len(first)} events, it {len(late)}')
        told = call(port, f'/api/s/tools/messages?from={state["messages"]}')
        if [m['participant'] for m in told] != ['user', 'assistant', 'user', 'assistant', 'user', 'assistant'] or words(told[4]) != queued:
            raise SystemExit(f'a message sent while a turn ran joined the log as {[(m["participant"], words(m)[:20]) for m in told]}')
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/s/tools/live', timeout=5) as r:
            if r.read():
                raise SystemExit('a page following no turn got events')
        try:
            say(port, 'tools', 'use tools forever', {'add': 'always'})
            raise SystemExit('a turn that never stops calling tools ended')
        except SystemExit as stopped:
            if 'stopped after 64 steps' not in str(stopped):
                raise
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
        return f'{len(seen["results"])} tool steps, {len(seen["asked"])} calls asked to run, {seen["tool memories"]} memory requests with tool calls, {state["messages"]} messages, multiplier {state["multiplier"]:.3f}'
    finally:
        app.terminate()
        mock.shutdown()


def chatted(turns, tls, ca, baseline) -> str:
    """The long chat, the fork and the settings; the main session's price multiplier goes to `baseline`."""
    mock_port, plain_port, port, root = free_port(), free_port(), free_port(), tempfile.mkdtemp()
    mock, plain = serve(mock_port, tls), serve(plain_port)
    app = subprocess.Popen([str(APP), '--root', root, '--addr', f'127.0.0.1:{port}', '--ca', ca, '--no-open'], stderr=subprocess.PIPE, text=True)
    try:
        for _ in range(100):
            try:
                call(port, '/api/sessions')
                break
            except OSError:
                time.sleep(0.05)
        configure(root, f'https://localhost:{mock_port}', WINDOW, 1948, 487)
        call(port, '/api/sessions', {'name': 'main'})
        state = chat(port, 'main', turns, 0)
        if abs(state['multiplier'] - 1) < 0.01:
            raise SystemExit(f'the price multiplier ({state["multiplier"]}) never calibrated to the prompt sizes the API reports')
        baseline.put(state['multiplier'])
        shown = [m['shown'] for m in call(port, '/api/s/main/messages')]     # memories, then raw chunks, then the tail, as the context shows them
        if 'recalled' not in shown or shown[-1] != 'tail' or shown != sorted(shown, key=['recalled', 'raw', 'tail'].index):
            raise SystemExit(f'the context shows the log as {shown}')
        if seen['problems']:
            raise SystemExit(f'the API would reject a request: {seen["problems"][0]}')
        if not seen['thought']:
            raise SystemExit('no reply streamed its reasoning')
        if not seen['forming']:
            raise SystemExit('the page never saw memories forming')
        levels = {level for level, _ in state['frontier'] or []}
        if len(state['layers']) < 3 or not levels - {0}:
            raise SystemExit(f'memories did not form or show: layers {state["layers"]}, frontier {state["frontier"]}')
        call(port, '/api/s/main/settings', {'system': 'You are being tested.'})
        if (made := call(port, '/api/s/main/fork', {'at': 20, 'name': 'branch'})).get('messages') != 21:   # the fork as the page then shows it
            raise SystemExit(f'the fork answered {str(made)[:80]}, not its state with 21 messages')
        logged = lambda name, query: [{k: v for k, v in m.items() if k != 'shown'} for m in call(port, f'/api/s/{name}/messages?{query}')]   # how each context shows them aside
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
        say(port, 'branch', 'with the saved key, over plain HTTP')
        settle(port, 'branch')
        shown, mode = call(port, '/api/settings')['providers'], os.stat(os.path.join(root, 'settings.json')).st_mode & 0o777
        listed = next((p for p in shown if p['name'] == 'plain'), None)
        kept, own = Path(root, 'settings.json').read_text(), Path(root, 'branch', 'session_settings.json').read_text()
        if (seen['key'], seen['model']) != ('k-123', 'mock-2') or listed != {'name': 'plain', 'base': f'http://127.0.0.1:{plain_port}/v1', 'key': True} \
                or 'k-123' in json.dumps(shown) + own or 'k-123' not in kept or mode != 0o600:
            raise SystemExit(f'settings: the API saw key {seen["key"]!r} and model {seen["model"]!r}, the page sees {shown}, settings.json has mode {mode:o}')
        if not next(p for p in call(port, '/api/settings', {'name': 'plain', 'base': f'http://127.0.0.1:{plain_port}/v1', 'key': ' '})['providers'] if p['name'] == 'plain')['key']:
            raise SystemExit('a provider saved without a key lost the one it had')
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
                            ('fork', {'at': 5, 'name': 'branch'})]:     # and a fork onto a session there is
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
            say(port, 'story', 'fail once please')
            raise SystemExit('a turn whose request failed went through')
        except SystemExit as failed:
            if 'Overloaded' not in str(failed):
                raise
        say(port, 'story', None)                                        # the request failed: the message gets its answer
        call(port, '/api/sessions', {'name': 'cut'})
        try:
            say(port, 'cut', 'cut off please')
            raise SystemExit('a reply whose stream was cut off went through')
        except SystemExit as cut:
            if 'ended before' not in str(cut) or [m['participant'] for m in call(port, '/api/s/cut/messages')] != ['user']:
                raise
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
        if told[:4] != [('user', 'tell me a long s'), ('assistant', 'Once upon a time'), ('assistant', ' and so the stor'), ('user', 'fail once please')] \
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


def part(name, run, done):
    """A part of the test in a process group of its own, which its apps and their servers join, so
    that killing the group stops them all: (name, its summary or None, why it failed or None, counts)."""
    os.setpgrp()
    try:
        done.put((name, run(), None, seen))
    except SystemExit as failed:
        done.put((name, None, str(failed), seen))
    except Exception as e:
        done.put((name, None, f'{type(e).__name__}: {e}', seen))


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
    parts = {'chat': lambda: chatted(turns, (cert, key), ca, baseline), 'tools': lambda: tooled((cert, key), ca, baseline), 'large': lambda: streamed_large((cert, key), ca)}
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
          f'(largest {totals["largest"] // 1024} KB); {summaries["tools"]}; {summaries["large"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
