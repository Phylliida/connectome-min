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
fork point; a key saved from the page reaches the API but never the page, in a file only its user
can read; another site's page can't change the settings; a server whose certificate names another
host gets nothing; the model's tool calls, through MCP servers, get their results and form
memories; and the app's peak memory doesn't grow with the contexts it streams.

    python3 tests/test_app.py [turns]
"""
import datetime
import json
import os
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
seen = {'chat': 0, 'memory': 0, 'bytes': 0, 'largest': 0, 'problems': [], 'key': None, 'thought': 0, 'results': [], 'tool memories': 0, 'events': [], 'asked': [], 'limit': None}


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
        seen['largest'] = max(seen['largest'], len(body))
        request = json.loads(body)
        if problem := invalid(request):
            return self.problem(problem)
        if request.get('stream') is not True:
            return self.problem('the request did not ask for a stream')
        if seen['limit'] and not request.get('tool_choice') and len(body) // 4 + request['max_tokens'] > seen['limit']:   # tokens as usage reports them
            return self.problem(f'prompt is too long: {len(body) // 4} tokens and {request["max_tokens"]} to answer in a window of {seen["limit"]}')
        messages = request['messages']
        last = ' '.join(b.get('text', '') for b in messages[-1]['content'])
        results = [b for b in messages[-1]['content'] if b['type'] == 'tool_result']
        asked = next(' '.join(b.get('text', '') for b in m['content']) for m in reversed(messages) if m['role'] == 'user' and any(b['type'] == 'text' for b in m['content']))
        stop = 'end_turn'
        if 'memory' in last and ('Write the memory' in last or 'consolidate' in last):
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
        self.stream(blocks, len(body) // 4, 0.25 if 'slowly' in last else 0, stop)

    def stream(self, blocks, input_tokens, pause, stop):
        """The reply as server-sent events, its text in five pieces `pause` seconds apart and a tool
        call's input in three; chunked or until the close, by turns."""
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


def invalid(request) -> str | None:
    """Why the Messages API would reject the request, if it would."""
    messages = request.get('messages') or []
    roles = [m['role'] for m in messages]
    if not messages or roles[0] != 'user' or roles[-1] != 'user' or any(a == b for a, b in zip(roles, roles[1:])):
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


def say(port, name, text, answers=None) -> tuple[str, str, list[float]]:
    """A chat turn as the page has it: the reply's text and reasoning, and when each piece of text
    arrived. A call that asks to run gets the answer `answers` has for its tool, else allow."""
    request = urllib.request.Request(f'http://127.0.0.1:{port}/api/s/{name}/say', data=json.dumps({'text': text}).encode(), headers={'x-connectome': '1'})
    pieces, thoughts, times, last, tools = [], [], [], {}, {}
    with urllib.request.urlopen(request, timeout=60) as r:
        for line in r:
            last = json.loads(line)
            if 'text' in last:
                pieces.append(last['text'])
                times.append(time.monotonic())
            if 'tool' in last or 'result' in last:
                seen['events'].append(last)
            if 'tool' in last:
                tools[last['tool']['id']] = last['tool']['name']
            if 'approve' in last:
                seen['asked'].append(tools[last['approve']])
                call(port, '/api/approve', {'id': last['approve'], 'answer': (answers or {}).get(tools[last['approve']], 'allow')})
            thoughts.append(last.get('thinking', ''))
    if not last.get('ok'):
        raise SystemExit(f'{name}: "{text[:40]}" got {last}')
    return ''.join(pieces), ''.join(thoughts), times


def settle(port, name):
    """The session's state once its memories have formed."""
    while (state := call(port, f'/api/s/{name}'))['forming']:
        time.sleep(0.05)
    return state


def peak_kb(pid) -> int:
    return int(next(line.split()[1] for line in open(f'/proc/{pid}/status') if line.startswith('VmHWM')))


def chat(port, name, turns, start, window=WINDOW):
    for t in range(start, start + turns):
        text = f'Message {t}: ' + ' '.join(f'word{t % 7}' for _ in range(150))
        reply, thinking, _ = say(port, name, text)
        state = settle(port, name)
        logged = call(port, f'/api/s/{name}/messages?from={state["messages"] - 1}')[0]
        if not reply.startswith('Reply ') or (logged['text'], logged['thinking']) != (reply, thinking):
            raise SystemExit(f'turn {t}: the page got {reply[:60]!r} after {thinking[:40]!r}, the log holds {logged}')
        seen['thought'] += bool(thinking)
        if state['armed'] and state['armed'] > window * 1.02:
            raise SystemExit(f'turn {t}: a context of {state["armed"]} tokens in a window of {window}')
    return state


def streamed_large(mock_port, ca) -> str:
    """A session of large messages in a huge window, each context streaming all of them: from the
    second turn to the last the contexts grow sixfold, and the app's peak memory may grow by less
    than three messages."""
    port, root, size = free_port(), tempfile.mkdtemp(), 900_000
    app = subprocess.Popen([str(APP), '--root', root, '--addr', f'127.0.0.1:{port}', '--base', f'https://localhost:{mock_port}', '--ca', ca,
                            '--model', 'mock', '--window', '20000000', '--reserve', '1000', '--no-open'], stderr=subprocess.PIPE, text=True)
    try:
        for _ in range(100):
            try:
                call(port, '/api/sessions', {'name': 'large'})
                break
            except OSError:
                time.sleep(0.05)
        peaks, contexts = [], []
        for t in range(12):
            words = ' '.join(f'w{t}x{i}' for i in range(size // 8))[:size]
            say(port, 'large', words)
            settle(port, 'large')
            peaks.append(peak_kb(app.pid))
            contexts.append(seen['largest'] // 1024)
        if peaks[-1] - peaks[1] > 3 * size // 1024:
            raise SystemExit(f'peak memory grew from {peaks[1]} KB to {peaks[-1]} KB as contexts grew from {contexts[1]} KB to {contexts[-1]} KB')
        return f'contexts of {contexts[1]} KB to {contexts[-1]} KB streamed at peaks of {peaks[1]} KB to {peaks[-1]} KB'

    finally:
        app.terminate()
        shutil.rmtree(root)


def tooled(mock_port, ca, multiplier) -> str:
    """A session whose model calls bash and the tools of two MCP servers (fake_mcp.py, the second, whose
    tools are wordy, shadowing the first's add), and of none: each call runs once the user allows it, or
    always allows its tool, and fails if they deny it, the tools always allowed kept across a restart; a
    command past its timeout is killed with the processes it started, and it or an MCP call past its
    timeout fails without holding up the next call; a server that stops starts again. Every call gets its result,
    in the log and on the page; the requests fit the window with the tool definitions counted, which
    leave the price multiplier near `multiplier`, a session's without them; once a turn never stops
    calling tools; and memories of the calls form with the tools declared but not callable."""
    port, root, work, fake, window = free_port(), tempfile.mkdtemp(), tempfile.mkdtemp(), str(HERE / 'fake_mcp.py'), 12000
    servers = {'first': {'command': sys.executable, 'args': [fake]}, 'second': {'command': sys.executable, 'args': [fake, '--shadow', '--wordy']},
               'broken': {'command': '/nonexistent/server'}}
    Path(root, 'mcp.json').write_text(json.dumps({'mcpServers': servers}))

    def start():
        app = subprocess.Popen([str(APP), '--root', root, '--addr', f'127.0.0.1:{port}', '--base', f'https://localhost:{mock_port}', '--ca', ca, '--model', 'mock',
                                '--window', str(window), '--reserve', '1000', '--workdir', work, '--tool-timeout', '2', '--no-open'], stderr=subprocess.PIPE, text=True)
        for _ in range(100):
            try:
                call(port, '/api/sessions')
                break
            except OSError:
                time.sleep(0.05)
        return app
    app, seen['limit'] = start(), window

    def results(text, answers=None, asked=None):
        """The results the turn's tool calls got, by the call's place in the reply."""
        before = len(seen['asked'])
        say(port, 'tools', text, answers)
        settle(port, 'tools')
        if asked is not None and seen['asked'][before:] != asked:
            raise SystemExit(f'"{text}" asked to run {seen["asked"][before:]}, not {asked}')
        return {r['tool_use_id'].split('-')[1]: r for r in seen['results'][-1]}
    try:
        call(port, '/api/sessions', {'name': 'tools'})
        state = call(port, '/api/s/tools')
        if (names := [t['name'] for t in state['tools']]) != ['bash', 'fail', 'picture', 'hang', 'die', 'add'] or state['overhead'] < 2000:
            raise SystemExit(f'tools {names} ({state["overhead"]} tokens): expected bash, the first server\'s but its add, then the second\'s wordy add')
        got = results('please use tools now', {'fail': 'deny', 'picture': 'always'}, ['add', 'fail', 'picture'])
        expected = {'0': [{'type': 'text', 'text': '5 (shadow)'}], '1': "the user didn't allow this call",
                    '2': [{'type': 'text', 'text': 'a picture:'}, {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'iVBORw0KGgo='}}],
                    '3': 'no tool nope'}
        if {k: r['content'] for k, r in got.items()} != expected or [k for k, r in sorted(got.items()) if r.get('is_error')] != ['1', '3']:
            raise SystemExit(f'tool results: {got}')
        if [e.get('tool', e.get('result'))['id'][-1] for e in seen['events']] != ['0', '0', '1', '1', '2', '2', '3', '3']:
            raise SystemExit(f'the page got {seen["events"]}')
        allowed = json.loads(Path(root, 'settings.json').read_text())['allow']
        if allowed != ['picture'] or [t['name'] for t in call(port, '/api/s/tools')['tools'] if t['allowed']] != ['picture']:
            raise SystemExit(f'always allowed: {allowed}')
        notes = ' '.join(m['text'] for m in call(port, '/api/s/tools/messages'))
        if '[tool call add {"a":2,"b":3}]' not in notes or "[tool result, failed] the user didn't allow this call" not in notes:
            raise SystemExit(f'the log shows {notes[:400]}')
        got = results('call [["bash", {"command": "pwd; echo out; echo err >&2; exit 3"}]]', {'bash': 'always'}, ['bash'])
        if got['0']['content'] != f'{work}\nout\nerr\n\n[exit code 3]' or got['0'].get('is_error'):
            raise SystemExit(f'bash: {got}')
        started = time.monotonic()
        got = results('call [["bash", {"command": "(sleep 1.5; touch survived) & sleep 5; echo never", "timeout": 1}]]', asked=[])
        if got['0']['content'] != '\n[killed after 1 s]' or not got['0'].get('is_error') or time.monotonic() - started > 4:
            raise SystemExit(f'bash past its timeout: {got} after {time.monotonic() - started:.1f} s')
        time.sleep(max(0.0, started + 2.5 - time.monotonic()))
        if Path(work, 'survived').exists():
            raise SystemExit('a process that a command past its timeout started outlived it')
        got = results('call [["hang", {}], ["picture", {}]]', asked=['hang'])
        if got['0']['content'] != 'MCP server first: tools/call timed out after 2 s' or got['1'].get('is_error'):
            raise SystemExit(f'a call past its timeout, then another of the same server: {got}')
        got = results('call [["die", {}], ["picture", {}]]', asked=['die'])
        if got['0']['content'] != 'MCP server first: the server stopped' or got['1'].get('is_error'):
            raise SystemExit(f'a server that stops, then a call that starts it again: {got}')
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
        if state['multiplier'] > multiplier + 0.1:
            raise SystemExit(f'the tool definitions skewed the price multiplier: {state["multiplier"]:.3f}, without them {multiplier:.3f}')
        call(port, '/api/settings', {'allow': ['bash']})
        app.terminate()
        app.wait()
        app = start()
        if (allowed := [t['name'] for t in call(port, '/api/s/tools')['tools'] if t['allowed']]) != ['bash']:
            raise SystemExit(f'after a restart the tools always allowed are {allowed}, not bash, as the page set them')
        return f'{len(seen["results"])} tool steps, {len(seen["asked"])} calls asked to run, {seen["tool memories"]} memory requests with tool calls, {state["messages"]} messages, multiplier {state["multiplier"]:.3f}'
    finally:
        seen['limit'] = None
        app.terminate()
        shutil.rmtree(root)
        shutil.rmtree(work)


def main(turns=60):
    turns = int(turns)
    build = subprocess.run(['cargo', 'build', '--release', '--quiet', '-p', 'connectome-web'], cwd=HERE.parent, capture_output=True, text=True)
    if build.returncode and not os.environ.get('CONNECTOME_WEB'):
        print(build.stderr)
        return 1
    mock_port, plain_port, port, root, keys = free_port(), free_port(), free_port(), tempfile.mkdtemp(), tempfile.mkdtemp()
    ca, cert, key = certificates(keys)
    mock, plain = serve(mock_port, (cert, key)), serve(plain_port)
    app = subprocess.Popen([str(APP), '--root', root, '--addr', f'127.0.0.1:{port}', '--base', f'https://localhost:{mock_port}', '--ca', ca,
                            '--model', 'mock', '--window', str(WINDOW), '--reserve', '1000', '--no-open'], stderr=subprocess.PIPE, text=True)
    try:
        for _ in range(100):
            try:
                call(port, '/api/sessions')
                break
            except OSError:
                time.sleep(0.05)
        call(port, '/api/sessions', {'name': 'main'})
        state = chat(port, 'main', turns, 0)
        if abs(state['multiplier'] - 1) < 0.01:
            raise SystemExit(f'the price multiplier ({state["multiplier"]}) never calibrated to the prompt sizes the API reports')
        if seen['problems']:
            raise SystemExit(f'the API would reject a request: {seen["problems"][0]}')
        if not seen['thought']:
            raise SystemExit('no reply streamed its reasoning')
        levels = {level for level, _ in state['frontier'] or []}
        if len(state['layers']) < 3 or not levels - {0}:
            raise SystemExit(f'memories did not form or show: layers {state["layers"]}, frontier {state["frontier"]}')
        call(port, '/api/s/main/fork', {'at': 20, 'name': 'branch'})
        parent, fork = call(port, '/api/s/main/messages?from=0&to=21'), call(port, '/api/s/branch/messages?from=0')
        if fork != parent:
            raise SystemExit('the fork does not share its parent\'s messages')
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
        call(port, '/api/settings', {'key': 'k-123', 'model': 'mock-2', 'base': f'http://127.0.0.1:{plain_port}/v1'})   # a base with /v1, as OpenRouter's
        say(port, 'branch', 'with the saved key, over plain HTTP')
        settle(port, 'branch')
        settings, mode = call(port, '/api/settings'), os.stat(os.path.join(root, 'settings.json')).st_mode & 0o777
        if seen['key'] != 'k-123' or settings != {'base': f'http://127.0.0.1:{plain_port}/v1', 'model': 'mock-2', 'key': True} or mode != 0o600:
            raise SystemExit(f'settings: the API saw key {seen["key"]!r}, the page sees {settings}, settings.json has mode {mode:o}')
        call(port, '/api/settings', {'base': f'https://127.0.0.1:{mock_port}'})
        try:
            say(port, 'branch', 'to a name the certificate does not hold')
            raise SystemExit('the app talked to a server whose certificate names another host')
        except SystemExit as refused:
            if 'certificate' not in str(refused).lower():
                raise
        call(port, '/api/settings', {'base': f'https://localhost:{mock_port}'})
        settle(port, 'branch')
        _, _, times = say(port, 'branch', 'answer slowly please')
        if len(times) < 5 or times[-1] - times[0] < 0.75:
            raise SystemExit(f'the reply reached the page all at once: {len(times)} pieces over {times[-1] - times[0]:.2f} s')
        peak = peak_kb(app.pid)
        tooling = tooled(mock_port, ca, state['multiplier'])
        large = streamed_large(mock_port, ca)
        print(f'{seen["chat"]} chat turns and {seen["memory"]} memory requests; main: {state["messages"]} messages, layers {state["layers"]}, '
              f'context {state["armed"]:.0f}/{WINDOW} tokens, multiplier {state["multiplier"]:.3f}; branch: {forked["messages"]} messages, layers {forked["layers"]}; '
              f'{seen["bytes"] // 1024} KB of requests streamed (largest {seen["largest"] // 1024} KB), app peak memory {peak} KB; {tooling}; {large}')
        return 0
    finally:
        app.terminate()
        mock.shutdown()
        plain.shutdown()
        shutil.rmtree(root)
        shutil.rmtree(keys)


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
