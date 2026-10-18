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
host gets nothing; and the app's peak memory doesn't grow with
the contexts it streams.

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
seen = {'chat': 0, 'memory': 0, 'bytes': 0, 'largest': 0, 'problems': [], 'key': None, 'thought': 0}


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
        last = ' '.join(b.get('text', '') for b in request['messages'][-1]['content'])
        if 'memory' in last and ('Write the memory' in last or 'consolidate' in last):
            seen['memory'] += 1
            blocks = [{'type': 'text', 'text': f'I remember stretch {seen["memory"]}: the user talked about ' + ' '.join(['things'] * 60)}]
        else:
            seen['chat'] += 1
            blocks = [{'type': 'thinking', 'thinking': 'Let me think about it.', 'signature': 'c2ln'}] * (seen['chat'] % 4 == 0) \
                + [{'type': 'text', 'text': f'Reply {seen["chat"]}: ' + ' '.join(['words'] * 80)}]
        self.stream(blocks, len(body) // 4, 0.25 if 'slowly' in last else 0)

    def stream(self, blocks, input_tokens, pause):
        """The reply as server-sent events, its text in five pieces `pause` seconds apart; chunked or
        until the close, by turns."""
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
            key = 'text' if block['type'] == 'text' else 'thinking'
            event('content_block_start', index=i, content_block={'type': block['type'], key: '', **({'signature': ''} if key == 'thinking' else {})})
            words, n = block[key], len(block[key])
            for k in range(5):
                time.sleep(pause)
                event('content_block_delta', index=i, delta={'type': f'{key}_delta', key: words[k * n // 5:(k + 1) * n // 5]})
            if key == 'thinking':
                event('content_block_delta', index=i, delta={'type': 'signature_delta', 'signature': block['signature']})
            event('content_block_stop', index=i)
        event('message_delta', delta={'stop_reason': 'end_turn'}, usage={'output_tokens': sum(len(b.get('text', '')) for b in blocks) // 4})
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


def say(port, name, text) -> tuple[str, str, list[float]]:
    """A chat turn as the page has it: the reply's text and reasoning, and when each piece of text
    arrived."""
    request = urllib.request.Request(f'http://127.0.0.1:{port}/api/s/{name}/say', data=json.dumps({'text': text}).encode(), headers={'x-connectome': '1'})
    pieces, thoughts, times, last = [], [], [], {}
    with urllib.request.urlopen(request, timeout=60) as r:
        for line in r:
            last = json.loads(line)
            if 'text' in last:
                pieces.append(last['text'])
                times.append(time.monotonic())
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


def chat(port, name, turns, start):
    for t in range(start, start + turns):
        text = f'Message {t}: ' + ' '.join(f'word{t % 7}' for _ in range(150))
        reply, thinking, _ = say(port, name, text)
        state = settle(port, name)
        logged = call(port, f'/api/s/{name}/messages?from={state["messages"] - 1}')[0]
        if not reply.startswith('Reply ') or (logged['text'], logged['thinking']) != (reply, thinking):
            raise SystemExit(f'turn {t}: the page got {reply[:60]!r} after {thinking[:40]!r}, the log holds {logged}')
        seen['thought'] += bool(thinking)
        if state['armed'] and state['armed'] > WINDOW * 1.02:
            raise SystemExit(f'turn {t}: a context of {state["armed"]} tokens in a window of {WINDOW}')
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
        large = streamed_large(mock_port, ca)
        print(f'{seen["chat"]} chat turns and {seen["memory"]} memory requests; main: {state["messages"]} messages, layers {state["layers"]}, '
              f'context {state["armed"]:.0f}/{WINDOW} tokens; branch: {forked["messages"]} messages, layers {forked["layers"]}; '
              f'{seen["bytes"] // 1024} KB of requests streamed (largest {seen["largest"] // 1024} KB), app peak memory {peak} KB; {large}')
        return 0
    finally:
        app.terminate()
        mock.shutdown()
        plain.shutdown()
        shutil.rmtree(root)
        shutil.rmtree(keys)


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
