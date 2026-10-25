"""A fake of Claude Code's headless mode, as connectome-web runs it (app/src/claude.rs): JSON lines in
and out, the Agent SDK's control requests among them. It answers the initialize request, says it has
started (its session id, from --session-id or --resume), and takes a turn on each message it's sent,
in order, echoing each as it takes it in (--replay-user-messages): one sent during a turn after the
step it's on (a call's result), or as a turn of its own once the turn's reply is done. Its turns:

- 'call TOOL': a reply of text, then a call to TOOL that it asks about (can_use_tool), its result
  what the answer let it do, then text saying so;
- 'delegate TOOL': text, then a call to a subagent (Agent), whose own call to TOOL it asks about, the
  subagent's messages and stream marked as the subagent's (parent_tool_use_id);
- 'wait': text, then a call to Bash that doesn't ask, whose result comes once a message has come
  (or after 10 s), which it takes in then, its reply then saying what that message said;
- 'linger': text that takes 1.5 s to finish, so a message comes during it;
- 'hang': text, then nothing, as a turn the user will stop;
- anything else: 'You said: …'.

Each run first writes into its working directory's claude-runs.jsonl what it was run with (its
arguments, whether a write outside that directory failed, as in a sandbox, and the status of a request
to its ANTHROPIC_BASE_URL with ANTHROPIC_CUSTOM_HEADERS), then streams its reply in pieces and gives
each finished block as Claude Code does. It ends with a result once the turn has, and exits when its
input closes."""
import json
import os
import queue
import sys
import threading
import time
import urllib.request

args = sys.argv[1:]
session = args[args.index('--session-id') + 1] if '--session-id' in args else args[args.index('--resume') + 1]


def say(line):
    print(json.dumps(line), flush=True)


inbox, waiting = queue.Queue(), []   # what comes on stdin, a line each (None at its end); messages come during a step, not yet taken in


def reading():
    for line in sys.stdin:
        inbox.put(json.loads(line))
    inbox.put(None)


def answer():
    """The answer to the control request just made; a message that comes first waits for the step's end."""
    while (line := inbox.get())['type'] != 'control_response':
        waiting.append(line)
    return line['response']['response']


def take_in():
    """The messages that came during the step, echoed as they're taken in; their text."""
    texts = []
    while waiting:
        message = waiting.pop(0)
        say({**message, 'isReplay': True, 'session_id': session})
        texts.append(' '.join(b['text'] for b in message['message']['content'] if b['type'] == 'text'))
    return texts


def gateway():
    headers = dict(h.split(': ', 1) for h in os.environ.get('ANTHROPIC_CUSTOM_HEADERS', '').split('\n') if h)
    body = json.dumps({'model': 'fake', 'messages': [{'role': 'user', 'content': 'hi'}]}).encode()
    request = urllib.request.Request(os.environ['ANTHROPIC_BASE_URL'] + '/v1/messages', data=body,
                                     headers={**headers, 'x-claude-code-session-id': session, 'content-type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=10) as r:
            return r.status
    except Exception as e:
        return repr(e)


def outside():
    """Whether a write next to the working directory, outside it, failed."""
    try:
        with open(os.path.join(os.path.dirname(os.getcwd()), 'claude-outside'), 'w') as f:
            f.write('out')
        return False
    except OSError:
        return True


def streamed(i, block, pieces):
    """Block `i` of the reply, streamed in `pieces` (text, or a call's input JSON), then given whole."""
    say({'type': 'stream_event', 'event': {'type': 'content_block_start', 'index': i, 'content_block': {**block, **({'input': {}} if block['type'] == 'tool_use' else {'text': ''})}}, 'parent_tool_use_id': None})
    for piece in pieces:
        delta = {'type': 'input_json_delta', 'partial_json': piece} if block['type'] == 'tool_use' else {'type': 'text_delta', 'text': piece}
        say({'type': 'stream_event', 'event': {'type': 'content_block_delta', 'index': i, 'delta': delta}, 'parent_tool_use_id': None})
    say({'type': 'assistant', 'message': {'id': reply_id, 'role': 'assistant', 'content': [block]}, 'parent_tool_use_id': None, 'session_id': session})
    say({'type': 'stream_event', 'event': {'type': 'content_block_stop', 'index': i}, 'parent_tool_use_id': None})


threading.Thread(target=reading, daemon=True).start()
with open('claude-runs.jsonl', 'a') as runs:
    runs.write(json.dumps({'args': args, 'outside_failed': outside(), 'gateway': gateway()}) + '\n')
initialize = inbox.get()
say({'type': 'control_response', 'response': {'subtype': 'success', 'request_id': initialize['request_id'], 'response': {}}})
say({'type': 'system', 'subtype': 'init', 'session_id': session, 'claude_code_version': 'fake', 'tools': ['Bash', 'Write']})
while (message := waiting.pop(0) if waiting else inbox.get()) is not None:
    waiting.insert(0, message)
    text = take_in()[0]
    reply_id = f'msg_{time.monotonic_ns()}'
    say({'type': 'stream_event', 'event': {'type': 'message_start', 'message': {'id': reply_id, 'usage': {'input_tokens': 10}}}, 'parent_tool_use_id': None})
    if text.startswith('call '):
        tool = text.split()[1]
        streamed(0, {'type': 'text', 'text': 'Calling.'}, ['Call', 'ing.'])
        call = {'type': 'tool_use', 'id': 'toolu_1', 'name': tool, 'input': {'file_path': 'x.txt'}}
        streamed(1, call, ['{"file_path":', ' "x.txt"}'])
        say({'type': 'control_request', 'request_id': 'ask-1', 'request': {'subtype': 'can_use_tool', 'tool_name': tool, 'input': call['input'], 'tool_use_id': 'toolu_1'}})
        decided = answer()
        allowed = decided['behavior'] == 'allow'
        result = {'tool_use_id': 'toolu_1', 'type': 'tool_result', 'content': 'did it' if allowed else decided['message'], 'is_error': not allowed}
        say({'type': 'user', 'message': {'role': 'user', 'content': [result]}, 'parent_tool_use_id': None, 'session_id': session})
        reply_id = f'msg_{time.monotonic_ns()}'
        streamed(0, {'type': 'text', 'text': 'It ran.' if allowed else 'Not allowed.'}, ['It ran.' if allowed else 'Not allowed.'])
        said = 'done'
    elif text.startswith('delegate '):
        tool = text.split()[1]
        streamed(0, {'type': 'text', 'text': 'Delegating.'}, ['Delegating.'])
        streamed(1, {'type': 'tool_use', 'id': 'toolu_agent', 'name': 'Agent', 'input': {'prompt': 'touch sub'}}, ['{"prompt": "touch sub"}'])
        sub = {'type': 'tool_use', 'id': 'toolu_sub', 'name': tool, 'input': {'command': 'touch sub'}}
        say({'type': 'stream_event', 'event': {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'SUBAGENT'}}, 'parent_tool_use_id': 'toolu_agent'})
        say({'type': 'assistant', 'message': {'id': 'msg_sub', 'role': 'assistant', 'content': [sub]}, 'parent_tool_use_id': 'toolu_agent', 'session_id': session})
        say({'type': 'control_request', 'request_id': 'ask-sub', 'request': {'subtype': 'can_use_tool', 'tool_name': tool, 'input': sub['input'], 'tool_use_id': 'toolu_sub', 'agent_id': 'agent-1'}})
        allowed = answer()['behavior'] == 'allow'
        say({'type': 'user', 'message': {'role': 'user', 'content': [{'tool_use_id': 'toolu_sub', 'type': 'tool_result', 'content': 'ok'}]}, 'parent_tool_use_id': 'toolu_agent', 'session_id': session})
        done = 'The subagent did it.' if allowed else 'The subagent was denied.'
        say({'type': 'user', 'message': {'role': 'user', 'content': [{'tool_use_id': 'toolu_agent', 'type': 'tool_result', 'content': done}]}, 'parent_tool_use_id': None, 'session_id': session})
        reply_id = f'msg_{time.monotonic_ns()}'
        streamed(0, {'type': 'text', 'text': done}, [done])
        said = 'done'
    elif text == 'wait':
        streamed(0, {'type': 'text', 'text': 'Working.'}, ['Working.'])
        streamed(1, {'type': 'tool_use', 'id': 'toolu_wait', 'name': 'Bash', 'input': {'command': 'sleep'}}, ['{"command": "sleep"}'])
        try:
            if not waiting:
                waiting.append(inbox.get(timeout=10))
        except queue.Empty:
            pass
        say({'type': 'user', 'message': {'role': 'user', 'content': [{'tool_use_id': 'toolu_wait', 'type': 'tool_result', 'content': 'slept'}]}, 'parent_tool_use_id': None, 'session_id': session})
        extra = take_in()
        reply_id = f'msg_{time.monotonic_ns()}'
        said = f'Also: {" ".join(extra)}.' if extra else 'Nothing more.'
        streamed(0, {'type': 'text', 'text': said}, [said])
    elif text == 'linger':
        say({'type': 'stream_event', 'event': {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}}, 'parent_tool_use_id': None})
        say({'type': 'stream_event', 'event': {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'Lingering'}}, 'parent_tool_use_id': None})
        time.sleep(1.5)
        say({'type': 'stream_event', 'event': {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': ' done.'}}, 'parent_tool_use_id': None})
        say({'type': 'assistant', 'message': {'id': reply_id, 'role': 'assistant', 'content': [{'type': 'text', 'text': 'Lingering done.'}]}, 'parent_tool_use_id': None, 'session_id': session})
        said = 'lingered'
    elif text == 'hang':
        streamed(0, {'type': 'text', 'text': 'Working'}, ['Working'])
        time.sleep(60)
        said = 'woke'
    else:
        streamed(0, {'type': 'text', 'text': f'You said: {text}'}, ['You said: ', text])
        said = text
    say({'type': 'result', 'subtype': 'success', 'is_error': False, 'result': said, 'session_id': session, 'total_cost_usd': 0.01,
         'usage': {'input_tokens': 10, 'cache_read_input_tokens': 5, 'cache_creation_input_tokens': 0, 'output_tokens': 7}})
