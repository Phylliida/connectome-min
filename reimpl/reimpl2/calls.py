"""The summarizer's calls (runAttempt and its transport) and how the strategy reads what comes back.
A call returns a Reply or a Failure and never raises, so each caller states what a failure does."""
import math
import re
from typing import Callable, NamedTuple

from blocks import REASONING, well_formed
from config import Config
from js import JS_SPACE, js_len, js_trim, sha256
from state import Entry, State
from turns import strip_reasoning

STOPS = {'end_turn', 'max_tokens', 'stop_sequence', 'tool_use', 'refusal', 'abort'}
_WRAPPED = re.compile(rf'[{JS_SPACE}]*<thinking>.*?</thinking>[{JS_SPACE}]*', re.I | re.A | re.S)
_OPENS = re.compile(rf'[{JS_SPACE}]*<thinking>', re.I | re.A)


class Host(NamedTuple):
    """The host's side of a call."""
    complete: Callable                    # request -> response, as membrane normalizes them; raises on failure,
                                          # with the HTTP status as .status, the kind as .type, and .retryable
    tools: list | None = None             # the agent's live tool definitions, once declared
    system: str | None = None             # the host's system prompt, once declared


class Reply(NamedTuple):
    """A call's result: the request the provider took, and its response."""
    sent: dict
    response: dict

    @property
    def stop(self):
        return self.response.get('stopReason')

    @property
    def verdict(self) -> tuple[str, str | None]:
        """assessFallbackCompressionResponse: (outcome, stopReason). Only 'valid', a complete end_turn of
        well-formed blocks with text, may become a memory."""
        stop, content = s if isinstance(s := self.stop, str) else None, self.response.get('content')
        if stop not in STOPS or not isinstance(content, list) or not all(map(well_formed, content)):
            return 'provider_error', stop
        if stop == 'refusal':
            return 'refusal', stop
        if not js_trim('\n'.join(b['text'] for b in content if b['type'] == 'text')):
            return 'unusable_empty', stop
        return 'valid' if stop == 'end_turn' else 'incomplete', stop

    @property
    def text(self) -> str:
        """The memory's text: the reply's text blocks, less a leading <thinking>…</thinking>
        (stripThinkingPreamble); a reply that opens one and never closes it was all thinking."""
        text = '\n'.join(b.get('text') or '' for b in self.response['content'] if b['type'] == 'text')
        if m := _WRAPPED.match(text):
            return text[m.end():]
        return '' if _OPENS.match(text) else text       # an opening _WRAPPED didn't match never closes


class Failure(NamedTuple):
    """A call that raised: the provider's error, with the fields the library reads off it."""
    error: Exception

    @property
    def retryable(self):
        return getattr(self.error, 'retryable', None)

    @property
    def kind(self):
        return getattr(self.error, 'type', None)


def call(host: Host, request: dict) -> Reply | Failure:
    """runAttempt's transport. A request the provider rejects for its reasoning blocks goes once more
    without them."""
    result = attempt(host, request)
    if isinstance(result, Failure) and carrier_rejected(result.error, request):
        return attempt(host, {**request, 'messages': [{**t, 'content': strip_reasoning(t['content'])} for t in request['messages']]})   # none empties
    return result


def attempt(host: Host, request: dict) -> Reply | Failure:
    try:
        return Reply(request, host.complete(request))
    except Exception as error:
        return Failure(error)


def carrier_rejected(error: Exception, request: dict) -> bool:
    """isCarrierTransportRejection: an invalid-request error (a 400, an 'invalid' type, or
    'invalid_request' in its message) whose message names thinking or reasoning, for a request that
    carries reasoning blocks."""
    message, kind = str(error).lower(), getattr(error, 'type', None)
    rejected = getattr(error, 'status', None) == 400 or isinstance(kind, str) and 'invalid' in kind.lower() or 'invalid_request' in message
    return (rejected and ('thinking' in message or 'reasoning' in message)
            and any(b['type'] in REASONING for t in request['messages'] for b in t['content']))


def input_tokens(response: dict) -> float | None:
    """compressionResponseInputTokens: the provider's count of the request's input, if it gave one above
    zero. Anthropic and Bedrock count cache reads and writes apart from it."""
    obj = lambda value: value if isinstance(value, dict) else {}
    details = obj(response.get('details'))
    usage = details['usage'] if isinstance(details.get('usage'), dict) else obj(response.get('usage'))
    read = lambda key: v if type(v := usage.get(key)) in (int, float) and v >= 0 else None   # JSON carries no infinity
    provider = obj(details.get('model')).get('provider')
    if (n := read('inputTokens')) is not None and isinstance(provider, str) and provider.lower() in ('anthropic', 'bedrock'):
        n += (read('cacheCreationTokens') or 0) + (read('cacheReadTokens') or 0)
    return n or None


def written(state: State, level: int, sources: tuple, span: tuple, reply: Reply, cfg: Config) -> Entry:
    """The memory a reply wrote. Its tokens are the reply's output tokens, else chars/3; it keeps the
    reply's reasoning and text blocks if it reasoned (captureResponseContent), and the hash of the
    request the provider took."""
    out, content, text = (reply.response.get('usage') or {}).get('outputTokens'), reply.response['content'], reply.text
    return Entry(f'L{level}-{len(state.entries)}', level, text, out if out and out > 0 else math.ceil(js_len(text) / 3), sources, span,
                 blocks=tuple(b for b in content if b['type'] in REASONING + ('text',)) if any(b['type'] in REASONING for b in content) else None,
                 provenance=('end_turn', sha256(reply.sent), cfg.compression_model))
