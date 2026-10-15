"""The summarizer's calls and how the strategy reads what comes back. A call returns a Reply or a
Failure and never raises."""
import re
from typing import Callable, NamedTuple

from blocks import well_formed
from js import JS_SPACE, js_trim

STOPS = {'end_turn', 'max_tokens', 'stop_sequence', 'tool_use', 'refusal', 'abort'}
_WRAPPED = re.compile(rf'[{JS_SPACE}]*<thinking>.*?</thinking>[{JS_SPACE}]*', re.I | re.A | re.S)
_OPENS = re.compile(rf'[{JS_SPACE}]*<thinking>', re.I | re.A)


class Host(NamedTuple):
    """The host's side of a call."""
    complete: Callable                    # request -> response, as membrane normalizes them; raises on failure
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
    """A call that raised: the provider's error."""
    error: Exception


def call(host: Host, request: dict) -> Reply | Failure:
    try:
        return Reply(request, host.complete(request))
    except Exception as error:
        return Failure(error)
