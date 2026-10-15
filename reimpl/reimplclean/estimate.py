"""What a message costs in @animalabs/context-manager 0.10.1, reproduced to the integer. A message
has two prices, each summed over its top-level blocks (blocks.py):
  text_only  sizes chunks (estimateTextOnlyTokens): chars/4, uncalibrated, blind to tool_result block lists.
  store      prices the windows (MessageStore.estimateTokens): density-aware chars/token, each top-level
             block scaled by the calibration multiplier and rounded.
"""
from typing import NamedTuple

from blocks import image_bytes, raw_tokens, text_only_tokens
from js import js_round


class Priced(NamedTuple):
    """What the windows and the chunker read from a message. Messages never change, so this is computed
    once; the multiplier applies on read, as in the library's per-block cache."""
    id: object
    kinds: frozenset                      # top-level block types
    raw: tuple                            # raw_tokens of each top-level block
    images: tuple                         # (raw_tokens, base64 length) of each top-level image
    text_only: int


def price(message) -> Priced:
    """message: {'id': ..., 'content': [blocks]}."""
    content = message['content']
    raw = tuple(map(raw_tokens, content))
    images = tuple((r, image_bytes(b)) for r, b in zip(raw, content) if b['type'] == 'image')
    return Priced(message['id'], frozenset(b['type'] for b in content), raw, images,
                  sum(map(text_only_tokens, content)))


def store_tokens(message: Priced, multiplier: float = 1.0) -> int:
    """Rounded per block, then summed: rounding the sum instead gives different totals."""
    return sum(js_round(r * multiplier) for r in message.raw)


class Calibration(NamedTuple):
    """The store multiplier, learned from usage (reportRealInputTokens). A committed compile arms one
    sample with the window total it rendered; only the next usage report counts, since later calls in
    the turn send a grown request. The harness keeps the multiplier in memory, so it restarts at 1.0;
    Connectome persists it, clamps it to 0.6..1.8 on load, and applies it once the first compile has
    rebuilt its chunks."""
    multiplier: float = 1.0
    armed: float = 0

    def arm(self, window_tokens: float) -> 'Calibration':
        return self._replace(armed=window_tokens)

    def observe(self, real_tokens: float) -> 'Calibration':
        """real_tokens: input + cache-read + cache-write. A ratio outside 0.6..1.8 is a request the
        compile didn't build: it disarms without learning."""
        if not (real_tokens > 0 and self.armed > 0): return self
        ratio, m = real_tokens / self.armed, self.multiplier
        if not 0.6 <= ratio <= 1.8: return Calibration(m)
        return Calibration(min(1.8, max(0.6, m + 0.2 * (ratio * m - m))))   # JS's operation order
