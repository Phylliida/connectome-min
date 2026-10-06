"""The compression boundary: one request builder, one model client, one gate.

    compress = Summarizer(MockModel())        # offline and deterministic: the demo's path
    compress = Summarizer(HttpModel())        # one POST; CONNECTOME_* env (see README.md)

`minisystem.step()` sees only `(text | None, reason | None)`, every reason a key of `config.ACTIONS`."""
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path

from config import (DEFAULT_BASE_URL, DEFAULT_MODEL,
                    DEFAULT_TIMEOUT, DIGEST_WORDS, BOUND_PER_MESSAGE, BOUND_RESERVE, CALIBRATION_ALPHA, CALIBRATION_BAND,
                    CONTEXT_BUDGET, SUMMARY_TARGET_TOKENS)   # config.py is the single source

PROMPTS = Path(__file__).resolve().parent / "prompts"

TARGET_TOKENS = SUMMARY_TARGET_TOKENS  # 200: production's 2000, scaled with the budget (config.py)
OUTPUT_FLOOR_TOKENS = 1600             # production's 16000, scaled /10 -- the term that binds here
MAX_OUTPUT_TOKENS = max(OUTPUT_FLOOR_TOKENS, int(TARGET_TOKENS * 1.5))   # production's expression
# Provider stop reasons -> production's vocabulary; anything else passes through as truncated.
STOP_ALIASES = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use",
                "function_call": "tool_use", "content_filter": "refusal"}


def prompt_text(name, prompts=PROMPTS, **fields):
    """Read a prompt file and fill its placeholders: `#` comment lines document them for us, not the model."""
    body = "\n".join(line for line in (prompts / name).read_text().splitlines()
                     if not line.lstrip().startswith("#")).strip()
    return body.format(**fields) if fields else body


def build_request(kind, target, source, *, prefix="", reading_tokens=None, source_only=False,
                  model, target_tokens=TARGET_TOKENS, max_tokens=MAX_OUTPUT_TOKENS, prompts=PROMPTS):
    """THE one place a request is assembled, for L1 and merge alike: production's six sections for an L1 and
    its three parts for a merge. `source_only` selects the refusal rung, sections 4-6 of the L1 builder by
    construction. README.md `## summarizer.py`."""
    level = int(target.split(":")[0][1:])
    if kind == "L1":
        fields = {"target_tokens": target_tokens}
        if reading_tokens:                            # section 6 is doc-aware: one shard of a
            fields["total_tokens"] = reading_tokens   # larger message is a READING, not an event
        instruction = prompt_text("reading_l1.txt" if reading_tokens else "l1_chunk.txt",
                                  prompts, **fields)
        head = "" if source_only else prefix          # sections 1-3 skipped structurally
        user = "\n\n".join(part for part in (head, prompt_text("marker.txt", prompts), source,
                                             instruction) if part)
        scope = "source-only" if source_only else "six-section"
    else:
        shown = max(0, level - 2)
        if reading_tokens is None:
            seen = ("the slices of recent experience above (raw conversation)" if not shown
                    else f"the L{shown} memories above")
            instruction = prompt_text("merge.txt", prompts, target_level=level,
                                      target_tokens=target_tokens, seen_description=seen)
        else:
            seen = ("the portions of text you read above (raw passages from a larger piece)"
                    if not shown else
                    f"your earlier L{shown} reflections above on portions you read")
            instruction = prompt_text("reading_merge.txt", prompts, target_level=level,
                                      target_tokens=target_tokens, seen_description=seen,
                                      total_tokens=reading_tokens)
        user = "\n\n".join(part for part in (prefix, source, instruction) if part)
        scope = "prefix+one-level-deeper"
    # tools=None is our deviation: production declares the agent's tools (:5631-5640). system.txt
    # is our stand-in for the host's live identity prompt, which this artifact has no analogue of.
    return {"kind": kind, "target": target, "model": model,
            "system": prompt_text("system.txt", prompts), "user": user, "tools": None,
            "scope": scope, "max_tokens": max_tokens, "source": source, "prefix": prefix}


def wire_payload(request):
    """The JSON object a client dispatches: one system turn, one user turn, and no `source`/`scope`."""
    return {"model": request["model"], "max_tokens": request["max_tokens"], "stream": False,
            "messages": [{"role": "system", "content": request["system"]},
                         {"role": "user", "content": request["user"]}]}


def wire_body(request):
    """The exact bytes of that payload, UTF-8 and unescaped -- one serializer for client and bound."""
    return json.dumps(wire_payload(request), ensure_ascii=False).encode("utf-8")


def request_input_bound(request):
    """Admission's fail-closed bound, production's term for term: the COMPLETE dispatched request as UTF-8
    bytes, plus BOUND_RESERVE and one BOUND_PER_MESSAGE per message. Bytes, not tokens, so no tokenizer is
    needed and it cannot under-count. The constants are `config.py`'s."""
    payload = wire_payload(request)
    return len(wire_body(request)) + BOUND_RESERVE + BOUND_PER_MESSAGE * len(payload["messages"])


def request_input_tokens(request):
    """The request's input as an ESTIMATE, the wire payload's UTF-8 bytes -- metadata, not accounting,
    as production labels its own. One unit for both numbers: the bound adds only its reserves."""
    return len(wire_body(request))


@dataclass(frozen=True)
class Reply:
    """One normalized reply, before the gate judges it. `error` is a failure that predates any disposition."""
    text: str = ""
    stop: str | None = None
    tool_call: bool = False
    error: str | None = None
    retryable: bool = True
    detail: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_tokens: int = 0                 # metadata, never accounting


def gate(reply):
    """The terminal-disposition gate: a complete stop with usable text and no tool call, or a failure the policy
    routes. The ORDER of these tests is load-bearing; every reason is a key of `config.ACTIONS`."""
    if reply.error:                                  # transport or admission: already a reason
        return None, reply.error
    if reply.stop == "refusal":                      # outranks its own text (:2784-2786)
        return None, "refusal"
    if reply.tool_call or reply.stop == "tool_use":  # before empty: our retry line keys on it
        return None, "tool_call"
    if not reply.text.strip():
        return None, "empty"
    if reply.stop != "end_turn":
        return None, "truncated"
    return reply.text, None


# reason -> (text, or None for a prefix of the source | stop | tool call | error). `overflow` is the
# admission decision, not a provider disposition, scripted so the demo's reshape rung runs offline.
MOCK_REPLIES = {
    "refusal": ("I can't write a memory of this material.", "refusal", False, None),
    "empty": (" \n ", "end_turn", False, None),
    "truncated": (None, "max_tokens", False, None),
    "tool_call": ("", "tool_use", True, None),
    "provider_error": ("", None, False, "provider_error"),
    "overflow": ("", None, False, "overflow"),
}


def mock_reply(what, request):
    """One canned reply per class, `None` being success: a digest of the source, not a memory."""
    text, stop, tool, error = ((" ".join(request["source"].split()[:DIGEST_WORDS]), "end_turn",
                                False, None) if what is None
                               else MOCK_REPLIES[what])   # KeyError: not a policy reason
    if text is None:                                      # truncated: memory cut off mid-thought
        text = " ".join(request["source"].split()[:6])
    answered = what not in ("provider_error", "overflow")
    return Reply(text=text, stop=stop, tool_call=tool, error=error,
                 retryable=error in (None, "provider_error"), detail=f"mock: {error}" if error else "",
                 input_tokens=(request_input_tokens(request)  # the request's own byte estimate,
                               if answered else 0),          # so the demo's calibration stays 1.0
                 output_tokens=max(1, len(text) // 4) if answered and text.strip() else 0)


class MockModel:
    """Deterministic offline stand-in: `plan` maps a work id to a queue of outcomes, one per call."""
    def __init__(self, plan=None, model="mock-1"):
        self.plan, self.model = {k: list(v) for k, v in (plan or {}).items()}, model

    def complete(self, request):
        queue = self.plan.get(request["target"].split(":")[-1].split(".")[0])
        what = queue.pop(0) if queue else None
        return what if isinstance(what, Reply) else mock_reply(what, request)


def urlopen_transport(url, headers, body, timeout):
    """The client's dependency surface: an HTTPError is a normal outcome; timeouts propagate."""
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as rejected:
        return rejected.code, rejected.read().decode("utf-8", "replace")


def _count(value):
    return value if isinstance(value, int) and value >= 0 else 0


class HttpModel:
    """One POST to an OpenAI-compatible endpoint, stdlib only. Every outcome maps into the failure vocabulary
    the rest of the system already routes: README.md's table, config.py's "model boundary"."""
    def __init__(self, model=None, base_url=None, api_key=None, timeout=None,
                 transport=urlopen_transport):
        env = os.environ.get
        self.model = model or env("CONNECTOME_MODEL") or DEFAULT_MODEL
        self.base_url = base_url or env("CONNECTOME_BASE_URL") or DEFAULT_BASE_URL
        self.api_key = api_key if api_key is not None else env("CONNECTOME_API_KEY")
        self.timeout, self.transport = float(timeout or env("CONNECTOME_TIMEOUT")
                                             or DEFAULT_TIMEOUT), transport

    def complete(self, request):
        headers = {"content-type": "application/json", "accept": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        url, body = f"{self.base_url.rstrip('/')}/chat/completions", wire_body(request)
        try:
            status, text = self.transport(url, headers, body, self.timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as fault:
            return Reply(error="provider_error", retryable=True, detail=f"transport: {fault}")
        if 200 <= status < 300:
            return self.parse(text)
        # 429/408/5xx are transient; any other 4xx is deterministic and feeds bounded attempts.
        try:
            error = json.loads(text).get("error") or {}
            detail = f"{error.get('type', '')} {error.get('message', '')}".strip()
        except (AttributeError, ValueError):
            detail = " ".join(text.split())
        return Reply(error="provider_error", retryable=status in (408, 429) or status >= 500,
                     detail=f"http {status}: {detail}"[:240])

    def parse(self, text):
        """A malformed 2xx body is a provider_error, as production's `malformed_response` is."""
        try:
            body = json.loads(text)
            usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
            choice = (body.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            finish = choice.get("finish_reason")
        except (AttributeError, IndexError, TypeError, ValueError) as fault:
            return Reply(error="provider_error", retryable=True, detail=f"malformed: {fault}")
        if finish is None:                            # production: a missing stop is malformed
            return Reply(error="provider_error", retryable=True, detail="malformed: no finish_reason",
                         input_tokens=_count(usage.get("prompt_tokens")))
        content = message.get("content")
        if isinstance(content, list):                 # some gateways answer with content parts
            content = "\n".join(part.get("text", "") for part in content
                                if isinstance(part, dict) and part.get("type") in (None, "text"))
        calls = message.get("tool_calls") or message.get("function_call")
        return Reply(text=content if isinstance(content, str) else "",
                     stop=STOP_ALIASES.get(finish, finish), tool_call=bool(calls),
                     input_tokens=_count(usage.get("prompt_tokens")),
                     output_tokens=_count(usage.get("completion_tokens")))


class Summarizer:
    """Assemble -> admit -> complete -> gate -> account, and the only thing minisystem imports."""
    def __init__(self, model, *, target_tokens=TARGET_TOKENS, prompts=PROMPTS,
                 budget=CONTEXT_BUDGET):
        self.model, self.target_tokens, self.prompts, self.budget = (
            model, target_tokens, prompts, budget)
        self.model_name = getattr(model, "model", "unset")
        self.calls, self.input_tokens, self.output_tokens = 0, 0, 0
        self.reasons, self.faults = {}, {"retryable": 0, "non_retryable": 0}
        self.estimated_tokens, self.calibration = 0, 1.0
        self.last_request = self.last_reply = None

    def request(self, kind, target, source, prefix="", reading_tokens=None, source_only=False):
        """The request alone, built and not dispatched: the seam a test asserts the assembly on."""
        self.last_request = build_request(kind, target, source, prefix=prefix,
                                          reading_tokens=reading_tokens, source_only=source_only,
                                          model=self.model_name, target_tokens=self.target_tokens,
                                          prompts=self.prompts)
        return self.last_request

    def estimate(self, request):
        """The byte estimate in calibrated units: calibration never touches `request_input_bound`."""
        return round(request_input_tokens(request) * self.calibration)

    def observe(self, reply, estimated):
        """Closed-loop calibration (production's `reportRealInputTokens`): one sample per dispatched call, paired
        with ITS OWN request's estimate -- structural here, so production's arm-once-per-compile trap cannot
        arise. An out-of-band ratio is a mismatch rather than evidence: drop it, do not learn."""
        if reply.input_tokens <= 0 or estimated <= 0:
            return
        low, high = CALIBRATION_BAND
        ratio = reply.input_tokens / estimated
        if not low <= ratio <= high:
            return
        observed = ratio * self.calibration         # back out the multiplier already applied
        step = CALIBRATION_ALPHA * (observed - self.calibration)
        self.calibration = min(high, max(low, self.calibration + step))

    def admit(self, request):
        """Pre-send admission: the fail-closed BYTE bound plus the request's own output reserve against the budget.
        Never the estimate -- one that is too low must not admit a request the provider would refuse -- and the
        refusal's detail carries both numbers. README.md `## summarizer.py`."""
        bound = request_input_bound(request)
        used = bound + request["max_tokens"]
        if used <= self.budget:
            return None
        return Reply(error="overflow", retryable=False,
                     detail=f"admission: bound {bound} + {request['max_tokens']} reserved = "
                            f"{used} > budget {self.budget} (estimate {self.estimate(request)})")

    def __call__(self, kind, target, source, prefix="", reading_tokens=None, source_only=False):
        request = self.request(kind, target, source, prefix, reading_tokens, source_only)
        estimated = self.estimate(request)       # metadata for the receipt; nothing gates on it
        reply = replace(self.admit(request) or self.model.complete(request),
                        estimated_tokens=estimated)
        self.calls += 1                          # an attempt was spent, dispatched or not
        self.estimated_tokens += estimated
        self.input_tokens += reply.input_tokens  # a rejected request bills nothing: it never left
        self.output_tokens += reply.output_tokens
        self.last_reply = reply
        self.observe(reply, estimated)
        text, reason = gate(reply)
        if reason:
            self.reasons[reason] = self.reasons.get(reason, 0) + 1
            if reason == "provider_error":
                self.faults["retryable" if reply.retryable else "non_retryable"] += 1
        return text, reason

    def usage(self):
        return dict(calls=self.calls, input_tokens=self.input_tokens,
                    output_tokens=self.output_tokens, estimated_tokens=self.estimated_tokens,
                    calibration=round(self.calibration, 3), reasons=dict(self.reasons),
                    faults=dict(self.faults))
