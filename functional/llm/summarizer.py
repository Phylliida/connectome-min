"""The compression boundary: one request builder, one model client, one gate -- functional port
of ../../llm/summarizer.py.

    boundary = summarizer(mock_model())         # offline and deterministic: the demo's path
    boundary = summarizer(http_model())         # one POST; CONNECTOME_* env (see README.md)
    (text, reason), boundary = complete(boundary, "L1", "L1:c3", src)

`minisystem.step()` sees only `(text | None, reason | None)`, every reason a key of
`config.ACTIONS`.

Where the original kept state ON objects (the mock's queues popped in place, the Summarizer's
counters and calibration written per call), the port makes state a VALUE:

  * A model is `Model(kind, name, complete, state)` with `complete: request -> (Reply, Model)`
    -- a Mealy machine. `mock_model`'s scripted queues live in `state` and the NEXT model
    carries what is left; `http_model`'s `state` is its resolved `HttpConfig`, constant, so it
    returns itself.
  * `Summarizer` is a frozen dataclass; `complete` (the original's `__call__`) returns the next
    one alongside the gated outcome. Counters, reasons, faults, calibration, last_request and
    last_reply are all fields of the value, updated by `dataclasses.replace`.

The transport itself (`urlopen_transport`) does network I/O -- an effect, but not state
mutation -- so no function here needs the folder's `_` prefix either.
"""
import json
import os
import urllib.error
import urllib.request
from collections import namedtuple
from dataclasses import dataclass, field, replace
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


# A model is a Mealy machine: `complete` answers one request and returns the model that answers
# the NEXT one. `kind` is the constructor's name ("MockModel" / "HttpModel"), kept for the
# demo's narrative line; `name` is the provider-side model id the request carries; `state` is the
# machine's inspectable remainder -- the mock's surviving queues, the HTTP client's resolved
# HttpConfig -- so a test asserts on the value instead of reaching into a closure.
Model = namedtuple("Model", "kind name complete state")


def mock_model(plan=None, name="mock-1"):
    """Deterministic offline stand-in: `plan` maps a work id to a queue of outcomes, one per
    call. The original POPPED the queue off the instance; here the remaining queue is bound
    into the returned model, so answering a request consumes nothing anyone else can see."""
    return mock_with({k: tuple(v) for k, v in (plan or {}).items()}, name)


def mock_with(queues, name):
    def complete(request):
        key = request["target"].split(":")[-1].split(".")[0]
        queue = queues.get(key, ())
        what = queue[0] if queue else None
        reply = what if isinstance(what, Reply) else mock_reply(what, request)
        return reply, mock_with({**queues, key: queue[1:]}, name)
    return Model("MockModel", name, complete, queues)


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


@dataclass(frozen=True)
class HttpConfig:
    """Everything an HTTP call needs, resolved once (env defaults included) and never changed."""
    model: str
    base_url: str
    api_key: str | None
    timeout: float
    transport: object


def http_model(model=None, base_url=None, api_key=None, timeout=None,
               transport=urlopen_transport):
    """One POST to an OpenAI-compatible endpoint, stdlib only. Every outcome maps into the failure vocabulary
    the rest of the system already routes: README.md's table, config.py's "model boundary". Stateless, so
    `complete` returns the same model -- a constant Mealy machine."""
    env = os.environ.get
    cfg = HttpConfig(model=model or env("CONNECTOME_MODEL") or DEFAULT_MODEL,
                     base_url=base_url or env("CONNECTOME_BASE_URL") or DEFAULT_BASE_URL,
                     api_key=api_key if api_key is not None else env("CONNECTOME_API_KEY"),
                     timeout=float(timeout or env("CONNECTOME_TIMEOUT") or DEFAULT_TIMEOUT),
                     transport=transport)

    def complete(request):
        return http_complete(cfg, request), machine

    machine = Model("HttpModel", cfg.model, complete, cfg)
    return machine


def http_complete(cfg, request):
    """The single POST, config in and one normalized Reply out."""
    headers = {"content-type": "application/json", "accept": "application/json"}
    if cfg.api_key:
        headers["authorization"] = f"Bearer {cfg.api_key}"
    url, body = f"{cfg.base_url.rstrip('/')}/chat/completions", wire_body(request)
    try:
        status, text = cfg.transport(url, headers, body, cfg.timeout)
    except (urllib.error.URLError, TimeoutError, OSError) as fault:
        return Reply(error="provider_error", retryable=True, detail=f"transport: {fault}")
    if 200 <= status < 300:
        return parse_reply(text)
    # 429/408/5xx are transient; any other 4xx is deterministic and feeds bounded attempts.
    try:
        error = json.loads(text).get("error") or {}
        detail = f"{error.get('type', '')} {error.get('message', '')}".strip()
    except (AttributeError, ValueError):
        detail = " ".join(text.split())
    return Reply(error="provider_error", retryable=status in (408, 429) or status >= 500,
                 detail=f"http {status}: {detail}"[:240])


def parse_reply(text):
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


@dataclass(frozen=True)
class Summarizer:
    """Assemble -> admit -> complete -> gate -> account, as a VALUE: the original's per-call
    field writes are `replace` steps here, and `complete` returns the next Summarizer with the
    gated outcome. The only thing minisystem imports."""
    model: Model
    target_tokens: int = TARGET_TOKENS
    prompts: object = PROMPTS
    budget: int = CONTEXT_BUDGET
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasons: dict = field(default_factory=dict)
    faults: dict = field(default_factory=lambda: {"retryable": 0, "non_retryable": 0})
    estimated_tokens: int = 0
    calibration: float = 1.0
    last_request: object = None
    last_reply: object = None


def summarizer(model, *, target_tokens=TARGET_TOKENS, prompts=PROMPTS, budget=CONTEXT_BUDGET):
    """The constructor, as a function: a fresh Summarizer value around a Model."""
    return Summarizer(model=model, target_tokens=target_tokens, prompts=prompts, budget=budget)


def request(s, kind, target, source, prefix="", reading_tokens=None, source_only=False):
    """The request alone, built and not dispatched -- the seam a test asserts the assembly on.
    Returns (request, s') where s' records it as `last_request`, as the original's field write did."""
    req = build_request(kind, target, source, prefix=prefix, reading_tokens=reading_tokens,
                        source_only=source_only, model=s.model.name,
                        target_tokens=s.target_tokens, prompts=s.prompts)
    return req, replace(s, last_request=req)


def estimate(s, request):
    """The byte estimate in calibrated units: calibration never touches `request_input_bound`."""
    return round(request_input_tokens(request) * s.calibration)


def observe(s, reply, estimated):
    """Closed-loop calibration (production's `reportRealInputTokens`): one sample per dispatched call, paired
    with ITS OWN request's estimate -- structural here, so production's arm-once-per-compile trap cannot
    arise. An out-of-band ratio is a mismatch rather than evidence: drop it, do not learn.
    Summarizer -> Summarizer; the calibration field is the only thing that can move."""
    if reply.input_tokens <= 0 or estimated <= 0:
        return s
    low, high = CALIBRATION_BAND
    ratio = reply.input_tokens / estimated
    if not low <= ratio <= high:
        return s
    observed = ratio * s.calibration              # back out the multiplier already applied
    step = CALIBRATION_ALPHA * (observed - s.calibration)
    return replace(s, calibration=min(high, max(low, s.calibration + step)))


def admit(s, request):
    """Pre-send admission: the fail-closed BYTE bound plus the request's own output reserve against the budget.
    Never the estimate -- one that is too low must not admit a request the provider would refuse -- and the
    refusal's detail carries both numbers. README.md `## summarizer.py`."""
    bound = request_input_bound(request)
    used = bound + request["max_tokens"]
    if used <= s.budget:
        return None
    return Reply(error="overflow", retryable=False,
                 detail=f"admission: bound {bound} + {request['max_tokens']} reserved = "
                        f"{used} > budget {s.budget} (estimate {estimate(s, request)})")


def complete(s, kind, target, source, prefix="", reading_tokens=None, source_only=False):
    """One call through the boundary, Summarizer -> ((text | None, reason | None), Summarizer).
    The original's `__call__`: the same five steps, with the counters, the model's next state
    and the calibration threaded through the returned value instead of written to fields."""
    req, s = request(s, kind, target, source, prefix, reading_tokens, source_only)
    estimated = estimate(s, req)               # metadata for the receipt; nothing gates on it
    refused = admit(s, req)
    if refused is not None:
        reply = refused
    else:
        reply, model = s.model.complete(req)
        s = replace(s, model=model)
    reply = replace(reply, estimated_tokens=estimated)
    s = observe(replace(s,
                        calls=s.calls + 1,                       # an attempt was spent,
                        estimated_tokens=s.estimated_tokens + estimated,   # dispatched or not
                        input_tokens=s.input_tokens + reply.input_tokens,  # a rejected request
                        output_tokens=s.output_tokens + reply.output_tokens,  # bills nothing:
                        last_reply=reply),                                   # it never left
                reply, estimated)
    text, reason = gate(reply)
    if reason:
        s = replace(s, reasons={**s.reasons, reason: s.reasons.get(reason, 0) + 1})
        if reason == "provider_error":
            which = "retryable" if reply.retryable else "non_retryable"
            s = replace(s, faults={**s.faults, which: s.faults[which] + 1})
    return (text, reason), s


def usage(s):
    return dict(calls=s.calls, input_tokens=s.input_tokens,
                output_tokens=s.output_tokens, estimated_tokens=s.estimated_tokens,
                calibration=round(s.calibration, 3), reasons=dict(s.reasons),
                faults=dict(s.faults))
