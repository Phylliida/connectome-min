
# these two helpers could be made more robust but it is not worth the complexity, they're good enough'
def chars_in_fenced_code_blocks(body: str) -> int:
    return sum(len(f) for f in body.split("```")[1::2])

def chars_in_bare_link_lines(body: str) -> int:
    return sum(len(l.strip()) for l in body.split("\n")
               if l.strip().startswith(("http://", "https://")))

def text_to_tokens(text):
    // TODO : use a tokenizer
    return len(text or "") // CHARS_PER_TOKEN

def message_data_to_tokens(data: MessageData):
    return IMAGE_TOKENS if data.kind == MessageDataKind.IMAGE else text_to_tokens(d.data)

def message_data_to_rederivable_tokens(data: MessageData):
    if data.kind == MessageDataKind.TEXT: # code/data that exists elsewhere (probably)
        return chars_in_fenced_code_blocks(body) + chars_in_bare_link_lines(body)
    else: # tool payloads, images, files
        return message_data_to_tokens(data)

def priced_tokens(message: Message) -> int:
    return sum(map(message_data_to_tokens, messsage.data))

def salience(message: Message) -> int:
    total = priced_tokens(message)
    external = sum(map(message_data_to_rederivable_tokens, message.data)) # stuff that we can rederive so it's less important'
    1.0 if total == 0 else max(0.2, 1 - 0.8 * (external / total))

def estimate_tokens(msg) -> int:
    tokens = 0
    for block in msg.content:                 # top-level blocks only
        tokens += estimate_block_tokens(block)
    return tokens


def estimate_block_tokens(block) -> int:
    return round(estimate_block_tokens_raw(block) * token_calibration)   # default 1.0


def estimate_block_tokens_raw(block):
    if block in _rawBlockTokens: return cached           # blocks are immutable; edits make new objects
    match block.type:
        "text"              -> text_estimator(block.text)
        "thinking"          -> block.tokenEstimate                    if stamped
                            or 600 (HIDDEN_THINKING_TOKENS_DEFAULT)   if signed but empty text
                            else text_estimator(block.thinking)
        "redacted_thinking" -> block.tokenEstimate                    if stamped
                            or round(len(block.data) / 6)             # ENCRYPTED_CARRIER_CHARS_PER_TOKEN
                            else 600
        "tool_use"          -> json_estimator(JSON.stringify(block.input)) + 20   # name + id overhead
        "tool_result"       -> 0                                      if falsy
                            or json_estimator(block.content)          if string
                            or sum(estimate_block_tokens_raw(b) for b in block.content)   # nested, recursive
        "image"             -> block.tokenEstimate ?? 1600             # ~1568px image
        "document"|"audio"|"video" -> 1000
        _                   -> 0                                      # unknown types are FREE
