
import math
import json

def estimate_text_tokens(text: str):
    return math.ceil(len(text.encode("utf-16-le")) / 4.)

def estimate_block_tokens(block: Block):
    estimate = 0
    if block.text: estimate += estimate_text_tokens(block.text) # thinking, text, sometimes tools
    if block.data: estimate += estimate_text_tokensjson.dumps(block.data, separators=(',', ':'), ensure_ascii=False) # tool calls and results
    if block.kind == BlockKind.IMAGE: estimate += IMAGE_TOKENS
    if block.kind == BlockKind.AUDIO: estimate += AUDIO_TOKENS
    if block.kind == BlockKind.VIDEO: estimate += VIDEO_TOKENS
    if block.kind == BlockKind.DOCUMENT: estimate += DOCUMENT_TOKENS
    return estimate


def estimate_message_tokens(message: Message):
    return sum(map(estimate_block_tokens, message.blocks))
