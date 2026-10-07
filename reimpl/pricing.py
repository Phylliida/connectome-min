
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
