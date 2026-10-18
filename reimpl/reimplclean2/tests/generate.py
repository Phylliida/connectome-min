"""Generated agent sessions, shared by test_session and test_reference, which runs them through both
this version and reimplclean: nothing here imports either.
"""
WORDS = 'the a memory tool file read wrote error fixed test passes ok so then é 漢字'.split()


def turn(r, t):
    """One agent step: text and maybe a tool call and its result (now and then interrupted, or after
    user messages that came in while it ran; or followed by a stray result), now and then an image or a
    user message."""
    say = lambda n: ' '.join(r.choice(WORDS) for _ in range(n))
    image = {'type': 'image', 'source': {'type': 'base64', 'mediaType': 'image/png', 'data': 'iVBOR' * r.randint(1, 50)}}
    out = [{'participant': 'user', 'content': [{'type': 'text', 'text': say(r.randint(1, 60))}] + [image] * (r.random() < 0.1)}] \
        * (r.random() < 0.2)
    if r.random() < 0.6:
        result = say(int(r.lognormvariate(4.5, 1.2))) if r.random() < 0.9 else [{'type': 'text', 'text': say(5)}] + [image] * r.randint(1, 3)
        call = {'participant': 'assistant', 'content': [{'type': 'text', 'text': say(r.randint(5, 80))},
                                                        {'type': 'tool_use', 'id': f't{t}', 'name': 'read', 'input': {'p': say(2)}}]}
        interjections = [{'participant': 'user', 'content': [{'type': 'text', 'text': say(3)}]}] * (r.randint(1, 7) if r.random() < 0.05 else 0)
        out += [call, *interjections, {'participant': 'user', 'content': [{'type': 'tool_result', 'toolUseId': f't{t}', 'content': result}]}
                ][:1 if r.random() < 0.05 else None]             # an interrupted call has no result
    if r.random() < 0.03:                                          # a stray result, for a call the log doesn't hold
        out += [{'participant': 'user', 'content': [{'type': 'tool_result', 'toolUseId': f'gone{t}', 'content': say(5)}]}]
    else:
        out += [{'participant': 'assistant', 'content': [{'type': 'text', 'text': ' '}] * (r.random() < 0.05)
                 + [{'type': 'text', 'text': say(r.randint(5, 200)), **({'cache_control': {'type': 'ephemeral'}} if r.random() < 0.03 else {})}]}]
    return out


def summarizer(r, refusals, inspect=lambda request: None):
    """A scripted summarizer that writes (now and then reasoning first, or in <thinking> tags), fails,
    replies empty and now and then refuses; it hands each request to `inspect` first."""
    def complete(request):
        inspect(request)
        kind = r.choices(['ok', 'raise', 'empty', 'refusal'], [85, 7, 6, refusals])[0]
        if kind == 'raise':
            raise RuntimeError('overloaded')
        text = {'ok': 'I remember ' + ' '.join(r.choice(WORDS) for _ in range(r.randint(10, 120))), 'empty': ' ', 'refusal': ''}[kind]
        text = r.choices(['', '<thinking>hm</thinking> ', ' <THINKING>a\nb</Thinking>\n', '<thinking>never closed '], [80, 8, 8, 4])[0] + text
        reasoned = [{'type': 'thinking', 'thinking': 'hm', 'signature': 's'}, {'type': 'text', 'text': ' '}] * (r.random() < 0.3)
        return {'stopReason': 'refusal' if kind == 'refusal' else 'end_turn', 'content': reasoned + [{'type': 'text', 'text': text}],
                'usage': {'outputTokens': r.randint(20, 200)}}
    return complete


def settings(r, Config):
    """A session's config and window, the agent's tools, and how often the summarizer refuses."""
    cfg = Config(target_chunk_tokens=r.choice([300, 1500]), recent_window_tokens=r.choice([500, 3000]), merge_threshold=r.choice([2, 3, 6]),
                 summary_target_tokens=200, recall_budget_tokens=r.choice([2000, 20000]), l1_holdback_chunks=r.choice([0, 1, 2]),
                 attempt_limit=r.choice([1, 3]), min_chunk_chars=r.choice([0, 200]), max_live_images=r.choice([0, 2, 6]),
                 max_live_image_bytes=r.choice([0, 600, 20 * 1024 * 1024]), compression_image_bytes=r.choice([0, 500, 12 * 1024 * 1024]))
    return cfg, r.choice([4000, 12000, 40000]), r.choice([None, [{'name': 'read'}]]), r.choice([0, 0, 1])
