import json
import mmap
import os

def last_jsonl_entry(path):
    for entry in iterate_jsonl_backwards(path): return entry
    return None

def iterate_jsonl_backwards(path):
    with open(path, "rb") as f:
        if os.fstat(f.fileno()).st_size == 0:
            return  # can't mmap an empty file
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            end = len(mm)
            while end > 0:
                nl = mm.rfind(b"\n", 0, end)  # last newline before end, C speed
                line = mm[nl + 1:end]
                if line.strip():
                    yield json.loads(line)
                end = nl  # -1 when no newline remains -> loop ends

def binary_search_jsonl(path, key, value, key_max=float("inf")):
    with open(path, "rb") as f:
        def read_json_at(offset):
            f.seek(offset)
            if offset:
                f.readline()  # skip partial line
            line = f.readline()
            return json.loads(line) if line else None

        def read_key_at(offset):
            obj = read_json_at(offset)
            return obj[key] if obj is not None and key in obj else key_max

        result = read_json_at(
            bisect.bisect_left(range(os.path.getsize(path)), value, key=read_key_at)
        )
        if result is None or result.get(key) != value:
            raise KeyError(f"no json with {key}={value!r} in {path}")
        return result

