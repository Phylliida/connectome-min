import bisect
import json
import mmap
import os

# ways of parsing append only jsonl files that don't require loading them all into memory

class JsonlFile(object):
    def __init__(self, path, class_factory=dict):
        repair_jsonl_tail(path)  # repair in case of issues
        open(path, "ab").close()  # create if absent so all readers work
        self.path = path
        self.class_factory = lambda data: None if data is None else class_factory(data)

    def append(self, value):
        append_jsonl_entry(self.path, value)

    def get(self, key, value):
        offset, entry = binary_search_jsonl(self.path, key, value)
        return self.class_factory(entry)

    def first(self):
        return self.class_factory(first_jsonl_entry(self.path))

    def last(self):
        return self.class_factory(last_jsonl_entry(self.path))

    def __iter__(self):
        for data in iterate_jsonl_forwards(self.path):
            yield self.class_factory(data)

    # iterates returning the given thing first and then stuff after that
    def iter_at(self, key, value):
        offset, entry = binary_search_jsonl(self.path, key, value)
        for data in iterate_jsonl_forwards(self.path, offset=offset):
            yield self.class_factory(data)

    def backwards(self):
        for data in iterate_jsonl_backwards(self.path):
            yield self.class_factory(data)


def first_jsonl_entry(path):
    with open(path, "rb") as f:
        try: return json.loads(f.readline())
        except json.JSONDecodeError: return None


def iterate_jsonl_forwards(path, offset=0):
    with open(path, "rb") as f:
        if offset: f.seek(offset)
        for line in f:
            if line.strip(): yield json.loads(line)


def repair_jsonl_tail(path):
    if not os.path.exists(path):  return  # nothing to repair
    with open(path, "r+b") as f:
        size = os.fstat(f.fileno()).st_size
        if size == 0:  return
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            end = size - 1 if mm[-1:] == b"\n" else size
            nl = mm.rfind(b"\n", 0, end)  # newline before the final line
            final_data = mm[nl + 1:end]
        try: json.loads(final_data)
        except json.JSONDecodeError: f.truncate(nl + 1)  # drop the broken line; nl=-1 empties the file
        else:
            if end == size:  # valid final line, missing its newline
                f.seek(0, os.SEEK_END)
                f.write(b"\n")


def append_jsonl_entry(path, entry):
    repair_jsonl_tail(path)  # repair first, otherwise we may just make it worse
    with open(path, "ab") as f:  # O_APPEND: writes always land at EOF
        f.write(json.dumps(entry).encode() + b"\n")


def last_jsonl_entry(path):
    return next(iterate_jsonl_backwards(path), None)


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
    """Returns (offset_of_line_start, entry). Raises KeyError if absent."""
    with open(path, "rb") as f:
        def read_json_at(offset):
            f.seek(offset)
            if offset:
                f.readline()  # skip partial line
            offset = f.tell()
            line = f.readline()
            return (offset, json.loads(line) if line else None)

        def read_key_at(offset):
            _, obj = read_json_at(offset)
            return obj[key] if obj is not None and key in obj else key_max

        offset, result = read_json_at(
            bisect.bisect_left(range(os.path.getsize(path)), value, key=read_key_at)
        )
        if result is None or result.get(key) != value:
            raise KeyError(f"no json with {key}={value!r} in {path}")
        return offset, result
