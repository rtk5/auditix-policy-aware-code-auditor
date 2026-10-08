from auditix.chunker import parse_code_chunks

SAMPLE = '''
import os

def top():
    return 1

class Service:
    def method(self):
        return 2

async def fetch():
    return 3
'''


def test_extracts_functions_classes_and_methods():
    names = {c.name for c in parse_code_chunks(SAMPLE, "svc.py")}
    assert names == {"top", "Service", "method", "fetch"}


def test_top_level_only_skips_methods():
    names = {c.name for c in parse_code_chunks(SAMPLE, "svc.py", top_level_only=True)}
    assert names == {"top", "Service", "fetch"}


def test_chunk_code_is_dedented_and_keeps_kind():
    chunks = {c.name: c for c in parse_code_chunks(SAMPLE, "svc.py")}
    assert chunks["method"].code.startswith("def method(self):")
    assert chunks["method"].kind == "FunctionDef"
    assert chunks["fetch"].kind == "AsyncFunctionDef"
    assert chunks["method"].path == "svc.py"


def test_syntax_error_returns_whole_file_as_one_chunk():
    broken = "def oops(:\n    pass\n"
    chunks = parse_code_chunks(broken, "broken.py")
    assert len(chunks) == 1
    assert chunks[0].kind == "File"
    assert chunks[0].code == broken


def test_file_without_definitions_is_kept_whole():
    chunks = parse_code_chunks("X = 1\nY = 2\n", "consts.py")
    assert len(chunks) == 1 and chunks[0].name == "consts.py"


def test_empty_file_produces_no_chunks():
    assert parse_code_chunks("   \n", "empty.py") == []
