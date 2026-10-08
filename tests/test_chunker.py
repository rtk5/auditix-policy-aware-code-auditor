from auditix.chunker import parse_code_chunks

# Sample module with a top-level function, a class with a method, and an async function.
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


# Every definition is found, including the method nested in the class.
def test_extracts_functions_classes_and_methods():
    names = {c.name for c in parse_code_chunks(SAMPLE, "svc.py")}
    assert names == {"top", "Service", "method", "fetch"}


# top_level_only=True drops the method and keeps the class and functions.
def test_top_level_only_skips_methods():
    names = {c.name for c in parse_code_chunks(SAMPLE, "svc.py", top_level_only=True)}
    assert names == {"top", "Service", "fetch"}


# Nested code is returned without its indentation and records the AST node type.
def test_chunk_code_is_dedented_and_keeps_kind():
    chunks = {c.name: c for c in parse_code_chunks(SAMPLE, "svc.py")}
    assert chunks["method"].code.startswith("def method(self):")
    assert chunks["method"].kind == "FunctionDef"
    assert chunks["fetch"].kind == "AsyncFunctionDef"
    assert chunks["method"].path == "svc.py"


# Broken Python falls back to a single chunk so nothing is silently skipped.
def test_syntax_error_returns_whole_file_as_one_chunk():
    broken = "def oops(:\n    pass\n"
    chunks = parse_code_chunks(broken, "broken.py")
    assert len(chunks) == 1
    assert chunks[0].kind == "File"
    assert chunks[0].code == broken


# A file with no functions or classes is still audited as one chunk.
def test_file_without_definitions_is_kept_whole():
    chunks = parse_code_chunks("X = 1\nY = 2\n", "consts.py")
    assert len(chunks) == 1 and chunks[0].name == "consts.py"


# Blank input produces no chunks, so there is nothing to audit.
def test_empty_file_produces_no_chunks():
    assert parse_code_chunks("   \n", "empty.py") == []
