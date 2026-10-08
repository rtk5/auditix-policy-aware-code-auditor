"""AST-based chunking: split a Python file into functions and classes.

Chunks are *semantic* units (one function or class each) instead of arbitrary
line windows, so each chunk can be judged against policies on its own.
"""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass

_DEF_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


@dataclass(frozen=True)
class CodeChunk:
    name: str
    kind: str   # "FunctionDef", "AsyncFunctionDef", "ClassDef" or "File"
    code: str   # dedented source text of the chunk
    path: str   # file the chunk came from (relative path in the report)

    def to_dict(self) -> dict:
        return {"name": self.name, "type": self.kind, "code": self.code, "path": self.path}


def parse_code_chunks(
    source_code: str,
    filename: str = "code.py",
    *,
    top_level_only: bool = False,
) -> list[CodeChunk]:
    """Extract functions and classes from ``source_code``.

    Args:
        source_code: the full text of a Python file.
        filename: used in the chunk ``path`` and as the name of whole-file chunks.
        top_level_only: if False (default, matches ``complete_code.ipynb``) every
            function/class at any depth becomes a chunk, so a method is audited
            both on its own and inside its class. If True, only module-level
            definitions are returned, which avoids counting code twice.

    If the file has a syntax error, or contains no definitions, the whole file is
    returned as a single chunk so that nothing is silently skipped.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return [CodeChunk(name=filename, kind="File", code=source_code, path=filename)]

    nodes = tree.body if top_level_only else ast.walk(tree)
    chunks: list[CodeChunk] = []
    for node in nodes:
        if isinstance(node, _DEF_NODES):
            segment = ast.get_source_segment(source_code, node)
            if segment:
                chunks.append(
                    CodeChunk(
                        name=node.name,
                        kind=type(node).__name__,
                        code=textwrap.dedent(segment),
                        path=filename,
                    )
                )

    if not chunks and source_code.strip():
        chunks.append(CodeChunk(name=filename, kind="File", code=source_code, path=filename))
    return chunks
