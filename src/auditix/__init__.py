"""Policy-aware code compliance auditor.

Public entry points:
    auditix.pipeline.run_audit_local / run_audit_repo  - run a full audit
    auditix.policies.load_policies                     - read policy statements
    auditix.chunker.parse_code_chunks                  - AST chunking
    auditix.hybrid_engine.audit_chunk_hybrid           - routing + verdict
"""

__version__ = "0.2.0"
