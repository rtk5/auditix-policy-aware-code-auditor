"""Allow ``python -m auditix`` as an alternative to the ``auditix`` command."""

from auditix.cli import main

raise SystemExit(main())
