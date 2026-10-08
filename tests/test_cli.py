from pathlib import Path

from auditix.cli import _settings_from_args, build_parser


def test_audit_local_flags_override_settings(monkeypatch):
    monkeypatch.delenv("AUDITIX_EXPLAIN_FAST_PATH", raising=False)
    args = build_parser().parse_args([
        "audit-local", "some/dir", "-o", "out", "--no-classifier", "--no-explain-fast-path",
    ])
    settings = _settings_from_args(args)
    assert settings.output_dir == Path("out")
    assert settings.use_classifier is False
    assert settings.explain_fast_path is False


def test_env_var_controls_fast_path(monkeypatch):
    from auditix.config import Settings

    monkeypatch.setenv("AUDITIX_EXPLAIN_FAST_PATH", "0")
    assert Settings.from_env().explain_fast_path is False


def test_subcommands_are_registered():
    parser = build_parser()
    for command in ["audit-local", "audit-repo", "build-index", "train"]:
        args = parser.parse_args([command] + (["x"] if command in {"audit-local", "audit-repo"} else []))
        assert args.command == command


def test_load_dotenv_file_does_not_override_shell(tmp_path, monkeypatch):
    from auditix.config import load_dotenv_file

    env_file = tmp_path / ".env"
    env_file.write_text('# comment\nAUDITIX_TEST_NEW="from-file"\nAUDITIX_TEST_SET=from-file\n', encoding="utf-8")
    monkeypatch.delenv("AUDITIX_TEST_NEW", raising=False)
    monkeypatch.setenv("AUDITIX_TEST_SET", "from-shell")
    assert load_dotenv_file(env_file) is True
    import os

    assert os.environ["AUDITIX_TEST_NEW"] == "from-file"
    assert os.environ["AUDITIX_TEST_SET"] == "from-shell"
    monkeypatch.delenv("AUDITIX_TEST_NEW", raising=False)


def test_load_dotenv_file_missing_is_noop(tmp_path):
    from auditix.config import load_dotenv_file

    assert load_dotenv_file(tmp_path / "missing.env") is False
