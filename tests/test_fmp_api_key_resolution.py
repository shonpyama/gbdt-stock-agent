from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gbdt_agent.fmp_client as fmp_client


def test_resolve_fmp_api_key_prefers_env(monkeypatch, tmp_path: Path) -> None:
    key_file = tmp_path / ".env_fmp"
    key_file.write_text("FILE_KEY")
    monkeypatch.setattr(fmp_client, "DEFAULT_API_KEY_FILES", [key_file])
    monkeypatch.setenv("FMP_API_KEY", "ENV_KEY")
    assert fmp_client.resolve_fmp_api_key() == "ENV_KEY"


def test_resolve_fmp_api_key_reads_plain_key_file(monkeypatch, tmp_path: Path) -> None:
    key_file = tmp_path / ".env_fmp"
    key_file.write_text("PLAIN_FILE_KEY\n")
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.delenv("FMP_API_KEY_FILE", raising=False)
    monkeypatch.setattr(fmp_client, "DEFAULT_API_KEY_FILES", [key_file])
    assert fmp_client.resolve_fmp_api_key() == "PLAIN_FILE_KEY"


def test_resolve_fmp_api_key_reads_export_format(monkeypatch, tmp_path: Path) -> None:
    key_file = tmp_path / "key.env"
    key_file.write_text('export FMP_API_KEY="EXPORTED_KEY"\n')
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.setenv("FMP_API_KEY_FILE", str(key_file))
    monkeypatch.setattr(fmp_client, "DEFAULT_API_KEY_FILES", [])
    assert fmp_client.resolve_fmp_api_key() == "EXPORTED_KEY"


def test_resolve_fmp_api_key_ignores_other_assignments(monkeypatch, tmp_path: Path) -> None:
    key_file = tmp_path / "mixed.env"
    key_file.write_text("OTHER_KEY=NOPE\nexport TOKEN=NOPE\nFMP_API_KEY=REAL_KEY\n")
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.setenv("FMP_API_KEY_FILE", str(key_file))
    monkeypatch.setattr(fmp_client, "DEFAULT_API_KEY_FILES", [])
    assert fmp_client.resolve_fmp_api_key() == "REAL_KEY"


def test_resolve_fmp_api_key_reads_spaced_assignment(monkeypatch, tmp_path: Path) -> None:
    key_file = tmp_path / "spaced.env"
    key_file.write_text('FMP_API_KEY = "SPACED_KEY"  # inline comment\n')
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.setenv("FMP_API_KEY_FILE", str(key_file))
    monkeypatch.setattr(fmp_client, "DEFAULT_API_KEY_FILES", [])
    assert fmp_client.resolve_fmp_api_key() == "SPACED_KEY"
