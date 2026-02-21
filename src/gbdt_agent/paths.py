from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    project_dir: Path

    @classmethod
    def from_project_dir(cls, project_dir: str | Path) -> "ProjectPaths":
        return cls(project_dir=Path(project_dir))

    @property
    def src_dir(self) -> Path:
        return self.project_dir / "src"

    @property
    def conf_dir(self) -> Path:
        return self.project_dir / "conf"

    @property
    def data_dir(self) -> Path:
        return self.project_dir / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def feature_store_dir(self) -> Path:
        return self.data_dir / "feature_store"

    @property
    def cache_http_dir(self) -> Path:
        return self.data_dir / "cache_http"

    @property
    def artifacts_dir(self) -> Path:
        return self.project_dir / "artifacts"

    @property
    def runs_dir(self) -> Path:
        return self.artifacts_dir / "runs"

    @property
    def state_dir(self) -> Path:
        return self.project_dir / "state"

    @property
    def logs_dir(self) -> Path:
        return self.project_dir / "logs"

    @property
    def reports_dir(self) -> Path:
        return self.project_dir / "reports"

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def run_log_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "logs" / "run.log"

    def run_metrics_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "metrics.json"

    def run_predictions_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "predictions.parquet"

    def run_backtest_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "backtest.parquet"

    def run_report_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "report.md"

    def ensure_base_dirs(self) -> None:
        for p in [
            self.src_dir,
            self.conf_dir,
            self.raw_dir,
            self.processed_dir,
            self.feature_store_dir,
            self.cache_http_dir,
            self.runs_dir,
            self.state_dir,
            self.logs_dir,
            self.reports_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)
