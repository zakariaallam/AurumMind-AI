"""Versioned model artifacts with atomic champion promotion and rollback."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib

from .config import load_config, resolve_path
from .database import Database, utc_now


class ModelRegistry:
    """Persist trusted local candidates and atomically select a champion."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        database: Database | None = None,
    ) -> None:
        self.config = config or load_config()
        self.database = database or Database()
        self.root = resolve_path(self.config["application"]["artifacts_dir"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.champion_file = self.root / "current_champion.json"

    def register(
        self,
        model_name: str,
        model: Any,
        feature_names: list[str],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Write and load-verify an immutable model artifact before indexing it."""
        version = f"{model_name}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
        temporary = self.root / f".{version}.tmp"
        destination = self.root / version
        temporary.mkdir(parents=True, exist_ok=False)
        bundle = {
            "model": model,
            "feature_names": feature_names,
            "class_mapping": {"0": "DOWN", "1": "NEUTRAL", "2": "UP"},
            "signal_threshold": metadata.get("signal_threshold"),
            "feature_schema_version": self.config["features"]["version"],
        }
        try:
            artifact = temporary / "model.joblib"
            joblib.dump(bundle, artifact)
            verified = joblib.load(artifact)
            if verified["feature_names"] != feature_names or not hasattr(
                verified["model"], "predict_proba"
            ):
                raise ValueError("Saved model artifact failed its load verification.")
            model_metadata = {
                **metadata,
                "version": version,
                "model_name": model_name,
                "registered_at": utc_now(),
                "artifact_path": str(destination / "model.joblib"),
                "trusted_artifact_warning": (
                    "Only load artifacts created and controlled by this local registry."
                ),
            }
            (temporary / "metadata.json").write_text(
                json.dumps(model_metadata, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(temporary, destination)
            model_metadata["artifact_path"] = str(destination / "model.joblib")
            self.database.record_model(model_metadata)
            return model_metadata
        except Exception:
            if temporary.exists():
                for item in temporary.iterdir():
                    item.unlink()
                temporary.rmdir()
            raise

    def load(self, version: str) -> dict[str, Any]:
        """Load a verified registered model; artifacts must be locally trusted."""
        record = self.database.get_model(version)
        if record is None:
            raise FileNotFoundError(f"Unknown model version: {version}")
        artifact = Path(record["artifact_path"])
        if not artifact.is_file() or self.root not in artifact.parents:
            raise ValueError("Model artifact path is missing or outside the registry.")
        bundle = joblib.load(artifact)
        metadata_path = artifact.with_name("metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["version"] != version:
            raise ValueError("Model metadata does not match the requested version.")
        return {**bundle, "metadata": metadata}

    def champion_version(self) -> str | None:
        """Return the active, indexed model version or None if not ready."""
        stored = self.database.get_state("champion_version")
        if not stored:
            return None
        self.load(stored)
        if self.champion_file.exists():
            pointer = json.loads(self.champion_file.read_text(encoding="utf-8"))
            if pointer.get("version") == stored:
                return stored
        self._write_pointer(
            {
                "version": stored,
                "previous_version": (self.database.get_state("champion_previous_version") or None),
                "recovered": True,
            }
        )
        return stored

    def promote(
        self,
        candidate: dict[str, Any],
        challenger_metrics: dict[str, float],
        incumbent_metrics: dict[str, float] | None,
        *,
        data_quality_score: float,
        leakage_checks_passed: bool,
        regime_macro_f1: dict[str, float],
        incumbent_regime_macro_f1: dict[str, float] | None = None,
    ) -> tuple[bool, str]:
        """Promote only a qualified candidate measured on the same validation set."""
        rules = self.config["training"]
        if data_quality_score < float(self.config["market_data"]["minimum_quality_score"]):
            return False, "Data quality is below the configured promotion threshold."
        if not leakage_checks_passed:
            return False, "Leakage/integrity checks did not pass."
        if challenger_metrics["macro_f1"] < float(rules["minimum_validation_macro_f1"]):
            return False, "Validation macro-F1 is below the configured minimum."
        if challenger_metrics["brier_score"] > float(rules["maximum_validation_brier_score"]):
            return False, "Validation Brier score is above the configured maximum."
        if any(value < 0.15 for value in regime_macro_f1.values()):
            return False, "At least one validation regime has unacceptable macro-F1."
        if incumbent_regime_macro_f1 is not None:
            for regime, score in regime_macro_f1.items():
                incumbent_score = incumbent_regime_macro_f1.get(regime)
                if incumbent_score is not None and score < incumbent_score - 0.05:
                    return (
                        False,
                        f"Validation performance degraded in regime {regime}.",
                    )
        minimum_gain = float(rules["minimum_promotion_improvement"])
        if incumbent_metrics is not None:
            if challenger_metrics["macro_f1"] < (incumbent_metrics["macro_f1"] + minimum_gain):
                return False, "Candidate did not beat the incumbent on this validation set."
            if challenger_metrics["brier_score"] > (incumbent_metrics["brier_score"] + 0.02):
                return False, "Candidate calibration degraded versus the incumbent."

        old_version = self.database.get_state("champion_version")
        pointer = {
            "version": candidate["version"],
            "previous_version": old_version,
            "promoted_at": utc_now(),
            "validation_metrics": challenger_metrics,
        }
        self.database.set_champion(candidate["version"], old_version)
        self._write_pointer(pointer)
        return True, "Candidate passed validation and was promoted."

    def rollback(self) -> str:
        """Atomically restore the most recently active champion."""
        current = self.champion_version()
        if current is None:
            raise RuntimeError("There is no champion pointer to roll back.")
        pointer = json.loads(self.champion_file.read_text(encoding="utf-8"))
        previous = pointer.get("previous_version")
        if not isinstance(previous, str) or not previous:
            raise RuntimeError("No previous champion version is available.")
        self.load(previous)
        next_pointer = {
            "version": previous,
            "previous_version": pointer.get("version"),
            "promoted_at": utc_now(),
            "rollback": True,
        }
        self.database.set_champion(previous, pointer.get("version"))
        self._write_pointer(next_pointer)
        return previous

    def _write_pointer(self, pointer: dict[str, Any]) -> None:
        temp_file = self.champion_file.with_name(f".{self.champion_file.name}.{uuid4().hex}.tmp")
        temp_file.write_text(json.dumps(pointer, indent=2), encoding="utf-8")
        os.replace(temp_file, self.champion_file)
