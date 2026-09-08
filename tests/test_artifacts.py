import json

from src.artifacts.loader import resolve_artifact_dir


def test_active_release_wins_over_flat_legacy_mirror(tmp_path):
    """Release pointer phải thắng model.joblib phẳng để không phục vụ nhầm version cũ."""
    version_dir = tmp_path / "models" / "release-v2"
    version_dir.mkdir(parents=True)
    (tmp_path / "model.joblib").write_bytes(b"legacy")
    (tmp_path / "active_release.json").write_text(
        json.dumps({"active_version": "release-v2"}),
        encoding="utf-8",
    )

    resolved = resolve_artifact_dir(tmp_path)

    assert resolved == version_dir.resolve()
