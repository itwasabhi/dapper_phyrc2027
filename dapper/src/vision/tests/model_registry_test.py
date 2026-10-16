import pytest

from vision import model_registry


def test_all_specs_are_pinned_to_commit_shas():
  for spec in model_registry.REGISTRY.values():
    assert len(spec.revision) == 40
    int(spec.revision, 16)


def test_local_path_returns_downloaded_dir(tmp_path):
  (tmp_path / "omdet_turbo").mkdir()
  (tmp_path / "omdet_turbo" / "config.json").write_text("{}")
  assert model_registry.local_path("omdet_turbo", tmp_path) == tmp_path / "omdet_turbo"


def test_local_path_missing_model_points_at_download_script(tmp_path):
  with pytest.raises(FileNotFoundError, match="download_models.py omdet_turbo"):
    model_registry.local_path("omdet_turbo", tmp_path)


def test_local_path_unknown_name(tmp_path):
  with pytest.raises(KeyError):
    model_registry.local_path("nope", tmp_path)
