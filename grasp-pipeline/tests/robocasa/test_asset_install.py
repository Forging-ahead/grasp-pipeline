"""Verify asset extraction cannot overwrite a differing file or escape its root."""
import zipfile

import pytest

from test_experiment_records import tool


def test_archive_hash_inventory_and_idempotent_install(tmp_path):
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("cabinet/model.xml", "<mujoco/>")
    downloader = tool("download_assets")
    root = tmp_path / "assets"
    inventory = downloader.extract_checked(archive, root)
    assert inventory["cabinet/model.xml"]["sha256"] == downloader.sha256(root / "cabinet/model.xml")
    assert downloader.extract_checked(archive, root) == inventory


def test_conflicting_existing_asset_is_preserved(tmp_path):
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("model.xml", "new")
    root = tmp_path / "assets"; root.mkdir()
    (root / "model.xml").write_text("original")
    with pytest.raises(FileExistsError):
        tool("download_assets").extract_checked(archive, root)
    assert (root / "model.xml").read_text() == "original"


@pytest.mark.parametrize("name", ["../escape.xml", "/absolute.xml"])
def test_unsafe_archive_member_is_rejected(tmp_path, name):
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(name, "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        tool("download_assets").extract_checked(archive, tmp_path / "assets")
