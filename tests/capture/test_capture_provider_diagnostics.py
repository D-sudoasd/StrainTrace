from pathlib import Path

import pytest

from straintrace_capture.cameras import gentl
from straintrace_capture.config import CaptureError
from straintrace_capture.presentation import format_provider_report


def test_driver_report_distinguishes_missing_bindings_from_vendor_cti():
    plain = format_provider_report({
        "pylon": {"state": "unavailable", "code": "PYLON_UNAVAILABLE"},
        "gentl": {"state": "unavailable", "code": "GENTL_NO_CTI"},
    }, frozen=True)
    assert "Basler · 缺少组件\n请使用 Windows 工业相机便携版。" in plain
    assert "GenTL · 缺少厂商接口文件\n请安装相机厂商的 64 位驱动或 SDK" in plain
    installed = format_provider_report({
        "gentl": {"state": "unavailable", "code": "GENTL_CTI_LOAD_FAILED", "message": "WinError 126"},
    }, frozen=True)
    assert "厂商接口加载失败" in installed
    assert "从厂商原安装目录选择" in installed
    assert "WinError 126" in installed


def test_cti_dependency_load_failure_identifies_the_vendor_file(tmp_path):
    cti = tmp_path / "Producer.cti"
    cti.touch()

    class Harvester:
        def add_file(self, path, *, check_existence, check_validity):
            assert check_existence and check_validity
            assert Path(path) == cti
            raise OSError("dependent library missing")

    with pytest.raises(CaptureError) as error:
        gentl._load(Harvester(), [cti])
    assert error.value.code == "GENTL_CTI_LOAD_FAILED"
    assert str(cti) in str(error.value)
