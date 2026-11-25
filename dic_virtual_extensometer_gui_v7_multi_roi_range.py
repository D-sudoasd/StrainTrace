#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DIC Virtual Extensometer GUI v7 Multi-ROI Range Preview
-----------------------------------------
用途：
    虚拟引伸计：多组 ROI pair，独立追踪，输出工程/真应变曲线。
    全场 2D DIC：子集网格 + IC-GN / IC-LM，输出位移场与 Green-Lagrange 应变图。

核心功能：
    1. 多组 ROI：可以添加 G01、G02、G03... 分别分析。
    2. 方向自动判断：auto 模式下，左右分开用 x，上下分开用 y，倾斜明显用 distance。
    3. 对齐辅助：
       - 水平对齐：强制 ROI1/ROI2 中心 y 相同，并建议用 x 应变。
       - 垂直对齐：强制 ROI1/ROI2 中心 x 相同，并建议用 y 应变。
       - 绘制 ROI2 时可自动对齐。
    4. 自适应追踪：
       - hard accept：相关系数高，应变增量连续。
       - adaptive accept：相关系数略低，但通过软阈值、应变连续、前后向检查。
       - rejected：不更新 ROI、不更新模板，应变写 NaN。
    5. 全场 2D DIC：矩形 ROI 内按 subset/step 布点，整数/模板匹配初值后 IC-GN 或 IC-LM
       一阶仿射细化（ZNSSD/ZNCC）。失败点保持 NaN，不插值填补。
    6. 由位移场窗口拟合 Green-Lagrange 与无穷小应变，可选高斯平滑；导出表与色图。
    7. Windows 中文路径兼容：使用 np.fromfile + cv2.imdecode 读取图片。
    8. 可预览任意帧，并设置分析起始帧/结束帧，方便避开无效前后段。

依赖：
    pip install opencv-python numpy pandas matplotlib pillow

运行：
    python dic_virtual_extensometer_gui_v7_multi_roi_range.py
"""


import contextlib
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import traceback
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2  # noqa: F401
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# The numerical/export implementation lives in the Tk-free core, and modular
# desktop UI helpers (tooltips, theme/DPI, image canvas, interactive viewers)
# live under ``ezdic_gui``.  Importing this module defines the ``MultiROIGUI``
# facade and preserves backwards-compatible module-level re-exports; no Tk
# root is created until ``main`` is called.
import ezdic_core as _core
from ezdic_gui import (
    CanvasMixin,
    ChoiceToolTip,
    DIC_COMPONENT_HELP,  # noqa: F401
    DIC_COMPONENT_LABELS,  # noqa: F401
    HelpNavigationToolbar,  # noqa: F401
    RECENT_CONFIG_ENV_VAR,
    RECENT_CONFIG_FILENAME,
    ThemeMixin,
    ToolTip,
    UI_BASE_FONT_SIZE,  # noqa: F401
    UI_FONT_FALLBACKS,  # noqa: F401
    UI_FONT_FAMILY,  # noqa: F401
    UI_LOG_FONT_SIZE,  # noqa: F401
    UI_MAX_TK_SCALING,  # noqa: F401
    UI_MIN_TK_SCALING,  # noqa: F401
    UI_PRIMARY_FONT_SIZE,  # noqa: F401
    UI_SCALE_ENV_VAR,  # noqa: F401
    UI_STEP_FONT_SIZE,  # noqa: F401
    UI_TITLE_FONT_SIZE,  # noqa: F401
    UI_TREE_ROWHEIGHT,  # noqa: F401
    UI_VIEWER_LABEL_FONT_SIZE,  # noqa: F401
    UI_VIEWER_LEGEND_FONT_SIZE,  # noqa: F401
    UI_VIEWER_TICK_FONT_SIZE,  # noqa: F401
    UI_VIEWER_TITLE_FONT_SIZE,  # noqa: F401
    ViewersMixin,
    _get_win32_monitor_api,  # noqa: F401
    choose_ui_font_family,  # noqa: F401
    enable_windows_dpi_awareness,
)

# 优先使用微软雅黑 / 黑体，并使用其他平台已有的中文字库作为回退。
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Noto Sans SC",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "PingFang SC",
    "Heiti SC",
    "Arial Unicode MS",
    "Arial",
    "Helvetica",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"


# ---------------------------------------------------------------------------
# Canonical compatibility re-exports from the Tk-free core
# ---------------------------------------------------------------------------
CoreError = _core.CoreError
PLOT_COLOR_CYCLE = _core.PLOT_COLOR_CYCLE
PLOT_EXPORT_FORMATS = _core.PLOT_EXPORT_FORMATS
PLOT_EXPORT_PRESETS = _core.PLOT_EXPORT_PRESETS
resolve_code_paths = _core.resolve_code_paths
resolve_source_paths = _core.resolve_source_paths
format_tracking_status_line = _core.format_tracking_status_line
write_image_checked = _core.write_image_checked
natural_sort_key = _core.natural_sort_key
safe_name = _core.safe_name
collect_images = _core.collect_images
image_sequence_fingerprint = _core.image_sequence_fingerprint
generate_synthetic_speckle = _core.generate_synthetic_speckle
warp_image_translation = _core.warp_image_translation
warp_image_deformation_gradient = _core.warp_image_deformation_gradient
green_lagrange_from_F = _core.green_lagrange_from_F
read_gray_image = _core.read_gray_image
normalize_to_uint8 = _core.normalize_to_uint8
get_display_image = _core.get_display_image
rect_normalize = _core.rect_normalize
clamp_rect = _core.clamp_rect
rect_center = _core.rect_center
move_rect_center = _core.move_rect_center
center_distance = _core.center_distance
roi_separation = _core.roi_separation
resolve_strain_mode = _core.resolve_strain_mode
length_between = _core.length_between
extract_patch = _core.extract_patch
has_nonzero_variance = _core.has_nonzero_variance
subpixel_peak = _core.subpixel_peak
integer_cc_guess = _core.integer_cc_guess
match_template_candidate = _core.match_template_candidate
match_template_candidate_diagnostic = _core.match_template_candidate_diagnostic
extract_patch_subpixel = _core.extract_patch_subpixel
update_template_from_rect = _core.update_template_from_rect
forward_backward_error = _core.forward_backward_error
draw_group_overlay = _core.draw_group_overlay
initialize_extensometer_group_state = _core.initialize_extensometer_group_state
track_extensometer_group_frame = _core.track_extensometer_group_frame
field_quality_summary = _core.field_quality_summary
DEFAULT_PEAK_MARGIN_MIN = _core.DEFAULT_PEAK_MARGIN_MIN
DEFAULT_PEAK_RATIO_MIN = _core.DEFAULT_PEAK_RATIO_MIN
DEFAULT_MAX_HESSIAN_CONDITION_NUMBER = _core.DEFAULT_MAX_HESSIAN_CONDITION_NUMBER
DEFAULT_MIN_CORRELATION_VALID_FRACTION = _core.DEFAULT_MIN_CORRELATION_VALID_FRACTION
DEFAULT_MIN_STRAIN_VALID_FRACTION = _core.DEFAULT_MIN_STRAIN_VALID_FRACTION
build_poi_grid = _core.build_poi_grid
poi_grid_is_usable = _core.poi_grid_is_usable
rect_is_inside_image = _core.rect_is_inside_image
validate_image_sequence_dimensions = _core.validate_image_sequence_dimensions
fullfield_field_has_finite_strain = _core.fullfield_field_has_finite_strain
roi_texture_metrics = _core.roi_texture_metrics
texture_is_ok = _core.texture_is_ok
texture_failure_code = _core.texture_failure_code
require_texture = _core.require_texture
TEXTURE_METRICS_VERSION = _core.TEXTURE_METRICS_VERSION
TEXTURE_DISCRIMINATOR_VERSION = _core.TEXTURE_DISCRIMINATOR_VERSION
TEXTURE_PREFLIGHT_VERSION = _core.TEXTURE_PREFLIGHT_VERSION
compute_reference_normalization = _core.compute_reference_normalization
compute_reference_normalization_bounds = _core.compute_reference_normalization_bounds
normalize_with_bounds = _core.normalize_with_bounds
normalize_sequence_frames = _core.normalize_sequence_frames
sha256_file = _core.sha256_file
file_identity = _core.file_identity
ordered_input_manifest = _core.ordered_input_manifest
canonicalize_json = _core.canonicalize_json
canonical_json_bytes = _core.canonical_json_bytes
canonical_json_hash = _core.canonical_json_hash
collect_environment = _core.collect_environment
code_fingerprint = _core.code_fingerprint
refine_subset_icgn = _core.refine_subset_icgn
refine_subset_iclm = _core.refine_subset_iclm
compute_strain_fields = _core.compute_strain_fields
run_2d_dic = _core.run_2d_dic
run_2d_dic_sequence = _core.run_2d_dic_sequence
dic_field_to_dataframe = _core.dic_field_to_dataframe
write_dic_field_txt = _core.write_dic_field_txt
write_dic_field_parameters = _core.write_dic_field_parameters
style_dic_colorbar = _core.style_dic_colorbar
render_dic_field_on_axes = _core.render_dic_field_on_axes
plot_dic_field_map = _core.plot_dic_field_map
add_dic_colorbar = _core.add_dic_colorbar
dic_color_limits = _core.dic_color_limits
overlay_dic_field_on_image = _core.overlay_dic_field_on_image
export_dic_field_outputs = _core.export_dic_field_outputs
normalize_roi_role = _core.normalize_roi_role
poisson_roles_are_configured = _core.poisson_roles_are_configured
get_poisson_role_groups = _core.get_poisson_role_groups
validate_poisson_role_groups = _core.validate_poisson_role_groups
build_core_strain_table = _core.build_core_strain_table
write_origin_txt = _core.write_origin_txt
build_mean_strain_table = _core.build_mean_strain_table
build_poisson_ratio_table = _core.build_poisson_ratio_table
build_all_groups_strain_table = _core.build_all_groups_strain_table
write_all_groups_origin_txt = _core.write_all_groups_origin_txt
write_mean_groups_origin_txt = _core.write_mean_groups_origin_txt
write_poisson_ratio_txt = _core.write_poisson_ratio_txt
build_origin_project_tables = _core.build_origin_project_tables
write_origin_opju_project = _core.write_origin_opju_project
get_plot_preset = _core.get_plot_preset
create_plot_figure = _core.create_plot_figure
style_publication_axes = _core.style_publication_axes
save_plot_figure = _core.save_plot_figure
publication_figure_paths = _core.publication_figure_paths
build_qc_summary = _core.build_qc_summary
write_qc_summary = _core.write_qc_summary
plot_engineering_strain = _core.plot_engineering_strain
plot_all_groups_engineering_strain = _core.plot_all_groups_engineering_strain
plot_poisson_ratio = _core.plot_poisson_ratio
plot_correlation_scores = _core.plot_correlation_scores


APP_NAME = "ezDIC"
APP_VERSION = "0.1.4"
APP_DEVELOPER = "Dr. Delun Gong"
APP_DOI = "10.5281/zenodo.20222465"
APP_DOI_URL = f"https://doi.org/{APP_DOI}"
APP_TITLE = f"{APP_NAME} v{APP_VERSION} - Developed by {APP_DEVELOPER} - DOI: {APP_DOI}"
ORIGIN_OPJU_FILENAME = "ezDIC_results.opju"

CITATION_TEXT = f"""Recommended citation:

Gong, D. (2026). ezDIC: A lightweight virtual extensometer for extracting linear strain from image sequences (Version {APP_VERSION}) [Computer software]. Zenodo. {APP_DOI_URL}
"""

USAGE_NOTICE = f"""ezDIC Attribution and Usage Notice

Developer:
{APP_DEVELOPER}

License:
MIT License (see LICENSE.txt and NOTICE_Attribution_and_Usage.txt)

DOI:
{APP_DOI}

This software was developed by {APP_DEVELOPER} for lightweight extraction of linear strain and 2D full-field DIC from image sequences.

{CITATION_TEXT.strip()}

Licensing and Attribution:
- Modification, commercial use, and redistribution are permitted under the MIT License.
- Please preserve the copyright notice and permission notice in all copies or substantial portions of the Software.
- Citation in scholarly work (thesis, paper, presentation, or report) is encouraged, not an additional license restriction.

Warranty Disclaimer:
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.
"""


IMAGE_EXTENSIONS = [
    "*.tif", "*.tiff", "*.TIF", "*.TIFF",
    "*.png", "*.jpg", "*.jpeg", "*.bmp"
]

TRACKING_PRESETS = {
    "标准": {
        "search_radius": 180,
        "hard_corr": 0.55,
        "soft_corr": 0.35,
        "max_frame_strain_jump": "0.01",
        "fb_tolerance_px": 12.0,
    },
    "低质量图像": {
        "search_radius": 220,
        "hard_corr": 0.45,
        "soft_corr": 0.30,
        "max_frame_strain_jump": "0.015",
        "fb_tolerance_px": 16.0,
    },
    "快速变形": {
        "search_radius": 300,
        "hard_corr": 0.50,
        "soft_corr": 0.30,
        "max_frame_strain_jump": "0.03",
        "fb_tolerance_px": 20.0,
    },
}

STRAIN_MODE_LABEL_TO_VALUE = {
    "自动判断": "auto",
    "横向应变": "x",
    "纵向应变": "y",
    "两点距离应变": "distance",
}
STRAIN_MODE_VALUE_TO_LABEL = {v: k for k, v in STRAIN_MODE_LABEL_TO_VALUE.items()}

ROI_ROLE_LABEL_TO_VALUE = {
    "普通": "none",
    "拉伸方向": "axial",
    "横向方向": "transverse",
}
ROI_ROLE_VALUE_TO_LABEL = {v: k for k, v in ROI_ROLE_LABEL_TO_VALUE.items()}
ROI_ROLE_VALUES = set(ROI_ROLE_VALUE_TO_LABEL)
POISSON_MIN_ABS_AXIAL_ENGINEERING_STRAIN = 1e-6

GROUP_TREE_HEADING_TEXTS = {
    "name": "组名",
    "role": "角色",
    "selected": "所选方向",
    "actual": "实际方向",
    "L0": "L0",
    "dx": "Δx",
    "dy": "Δy",
    "roi1": "ROI1",
    "roi2": "ROI2",
}
GROUP_TREE_COLUMN_WIDTHS = {
    "name": 56,
    "role": 52,
    "selected": 76,
    "actual": 76,
    "L0": 48,
    "dx": 44,
    "dy": 44,
    "roi1": 72,
    "roi2": 72,
}
TRACKING_ACCEPT_MODE_LABELS = {
    "initial": "初始",
    "hard": "硬接受",
    "adaptive": "自适应接受",
    "rejected": "已拒绝",
}

ANALYSIS_MODE_EXTENSOMETER = "extensometer"
ANALYSIS_MODE_FULLFIELD = "fullfield"
DIC_SOLVER_ICGN = "IC-GN"
DIC_SOLVER_ICLM = "IC-LM"
DIC_SOLVERS = (DIC_SOLVER_ICGN, DIC_SOLVER_ICLM)
DIC_FIELD_COMPONENTS = ("u", "v", "zncc", "Exx", "Eyy", "Exy", "exx", "eyy", "exy")


def open_output_folder(path):
    folder = Path(path)
    if not folder.exists():
        raise RuntimeError(f"结果目录不存在：{folder}")
    if not folder.is_dir():
        raise RuntimeError(f"结果路径不是文件夹：{folder}")
    if hasattr(os, "startfile"):
        os.startfile(str(folder))
        return
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    try:
        subprocess.Popen([opener, str(folder)])
    except OSError as exc:
        raise RuntimeError(f"无法使用 {opener} 打开结果目录：{exc}") from exc


FULLFIELD_GENERATED_FILE_RE = re.compile(
    r"^(?:"
    r"frame_\d{4}\.(?:txt|csv)|"
    r"frame_\d{4}_(?:u|v|Exx|Eyy|Exy|overlay)\.png|"
    r"frame_\d{4}_parameters\.txt"
    r")$",
    flags=re.IGNORECASE,
)


def _create_unique_timestamped_dir(root, *, prefix=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = f"{prefix}_{timestamp}" if prefix else timestamp
    candidate = root / name
    suffix = 1
    while candidate.exists():
        candidate = root / f"{name}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def _atomic_move_entries(entries, label, *, after_move=None):
    """Move exact file entries with best-effort inverse moves on failure."""
    normalized_entries = [(Path(source), Path(destination)) for source, destination in entries]
    moved = []
    try:
        for source, destination in normalized_entries:
            shutil.move(str(source), str(destination))
            moved.append((source, destination))
        if after_move is not None:
            after_move()
    except Exception as exc:
        rollback_errors = []
        for source, destination in reversed(moved):
            if not destination.exists():
                continue
            try:
                shutil.move(str(destination), str(source))
            except Exception as rollback_exc:
                # Path.replace bypasses a transient shutil failure while
                # remaining confined to the exact generated destination.
                try:
                    destination.replace(source)
                except Exception as replace_exc:
                    rollback_errors.append(
                        f"{destination} -> {source}: {rollback_exc}; fallback: {replace_exc}"
                    )
        detail = f"{label}失败：{exc}"
        if rollback_errors:
            detail += "；回滚失败：" + " | ".join(rollback_errors)
        raise RuntimeError(detail) from exc


def archive_previous_fullfield_outputs(dic_dir):
    """Move exact legacy full-field outputs under the core transaction guards.

    The legacy GUI path predates the manifest ledger, so it retains the
    conservative exact-name allowlist above.  It must nevertheless obey the
    same filesystem boundary as the manifest transaction: all output and
    destination ancestors are checked before enumeration or directory
    creation, and each move uses the core's no-replace primitive (which also
    applies the Windows directory-handle guards).  These are deliberately
    private core calls because this compatibility migration has no public
    transaction API of its own.
    """
    dic_dir = Path(dic_dir)
    output_root = dic_dir.parent
    previous_root = dic_dir / "_previous_runs"

    def assert_safe(path, *, allow_missing_leaf):
        _core._assert_no_reparse_components(path, allow_missing_leaf=allow_missing_leaf)

    def assert_contained(path, *, code):
        _core._assert_contained_path(path, output_root, code=code)

    # Resolve no links before any exists/is_dir check.  In particular, a
    # junction at output_root/dic must fail before it can be enumerated.
    assert_safe(output_root, allow_missing_leaf=True)
    assert_safe(dic_dir, allow_missing_leaf=True)
    assert_contained(dic_dir, code="LEGACY_OUTPUT_OUTSIDE_ROOT")
    assert_safe(previous_root, allow_missing_leaf=True)
    assert_contained(previous_root, code="LEGACY_ARCHIVE_OUTSIDE_ROOT")

    if not dic_dir.exists() or not dic_dir.is_dir():
        return None

    # Match the core's operation lock and in-process lock ordering.  The
    # second preflight closes the gap between the initial check and lock
    # acquisition before the directory is inspected or changed.
    with _core._operation_os_lock(output_root):
        with _core._transaction_lock(output_root):
            assert_safe(output_root, allow_missing_leaf=False)
            assert_safe(dic_dir, allow_missing_leaf=False)
            assert_contained(dic_dir, code="LEGACY_OUTPUT_OUTSIDE_ROOT")
            assert_safe(previous_root, allow_missing_leaf=True)
            assert_contained(previous_root, code="LEGACY_ARCHIVE_OUTSIDE_ROOT")

            if not dic_dir.exists() or not dic_dir.is_dir():
                return None

            # A non-directory destination parent is unusable and, like a
            # reparse point, must be rejected before source enumeration.
            if previous_root.exists() and not previous_root.is_dir():
                raise RuntimeError(f"全场旧结果归档失败：归档目录不是文件夹：{previous_root}")

            # Select and validate the eventual destination chain before
            # enumerating legacy files.  The parent may be missing, so this
            # remains read-only until all source checks have completed.
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            archive_name = timestamp
            suffix = 1
            while True:
                archive_dir = previous_root / archive_name
                assert_safe(archive_dir, allow_missing_leaf=True)
                assert_contained(archive_dir, code="LEGACY_ARCHIVE_OUTSIDE_ROOT")
                if not archive_dir.exists():
                    break
                archive_name = f"{timestamp}_{suffix:02d}"
                suffix += 1

            previous_files = []
            for path in dic_dir.iterdir():
                if not FULLFIELD_GENERATED_FILE_RE.fullmatch(path.name):
                    continue
                # A matching symlink/junction is not a user-owned exception:
                # reject it before creating an archive directory or moving a
                # different legacy file.
                assert_safe(path, allow_missing_leaf=False)
                if path.is_file():
                    assert_contained(path, code="LEGACY_OUTPUT_OUTSIDE_ROOT")
                    previous_files.append(path)
            if not previous_files:
                return None

            # Keep the destination parent containment check adjacent to the
            # source preflight; its reparse check above ran before enumeration.
            assert_contained(previous_root, code="LEGACY_ARCHIVE_OUTSIDE_ROOT")

            entries = []
            for source in sorted(previous_files, key=lambda p: p.name):
                destination = archive_dir / source.name
                assert_safe(source, allow_missing_leaf=False)
                # The final archive directory may not exist yet, so this
                # check validates its complete existing ancestor chain while
                # still keeping directory creation below the preflight.
                assert_safe(destination.parent, allow_missing_leaf=True)
                assert_contained(source, code="LEGACY_OUTPUT_OUTSIDE_ROOT")
                assert_contained(destination.parent, code="LEGACY_ARCHIVE_OUTSIDE_ROOT")
                if not source.is_file():
                    raise RuntimeError(f"全场旧结果归档失败：源文件已变化：{source}")
                entries.append((source, destination))

            # Recheck and create the destination only after all legacy source
            # and destination paths have passed the read-only preflight.
            assert_safe(previous_root, allow_missing_leaf=True)
            previous_root.mkdir(parents=True, exist_ok=True)
            assert_safe(previous_root, allow_missing_leaf=False)
            assert_contained(previous_root, code="LEGACY_ARCHIVE_OUTSIDE_ROOT")
            archive_dir.mkdir()
            assert_safe(archive_dir, allow_missing_leaf=False)
            assert_contained(archive_dir, code="LEGACY_ARCHIVE_OUTSIDE_ROOT")
            for _source, destination in entries:
                assert_safe(destination.parent, allow_missing_leaf=False)
                if destination.exists() or destination.is_symlink():
                    raise RuntimeError(f"全场旧结果归档失败：目标已存在：{destination}")

            moved = []
            try:
                for source, destination in entries:
                    # _move_exact supplies no-replace publication, containment
                    # rechecks, and Windows directory-handle protection.
                    _core._move_exact(source, destination, root=output_root)
                    moved.append((source, destination))
            except Exception as exc:
                rollback_errors = []
                for source, destination in reversed(moved):
                    try:
                        _core._move_exact(destination, source, root=output_root)
                    except Exception as rollback_exc:
                        rollback_errors.append(f"{destination} -> {source}: {rollback_exc}")
                try:
                    assert_safe(archive_dir, allow_missing_leaf=False)
                    assert_contained(archive_dir, code="LEGACY_ARCHIVE_OUTSIDE_ROOT")
                    archive_dir.rmdir()
                except OSError:
                    pass
                detail = f"全场旧结果归档失败：{exc}"
                if rollback_errors:
                    detail += "；回滚失败：" + " | ".join(rollback_errors)
                raise RuntimeError(detail) from exc
            return archive_dir


def commit_fullfield_staging(staging_dir, dic_dir):
    """Commit verified staged files into the full-field output root."""
    staging_dir = Path(staging_dir)
    dic_dir = Path(dic_dir)
    dic_dir.mkdir(parents=True, exist_ok=True)
    entries = sorted(staging_dir.iterdir(), key=lambda p: p.name)
    move_entries = []
    for entry in entries:
        destination = dic_dir / entry.name
        if destination.exists():
            raise RuntimeError(f"全场输出提交目标已存在：{destination}")
        move_entries.append((entry, destination))
    _atomic_move_entries(move_entries, "全场输出提交", after_move=staging_dir.rmdir)
    return dic_dir


def archive_failed_fullfield_staging(staging_dir, dic_dir):
    """Retain a failed run's staging contents under a unique failure archive."""
    staging_dir = Path(staging_dir)
    if not staging_dir.exists():
        return None
    failed_dir = _create_unique_timestamped_dir(Path(dic_dir) / "_failed_runs")
    entries = sorted(staging_dir.iterdir(), key=lambda p: p.name)
    move_entries = [(entry, failed_dir / entry.name) for entry in entries]
    _atomic_move_entries(move_entries, "全场失败运行归档", after_move=staging_dir.rmdir)
    return failed_dir


def validate_fullfield_snapshot(settings, *, check_sequence_dimensions=True):
    """Validate a frozen full-field settings snapshot without touching Tk.

    ``start_processing`` freezes all GUI values into ``settings`` before it
    starts the worker.  This helper is deliberately module-level and only
    consumes that mapping plus the selected image files, so it is safe to run
    from a worker thread.  It also keeps legacy-output migration behind the
    same read-only checks used by the GUI-side preflight.
    """
    if not isinstance(settings, Mapping):
        raise RuntimeError("全场 DIC 设置快照必须是映射。")

    raw_paths = settings.get("image_paths")
    if isinstance(raw_paths, (str, bytes)):
        raise RuntimeError("全场 DIC 图像路径快照无效。")
    try:
        image_paths = list(raw_paths or [])
    except TypeError as exc:
        raise RuntimeError("全场 DIC 图像路径快照无效。") from exc
    if not image_paths:
        raise RuntimeError("请先加载图像序列。")

    def snapshot_int(key, label, default=None):
        value = settings.get(key, default)
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(f"{label}必须是整数。") from exc

    def snapshot_float(key, label, default=None):
        value = settings.get(key, default)
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(f"{label}必须是数字。") from exc
        if not np.isfinite(value):
            raise RuntimeError(f"{label}必须是有限数字。")
        return value

    start_idx = snapshot_int("start_idx", "起始帧", 0)
    end_idx = snapshot_int("end_idx", "结束帧", len(image_paths) - 1)
    if start_idx < 0 or end_idx < start_idx or end_idx >= len(image_paths):
        raise RuntimeError("全场 DIC 的分析范围无效。")
    if end_idx <= start_idx:
        raise RuntimeError("分析范围至少应包含两帧（参考帧 + 变形帧）。")

    reference_frame = snapshot_int(
        "reference_frame_1based",
        "参考帧",
        start_idx + 1,
    )
    if reference_frame < start_idx + 1 or reference_frame > end_idx + 1:
        raise RuntimeError(
            "全场 DIC 的参考帧必须位于当前分析范围内："
            f"reference={reference_frame}, range={start_idx + 1}–{end_idx + 1}。"
        )
    field_ref = settings.get("field_roi_reference_frame_1based")
    if field_ref is None:
        raise RuntimeError("全场 ROI 缺少绘制参考帧记录，请在当前起始/参考帧重画。")
    try:
        field_ref = int(field_ref)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("全场 ROI 的参考帧记录必须是整数。") from exc
    if field_ref != reference_frame:
        raise RuntimeError(
            f"全场 ROI 在第 {field_ref} 帧定义，但冻结参考帧为第 {reference_frame} 帧，请重画全场 ROI。"
        )

    output_value = settings.get("output_dir")
    if output_value is None or not str(output_value).strip():
        raise RuntimeError("请设置输出文件夹。")
    try:
        output_path = Path(output_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("输出文件夹路径无效。") from exc
    if output_path.exists() and not output_path.is_dir():
        raise RuntimeError(f"输出路径已存在但不是文件夹：{output_path}")

    selected_paths = image_paths[start_idx : end_idx + 1]
    image_shape = validate_image_sequence_dimensions(selected_paths) if check_sequence_dimensions else None
    try:
        reference_raw = read_gray_image(image_paths[reference_frame - 1])
    except Exception as exc:
        raise RuntimeError(f"全场 DIC 无法读取参考图像：{exc}") from exc
    if image_shape is None:
        image_shape = reference_raw.shape[:2]

    roi = settings.get("field_roi")
    if roi is None:
        raise RuntimeError("请先绘制全场分析 ROI。")
    if not rect_is_inside_image(roi, image_shape):
        raise RuntimeError("全场 ROI 必须完整位于参考图像内。")

    try:
        normalization = settings.get("normalization")
        if normalization:
            reference8 = normalize_with_bounds(reference_raw, normalization)
        else:
            reference8 = normalize_to_uint8(reference_raw)
    except Exception as exc:
        raise RuntimeError(f"全场 DIC 参考图像归一化失败：{exc}") from exc

    min_texture_std = snapshot_float("min_texture_std", "最小纹理标准差", 8.0)
    min_texture_contrast = snapshot_float("min_texture_contrast", "最小纹理对比度", 25.0)
    max_saturated_frac = snapshot_float("max_saturated_frac", "最大近黑/近白比例", 0.20)
    min_structure_ratio = snapshot_float(
        "min_structure_ratio",
        "最小二维结构张量比值",
        _core.DEFAULT_TEXTURE_MIN_STRUCTURE_RATIO,
    )
    max_directional_coherence = snapshot_float(
        "max_directional_coherence",
        "最大方向梯度一致性",
        _core.DEFAULT_TEXTURE_MAX_DIRECTIONAL_COHERENCE,
    )
    min_periodicity_score = snapshot_float(
        "min_periodicity_score",
        "最小周期性分数",
        _core.DEFAULT_TEXTURE_MIN_PERIODICITY_SCORE,
    )
    metrics = roi_texture_metrics(reference8, roi)
    texture_code = texture_failure_code(
        metrics,
        min_texture_std,
        min_texture_contrast,
        max_saturated_frac,
        min_structure_ratio,
        max_directional_coherence,
        min_periodicity_score,
    )
    # The GUI adapter intentionally lets low/saturated texture reach the core
    # so it can retain a structured invalid-frame record.  Ambiguous texture
    # and malformed thresholds remain hard preflight failures.
    if texture_code == "AMBIGUOUS_TEXTURE":
        raise RuntimeError("AMBIGUOUS_TEXTURE：全场 ROI 的二维纹理不可辨识，请更换或增大 ROI。")
    if texture_code not in (None, "LOW_TEXTURE", "SATURATED_TEXTURE"):
        raise RuntimeError(f"全场 DIC 纹理预检失败：{texture_code}")

    subset = snapshot_int("dic_subset_size", "子集尺寸")
    step = snapshot_int("dic_step", "步长")
    window = snapshot_int("dic_strain_window", "应变窗口")
    search_radius = snapshot_int("dic_search_radius", "DIC 搜索半径")
    zncc_min = snapshot_float("dic_zncc_min", "ZNCC 下限")
    smooth_sigma = snapshot_float("dic_smooth_sigma", "高斯平滑 σ")
    pyramid_levels = snapshot_int("dic_pyramid_levels", "金字塔层数", 1)
    pyramid_scale = snapshot_float("dic_pyramid_scale", "金字塔缩放", 0.5)
    solver = str(settings.get("dic_solver", DIC_SOLVER_ICGN))
    if subset < 9:
        raise RuntimeError("子集尺寸必须 >= 9。")
    if subset % 2 == 0:
        raise RuntimeError("子集尺寸必须为奇数；请使用 21、23 等值。")
    if step < 1:
        raise RuntimeError("步长必须 >= 1。")
    if window < 3:
        raise RuntimeError("应变窗口必须 >= 3。")
    if window % 2 == 0:
        raise RuntimeError("应变窗口必须为奇数；请使用 5、7 等值。")
    if search_radius < 1:
        raise RuntimeError("DIC 搜索半径必须 > 0。")
    if not (0.0 <= zncc_min <= 1.0):
        raise RuntimeError("ZNCC 下限应在 0 到 1 之间。")
    if smooth_sigma < 0:
        raise RuntimeError("高斯平滑 σ 不能为负。")
    if not (1 <= pyramid_levels <= 8):
        raise RuntimeError("金字塔层数应在 1 到 8 之间。")
    if not (0.0 < pyramid_scale < 1.0):
        raise RuntimeError("金字塔缩放应在 0 到 1 之间。")
    if solver not in DIC_SOLVERS:
        raise RuntimeError("求解器必须是 IC-GN 或 IC-LM。")

    X, Y = build_poi_grid(roi, subset, step, image_shape)
    if not poi_grid_is_usable(X, Y, min_rows=3, min_cols=3):
        raise RuntimeError("当前 ROI / 子集 / 步长至少需要 3×3 个可分析的 2D POI，请增大 ROI 或减小步长。")
    if int(settings.get("strain_degree", 2)) not in (1, 2):
        raise RuntimeError("拟合最高阶次必须为 1 或 2。")
    outlier = float(settings.get("outlier_threshold_px", 1.0))
    if not np.isfinite(outlier) or outlier < 0:
        raise RuntimeError("异常位移阈值必须为非负有限数值。")
    mask, _, _ = _core._resolve_specimen_mask(reference8, roi, settings)
    if mask is not None:
        from scipy import ndimage
        supported = ndimage.minimum_filter(mask.astype(np.uint8), size=subset, mode="constant", cval=0)
        if np.count_nonzero(supported[Y.astype(int), X.astype(int)]) < 9:
            raise RuntimeError("遮罩内至少需要 9 个能容纳完整子集的测量点。请检查遮罩或减小子集。")
    display = settings.get("display")
    if isinstance(display, Mapping) and display:
        color_mode = display.get("color_mode", "range")
        if color_mode not in ("range", "symmetric", "manual"):
            raise RuntimeError("全场 DIC 色标范围模式无效。")
        if color_mode == "manual":
            try:
                vmin = float(display["vmin"])
                vmax = float(display["vmax"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("手动色标需要有限且递增的最小值和最大值。") from exc
            if not (math.isfinite(vmin) and math.isfinite(vmax)) or vmin >= vmax:
                raise RuntimeError("手动色标需要有限且递增的最小值和最大值。")


# ==========================
# GUI 主类
# ==========================


class MultiROIGUI(ThemeMixin, CanvasMixin, ViewersMixin):
    def __init__(self, root):
        self.root = root
        self.root.title("分析控制台 · " + APP_TITLE)
        self.configure_initial_window()

        self.image_folder = tk.StringVar()
        self.output_folder = tk.StringVar()

        self.search_radius = tk.IntVar(value=180)
        self.hard_corr = tk.DoubleVar(value=0.55)
        self.soft_corr = tk.DoubleVar(value=0.35)
        self.strain_mode = tk.StringVar(value="auto")
        self.strain_mode_display = tk.StringVar(value=STRAIN_MODE_VALUE_TO_LABEL["auto"])
        self.roi_role = tk.StringVar(value="none")
        self.roi_role_display = tk.StringVar(value=ROI_ROLE_VALUE_TO_LABEL["none"])
        self.tracking_preset = tk.StringVar(value="标准")
        self.preset_status_var = tk.StringVar(value="当前追踪模式：标准")
        self._applying_preset = False

        self.enable_adaptive = tk.BooleanVar(value=True)
        # Fixed-reference templates are the reproducible default.  Following
        # the current frame remains an explicit experimental option.
        self.use_prev_frame_template = tk.BooleanVar(value=False)
        self.template_alpha = tk.DoubleVar(value=0.70)
        self.max_frame_strain_jump = tk.StringVar(value="0.01")

        self.enable_fb_check = tk.BooleanVar(value=True)
        self.fb_tolerance_px = tk.DoubleVar(value=12.0)

        self.overlay_every = tk.IntVar(value=5)
        self.pixel_size_mm = tk.StringVar(value="")

        self.auto_align_roi2 = tk.BooleanVar(value=True)

        self.min_texture_std = tk.DoubleVar(value=8.0)
        self.min_texture_contrast = tk.DoubleVar(value=25.0)
        self.max_saturated_frac = tk.DoubleVar(value=0.20)
        self.advanced_visible = tk.BooleanVar(value=False)

        self.export_origin_txt = tk.BooleanVar(value=True)
        self.export_origin_opju = tk.BooleanVar(value=False)
        self.export_engineering_png = tk.BooleanVar(value=True)
        self.export_publication_figures = tk.BooleanVar(value=False)
        self.export_qc_summary = tk.BooleanVar(value=True)
        self.export_full_csv = tk.BooleanVar(value=False)
        self.export_corr_plot = tk.BooleanVar(value=False)
        self.export_overlays = tk.BooleanVar(value=False)
        self.export_parameters = tk.BooleanVar(value=False)

        self.image_paths = []
        self.loaded_image_folder = None
        self.loaded_image_sequence_fingerprint = None
        self.preview_raw = None
        self.preview_img8 = None  # 当前预览帧的 8-bit 图像，用于画布预览与交互绘制
        self.reference_raw = None
        self.reference_img8 = None  # 当前起始/参考帧（start_idx）对应的原始/8-bit 图像
        self._reference_cache_key = None
        self.current_fullres_img8 = None  # 用于动态缩放的原始分辨率图
        self._canvas_mask_overlay_rgb = None  # 遮罩预览的原始分辨率彩色叠加图
        self.display_img = None
        self.display_scale = 1.0
        self.photo = None
        self._resize_after_id = None
        self._has_shown_resize_hint = False

        # 暗色模式基础（可切换色板）
        self.dark_mode = tk.BooleanVar(value=False)

        # Dual workflow: virtual extensometer (default) or full-field 2D DIC
        self.analysis_mode = tk.StringVar(value=ANALYSIS_MODE_EXTENSOMETER)
        self.dic_subset_size = tk.IntVar(value=21)
        self.dic_step = tk.IntVar(value=5)
        self.dic_solver = tk.StringVar(value=DIC_SOLVER_ICGN)
        self.dic_strain_window = tk.IntVar(value=7)
        self.dic_smooth_sigma = tk.DoubleVar(value=0.0)
        self.dic_search_radius = tk.IntVar(value=20)
        self.dic_zncc_min = tk.DoubleVar(value=0.75)
        # Multiscale recovery controls.  Defaults preserve the historical
        # single-level solver and are persisted only as optional recent-state
        # fields, so older snapshots remain valid.
        self.dic_pyramid_levels = tk.IntVar(value=1)
        self.dic_pyramid_scale = tk.DoubleVar(value=0.5)
        self.dic_field_component = tk.StringVar(value="u")
        self.dic_strain_degree = tk.IntVar(value=2)
        self.dic_robust_strain = tk.BooleanVar(value=True)
        self.dic_reject_nonconverged = tk.BooleanVar(value=True)
        self.dic_outlier_threshold = tk.DoubleVar(value=1.0)
        self.dic_mask_mode = tk.StringVar(value="矩形 ROI")
        self.dic_mask_path = tk.StringVar(value="")
        self.dic_mask_exclusions = []
        self._drawing_mask_exclusion = False
        self.dic_view_background = tk.StringVar(value="无底图")
        self.dic_color_mode = tk.StringVar(value="数据范围")
        self.dic_color_min = tk.StringVar(value="")
        self.dic_color_max = tk.StringVar(value="")
        self.dic_percent = tk.BooleanVar(value=False)
        self.dic_display_style = tk.StringVar(value="连续云图")
        self.dic_colormap = tk.StringVar(value="RdBu_r")
        self.field_roi = None
        self.roi1_reference_frame_1based = None
        self.roi2_reference_frame_1based = None
        self.field_roi_reference_frame_1based = None
        self.dic_last_field = None
        self.dic_last_image = None
        self.dic_last_frame_1based = None
        self.dic_last_filename = None
        self.dic_last_reference_frame_1based = None
        self.dic_last_reference_filename = None
        self.field_viewer_context_var = tk.StringVar(value="")
        self._viewer_kind = "extensometer"
        self._committed_analysis_mode = ANALYSIS_MODE_EXTENSOMETER
        self._canvas_shows_field_overlay = False

        # 图像缩放状态
        self.zoom_factor = 1.0          # 相对于原始图像的缩放倍率
        self.auto_fit_enabled = True    # 是否跟随窗口自动适应

        self.preview_frame_1based = tk.IntVar(value=1)
        self.start_frame_1based = tk.IntVar(value=1)
        self.end_frame_1based = tk.IntVar(value=1)
        self.current_preview_index = 0  # 0-based

        self.roi1 = None
        self.roi2 = None
        self.current_roi_index = 1
        self.drag_start = None
        self.temp_rect_id = None

        self.roi_groups = []
        self.next_group_idx = 1
        self.group_name_var = tk.StringVar(value="")

        self._state_lock = threading.Lock()
        self.is_processing = False
        self._completion_pending = False
        self._run_generation = 0
        self._active_run_token = None
        self._worker_thread = None
        self._closing = False
        self._ui_poll_after_id = None
        self._group_context_menu = None
        self._worker_context = threading.local()
        self.tooltips = []
        self.ui_queue = queue.Queue()

        # In-app results viewer (Tier 0)
        self.results_df = None
        self.results_groups = None
        self.viewer_figure = None
        self.viewer_canvas = None
        self.viewer_toolbar = None
        self.last_qc_summary = None

        self.preflight_summary_var = tk.StringVar(value="")
        self.current_roi_summary_var = tk.StringVar(value="尚未绘制 ROI。")
        self.qc_overview_var = tk.StringVar(value="分析完成后显示 QC 总览。")
        self.recent_image_dir = ""
        self.recent_output_dir = ""
        self._recent_config_path = self.default_recent_config_path()
        self.load_recent_paths()

        for var in [
            self.search_radius,
            self.hard_corr,
            self.soft_corr,
            self.max_frame_strain_jump,
            self.fb_tolerance_px,
            self.template_alpha,
            self.min_texture_std,
            self.min_texture_contrast,
            self.max_saturated_frac,
        ]:
            var.trace_add("write", self.mark_tracking_custom)

        for var in [
            self.start_frame_1based,
            self.end_frame_1based,
            self.preview_frame_1based,
            self.strain_mode,
            self.roi_role,
            *self._export_option_vars(),
            self.image_folder,
            self.output_folder,
            self.dic_subset_size,
            self.dic_step,
            self.dic_solver,
            self.dic_strain_window,
            self.dic_smooth_sigma,
            self.dic_search_radius,
            self.dic_zncc_min,
            self.dic_pyramid_levels,
            self.dic_pyramid_scale,
            self.dic_strain_degree, self.dic_robust_strain, self.dic_outlier_threshold,
            self.dic_reject_nonconverged,
            self.dic_mask_mode, self.dic_mask_path,
        ]:
            var.trace_add("write", self._on_gui_state_change)

        self.configure_ui_style()
        self.build_ui()
        self.bind_common_shortcuts()
        self.configure_final_window_limits()
        self._compact_control_layout = None
        self.root.bind("<Configure>", self._on_control_window_configure, add="+")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close_window)
        self._on_control_window_configure()
        self.start_ui_queue_polling()

    @property
    def first_raw(self):
        """Backward-compatible alias for the currently previewed raw frame."""
        return getattr(self, "preview_raw", None)

    @first_raw.setter
    def first_raw(self, value):
        self.preview_raw = value

    @property
    def first_img8(self):
        """Backward-compatible alias for the currently previewed 8-bit frame."""
        return getattr(self, "preview_img8", None)

    @first_img8.setter
    def first_img8(self, value):
        self.preview_img8 = value

    def _state_lock_ctx(self):
        lock = getattr(self, "_state_lock", None)
        return lock if lock is not None else contextlib.nullcontext()

    def _invalidate_reference_cache(self):
        self.reference_raw = None
        self.reference_img8 = None
        self._reference_cache_key = None

    def _get_reference_frame_images(self, start_idx=None):
        """Return ``(reference_raw, reference_img8)`` for ``image_paths[start_idx]``.

        Never substitutes a non-reference preview frame when ``start_idx`` is valid.
        Falls back to ``preview_raw``/``preview_img8`` only when tests inject an
        in-memory image without loading ``image_paths``.
        """
        if not getattr(self, "image_paths", None):
            return getattr(self, "preview_raw", None), getattr(self, "preview_img8", None)

        if start_idx is None:
            start_1based, err = self._safe_int_var(self.start_frame_1based)
            if err is not None:
                return None, None
            start_idx = start_1based - 1

        if start_idx < 0 or start_idx >= len(self.image_paths):
            return None, None

        path = self.image_paths[start_idx]
        cache_key = (
            int(start_idx),
            str(path),
            getattr(self, "loaded_image_sequence_fingerprint", None),
        )
        if (
            getattr(self, "_reference_cache_key", None) == cache_key
            and getattr(self, "reference_raw", None) is not None
            and getattr(self, "reference_img8", None) is not None
        ):
            return self.reference_raw, self.reference_img8

        if (
            getattr(self, "current_preview_index", None) == start_idx
            and getattr(self, "preview_raw", None) is not None
            and getattr(self, "preview_img8", None) is not None
        ):
            raw = self.preview_raw
            img8 = self.preview_img8
        else:
            raw = read_gray_image(path)
            img8 = normalize_to_uint8(raw)

        self.reference_raw = raw
        self.reference_img8 = img8
        self._reference_cache_key = cache_key
        return raw, img8

    def _on_close_window(self):
        with self._state_lock_ctx():
            busy = bool(self.is_processing or self._completion_pending)
        busy = (
            busy
            or bool(getattr(getattr(self, "agent_page", None), "_busy", False))
            or bool(getattr(getattr(self, "campaign_page", None), "_busy", False))
        )
        if busy:
            if not messagebox.askyesno(
                "确认退出",
                "当前仍有后台分析任务正在运行，强行关闭可能中断结果写入。确定要退出吗？",
            ):
                return
        self._closing = True
        for attr in ("_ui_poll_after_id", "_resize_after_id"):
            job = getattr(self, attr, None)
            if job is not None:
                try:
                    self.root.after_cancel(job)
                except tk.TclError:
                    pass
                setattr(self, attr, None)
        menu = getattr(self, "_group_context_menu", None)
        if menu is not None:
            try:
                menu.destroy()
            except tk.TclError:
                pass
            self._group_context_menu = None
        try:
            self._dispose_viewer_plot()
        except tk.TclError:
            pass
        for tip in list(getattr(self, "tooltips", [])):
            try:
                tip.dispose()
            except tk.TclError:
                pass
        for page_attr in ("agent_page", "campaign_page"):
            page = getattr(self, page_attr, None)
            if page is None:
                continue
            for hook in ("shutdown", "close"):
                fn = getattr(page, hook, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        if sys.stderr is not None:
                            traceback.print_exc()
                    break
            try:
                if page.winfo_exists():
                    page.destroy()
            except tk.TclError:
                pass
        visual = getattr(self, "visual_window", None)
        if visual is not None:
            try:
                if visual.winfo_exists():
                    visual.destroy()
            except tk.TclError:
                pass
        try:
            if self.root.winfo_exists():
                self.root.destroy()
        except tk.TclError:
            pass

    # ---------- UI ----------

    def add_tooltip(self, widget, text, choices=None):
        def provider(event):
            return self._control_help(widget, text, event)

        tip = ChoiceToolTip(widget, provider, choices) if choices is not None else ToolTip(widget, provider)
        tip.owner = self.root
        widget._tooltip_text = tip.resolve_text()
        self.tooltips.append(tip)
        widget.bind("<Destroy>", lambda _event: self.tooltips.remove(tip) if tip in self.tooltips else None, add="+")
        return widget

    def _control_help(self, widget, text, event=None):
        explanation = text(event) if callable(text) else text
        if not explanation or not isinstance(widget, ttk.Widget) or not widget.instate(["disabled"]):
            return explanation
        if self.is_processing or self._completion_pending:
            reason = "正在计算或保存结果，请等本次分析完成后再操作。"
        elif widget == getattr(self, "start_button", None):
            reason = self._start_prerequisite_help()
        elif widget == getattr(self, "prev_frame_button", None) and self.image_paths:
            reason = "当前已是第一张图片。"
        elif widget == getattr(self, "next_frame_button", None) and self.image_paths:
            reason = "当前已是最后一张图片。"
        else:
            requirements = {
                "open_recent_output_button": "先选择输出文件夹，或完成一次分析。",
                "roi1_button": "先加载图像序列。", "roi2_button": "先加载图像序列。",
                "draw_field_roi_button": "先加载图像序列。",
                "show_preview_button": "先加载图像序列。",
                "prev_frame_button": "先加载图像序列。", "next_frame_button": "先加载图像序列。",
                "set_start_button": "先加载图像序列。", "set_end_button": "先加载图像序列。",
                "btn_zoom_in": "先加载图像序列。", "btn_zoom_out": "先加载图像序列。",
                "btn_fit": "先加载图像序列。", "btn_1to1": "先加载图像序列。",
                "align_x_button": "先画好两个测量框。", "align_y_button": "先画好两个测量框。",
                "add_group_button": "先在同一参考图片上画好两个测量框。",
                "load_group_button": "先在列表中单击选中一组。",
                "update_group_button": "先选中一组，并画好两个测量框。",
                "delete_group_button": "先在列表中单击选中一组。",
                "clear_rois_button": "先在图片上画出测量框。",
                "viewer_export_btn": "先完成分析，让这里显示结果图。",
                "viewer_clear_btn": "当前没有结果预览；完成分析后可使用。",
            }
            reason = next((message for name, message in requirements.items()
                           if widget == getattr(self, name, None)), "")
        return explanation + (f"\n暂不可用：{reason}" if reason else "")

    def _start_prerequisite_help(self):
        hint = self.workflow_hint_var.get()
        label, _, message = hint.partition("：")
        reasons = {
            "图像序列": "先选择存放图片的文件夹，再点“加载序列”。",
            "分析范围": "填写两张不同图片的起止序号：从 1 开始，终点要大于起点，并且不能超过已加载的图片总数。",
            "全场 ROI": "先在第一张要分析的图片上拖出矩形分析区域；整个矩形都要落在图片内。",
            "参考帧": (
                "先显示第一张要分析的图片，再重新画分析区域。"
                if self.is_fullfield_mode() else
                "测量框必须画在第一张要分析的图片上；请载入需要修改的分组，重画后点“更新选中”保存。"
            ),
            "纹理": "测量区域的明暗纹理无法同时确定左右和上下位置。请换一块纹理更清楚的区域，或画大一些再检查。",
            "子集尺寸": "每个测量点使用的正方形图片块，边长必须是至少 9 的奇数，如 21 个像素。",
            "步长": "相邻测量点的间距必须是大于 0 的整数，单位是像素。",
            "应变窗口": "计算局部伸缩所用的邻近测量点范围必须是至少 3 的奇数，如 7 行、7 列。",
            "高斯平滑": "平均附近移动量的范围不能为负。填 0 关闭平滑，或填写正数。",
            "求解器": "请从计算方法下拉列表选择一种方法。默认的 IC-GN 会逐步修正图片块的位置和形状。",
            "POI 网格": "区域内的测量点不足。请扩大分析区域、减小图片块边长或减小测量点间距，至少要有 3 行、3 列测量点。",
            "DIC 参数": "请给“子集尺寸”（图片块边长）、“步长”（测量点间距）、“应变窗口”（邻近点数）和“高斯平滑”（平均范围）填写有效数字。",
            "ROI 组": "先在第一张要分析的图片上画两个测量框，再点“添加 ROI 组”保存这一对。",
            "L0": "两个测量框的初始间距必须大于零且能读出数值。请检查测量方向，重新画框并更新该组。",
            "输出目录": (
                "当前输出路径指向已有文件。请换成文件夹路径，或点击“选择输出”。"
                if "不是文件夹" in message else
                "先填写结果文件夹路径，或点击“选择输出”设置保存位置。"
            ),
            "导出选项": "先勾选至少一种要保存的结果，例如数值表格或曲线图片。",
        }
        return reasons.get(label, hint.replace("ROI", "测量框").replace("参考帧", "参考图片"))

    def _workspace_help(self, event=None):
        if event is not None and getattr(event, "keysym", "") != "F1":
            try:
                index = self.workspace_notebook.index(f"@{event.x},{event.y}")
            except tk.TclError:
                return ""
        else:
            index = self.workspace_notebook.index("current")
        help_text = (
            "点击查看图片并画测量框。ROI 就是要追踪或计算的图片区域；先选参考图片，再按住左键拖出矩形。",
            "点击查看分析得到的曲线或彩色结果图。完成分析后，可在这里放大、移动或保存图表。",
            "点击查看 campaign 项目的首、中、末参考帧、自由段和引伸计取点；编辑仍在控制台批次项目页进行。",
            "查看当前批次的 frame-strain 曲线及已生成的图像叠图。",
            "查看自动分析项目的固定参考图；拖动绘制计算区域、统计 ROI、排除区和虚拟标距端点。",
        )
        return help_text[index] if 0 <= index < len(help_text) else ""

    def _group_table_help(self, event=None):
        action = "单击选中一组，双击载入编辑；右键可载入、更新或删除。"
        if event is None or getattr(event, "keysym", "") == "F1":
            return "每行是一对已保存的测量框，会单独计算距离变化。" + action
        region = self.group_tree.identify_region(event.x, event.y)
        if region == "separator":
            return "按住列标题之间的分隔线左右拖动，可调整这一列的宽度。"
        column = self.group_tree.identify_column(event.x)
        if not column or column == "#0":
            return action
        displayed = self.group_tree.cget("displaycolumns")
        key = displayed[int(column[1:]) - 1]
        explanation = {
            "name": "这对测量框的名称，也会用于结果文件名。",
            "role": "这组测量沿拉伸方向还是垂直于拉伸方向；两种角色一起用于计算泊松比，即横向收缩与纵向伸长的比值。",
            "selected": "添加这组时选择的距离计算方向；自动判断会根据两个框的位置确定方向。",
            "actual": "实际采用的距离方向：横向是左右，纵向是上下，两点距离是直线距离。",
            "L0": "L0 是两个测量框在参考图片上的初始间距，单位是像素。应变表示间距相对这个初始值变化了多少。",
            "dx": "Δx 是两个测量框中心在左右方向的初始间距，单位是像素。",
            "dy": "Δy 是两个测量框中心在上下方向的初始间距，单位是像素。",
            "roi1": "第一个测量框的位置和尺寸：左上角的左右坐标、上下坐标、宽、高，均为像素。",
            "roi2": "第二个测量框的位置和尺寸：左上角的左右坐标、上下坐标、宽、高，均为像素。",
        }[key]
        return explanation if region == "heading" else explanation + "\n" + action

    def _on_gui_state_change(self, *args):
        if getattr(self, "_closing", False):
            return
        try:
            self.update_workflow_action_states()
        except tk.TclError:
            pass
        except Exception:
            if sys.stderr is not None:
                traceback.print_exc()

    def bind_common_shortcuts(self):
        for window in (self.root, self.visual_window):
            window.bind("<Control-l>", lambda _event: self.load_first_image() or "break")
            window.bind("<Control-f>", lambda _event: self.fit_image_to_view() or "break")
            window.bind("<Control-Return>", lambda _event: self.start_processing() or "break")
            window.bind("<Escape>", lambda _event: self.clear_current_rois() or "break")
            window.bind("<Control-plus>", lambda _event: self.zoom_image(1.25) or "break")
            window.bind("<Control-equal>", lambda _event: self.zoom_image(1.25) or "break")
            window.bind("<Control-minus>", lambda _event: self.zoom_image(1 / 1.25) or "break")
            window.bind("<Control-i>", lambda _event: self.show_visual_window() or "break")

    def default_recent_config_path(self):
        configured = os.environ.get(RECENT_CONFIG_ENV_VAR)
        if configured:
            return Path(configured)
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / ".ezdic"
        return base / "ezDIC" / RECENT_CONFIG_FILENAME

    def load_recent_paths(self):
        try:
            path = Path(self._recent_config_path)
            if not path.exists():
                return
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return

        image_dir = str(data.get("image_dir", "") or "")
        output_dir = str(data.get("output_dir", "") or "")
        self.recent_image_dir = image_dir
        self.recent_output_dir = output_dir
        if image_dir and not self.image_folder.get().strip():
            self.image_folder.set(image_dir)
        if output_dir and not self.output_folder.get().strip():
            self.output_folder.set(output_dir)
        # Optional fields were added after the original two-path snapshot.
        # Ignore absent/invalid values so old snapshots remain loadable.
        try:
            if "pyramid_levels" in data:
                value = int(data["pyramid_levels"])
                if 1 <= value <= 8:
                    self.dic_pyramid_levels.set(value)
            if "pyramid_scale" in data:
                value = float(data["pyramid_scale"])
                if 0.0 < value < 1.0:
                    self.dic_pyramid_scale.set(value)
        except (TypeError, ValueError, tk.TclError):
            pass

    def save_recent_paths(self):
        try:
            pyramid_levels = int(self.dic_pyramid_levels.get())
            if not (1 <= pyramid_levels <= 8):
                pyramid_levels = 1
        except (tk.TclError, TypeError, ValueError):
            pyramid_levels = 1
        try:
            pyramid_scale = float(self.dic_pyramid_scale.get())
            if not math.isfinite(pyramid_scale) or not (0.0 < pyramid_scale < 1.0):
                pyramid_scale = 0.5
        except (tk.TclError, TypeError, ValueError):
            pyramid_scale = 0.5
        try:
            path = Path(self._recent_config_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "image_dir": self.recent_image_dir,
                "output_dir": self.recent_output_dir,
                "pyramid_levels": pyramid_levels,
                "pyramid_scale": pyramid_scale,
            }
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except (OSError, TypeError, ValueError) as exc:
            if sys.stderr is not None:
                print(f"Warning: failed to save recent paths: {exc}", file=sys.stderr)

    def remember_recent_paths(self, image_dir=None, output_dir=None):
        if image_dir:
            self.recent_image_dir = str(image_dir)
        if output_dir:
            self.recent_output_dir = str(output_dir)
        self.save_recent_paths()
        self.update_recent_output_button_state()

    def update_recent_output_button_state(self):
        if not hasattr(self, "open_recent_output_button"):
            return
        state = tk.NORMAL if self.recent_output_dir else tk.DISABLED
        self.open_recent_output_button.config(state=state)

    def open_recent_output_folder(self):
        if not self.recent_output_dir:
            messagebox.showinfo("无最近输出", "当前还没有可打开的最近输出目录。")
            return
        try:
            open_output_folder(self.recent_output_dir)
        except Exception as exc:
            messagebox.showerror("无法打开输出目录", str(exc))
            self.log(f"无法打开最近输出目录：{exc}")

    def _export_option_vars(self):
        return [
            self.export_origin_txt,
            self.export_origin_opju,
            self.export_engineering_png,
            self.export_publication_figures,
            self.export_qc_summary,
            self.export_full_csv,
            self.export_corr_plot,
            self.export_overlays,
            self.export_parameters,
        ]

    def has_any_export_option(self):
        return any(bool(var.get()) for var in self._export_option_vars())

    def _preflight_item(self, level, label, message):
        return {"level": level, "label": label, "message": message}

    def _safe_int_var(self, var):
        try:
            return int(var.get()), None
        except (tk.TclError, TypeError, ValueError) as exc:
            return None, exc

    def build_preflight_items(self, update_display=True):
        items = []
        has_sequence = bool(self.image_paths) and self.preview_img8 is not None
        frame_count = len(self.image_paths)

        if has_sequence:
            items.append(self._preflight_item("ok", "图像序列", f"已加载 {frame_count} 帧。"))
        else:
            items.append(self._preflight_item("block", "图像序列", "请先加载图像序列。"))

        start, start_err = self._safe_int_var(self.start_frame_1based)
        end, end_err = self._safe_int_var(self.end_frame_1based)
        if start_err or end_err:
            items.append(self._preflight_item("block", "分析范围", "起始帧和结束帧必须是整数。"))
        elif has_sequence and (start < 1 or end < 1 or start > frame_count or end > frame_count):
            items.append(self._preflight_item("block", "分析范围", f"范围必须位于 1 到 {frame_count} 帧之间。"))
        elif start is not None and end is not None and end <= start:
            items.append(self._preflight_item("block", "分析范围", "至少需要两帧：结束帧必须大于起始帧。"))
        elif start is not None and end is not None:
            items.append(self._preflight_item("ok", "分析范围", f"第 {start} 到 {end} 帧。"))

        ref_img8 = None
        if has_sequence and start_err is None and 1 <= start <= frame_count:
            try:
                _ref_raw, ref_img8 = self._get_reference_frame_images(start - 1)
            except Exception as exc:
                items.append(self._preflight_item("block", "参考帧", f"无法读取第 {start} 帧参考图像：{exc}"))

        is_ff = self.is_fullfield_mode()
        if is_ff:
            field_roi = self.field_roi
            if field_roi is None:
                items.append(self._preflight_item("block", "全场 ROI", "请在参考帧拖出全场分析 ROI。"))
            elif ref_img8 is not None and not rect_is_inside_image(field_roi, ref_img8.shape):
                items.append(self._preflight_item("block", "全场 ROI", "全场 ROI 必须完整位于参考图像内。"))
            else:
                x, y, w, h = field_roi
                items.append(self._preflight_item("ok", "全场 ROI", f"ROI {w}×{h} px @ ({x},{y})。"))

            if field_roi is not None and ref_img8 is not None and rect_is_inside_image(field_roi, ref_img8.shape):
                metrics, texture_code = self._core_texture_metrics_and_code(field_roi, image=ref_img8)
                if texture_code is not None:
                    level = "block" if texture_code == "AMBIGUOUS_TEXTURE" else "warn"
                    items.append(
                        self._preflight_item(
                            level,
                            "纹理",
                            f"{texture_code}：二维结构张量比值={metrics['structure_tensor_ratio']:.4f}。",
                        )
                    )
                else:
                    items.append(
                        self._preflight_item(
                            "ok",
                            "纹理",
                            f"二维纹理可辨识，结构张量比值={metrics['structure_tensor_ratio']:.4f}。",
                        )
                    )

            if field_roi is not None and start is not None:
                field_ref = self.field_roi_reference_frame_1based
                if field_ref is None:
                    items.append(
                        self._preflight_item(
                            "block",
                            "参考帧",
                            "全场 ROI 缺少绘制参考帧记录，请在当前起始/参考帧重画。",
                        )
                    )
                elif int(field_ref) != start:
                    items.append(
                        self._preflight_item(
                            "block",
                            "参考帧",
                            f"全场 ROI 在第 {field_ref} 帧定义，但当前起始/参考帧为第 {start} 帧，请重画全场 ROI。",
                        )
                    )
                else:
                    items.append(self._preflight_item("ok", "参考帧", "全场 ROI 与当前起始/参考帧一致。"))

            try:
                subset = int(self.dic_subset_size.get())
                step = int(self.dic_step.get())
                window = int(self.dic_strain_window.get())
                solver = str(self.dic_solver.get())
                smooth_sigma = float(self.dic_smooth_sigma.get())
                if subset < 9:
                    items.append(self._preflight_item("block", "子集尺寸", "子集尺寸必须 >= 9。"))
                elif subset % 2 == 0:
                    items.append(self._preflight_item("block", "子集尺寸", "子集尺寸必须为奇数；当前偶数值会改变实际窗口。"))
                elif step < 1:
                    items.append(self._preflight_item("block", "步长", "步长必须 >= 1。"))
                elif window < 3:
                    items.append(self._preflight_item("block", "应变窗口", "应变窗口必须 >= 3。"))
                elif window % 2 == 0:
                    items.append(self._preflight_item("block", "应变窗口", "应变窗口必须为奇数；当前偶数值会改变实际窗口。"))
                elif not math.isfinite(smooth_sigma) or smooth_sigma < 0:
                    items.append(self._preflight_item("block", "高斯平滑", "高斯平滑 σ 必须为非负有限数字。"))
                elif solver not in DIC_SOLVERS:
                    items.append(self._preflight_item("block", "求解器", "请选择 IC-GN 或 IC-LM。"))
                elif field_roi is not None and ref_img8 is not None:
                    X, Y = build_poi_grid(field_roi, subset, step, ref_img8.shape)
                    if not poi_grid_is_usable(X, Y, min_rows=3, min_cols=3):
                        items.append(self._preflight_item("block", "POI 网格", "当前 ROI / 子集 / 步长至少需要 3×3 个可分析的 2D POI。"))
                    else:
                        items.append(
                            self._preflight_item(
                                "ok",
                                "DIC 参数",
                                f"subset={subset} px, step={step} px, window={window}, {solver}；POI={X.size}。",
                            )
                        )
                else:
                    items.append(self._preflight_item("ok", "DIC 参数", f"subset={subset} px, step={step} px, window={window}, {solver}。"))
            except (tk.TclError, TypeError, ValueError):
                items.append(self._preflight_item("block", "DIC 参数", "子集、步长、应变窗口和高斯平滑必须是有效数字。"))

            try:
                search_radius = int(self.dic_search_radius.get())
                zncc_min = float(self.dic_zncc_min.get())
                pyramid_levels = int(self.dic_pyramid_levels.get())
                pyramid_scale = float(self.dic_pyramid_scale.get())
                strain_degree = int(self.dic_strain_degree.get())
                outlier_threshold = float(self.dic_outlier_threshold.get())
            except (tk.TclError, TypeError, ValueError):
                items.append(self._preflight_item("block", "DIC 高级参数", "搜索半径、ZNCC 下限、金字塔、拟合阶次与异常位移阈值必须是有效数字。"))
            else:
                if search_radius < 1:
                    items.append(self._preflight_item("block", "DIC 搜索半径", "DIC 搜索半径必须 > 0。"))
                if not math.isfinite(zncc_min) or not (0.0 <= zncc_min <= 1.0):
                    items.append(self._preflight_item("block", "ZNCC 下限", "ZNCC 下限必须位于 0 到 1 之间。"))
                if not (1 <= pyramid_levels <= 8):
                    items.append(self._preflight_item("block", "金字塔层数", "金字塔层数必须位于 1 到 8 之间。"))
                if not math.isfinite(pyramid_scale) or not (0.0 < pyramid_scale < 1.0):
                    items.append(self._preflight_item("block", "金字塔缩放", "金字塔缩放必须位于 0 到 1 之间（不含端点）。"))
                if strain_degree not in (1, 2):
                    items.append(self._preflight_item("block", "拟合阶次", "拟合最高阶次必须为 1 或 2。"))
                if not math.isfinite(outlier_threshold) or outlier_threshold < 0.0:
                    items.append(self._preflight_item("block", "异常位移阈值", "异常位移阈值必须为非负有限数字。"))

            try:
                mask_settings = self.dic_mask_settings()
            except RuntimeError as exc:
                items.append(self._preflight_item("block", "试样遮罩", str(exc)))
            else:
                if mask_settings["mode"] == "file":
                    mask_path = mask_settings.get("path") or ""
                    if not mask_path or not Path(mask_path).is_file():
                        items.append(self._preflight_item("block", "试样遮罩", "导入遮罩模式需要选择有效的遮罩图像文件。"))

            try:
                self.dic_display_options()
            except RuntimeError as exc:
                items.append(self._preflight_item("block", "显示设置", str(exc)))
        elif self.roi_groups:
            items.append(self._preflight_item("ok", "ROI 组", f"已添加 {len(self.roi_groups)} 组。"))
            if ref_img8 is not None:
                for group in self.roi_groups:
                    for label, rect in (("ROI1", group.get("roi1")), ("ROI2", group.get("roi2"))):
                        metrics, texture_code = self._core_texture_metrics_and_code(rect, image=ref_img8)
                        if texture_code is None:
                            continue
                        level = "block" if texture_code == "AMBIGUOUS_TEXTURE" else "warn"
                        items.append(
                            self._preflight_item(
                                level,
                                "纹理",
                                f"{group['name']} {label}: {texture_code}，结构张量比值={metrics['structure_tensor_ratio']:.4f}。",
                            )
                        )
        else:
            items.append(self._preflight_item("block", "ROI 组", "请至少添加一组 ROI1/ROI2。"))

        if (not is_ff) and self.roi_groups and start is not None:
            missing_ref = [
                g["name"]
                for g in self.roi_groups
                if g.get("reference_frame_1based") is None
            ]
            mismatched = [
                g["name"]
                for g in self.roi_groups
                if g.get("reference_frame_1based") is not None and int(g.get("reference_frame_1based")) != start
            ]
            if missing_ref:
                items.append(
                    self._preflight_item(
                        "block",
                        "参考帧",
                        f"ROI 组 {', '.join(missing_ref)} 缺少参考帧记录，请载入后在当前起始/参考帧重画并更新。",
                    )
                )
            elif mismatched:
                items.append(
                    self._preflight_item(
                        "block",
                        "参考帧",
                        f"ROI 组 {', '.join(mismatched)} 不是在当前起始/参考帧上定义的，请载入后重画或恢复参考帧。",
                    )
                )
            else:
                items.append(self._preflight_item("ok", "参考帧", "ROI 组与当前起始/参考帧一致。"))

        if (not is_ff) and self.roi_groups:
            invalid_l0 = [g["name"] for g in self.roi_groups if g.get("L0", 0) <= 0 or not np.isfinite(float(g.get("L0", np.nan)))]
            small_l0 = [g for g in self.roi_groups if g.get("L0", 0) > 0 and g.get("L0", 0) < 50]
            if invalid_l0:
                items.append(self._preflight_item("block", "L0", f"ROI 组 {', '.join(invalid_l0)} 的 L0 无效。"))
            elif small_l0:
                detail = ", ".join(f"{g['name']}={g['L0']:.1f}px" for g in small_l0)
                items.append(self._preflight_item("warn", "L0", f"L0 偏小：{detail}；应变噪声会被放大。"))
            else:
                min_l0 = min(float(g["L0"]) for g in self.roi_groups)
                items.append(self._preflight_item("ok", "L0", f"最小 L0={min_l0:.1f} px。"))

            actual_modes = sorted({str(g.get("actual_mode", "unknown")) for g in self.roi_groups})
            if "distance" in actual_modes:
                items.append(self._preflight_item("warn", "应变方向", "存在 distance 方向；请确认倾斜标距的物理意义。"))
            else:
                items.append(self._preflight_item("ok", "应变方向", "方向已解析为 " + ", ".join(actual_modes) + "。"))

        if not is_ff:
            try:
                poisson_enabled = validate_poisson_role_groups(self.roi_groups)
            except RuntimeError as exc:
                items.append(self._preflight_item("warn", "泊松比角色", str(exc).replace("\n", " ")))
            else:
                if poisson_enabled:
                    axial, transverse = get_poisson_role_groups(self.roi_groups)
                    if axial and transverse and axial[0].get("actual_mode") == transverse[0].get("actual_mode"):
                        items.append(self._preflight_item("warn", "泊松比角色", "轴向和横向组的 actual_mode 相同，请复核方向。"))
                    else:
                        items.append(self._preflight_item("ok", "泊松比角色", "轴向/横向角色已成对配置。"))
                else:
                    items.append(self._preflight_item("ok", "泊松比角色", "未启用泊松比导出角色。"))

        output_text = self.output_folder.get().strip()
        if not output_text:
            items.append(self._preflight_item("block", "输出目录", "请设置输出文件夹。"))
        else:
            output_path = Path(output_text)
            if output_path.exists() and not output_path.is_dir():
                items.append(self._preflight_item("block", "输出目录", "输出路径已存在但不是文件夹。"))
            else:
                items.append(self._preflight_item("ok", "输出目录", str(output_path)))

        if is_ff:
            items.append(self._preflight_item("ok", "全场输出", "固定保存数值表、完整数值文件与九个分量图；可选叠加图。"))
        elif self.has_any_export_option():
            items.append(self._preflight_item("ok", "导出选项", "至少已选择一种导出内容。"))
        else:
            items.append(self._preflight_item("block", "导出选项", "请至少选择一种导出内容。"))

        if update_display and hasattr(self, "preflight_summary_var"):
            self.preflight_summary_var.set(self.format_preflight_items(items))
        return items

    def format_preflight_items(self, items):
        prefix = {"ok": "通过", "warn": "警告", "block": "阻止"}
        return "\n".join(f"[{prefix.get(item['level'], item['level'])}] {item['label']}：{item['message']}" for item in items)

    def refresh_preflight_panel(self):
        return self.build_preflight_items(update_display=True)

    def _safe_texture_thresholds(self):
        """Read 1D texture thresholds safely, falling back to core defaults in fullfield or mid-edit."""
        if self.is_fullfield_mode():
            return 8.0, 25.0, 0.20
        defaults = (8.0, 25.0, 0.20)
        vars_ = (self.min_texture_std, self.min_texture_contrast, self.max_saturated_frac)
        resolved = []
        for var, default in zip(vars_, defaults):
            try:
                val = float(var.get())
                resolved.append(val if math.isfinite(val) else default)
            except (tk.TclError, TypeError, ValueError):
                resolved.append(default)
        return tuple(resolved)

    def _roi_texture_status_text(self, rect):
        img8 = self.preview_img8
        if img8 is None or rect is None:
            return "未检查"
        metrics = roi_texture_metrics(img8, rect)
        min_std, min_contrast, max_sat = self._safe_texture_thresholds()
        ok = texture_is_ok(metrics, min_std, min_contrast, max_sat)
        status = "良好" if ok else "偏弱"
        return f"{status}(std={metrics['std_gray']:.1f}, P95-P5={metrics['contrast_p95_p5']:.1f})"

    def build_current_roi_summary(self):
        if self.is_fullfield_mode():
            if self.field_roi is None:
                return "尚未绘制全场 ROI。请在当前参考帧上拖出矩形分析区域。"
            x, y, w, h = self.field_roi
            ref = self.field_roi_reference_frame_1based
            ref_text = f"；参考帧={ref}" if ref is not None else "；参考帧未记录"
            return f"全场 ROI {w}×{h} px @ ({x},{y}){ref_text}。"
        if self.roi1 is None and self.roi2 is None:
            return "尚未绘制 ROI。请在参考帧上依次绘制 ROI1 和 ROI2。"
        if self.roi1 is None:
            return "ROI1 未绘制；请先绘制 ROI1。"
        if self.roi2 is None:
            _, _, w, h = self.roi1
            return f"ROI1 {w}×{h} px；ROI2 未绘制；ROI1 纹理：{self._roi_texture_status_text(self.roi1)}。"

        _, _, w1, h1 = self.roi1
        _, _, w2, h2 = self.roi2
        actual_mode = resolve_strain_mode(self.roi1, self.roi2, self.strain_mode.get())
        dx, dy, dist = roi_separation(self.roi1, self.roi2)
        l0 = length_between(self.roi1, self.roi2, actual_mode)
        return (
            f"ROI1 {w1}×{h1} px；ROI2 {w2}×{h2} px；"
            f"dx={dx:.1f} px, dy={dy:.1f} px, distance={dist:.1f} px, L0={l0:.1f} px；"
            f"方向={actual_mode}；纹理：ROI1 {self._roi_texture_status_text(self.roi1)}，"
            f"ROI2 {self._roi_texture_status_text(self.roi2)}。"
        )

    def refresh_current_roi_summary(self):
        if hasattr(self, "current_roi_summary_var"):
            self.current_roi_summary_var.set(self.build_current_roi_summary())

    def format_qc_overview(self, summary):
        if not summary or not summary.get("groups"):
            return "分析完成后显示 QC 总览。"

        groups = summary["groups"]
        total_frames = int(sum(item["frames"] for item in groups.values()))
        rejected = int(summary["overall"].get("rejected_frames", 0))
        adaptive = int(summary["overall"].get("adaptive_accepted_frames", 0))
        rejected_ratio = rejected / total_frames if total_frames else 0.0
        levels = {"Good": 0, "Warning": 1, "Poor": 2}

        def group_key(pair):
            _name, item = pair
            ratio = item["rejected_frames"] / item["frames"] if item["frames"] else 0.0
            return levels.get(item["qc_level"], 0), ratio, item["adaptive_accepted_frames"]

        worst_name, worst = max(groups.items(), key=group_key)
        review_frames = sorted({int(frame) for frame in worst.get("rejected_frame_list", [])})
        review_text = ", ".join(str(frame) for frame in review_frames[:12]) if review_frames else "无"
        if len(review_frames) > 12:
            review_text += " ..."

        return (
            f"QC 总览：{summary['overall']['qc_level']}；ROI 组={summary['overall']['groups']}；"
            f"拒绝帧比例={rejected}/{total_frames} ({rejected_ratio:.1%})；自适应接受={adaptive}。\n"
            f"最需复核：{worst_name}（{worst['qc_level']}，拒绝 {worst['rejected_frames']}/{worst['frames']}）。\n"
            f"建议复核帧：{review_text}。"
        )

    def update_qc_overview(self, summary):
        self.last_qc_summary = summary
        if hasattr(self, "qc_overview_var"):
            self.qc_overview_var.set(self.format_qc_overview(summary))

    def format_fullfield_qc_overview(self, frames):
        valid_count = sum(frame.get("status") == "scientific_valid" for frame in frames)
        failed = [str(frame.get("frame_global_1based", "未知")) for frame in frames
                  if frame.get("status") != "scientific_valid"]
        def minimum_fraction(key):
            values = [float(frame[key]) for frame in frames
                      if isinstance(frame.get(key), (int, float)) and np.isfinite(frame[key])]
            return f"{min(values):.1%}" if values else "未提供"
        correlation = minimum_fraction("correlation_valid_fraction")
        strain = minimum_fraction("strain_valid_fraction")
        failed_text = "、".join(failed[:12]) + ("…" if len(failed) > 12 else "") if failed else "无"
        return (f"全场 DIC：{valid_count}/{len(frames)} 个变形帧通过有效性检查。\n"
                f"已计算帧的最低有效点比例：相关 {correlation}；应变 {strain}。\n"
                f"未通过的帧：{failed_text}。")

    def get_int_setting(self, var, label):
        try:
            return int(var.get())
        except (tk.TclError, TypeError, ValueError):
            raise RuntimeError(f"{label}必须是整数。")

    def get_float_setting(self, var, label):
        try:
            value = float(var.get())
        except tk.TclError as exc:
            if "not a number" in str(exc).lower():
                raise RuntimeError(f"{label}必须是有限数字。") from exc
            raise RuntimeError(f"{label}必须是数字。") from exc
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{label}必须是数字。") from exc
        if not math.isfinite(value):
            raise RuntimeError(f"{label}必须是有限数字。")
        return value

    def build_ui(self):
        self.root.configure(background=self.ui_bg)
        self.main_frame = ttk.Frame(self.root, style="App.TFrame", padding=(16, 12, 16, 12))
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(2, weight=1)

        self._build_app_header(self.main_frame)
        self._build_project_section(self.main_frame)
        self._build_workspace(self.main_frame)
        self.update_workflow_action_states()

    def _build_app_header(self, parent):
        workbench_menu = tk.Menu(self.root, tearoff=0)
        capture_menu = tk.Menu(workbench_menu, tearoff=0)
        capture_menu.add_command(label="打开采集实验…", command=self._choose_capture_experiment)
        capture_menu.add_command(label="跟随采集与在线分析…", command=lambda: self._choose_capture_experiment(live=True))
        capture_menu.add_command(label="打开 Capture 工作台", command=self._open_capture_workbench)
        workbench_menu.add_cascade(label="采集联动", menu=capture_menu)
        self.root.configure(menu=workbench_menu)
        header = ttk.Frame(parent, style="App.TFrame")
        self.app_header = header
        header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="StrainTrace", style="Brand.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="分析控制台 · 序列、参数与质量", style="AppHint.TLabel").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self.open_visual_button = ttk.Button(
            header, text="图像与结果 ↗", command=self.show_visual_window, style="Secondary.TButton")
        self.open_visual_button.grid(row=0, column=1, sticky="e", padx=(8, 6))
        self.add_tooltip(self.open_visual_button, "点击打开独立的图像与结果窗口（Ctrl + I），查看图片、绘制 ROI 或检查结果。关闭该窗口不会清空数据或停止分析。")
        self.open_recent_output_button = ttk.Button(
            header, text="打开输出", command=self.open_recent_output_folder, style="Compact.TButton")
        self.open_recent_output_button.grid(row=0, column=3, padx=(0, 6))
        self.dark_mode_btn = ttk.Button(header, text="暗色模式", command=self.toggle_dark_mode, style="Compact.TButton")
        self.dark_mode_btn.grid(row=0, column=2, padx=(0, 6))
        self.usage_notice_button = ttk.Button(header, text="关于 / 引用", command=self.show_usage_notice, style="Compact.TButton")
        self.usage_notice_button.grid(row=0, column=4)
        self.add_tooltip(self.usage_notice_button, "点击查看版本、作者和使用说明，以及论文中引用本软件时可用的信息。")
        self.add_tooltip(self.dark_mode_btn, "点击切换浅色或深色界面，结果图也会换色。")
        self.add_tooltip(self.open_recent_output_button, "点击在文件管理器中打开最近选择的结果文件夹，查看已保存的表格和图片。")
        self.update_recent_output_button_state()

    def _build_project_section(self, parent):
        self.project_frame = ttk.LabelFrame(parent, text="01  图像序列", padding=(12, 8))
        self.project_frame.grid(row=1, column=0, sticky="ew")
        self.project_frame.columnconfigure(1, weight=1)
        self.project_frame.columnconfigure(3, weight=0)

        image_folder_tip = (
            "填写连续拍摄同一样品、同一位置的图片文件夹路径。加载时会按文件名中的数字排序，如 1、2、10。"
        )
        self.image_folder_label = ttk.Label(self.project_frame, text="图像文件夹：", style="Key.TLabel")
        self.image_folder_label.grid(row=0, column=0, sticky="w", pady=1)
        self.add_tooltip(self.image_folder_label, image_folder_tip)
        self.image_folder_entry = ttk.Entry(self.project_frame, textvariable=self.image_folder, width=42)
        self.image_folder_entry.grid(
            row=0, column=1, sticky="ew", padx=(6, 8), pady=1
        )
        self.add_tooltip(self.image_folder_entry, image_folder_tip)
        self.select_image_button = ttk.Button(
            self.project_frame,
            text="选择图像",
            command=self.select_image_folder,
            style="Secondary.TButton",
        )
        self.select_image_button.grid(row=0, column=2, sticky="ew", pady=1)
        self.add_tooltip(self.select_image_button, "点击选择存放连续拍摄图片的文件夹，再点“加载序列”显示图片。")

        output_folder_tip = (
            "填写结果文件夹路径，分析后的表格和图片会保存在这里。开始分析前必须设置；也可点击“选择输出”设置保存位置。"
        )
        self.output_folder_label = ttk.Label(self.project_frame, text="输出文件夹：")
        self.output_folder_label.grid(row=1, column=0, sticky="w", pady=1)
        self.add_tooltip(self.output_folder_label, output_folder_tip)
        self.output_folder_entry = ttk.Entry(self.project_frame, textvariable=self.output_folder, width=42)
        self.output_folder_entry.grid(
            row=1, column=1, sticky="ew", padx=(6, 8), pady=1
        )
        self.add_tooltip(self.output_folder_entry, output_folder_tip)
        self.select_output_button = ttk.Button(
            self.project_frame,
            text="选择输出",
            command=self.select_output_folder,
            style="Secondary.TButton",
        )
        self.select_output_button.grid(row=1, column=2, sticky="ew", pady=1)
        self.add_tooltip(self.select_output_button, "点击选择分析结果的保存位置。选好后，顶部“打开输出”可打开这个文件夹。")

        self.load_images_button = ttk.Button(
            self.project_frame,
            text="加载序列",
            command=self.load_first_image,
            style="Secondary.TButton",
        )
        self.load_images_button.grid(row=0, column=3, rowspan=2, padx=(8, 0), pady=1, sticky="nsew")
        self.add_tooltip(
            self.load_images_button,
            "点击加载所选文件夹的图片并显示预览（Ctrl + L）。首次加载会选中全部图片；更换图片序列会清空原有测量框和分组。",
        )

    def _build_workspace(self, parent):
        workspace = ttk.Frame(parent, style="App.TFrame")
        self.control_workspace = workspace
        workspace.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
        workspace.columnconfigure(0, weight=1)
        workspace.rowconfigure(0, weight=1)
        self.control_notebook = ttk.Notebook(workspace)
        self.control_notebook.grid(row=0, column=0, sticky="nsew")
        self.settings_page = ttk.Frame(self.control_notebook, style="Card.TFrame")
        self.quality_page = ttk.Frame(self.control_notebook, style="Card.TFrame", padding=12)
        for page in (self.settings_page, self.quality_page):
            page.columnconfigure(0, weight=1)
            page.rowconfigure(0, weight=1)
        self.control_notebook.add(self.settings_page, text="分析设置")
        self.control_notebook.add(self.quality_page, text="质量与日志")
        # Campaign support is a separate workflow surface. Import only its Tk
        # page here; the campaign service layer remains lazy until an action is
        # started, preserving the legacy single-sequence import path.
        from straintrace_campaign.gui import CampaignPage

        self.campaign_page = CampaignPage(self.control_notebook, app=self)
        self.control_notebook.add(self.campaign_page, text="批次项目")
        from straintrace_agent.gui import AgentPage

        self.agent_page = AgentPage(self.control_notebook, app=self)
        self.control_notebook.add(self.agent_page, text="自动分析")
        self.add_tooltip(self.control_notebook, self._control_workspace_help)

        self.visual_window = tk.Toplevel(self.root)
        self.visual_window.title("StrainTrace · 图像与结果")
        self.visual_window.configure(background=self.ui_bg)
        self.visual_window.protocol("WM_DELETE_WINDOW", self.hide_visual_window)
        visual_frame = ttk.Frame(self.visual_window, style="App.TFrame", padding=(16, 12))
        visual_frame.pack(fill=tk.BOTH, expand=True)
        visual_frame.columnconfigure(0, weight=1)
        visual_frame.rowconfigure(1, weight=1)
        header = ttk.Frame(visual_frame, style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="图像与结果", style="Brand.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="ROI · 遮罩 · 位移 · 应变", style="AppHint.TLabel").grid(
            row=0, column=1, sticky="w", padx=(16, 8))
        self.arrange_windows_button = ttk.Button(
            header, text="排列窗口", command=self.arrange_windows, style="Compact.TButton")
        self.arrange_windows_button.grid(row=0, column=2, padx=(0, 6))
        self.add_tooltip(self.arrange_windows_button, "点击把控制台和图像窗口排列到当前屏幕内；屏幕足够宽时并排显示。之后仍可独立移动、缩放或放到另一块屏幕。")
        self.return_control_button = ttk.Button(
            header, text="分析控制台 ↗", command=self.show_control_window, style="Secondary.TButton")
        self.return_control_button.grid(row=0, column=3)
        self.add_tooltip(self.return_control_button, "点击回到分析控制台，调整图片范围、计算参数或查看质量与日志。图像窗口的位置和内容保留。")
        self.workspace_notebook = ttk.Notebook(visual_frame)
        self.workspace_notebook.grid(row=1, column=0, sticky="nsew")
        self.image_page = ttk.Frame(self.workspace_notebook, style="Card.TFrame")
        self.results_page = ttk.Frame(self.workspace_notebook, style="Card.TFrame")
        self.campaign_visual_page = ttk.Frame(self.workspace_notebook, style="Card.TFrame")
        for page in (self.image_page, self.results_page):
            page.columnconfigure(0, weight=1)
            page.rowconfigure(0, weight=1)
        self.workspace_notebook.add(self.image_page, text="图像与 ROI")
        self.workspace_notebook.add(self.results_page, text="分析结果")
        self.campaign_visual_page.columnconfigure(0, weight=1)
        self.campaign_visual_page.rowconfigure(1, weight=1)
        ttk.Label(self.campaign_visual_page, text="批次 DIC 参考帧与已审核取点").grid(
            row=0, column=0, sticky="w", padx=10, pady=(8, 2))
        self.campaign_preview_info_var = tk.StringVar(value="请在控制台的“批次项目”页选择批次和参考帧。")
        ttk.Label(self.campaign_visual_page, textvariable=self.campaign_preview_info_var).grid(
            row=2, column=0, sticky="w", padx=10, pady=(2, 8))
        self.campaign_preview_canvas = tk.Canvas(
            self.campaign_visual_page, background="#111827", height=410, highlightthickness=0)
        self.campaign_preview_canvas.grid(row=1, column=0, sticky="nsew", padx=6)
        self.add_tooltip(self.campaign_preview_canvas,
                         "查看批次参考帧及已审核 ROI；取点坐标对应参考图像，图像或几何变化后需重新审核。")
        self.campaign_preview_canvas.bind(
            "<Configure>", lambda _event: self.campaign_page.render_roi_preview()
            if hasattr(self, "campaign_page") else None)
        self.workspace_notebook.add(self.campaign_visual_page, text="批次图像")
        self.campaign_result_visual_page = ttk.Frame(self.workspace_notebook, style="Card.TFrame")
        self.campaign_result_visual_page.columnconfigure(0, weight=1)
        self.campaign_result_visual_page.rowconfigure(1, weight=1)
        ttk.Label(self.campaign_result_visual_page, text="已对齐的 Frame-strain 曲线与结果叠图").grid(
            row=0, column=0, sticky="w", padx=10, pady=(8, 2))
        campaign_results = ttk.Panedwindow(self.campaign_result_visual_page, orient=tk.HORIZONTAL)
        campaign_results.grid(row=1, column=0, sticky="nsew", padx=6)
        campaign_plot_frame = ttk.LabelFrame(campaign_results, text="Frame-strain")
        campaign_overlay_frame = ttk.LabelFrame(campaign_results, text="DIC 叠图附件")
        campaign_results.add(campaign_plot_frame, weight=2)
        campaign_results.add(campaign_overlay_frame, weight=1)
        campaign_plot_frame.columnconfigure(0, weight=1)
        campaign_plot_frame.rowconfigure(0, weight=1)
        self.campaign_result_canvas = tk.Canvas(campaign_plot_frame, background=self.card_bg, height=390,
                                                highlightthickness=0)
        self.campaign_result_canvas.grid(row=0, column=0, sticky="nsew")
        self.add_tooltip(self.campaign_result_canvas,
                         "查看所选批次的 Frame 应变曲线；只绘制有效有限值，完整逐帧数据和质量原因见项目结果文件。")
        self.campaign_result_canvas.bind(
            "<Configure>", lambda _event: self.campaign_page._draw_result_curve()
            if hasattr(self, "campaign_page") else None)
        campaign_overlay_frame.columnconfigure(0, weight=1)
        campaign_overlay_frame.rowconfigure(0, weight=1)
        self.campaign_result_overlay_canvas = tk.Canvas(
            campaign_overlay_frame, background="#111827", height=390, highlightthickness=0)
        self.campaign_result_overlay_canvas.grid(row=0, column=0, sticky="nsew")
        self.add_tooltip(self.campaign_result_overlay_canvas,
                         "查看所选分析阶段输出的关键帧叠图；无叠图附件时显示空状态。")
        self.campaign_result_overlay_canvas.bind(
            "<Configure>", lambda _event: self.campaign_page._redraw_result_overlay()
            if hasattr(self, "campaign_page") else None)
        self.campaign_result_info_var = tk.StringVar(value="选择有结果的批次后显示可用曲线和叠图。")
        ttk.Label(self.campaign_result_visual_page, textvariable=self.campaign_result_info_var).grid(
            row=2, column=0, sticky="w", padx=10, pady=(2, 8))
        self.workspace_notebook.add(self.campaign_result_visual_page, text="批次结果")
        self.agent_page.attach_visual(self.workspace_notebook)
        self.workspace_notebook.bind("<<NotebookTabChanged>>", self._on_workspace_tab_changed)
        self.add_tooltip(self.workspace_notebook, self._workspace_help)
        self._build_image_section(self.image_page)
        self._build_scrollable_controls(self.settings_page)
        self.control_notebook.bind("<<NotebookTabChanged>>", self._on_control_tab_changed)
        ttk.Label(visual_frame, textvariable=self.status_var, style="AppHint.TLabel",
                  wraplength=700, justify=tk.LEFT).grid(row=2, column=0, sticky="ew", pady=(8, 0))

    def _on_control_tab_changed(self, _event=None):
        """The automatic and campaign tabs own their own source and run controls."""
        selected = self.control_notebook.select()
        standalone = selected in (str(self.agent_page), str(self.campaign_page))
        for frame in (self.project_frame, self.run_frame):
            frame.grid_remove() if standalone else frame.grid()

    def _control_workspace_help(self, event=None):
        try:
            index = (self.control_notebook.index(f"@{event.x},{event.y}")
                     if event is not None and getattr(event, "keysym", "") != "F1"
                     else self.control_notebook.index("current"))
        except tk.TclError:
            return ""
        help_text = (
            "点击设置分析模式、参考图片、ROI 和计算参数；绘图操作在独立的图像窗口中进行。",
            "点击查看分析前缺少什么、哪些图片未能算出结果，以及运行中的详细记录。",
            "管理多个采集批次的身份、ROI 审核、时间与载荷对齐，以及可恢复的项目运行。",
            "发现实验输入，人工编辑多个统计 ROI、虚拟标距与参数，生成计划并运行；查看质量状态和报告。",
        )
        return help_text[index] if 0 <= index < len(help_text) else ""

    def show_control_window(self):
        self.root.deiconify()
        self.root.lift()

    def show_visual_window(self):
        self.visual_window.deiconify()
        self.visual_window.lift()
        self.root.after_idle(self._rescale_display_to_current_size)

    def hide_visual_window(self):
        self.visual_window.withdraw()

    def arrange_windows(self):
        """Place both independently resizable windows inside the primary screen."""
        screen_w, screen_h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        control_min_w, control_min_h = self.root.minsize()
        visual_min_w, visual_min_h = self.visual_window.minsize()
        control_w = max(control_min_w, min(650, screen_w - 80))
        height = max(control_min_h, visual_min_h, min(900, screen_h - 100))
        if screen_w >= control_w + visual_min_w + 60:
            visual_w = min(1360, screen_w - control_w - 60)
            visual_x = control_w + 40
        else:
            visual_w = max(visual_min_w, min(1120, screen_w - 80))
            visual_x = max(20, screen_w - visual_w - 20)
        self.root.geometry(f"{control_w}x{height}+20+30")
        self.visual_window.geometry(f"{visual_w}x{height}+{visual_x}+30")
        self.root.deiconify()
        self.visual_window.deiconify()

    def _build_scrollable_controls(self, parent):
        sidebar_width = round(370 * max(1.0, self.ui_scaling / (96 / 72)))
        self.controls_frame = ttk.Frame(parent, style="Card.TFrame", width=sidebar_width)
        self.controls_frame.grid(row=0, column=0, sticky="nsew")
        self.controls_frame.grid_propagate(False)
        self.controls_frame.columnconfigure(0, weight=1)
        self.controls_frame.rowconfigure(1, weight=1)

        self._build_workflow_guide(self.controls_frame)

        self.controls_canvas = tk.Canvas(
            self.controls_frame,
            width=sidebar_width,
            height=560,
            bg=self.card_bg,
            highlightthickness=0,
            highlightbackground=self.border_color,
            borderwidth=0,
        )
        controls_scrollbar = ttk.Scrollbar(self.controls_frame, orient=tk.VERTICAL, command=self.controls_canvas.yview)
        self.controls_canvas.configure(yscrollcommand=controls_scrollbar.set)
        self.controls_canvas.grid(row=1, column=0, sticky="nsew")
        controls_scrollbar.grid(row=1, column=1, sticky="ns")
        self.add_tooltip(controls_scrollbar, "上下拖动查看被隐藏的设置。在控制台的设置区滚动鼠标滚轮，也可上下移动。")
        self.add_tooltip(self.controls_canvas, "在这里滚动鼠标滚轮，查看下方的测量和导出设置。")

        self.controls_panel = ttk.Frame(self.controls_canvas, style="Card.TFrame", padding=(10, 6, 10, 12))
        self.controls_panel.columnconfigure(0, weight=1)
        self.controls_window = self.controls_canvas.create_window((0, 0), window=self.controls_panel, anchor="nw")
        self.controls_panel.bind(
            "<Configure>",
            lambda _event: self.controls_canvas.configure(scrollregion=self.controls_canvas.bbox("all")),
        )
        self.controls_canvas.bind(
            "<Configure>",
            self._on_controls_configure,
        )
        self.workflow_canvas = self.controls_canvas
        self.workflow_panel = self.controls_panel

        self._build_measure_section(self.controls_panel)
        self._build_fullfield_section(self.controls_panel)
        self._build_roi_section(self.controls_panel)
        self._build_analysis_section(self.controls_panel)
        self._build_run_section(self.main_frame, row=3)
        self._build_quality_section(self.quality_page)
        self._build_results_section(self.results_page)
        self._bind_workflow_scroll_handler()
        self.set_analysis_mode()

    def _on_controls_configure(self, event):
        self.controls_canvas.itemconfigure(self.controls_window, width=event.width)
        wrap = max(180, event.width - 52)
        for label_name in ("preset_status_label", "current_roi_summary_label", "dic_field_summary_label",
                           "export_hint_label", "fullfield_export_info_label", "workflow_hint_label", "status_label"):
            label = getattr(self, label_name, None)
            if label is not None:
                label.configure(wraplength=wrap)

    def _on_workspace_tab_changed(self, _event=None):
        if self.workspace_notebook.select() == str(self.image_page):
            self.root.after_idle(self._rescale_display_to_current_size)

    def _show_image_workspace(self):
        if hasattr(self, "workspace_notebook"):
            self.show_visual_window()
            self.workspace_notebook.select(self.image_page)

    def _bind_workflow_scroll_handler(self):
        """Enable mouse-wheel scrolling inside the right workflow panel."""
        seen = set()

        def bind_tree(widget):
            widget_key = str(widget)
            if widget_key in seen:
                return
            seen.add(widget_key)
            widget.bind("<MouseWheel>", self._on_workflow_mouse_wheel, add="+")
            widget.bind("<Button-4>", self._on_workflow_mouse_wheel, add="+")
            widget.bind("<Button-5>", self._on_workflow_mouse_wheel, add="+")

            for child in widget.winfo_children():
                bind_tree(child)

        bind_tree(self.controls_canvas)
        bind_tree(self.controls_panel)
        if hasattr(self, "workflow_guide_frame"):
            bind_tree(self.workflow_guide_frame)

    def _on_workflow_mouse_wheel(self, event):
        """Scroll the workflow panel; image-canvas wheel events keep zoom behavior."""
        delta = getattr(event, "delta", 0)
        button_num = getattr(event, "num", 0)
        if button_num == 4:
            amount = -1
        elif button_num == 5:
            amount = 1
        elif delta > 0:
            amount = -max(1, abs(int(delta / 120)))
        elif delta < 0:
            amount = max(1, abs(int(delta / 120)))
        else:
            return "break"

        if event.widget == getattr(self, "group_tree", None) and self.group_tree.yview() != (0.0, 1.0):
            self.group_tree.yview_scroll(amount, "units")
        else:
            self.controls_canvas.yview_scroll(amount, "units")
        return "break"

    def _build_workflow_guide(self, parent):
        self.workflow_guide_frame = ttk.Frame(parent, style="Card.TFrame", padding=(12, 10))
        self.workflow_guide_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.workflow_guide_frame.columnconfigure(0, weight=1)

        self.workflow_step_texts = [
            "1. 选择图像文件夹和输出文件夹，加载序列",
            "2. 设置参考帧、分析范围和测量方向",
            "3. 画 ROI1/ROI2，添加 ROI 组",
            "4. 确认导出内容，点击开始分析",
        ]
        self.workflow_labels = []
        self.workflow_steps_label = ttk.Label(
            self.workflow_guide_frame,
            text=self._visible_workflow_steps_text(),
            style="Hint.TLabel",
            font=(self.ui_font_family, 9),
            justify=tk.LEFT,
            wraplength=430,
        )
        self.workflow_steps_label.grid(row=0, column=0, sticky="ew")
        self.workflow_labels.append(self.workflow_steps_label)
        self.add_tooltip(
            self.workflow_steps_label,
            lambda _event: ("先加载图片，选一张作为比较基准，再画要分析的矩形区域，最后开始计算整片区域的移动和伸缩。"
                            if self.is_fullfield_mode() else
                            "先加载图片，选一张作为比较基准，再画两个测量框并添加为一组，最后计算它们之间的距离变化。"),
        )

        self.workflow_hint_var = tk.StringVar(value="请选择并加载图像序列。")
        self.workflow_guide_frame.bind(
            "<Configure>",
            lambda event: self._sync_workflow_guide_wraplength(event.width),
            add="+",
        )

    def _sync_workflow_guide_wraplength(self, width):
        wrap = max(int(width) - 48, 220)
        self.workflow_steps_label.configure(wraplength=wrap)

    def _visible_workflow_steps_text(self):
        """Keep the current workflow visible without repeating the instructions."""
        if str(self.analysis_mode.get()) == ANALYSIS_MODE_FULLFIELD:
            return "选参考图片 → 在图像窗口画分析区域 → 开始分析 · 悬停或按 F1 查看说明"
        return "选参考图片 → 在图像窗口画两个测量框 → 开始分析 · 悬停或按 F1 查看说明"

    def _build_measure_section(self, parent):
        mode_bar = ttk.Frame(parent, style="Card.TFrame")
        mode_bar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        mode_bar.columnconfigure((0, 1), weight=1, uniform="mode")
        self.mode_extensometer_radio = ttk.Radiobutton(
            mode_bar,
            text="虚拟引伸计",
            variable=self.analysis_mode,
            value=ANALYSIS_MODE_EXTENSOMETER,
            command=self.set_analysis_mode,
            style="Mode.TRadiobutton",
        )
        self.mode_extensometer_radio.grid(row=0, column=0, sticky="ew")
        self.add_tooltip(
            self.mode_extensometer_radio,
            "选择后，在参考图片上画两个测量框（ROI），再添加为一组。程序追踪这两个框，画出它们之间的距离随时间伸长或缩短的曲线。",
        )
        self.mode_fullfield_radio = ttk.Radiobutton(
            mode_bar,
            text="全场 2D DIC",
            variable=self.analysis_mode,
            value=ANALYSIS_MODE_FULLFIELD,
            command=self.set_analysis_mode,
            style="Mode.TRadiobutton",
        )
        self.mode_fullfield_radio.grid(row=0, column=1, sticky="ew")
        self.add_tooltip(
            self.mode_fullfield_radio,
            "选择后，在参考图片上画一片矩形区域（ROI）。DIC 是比较图片纹理来测量变形的方法；程序会画出这片区域各处的移动和伸缩。",
        )

        self.measure_frame = ttk.LabelFrame(parent, text="02  测量设置", padding=(10, 8))
        self.measure_frame.grid(row=1, column=0, sticky="ew")
        self.measure_frame.columnconfigure(0, weight=1)

        frame_range = ttk.Frame(self.measure_frame, style="Card.TFrame")
        frame_range.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        frame_range.columnconfigure(0, weight=1)

        preview_row = ttk.Frame(frame_range, style="Card.TFrame")
        preview_row.grid(row=0, column=0, sticky="ew", pady=(0, 3))
        preview_row.columnconfigure(3, weight=1)

        preview_tip = (
            "输入要查看的图片序号，再点“显示”。帧就是序列中的一张图片，编号从 1 开始；这里只切换预览图片。"
        )
        self.preview_frame_label = ttk.Label(preview_row, text="预览帧：")
        self.preview_frame_label.grid(row=0, column=0, padx=(0, 4), sticky="w")
        self.add_tooltip(self.preview_frame_label, preview_tip)
        self.preview_frame_entry = ttk.Entry(preview_row, textvariable=self.preview_frame_1based, width=6)
        self.preview_frame_entry.grid(row=0, column=1, padx=(0, 5), sticky="w")
        self.add_tooltip(self.preview_frame_entry, preview_tip)
        self.show_preview_button = ttk.Button(
            preview_row,
            text="显示",
            command=self.go_to_preview_frame,
            style="Compact.TButton",
        )
        self.show_preview_button.grid(row=0, column=2, padx=(0, 5), sticky="w")
        self.add_tooltip(self.show_preview_button, "点击显示左侧序号对应的图片；编号从 1 开始。只切换预览，不会开始分析。")

        self.prev_frame_button = ttk.Button(
            preview_row,
            text="上一帧",
            command=lambda: self.step_preview_frame(-1),
            style="Compact.TButton",
        )
        self.prev_frame_button.grid(row=0, column=3, padx=(0, 5), sticky="e")
        self.add_tooltip(self.prev_frame_button, "点击查看上一张图片，检查画框的位置是否合适；分析范围保持原来的设置。")

        self.next_frame_button = ttk.Button(
            preview_row,
            text="下一帧",
            command=lambda: self.step_preview_frame(1),
            style="Compact.TButton",
        )
        self.next_frame_button.grid(row=0, column=4, sticky="e")
        self.add_tooltip(self.next_frame_button, "点击查看下一张图片，检查测量部位是否仍然清晰可见；分析范围保持原来的设置。")

        range_row = ttk.Frame(frame_range, style="Card.TFrame")
        range_row.grid(row=1, column=0, sticky="ew")
        range_row.columnconfigure(6, weight=1)

        analysis_range_tip = (
            "填写要分析的第一张和最后一张图片序号，包含两端，编号从 1 开始。起始图片也是比较变形的基准；测量框要在它上面绘制。"
        )
        self.analysis_range_label = ttk.Label(range_row, text="分析范围：", style="Key.TLabel")
        self.analysis_range_label.grid(row=0, column=0, padx=(0, 4), sticky="w")
        self.add_tooltip(self.analysis_range_label, analysis_range_tip)
        self.start_frame_entry = ttk.Entry(range_row, textvariable=self.start_frame_1based, width=6)
        self.start_frame_entry.grid(row=0, column=1, padx=(0, 4), sticky="w")
        self.add_tooltip(self.start_frame_entry, "填写第一张要分析的图片序号（从 1 开始）。它同时作为比较基准；改动后，请在新的参考图片上重新画测量框。")
        ttk.Label(range_row, text="到").grid(row=0, column=2, padx=(0, 4), sticky="w")
        self.end_frame_entry = ttk.Entry(range_row, textvariable=self.end_frame_1based, width=6)
        self.end_frame_entry.grid(row=0, column=3, padx=(0, 8), sticky="w")
        self.add_tooltip(self.end_frame_entry, "填写最后一张要分析的图片序号，计算会包含这张图片。序号不能小于起始值，也不能超过图片总数。")

        self.set_start_button = ttk.Button(
            range_row,
            text="设为起始/参考",
            command=self.set_start_to_current,
            style="Compact.TButton",
        )
        self.set_start_button.grid(row=1, column=1, columnspan=2, padx=(0, 5), pady=(2, 0), sticky="w")
        self.add_tooltip(
            self.set_start_button,
            "点击把正在预览的图片设为分析起点和比较基准。已有测量框时会询问是否清空；确认后，请在这张参考图片上重新画框。",
        )

        self.set_end_button = ttk.Button(
            range_row,
            text="设为结束帧",
            command=self.set_end_to_current,
            style="Compact.TButton",
        )
        self.set_end_button.grid(row=1, column=3, columnspan=2, pady=(2, 0), sticky="w")
        self.add_tooltip(
            self.set_end_button,
            "点击把正在预览的图片设为分析终点，计算会包含这张图片。可用来跳过后面模糊、断裂或样品离开画面的图片。",
        )

        measure_core = ttk.Frame(self.measure_frame, style="Card.TFrame", padding=(0, 10, 0, 0))
        self.measure_core_frame = measure_core
        measure_core.grid(row=1, column=0, sticky="ew")
        measure_core.columnconfigure(1, weight=1)

        strain_mode_tip = (
            "选择两个测量框之间要比较的距离方向，添加或更新分组后生效。应变就是这段距离相对初始值伸长或缩短的比例。"
        )
        self.strain_mode_label = ttk.Label(measure_core, text="应变方向", style="Key.TLabel")
        self.strain_mode_label.grid(row=0, column=0, sticky="w", padx=(0, 6), pady=1)
        self.add_tooltip(self.strain_mode_label, strain_mode_tip)
        self.strain_mode_box = ttk.Combobox(
            measure_core,
            textvariable=self.strain_mode_display,
            values=list(STRAIN_MODE_LABEL_TO_VALUE.keys()),
            width=12,
            state="readonly",
        )
        self.strain_mode_box.grid(row=0, column=1, columnspan=3, sticky="ew", pady=1)
        self.strain_mode_box.bind("<<ComboboxSelected>>", self.sync_strain_mode_from_display)
        self.add_tooltip(self.strain_mode_box, strain_mode_tip, choices={
            "自动判断": "按两个框的位置自动选左右、上下或直线距离，首次使用可选此项。",
            "横向应变": "只比较两个框在图片左右方向的距离变化，适用于左右排列的测量部位。",
            "纵向应变": "只比较两个框在图片上下方向的距离变化，适用于上下排列的测量部位。",
            "两点距离应变": "比较两个框中心之间的直线距离，适用于倾斜排列的测量部位。",
        })

        tracking_preset_tip = (
            "选择后自动填写一组追踪参数，用于在后续图片中寻找两个测量框。首次使用选“标准”，再根据检查结果调整。"
        )
        self.tracking_preset_label = ttk.Label(measure_core, text="追踪模式", style="Key.TLabel")
        self.tracking_preset_label.grid(row=1, column=0, sticky="w", padx=(0, 6), pady=1)
        self.add_tooltip(self.tracking_preset_label, tracking_preset_tip)
        self.tracking_preset_box = ttk.Combobox(
            measure_core,
            textvariable=self.tracking_preset,
            values=list(TRACKING_PRESETS.keys()) + ["自定义"],
            width=12,
            state="readonly",
        )
        self.tracking_preset_box.grid(row=1, column=1, columnspan=3, sticky="ew", pady=1)
        self.tracking_preset_box.bind("<<ComboboxSelected>>", self.apply_tracking_preset)
        self.add_tooltip(self.tracking_preset_box, tracking_preset_tip, choices={
            "标准": "使用常规搜索范围和检查要求，适合纹理清晰、变化较缓慢的图片。",
            "低质量图像": "放宽纹理相似程度的要求，让较难识别的图片有机会算出结果；需核对追踪位置。",
            "快速变形": "扩大寻找范围，并允许相邻图片的伸缩变化更大；计算可能更慢。",
            "自定义": "保留当前参数。展开“高级设置”可逐项修改，修改后这里也会自动显示自定义。",
        })
        self.preset_status_label = ttk.Label(measure_core, textvariable=self.preset_status_var, style="Hint.TLabel", wraplength=430)
        self.preset_status_label.grid(
            row=2, column=0, columnspan=4, sticky="ew", pady=(0, 2)
        )
        self.add_tooltip(self.preset_status_label, tracking_preset_tip)

        pixel_size_tip = (
            "填写一个像素对应多少毫米。例如 100 个像素等于 1 毫米，就填 0.01。用于记录实际初始间距；不清楚时可留空，仍能计算伸缩比例。"
        )
        self.pixel_size_label = ttk.Label(measure_core, text="标定 mm/px，可空")
        self.pixel_size_label.grid(row=3, column=0, sticky="w", padx=(0, 6), pady=1)
        self.add_tooltip(self.pixel_size_label, pixel_size_tip)
        self.pixel_size_entry = ttk.Entry(measure_core, textvariable=self.pixel_size_mm, width=9)
        self.pixel_size_entry.grid(row=3, column=1, sticky="w", pady=1)
        self.add_tooltip(self.pixel_size_entry, pixel_size_tip)
        auto_align_tip = (
            "勾选后，画完第二个测量框时会按测量方向把两个框对齐，便于测左右或上下间距。要保留倾斜位置时取消勾选。"
        )
        self.auto_align_roi2_check = ttk.Checkbutton(measure_core, text="绘制 ROI2 后自动对齐", variable=self.auto_align_roi2)
        self.auto_align_roi2_check.grid(
            row=4, column=0, columnspan=4, sticky="w", pady=1
        )
        self.add_tooltip(self.auto_align_roi2_check, auto_align_tip)

        self.advanced_toggle_btn = ttk.Button(
            self.measure_frame,
            text="显示高级设置",
            command=self.toggle_advanced_settings,
            style="Compact.TButton",
        )
        self.advanced_toggle_btn.grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.add_tooltip(
            self.advanced_toggle_btn,
            "点击展开或收起搜索范围、图像相似程度和纹理检查等设置。首次使用可保留默认值；每项都有具体说明。",
        )

        self.advanced_frame = ttk.LabelFrame(self.measure_frame, text="高级设置", padding=(10, 8))
        self.advanced_frame.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        self._build_advanced_controls()
        self.advanced_frame.grid_remove()
        self.pixel_size_label.grid_remove()
        self.pixel_size_entry.grid_remove()
        self.preset_status_label.grid_remove()

    def _build_fullfield_section(self, parent):
        self.fullfield_frame = ttk.LabelFrame(parent, text="03  全场 ROI 与 DIC 参数", padding=(10, 8))
        self.fullfield_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        self.fullfield_frame.columnconfigure(1, weight=1)

        subset_tip = (
            "每个测量点用多大的正方形图片块来寻找移动位置，边长以像素计。填至少 9 的奇数，如 21；调大通常更容易识别，但细小变化会被平均。"
        )
        ttk.Label(self.fullfield_frame, text="子集 px", style="Key.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 6), pady=1
        )
        self.dic_subset_size_entry = ttk.Entry(self.fullfield_frame, textvariable=self.dic_subset_size, width=8)
        self.dic_subset_size_entry.grid(row=0, column=1, sticky="w", pady=1)
        self.add_tooltip(self.dic_subset_size_entry, subset_tip)

        step_tip = (
            "相邻测量点之间隔多少个像素。填大于 0 的整数；调小会测更多位置、计算更慢，调大会让结果图中的测量点更稀。"
        )
        ttk.Label(self.fullfield_frame, text="步长 px", style="Key.TLabel").grid(
            row=1, column=0, sticky="w", padx=(0, 6), pady=1
        )
        self.dic_step_entry = ttk.Entry(self.fullfield_frame, textvariable=self.dic_step, width=8)
        self.dic_step_entry.grid(row=1, column=1, sticky="w", pady=1)
        self.add_tooltip(self.dic_step_entry, step_tip)

        solver_tip = (
            "选择逐步修正图片块位置和形状的计算方法，下一次分析时生效。通常先用 IC-GN；难以找到位置时可尝试 IC-LM。"
        )
        ttk.Label(self.fullfield_frame, text="求解器", style="Key.TLabel").grid(
            row=2, column=0, sticky="w", padx=(0, 6), pady=1
        )
        self.dic_solver_box = ttk.Combobox(
            self.fullfield_frame,
            textvariable=self.dic_solver,
            values=list(DIC_SOLVERS),
            width=10,
            state="readonly",
        )
        self.dic_solver_box.grid(row=2, column=1, sticky="w", pady=1)
        self.add_tooltip(self.dic_solver_box, solver_tip, choices={
            DIC_SOLVER_ICGN: "高斯–牛顿方法：逐步修正图片块的位置和形状，通常计算较快。",
            DIC_SOLVER_ICLM: "带阻尼的修正方法：限制每一步的改动，初始位置不够准确时可尝试，通常较慢。",
        })

        ttk.Label(self.fullfield_frame, text="应变窗口（点）").grid(row=3, column=0, sticky="w", padx=(0, 6), pady=1)
        self.dic_strain_window_entry = ttk.Entry(
            self.fullfield_frame, textvariable=self.dic_strain_window, width=8
        )
        self.dic_strain_window_entry.grid(row=3, column=1, sticky="w", pady=1)
        self.add_tooltip(
            self.dic_strain_window_entry,
            "计算一处伸缩时，使用周围多少行、多少列的测量点。填至少 3 的奇数，如 7；调大会减少杂乱起伏，也会减弱细小的局部变化。",
        )

        ttk.Label(self.fullfield_frame, text="平滑 σ").grid(row=4, column=0, sticky="w", padx=(0, 6), pady=1)
        self.dic_smooth_sigma_entry = ttk.Entry(
            self.fullfield_frame, textvariable=self.dic_smooth_sigma, width=8
        )
        self.dic_smooth_sigma_entry.grid(row=4, column=1, sticky="w", pady=1)
        self.add_tooltip(
            self.dic_smooth_sigma_entry,
            "计算伸缩前，先把邻近位置的移动量做加权平均。σ 表示平均范围，以测量点间距为单位；填 0 关闭，调大会减少起伏，也会压低尖峰。",
        )

        ttk.Label(self.fullfield_frame, text="金字塔层数").grid(row=5, column=0, sticky="w", padx=(0, 6), pady=1)
        self.dic_pyramid_levels_entry = ttk.Entry(
            self.fullfield_frame, textvariable=self.dic_pyramid_levels, width=8
        )
        self.dic_pyramid_levels_entry.grid(row=5, column=1, sticky="w", pady=1)
        self.add_tooltip(
            self.dic_pyramid_levels_entry,
            "先在缩小的图片上找位置，再回到大图细算的层数。填 1 只用原图；移动较大时可试 2 到 4 层，所画区域需能容纳缩小后的图片块。",
        )

        ttk.Label(self.fullfield_frame, text="金字塔缩放").grid(row=6, column=0, sticky="w", padx=(0, 6), pady=1)
        self.dic_pyramid_scale_entry = ttk.Entry(
            self.fullfield_frame, textvariable=self.dic_pyramid_scale, width=8
        )
        self.dic_pyramid_scale_entry.grid(row=6, column=1, sticky="w", pady=1)
        self.add_tooltip(
            self.dic_pyramid_scale_entry,
            "每层图片边长缩为上一层的多少倍，填 0 到 1 之间的数，不能等于两端。0.5 表示宽和高各减半；只有层数大于 1 时生效。",
        )

        ttk.Label(self.fullfield_frame, text="搜索 px").grid(row=7, column=0, sticky="w", padx=(0, 6), pady=1)
        self.dic_search_radius_entry = ttk.Entry(
            self.fullfield_frame, textvariable=self.dic_search_radius, width=8
        )
        self.dic_search_radius_entry.grid(row=7, column=1, sticky="w", pady=1)
        self.add_tooltip(
            self.dic_search_radius_entry,
            "在预计位置周围最多寻找多少个像素，填大于 0 的整数。图片中的移动较大时可调大，计算会更慢；太小可能找不到对应纹理。",
        )

        ttk.Label(self.fullfield_frame, text="ZNCC 下限").grid(row=8, column=0, sticky="w", padx=(0, 6), pady=1)
        self.dic_zncc_min_entry = ttk.Entry(
            self.fullfield_frame, textvariable=self.dic_zncc_min, width=8
        )
        self.dic_zncc_min_entry.grid(row=8, column=1, sticky="w", pady=1)
        self.add_tooltip(
            self.dic_zncc_min_entry,
            "ZNCC 表示两个图片块的纹理相似程度，越接近 1 越相似。填 0 到 1；提高下限会拒绝更多不相似的位置，这些位置留空，不当作零。",
        )
        self.fullfield_frame.columnconfigure(3, weight=1)
        for widget in self.fullfield_frame.winfo_children():
            info = widget.grid_info()
            row, col = int(info["row"]), int(info["column"])
            if isinstance(widget, ttk.Label):
                entry = self.fullfield_frame.grid_slaves(row=row, column=1)[0]
                self.add_tooltip(widget, entry._tooltip.resolve_text)
            widget.grid_configure(row=row // 2, column=(row % 2) * 2 + col,
                                  sticky="ew" if col else "w", padx=(0, 8), pady=3)
            if isinstance(widget, ttk.Entry):
                widget.configure(width=5)

        draw_row = ttk.Frame(self.fullfield_frame, style="Card.TFrame")
        draw_row.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        draw_row.columnconfigure(1, weight=1)
        self.draw_field_roi_button = ttk.Button(
            draw_row,
            text="画全场 ROI",
            command=lambda: self.set_roi_mode(1),
            style="Secondary.TButton",
        )
        self.draw_field_roi_button.grid(row=0, column=0, sticky="w")
        self.add_tooltip(
            self.draw_field_roi_button,
            "点击后，在参考图片上按住左键拖出要分析的矩形区域（ROI）。程序将在框内逐点测量移动和伸缩；太靠近边缘、放不下完整图片块的位置会跳过。",
        )
        self.dic_field_summary_var = tk.StringVar(value="尚未绘制全场 ROI。")
        self.dic_field_summary_label = ttk.Label(
            draw_row,
            textvariable=self.dic_field_summary_var,
            style="Hint.TLabel",
            wraplength=280,
        )
        self.dic_field_summary_label.grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.add_tooltip(self.dic_field_summary_label, "显示所画分析区域的宽、高和左上角位置，单位都是像素。要更换区域，可再次点击“画全场 ROI”后重新拖框。")
        options = ttk.Frame(self.fullfield_frame, style="Card.TFrame")
        options.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        options.columnconfigure(1, weight=1)
        ttk.Label(options, text="拟合最高阶次").grid(row=0, column=0, sticky="w")
        self.dic_degree_box = ttk.Combobox(options, textvariable=self.dic_strain_degree, values=(1,2), width=4, state="readonly")
        self.dic_degree_box.grid(row=0, column=1, sticky="w")
        self.add_tooltip(self.dic_degree_box, "选择如何用周围测量点的移动量计算局部伸缩，下一次分析时生效。", choices={
            1: "用直线变化近似周围的移动量，适合较均匀的变形。",
            2: "优先考虑弯曲变化，适合局部变形不均匀的区域；可用点不足时会尝试一阶计算。",
        })
        robust_check = ttk.Checkbutton(options, text="稳健拟合", variable=self.dic_robust_strain)
        robust_check.grid(row=0, column=2, sticky="w")
        self.add_tooltip(robust_check, "勾选后，计算局部伸缩时会减小少数偏离较大的测量点的影响。取消后，周围各点按普通方式参与计算。")
        ttk.Label(options, text="异常位移下限 px").grid(row=1, column=0, sticky="w", pady=4)
        self.dic_outlier_entry = ttk.Entry(options, textvariable=self.dic_outlier_threshold, width=6)
        self.dic_outlier_entry.grid(row=1, column=1, sticky="w")
        self.add_tooltip(self.dic_outlier_entry, "检查某点是否比周围位置预计的移动量偏离过多，数值以像素计，并结合周围点的起伏判断。填 0 关闭；被排除点的原始移动量仍会保存。")
        ttk.Label(options, text="试样遮罩").grid(row=2, column=0, sticky="w")
        self.dic_mask_box = ttk.Combobox(options, textvariable=self.dic_mask_mode, values=("矩形 ROI", "自动纹理", "导入遮罩"), width=10, state="readonly")
        self.dic_mask_box.grid(row=2, column=1, sticky="w")
        self.add_tooltip(self.dic_mask_box, "遮罩是一张指定哪些位置参与计算的黑白图。选择范围后，可点“预览遮罩”检查，下一次分析时生效。", choices={
            "矩形 ROI": "分析所画矩形内的位置，并跳过手动画出的排除区。ROI 就是在图上画出的分析区域。",
            "自动纹理": "根据明暗纹理自动排除平坦背景和孔洞；请先预览，检查样品边缘是否被误排除。",
            "导入遮罩": "使用与参考图片尺寸相同的黑白图片，白色保留、黑色排除；点旁边“导入”选择文件。",
        })
        self.dic_mask_load_button = ttk.Button(options, text="导入", command=self.select_dic_mask, style="Compact.TButton")
        self.dic_mask_load_button.grid(row=2, column=2, sticky="ew")
        self.add_tooltip(self.dic_mask_load_button, "点击选择与参考图片同样宽、高的黑白遮罩图，白色位置参与计算，黑色位置跳过。选好后会切换为“导入遮罩”并预览。")
        self.dic_mask_path_label = ttk.Label(options, textvariable=self.dic_mask_path, wraplength=280, style="Hint.TLabel")
        self.dic_mask_path_label.grid(row=3, column=0, columnspan=3, sticky="ew", pady=4)
        self.add_tooltip(self.dic_mask_path_label, "这里显示所选黑白遮罩图片的完整路径。要换文件，请点击“导入”。")
        actions = ttk.Frame(options, style="Card.TFrame")
        actions.grid(row=4, column=0, columnspan=3, sticky="ew")
        for column, (label, action, tip) in enumerate((
            ("预览遮罩", self.preview_dic_mask, "先画好分析区域，再点击查看哪些位置被排除。参考图片上的红色区域将被跳过；这里只预览，不开始计算。"),
            ("画排除区", self.draw_dic_exclusion, "先画好分析区域，再点击并在参考图片上拖出矩形，排除孔洞、背景等不想计算的位置。可重复画多个框。"),
            ("清空排除区", self.clear_dic_exclusions, "点击移除所有手动画出的排除框，并更新预览。自动纹理或导入黑白图指定的排除位置仍然有效。"),
        )):
            button = ttk.Button(actions, text=label, command=action, style="Compact.TButton")
            button.grid(row=0, column=column, padx=(0,4), sticky="ew")
            self.add_tooltip(button, tip)
        for row in range(3):
            label = options.grid_slaves(row=row, column=0)[0]
            entry = options.grid_slaves(row=row, column=1)[0]
            self.add_tooltip(label, entry._tooltip.resolve_text)
        convergence_check = ttk.Checkbutton(options, text="仅接受收敛子集", variable=self.dic_reject_nonconverged)
        convergence_check.grid(row=5, column=0, columnspan=3, sticky="w", pady=3)
        self.add_tooltip(convergence_check, "收敛表示反复修正后，图片块位置和形状的改动已小到可以停止。勾选后，只使用达到停止条件的点；取消后可查看未达到条件的候选结果，计算状态仍会保存。")
        ttk.Label(options, text="遮罩中白色保留、黑色排除。\n计算范围越大，起伏越少，细小变化也越不明显。", style="Hint.TLabel", wraplength=300).grid(row=6, column=0, columnspan=3, sticky="w", pady=5)
        self.fullfield_frame.grid_remove()

    def dic_mask_settings(self):
        mode_label = self.dic_mask_mode.get()
        mode_map = {"矩形 ROI": "none", "自动纹理": "auto", "导入遮罩": "file"}
        if mode_label not in mode_map:
            raise RuntimeError(f"未知的试样遮罩模式：{mode_label}")
        mode = mode_map[mode_label]
        return {"mode": mode, "path": self.dic_mask_path.get().strip() if mode == "file" else None,
                "texture_threshold": 3.0, "exclusions": [list(rect) for rect in self.dic_mask_exclusions]}

    def dic_display_options(self):
        bg_label = self.dic_view_background.get()
        bg_map = {"无底图": "none", "参考图": "reference", "变形图": "deformed"}
        if bg_label not in bg_map:
            raise RuntimeError(f"未知的显示底图选项：{bg_label}")
        color_label = self.dic_color_mode.get()
        color_map = {"数据范围": "range", "零点对称": "symmetric", "手动范围": "manual"}
        if color_label not in color_map:
            raise RuntimeError(f"未知的色标范围模式：{color_label}")
        options = {"style": "contour" if self.dic_display_style.get() == "连续云图" else "points",
                   "background": bg_map[bg_label],
                   "color_mode": color_map[color_label],
                   "percent": bool(self.dic_percent.get()), "cmap": self.dic_colormap.get(), "alpha": .65}
        if options["color_mode"] == "manual":
            try:
                options["vmin"], options["vmax"] = float(self.dic_color_min.get()), float(self.dic_color_max.get())
                if not np.isfinite([options["vmin"], options["vmax"]]).all() or options["vmin"] >= options["vmax"]:
                    raise ValueError()
            except ValueError as exc:
                raise RuntimeError("手动色标需要有限且递增的最小值和最大值。") from exc
        return options

    def is_fullfield_mode(self):
        return str(self.analysis_mode.get()) == ANALYSIS_MODE_FULLFIELD

    def set_analysis_mode(self, *_args):
        requested = str(self.analysis_mode.get())
        committed = getattr(self, "_committed_analysis_mode", ANALYSIS_MODE_EXTENSOMETER)
        if getattr(self, "is_processing", False) or getattr(self, "_completion_pending", False):
            if requested != committed:
                self.analysis_mode.set(committed)
                if hasattr(self, "status_var"):
                    self.status_var.set("正在处理或收尾：不能切换分析模式。")
                if hasattr(self, "log"):
                    self.log("分析进行中，已保持当前模式。")
            return
        self._committed_analysis_mode = requested
        self._drawing_mask_exclusion = False
        self.drag_start = None
        if getattr(self, "temp_rect_id", None) is not None and hasattr(self, "canvas"):
            try:
                self.canvas.delete(self.temp_rect_id)
            except tk.TclError:
                pass
            self.temp_rect_id = None
        is_ff = self.is_fullfield_mode()
        previous_kind = getattr(self, "_viewer_kind", "extensometer")
        target_kind = "fullfield" if is_ff else "extensometer"
        canvas_restored = False
        if previous_kind != target_kind and (
            getattr(self, "viewer_figure", None) is not None
            or getattr(self, "results_df", None) is not None
            or getattr(self, "dic_last_field", None) is not None
        ):
            self.last_qc_summary = None
            if hasattr(self, "qc_overview_var"):
                self.qc_overview_var.set("分析完成后显示 QC 总览。")
            if hasattr(self, "clear_viewer"):
                self.clear_viewer(keep_placeholder=True)
            self._restore_sequence_preview()
            canvas_restored = True
        if not is_ff and getattr(self, "_canvas_mask_overlay_rgb", None) is not None:
            self._canvas_mask_overlay_rgb = None
            if not canvas_restored:
                self._restore_sequence_preview()
                canvas_restored = True
        if hasattr(self, "fullfield_frame"):
            if is_ff:
                self.fullfield_frame.grid()
            else:
                self.fullfield_frame.grid_remove()
        # The frame range is shared.  All 1-D-only controls stay hidden in
        # full-field mode so stale ROI/export settings cannot look actionable.
        if hasattr(self, "measure_core_frame"):
            if is_ff:
                self.measure_core_frame.grid_remove()
                self.advanced_toggle_btn.grid_remove()
                self.advanced_frame.grid_remove()
            else:
                self.measure_core_frame.grid()
                self.advanced_toggle_btn.grid()
                if self.advanced_visible.get():
                    self.advanced_frame.grid()
        if hasattr(self, "roi_group_frame"):
            (self.roi_group_frame.grid_remove if is_ff else self.roi_group_frame.grid)()
        if hasattr(self, "export_frame"):
            (self.export_frame.grid_remove if is_ff else self.export_frame.grid)()
        if hasattr(self, "fullfield_export_info_frame"):
            (self.fullfield_export_info_frame.grid if is_ff else self.fullfield_export_info_frame.grid_remove)()
        if is_ff:
            self.current_roi_index = 1
            if hasattr(self, "status_var"):
                self.status_var.set("全场 2D DIC：请在参考帧拖出全场 ROI，再开始分析。")
            if hasattr(self, "export_hint_label"):
                self.export_hint_label.configure(
                    text="确认参考帧、全场 ROI 和 DIC 参数后开始；全场核心输出固定生成。"
                )
            if hasattr(self, "workflow_step_texts"):
                self.workflow_step_texts = [
                    "1. 选择图像文件夹和输出文件夹，加载序列",
                    "2. 设置参考帧与分析范围",
                    "3. 画全场 ROI，设置子集/步长/IC-GN 或 IC-LM",
                    "4. 开始分析，查看 u/v/应变图",
                ]
        else:
            if hasattr(self, "export_hint_label"):
                self.export_hint_label.configure(
                    text="确认参考帧、ROI 方向和导出内容后再开始。"
                )
            if hasattr(self, "workflow_step_texts"):
                self.workflow_step_texts = [
                    "1. 选择图像文件夹和输出文件夹，加载序列",
                    "2. 设置参考帧、分析范围和测量方向",
                    "3. 画 ROI1/ROI2，添加 ROI 组",
                    "4. 确认导出内容，点击开始分析",
                ]
        if hasattr(self, "workflow_steps_label"):
            self.workflow_steps_label.configure(text=self._visible_workflow_steps_text())
        if hasattr(self, "controls_canvas"):
            try:
                self.controls_panel.update_idletasks()
                self.controls_canvas.configure(scrollregion=self.controls_canvas.bbox("all"))
            except tk.TclError:
                pass
        if not canvas_restored and getattr(self, "display_img", None) is not None and hasattr(self, "canvas"):
            self.show_image()
        self.update_workflow_action_states()

    def _build_advanced_controls(self):
        advanced_fields = [
            (
                "search_radius_entry",
                "搜索半径 px",
                self.search_radius,
                7,
                "在上一次找到的位置周围，最多再找多少个像素。填正整数；移动较大时可调大，计算会更慢，也更可能找到相似但错误的位置。",
            ),
            (
                "hard_corr_entry",
                "严格接受阈值",
                self.hard_corr,
                7,
                "两个测量框与原有纹理至少要多相似才通过检查，分数越接近 1 越相似。填 -1 到 1；调高会拒绝更多图片，调低也更容易接受错误位置。",
            ),
            (
                "soft_corr_entry",
                "弱接受阈值",
                self.soft_corr,
                7,
                "允许较难识别的图片再接受额外检查时，纹理相似分数的最低值。不得高于严格接受阈值；只有勾选“启用自适应弱接受”时生效。",
            ),
            (
                "max_frame_strain_jump_entry",
                "单帧应变突变上限",
                self.max_frame_strain_jump,
                8,
                "相邻两张有效图片之间，允许伸缩比例改变多少。0.01 表示相差 1 个百分点；调大允许变化更快，留空关闭这项检查。",
            ),
            (
                "fb_tolerance_entry",
                "FB 容差 px",
                self.fb_tolerance_px,
                7,
                "FB 是先向后追踪、再返回前一张图片检查位置。这里填返回位置允许偏离原位置多少个像素；调小更严格，仅在勾选前后向检查时生效。",
            ),
            (
                "template_alpha_entry",
                "模板跟随系数",
                self.template_alpha,
                7,
                "找到新位置后，用当前图片的纹理替换原有追踪模板的比例。填 0 到 1；0 保留原模板，1 完全替换。仅在勾选模板跟随时生效，过大可能逐渐跟错位置。",
            ),
            (
                "min_texture_std_entry",
                "最小灰度标准差",
                self.min_texture_std,
                8,
                "灰度标准差表示测量框里明暗变化有多明显。低于填写值时会提醒纹理太平坦、难以追踪；调高会更容易出现提醒，不会让图片本身变清晰。",
            ),
            (
                "min_texture_contrast_entry",
                "最小 P95-P5 对比度",
                self.min_texture_contrast,
                8,
                "P95-P5 是较亮与较暗像素的灰度差，忽略最亮和最暗各 5%。差值低于这里填写的数时，提醒测量框的明暗对比不足；调高会更容易提醒。",
            ),
            (
                "max_saturated_frac_entry",
                "最大近黑/近白比例",
                self.max_saturated_frac,
                8,
                "允许测量框里接近全黑或全白的像素占多少，填 0 到 1，例如 0.2 是 20%。超过时提醒可能曝光不足或过亮；调小会更早提醒。",
            ),
            (
                "overlay_every_entry",
                "overlay 间隔",
                self.overlay_every,
                7,
                "勾选“叠加图像”后，每隔多少张图片保存一次带测量框的检查图。填正整数；数值小会保存更多图片，便于查看是否跟错位置。",
            ),
        ]
        self.advanced_frame.columnconfigure(1, weight=1)
        for idx, (attr_name, label, variable, width, tooltip) in enumerate(advanced_fields):
            row = idx
            col = 0
            label_widget = ttk.Label(
                self.advanced_frame,
                text=label,
                style="Key.TLabel" if idx < 4 else "TLabel",
            )
            label_widget.grid(row=row, column=col, sticky="w", padx=(0, 4), pady=3)
            self.add_tooltip(label_widget, tooltip)
            entry = ttk.Entry(self.advanced_frame, textvariable=variable, width=width)
            entry.grid(
                row=row, column=col + 1, sticky="ew", padx=(0, 12), pady=3
            )
            setattr(self, attr_name, entry)
            self.add_tooltip(entry, tooltip)

        option_row = len(advanced_fields)
        adaptive_tip = (
            "勾选后，纹理不够相似的图片仍可按较低相似下限和其他已启用检查判断。取消后只接受严格检查通过的图片，未通过的位置会留空。"
        )
        self.enable_adaptive_check = ttk.Checkbutton(self.advanced_frame, text="启用自适应弱接受", variable=self.enable_adaptive)
        self.enable_adaptive_check.grid(
            row=option_row, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        self.add_tooltip(self.enable_adaptive_check, adaptive_tip)
        template_follow_tip = (
            "勾选后，用每次找到的图片块逐渐更新追踪纹理，更新比例由“模板跟随系数”决定。取消后保留参考图片的纹理；跟随可适应外观变化，也可能逐渐跟错位置。"
        )
        self.use_prev_frame_template_check = ttk.Checkbutton(
            self.advanced_frame,
            text="使用前一帧模板跟随",
            variable=self.use_prev_frame_template,
        )
        self.use_prev_frame_template_check.grid(
            row=option_row + 1, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        self.add_tooltip(self.use_prev_frame_template_check, template_follow_tip)
        fb_tip = (
            "勾选后，找到新位置还会反向追踪回上一张有效图片；返回位置偏差超过“FB 容差”就不接受。取消可减少计算，但少了这项位置核对。"
        )
        self.enable_fb_check_check = ttk.Checkbutton(self.advanced_frame, text="前后向一致性检查", variable=self.enable_fb_check)
        self.enable_fb_check_check.grid(
            row=option_row + 2, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        self.add_tooltip(self.enable_fb_check_check, fb_tip)

    def _build_roi_section(self, parent):
        group_frame = ttk.LabelFrame(parent, text="03  ROI 配对与分组", padding=(10, 8))
        self.roi_group_frame = group_frame
        group_frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        group_frame.columnconfigure(0, weight=1)

        tool_row = ttk.Frame(group_frame, style="Card.TFrame")
        tool_row.grid(row=0, column=0, sticky="ew")
        tool_row.columnconfigure((0, 1), weight=1, uniform="roi")
        self.roi1_button = ttk.Button(tool_row, text="画 ROI 1", command=lambda: self.set_roi_mode(1), style="Compact.TButton")
        self.roi1_button.grid(row=0, column=0, padx=(0, 6), pady=2, sticky="ew")
        self.add_tooltip(
            self.roi1_button,
            "点击后，在参考图片上按住左键拖出第一个测量框（ROI 1）。框内要有清楚的斑点或纹理，程序将追踪这个位置；画完会切换到第二个框。",
        )
        self.roi2_button = ttk.Button(tool_row, text="画 ROI 2", command=lambda: self.set_roi_mode(2), style="Compact.TButton")
        self.roi2_button.grid(row=0, column=1, pady=2, sticky="ew")
        self.add_tooltip(
            self.roi2_button,
            "点击后，在参考图片上按住左键拖出第二个测量框（ROI 2）。程序比较两个框中心之间的距离变化；画好后点“添加 ROI 组”保存这一对。",
        )
        self.align_x_button = ttk.Button(tool_row, text="水平对齐 · x", command=lambda: self.align_current_pair("x", set_mode=True), style="Compact.TButton")
        self.align_x_button.grid(row=1, column=0, padx=(0, 6), pady=2, sticky="ew")
        self.add_tooltip(
            self.align_x_button,
            "点击把两个测量框移到同一水平线上，并切换为测量左右间距（x 方向）。位置会改变，适合测量部位左右排列的图片。",
        )
        self.align_y_button = ttk.Button(tool_row, text="垂直对齐 · y", command=lambda: self.align_current_pair("y", set_mode=True), style="Compact.TButton")
        self.align_y_button.grid(row=1, column=1, pady=2, sticky="ew")
        self.add_tooltip(
            self.align_y_button,
            "点击把两个测量框移到同一垂直线上，并切换为测量上下间距（y 方向）。位置会改变，适合测量部位上下排列的图片。",
        )

        roi_summary_frame = ttk.Frame(group_frame, style="Card.TFrame", padding=(0, 4))
        roi_summary_frame.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        roi_summary_frame.columnconfigure(0, weight=1)
        self.current_roi_summary_label = ttk.Label(
            roi_summary_frame,
            textvariable=self.current_roi_summary_var,
            style="Hint.TLabel",
            justify=tk.LEFT,
            wraplength=430,
        )
        self.current_roi_summary_label.grid(row=0, column=0, sticky="ew")
        self.add_tooltip(
            self.current_roi_summary_label,
            "显示当前两个测量框的尺寸、测量方向和图像是否容易识别。L0 是参考图片上的初始间距；添加为一组前，可在这里核对。",
        )

        form_row = ttk.Frame(group_frame, style="Card.TFrame")
        form_row.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        form_row.columnconfigure((1, 3), weight=1)
        group_name_tip = (
            "给这一对测量框起个名称，如“样品左侧”。添加或更新时会使用这个名称；留空会自动编号为 G01、G02 等，也会用于结果文件名。"
        )
        self.group_name_label = ttk.Label(form_row, text="组名：")
        self.group_name_label.grid(row=0, column=0, padx=(0, 4), pady=2, sticky="w")
        self.add_tooltip(self.group_name_label, group_name_tip)
        self.group_name_entry = ttk.Entry(form_row, textvariable=self.group_name_var, width=8)
        self.group_name_entry.grid(row=0, column=1, padx=(0, 8), pady=2, sticky="ew")
        self.add_tooltip(self.group_name_entry, group_name_tip)
        role_tip = (
            "说明这组测量沿拉伸方向还是垂直于拉伸方向，添加或更新后生效。泊松比表示横向收缩与纵向伸长的比值，需两种角色同时存在。"
        )
        self.roi_role_label = ttk.Label(form_row, text="角色：", style="Key.TLabel")
        self.roi_role_label.grid(row=0, column=2, padx=(0, 4), pady=2, sticky="w")
        self.add_tooltip(self.roi_role_label, role_tip)
        self.roi_role_box = ttk.Combobox(
            form_row,
            textvariable=self.roi_role_display,
            values=list(ROI_ROLE_LABEL_TO_VALUE.keys()),
            width=8,
            state="readonly",
        )
        self.roi_role_box.grid(row=0, column=3, pady=2, sticky="ew")
        self.roi_role_box.bind("<<ComboboxSelected>>", self.sync_roi_role_from_display)
        self.add_tooltip(self.roi_role_box, role_tip, choices={
            "普通": "只计算这组的距离变化，不参加泊松比计算，首次只测伸缩可选此项。",
            "拉伸方向": "这对框测量沿样品受拉方向的伸长，作为泊松比计算中的纵向测量。",
            "横向方向": "这对框测量垂直于样品受拉方向的收缩，需同时添加拉伸方向组才能计算泊松比。",
        })
        self.add_group_button = ttk.Button(form_row, text="添加 ROI 组", command=self.add_current_group, style="Secondary.TButton")
        self.add_group_button.grid(row=1, column=0, columnspan=4, pady=(6, 2), sticky="ew")
        self.add_tooltip(
            self.add_group_button,
            "点击把当前两个测量框、名称、方向和角色保存到下方列表。每组会单独计算一条伸缩曲线；先在同一参考图片上画好两个框。",
        )

        action_row = ttk.Frame(group_frame, style="Card.TFrame")
        action_row.grid(row=3, column=0, sticky="ew", pady=(4, 0))
        action_row.columnconfigure((0, 1), weight=1, uniform="group")
        self.update_group_button = ttk.Button(action_row, text="更新选中", command=self.update_selected_group, style="Compact.TButton")
        self.update_group_button.grid(row=0, column=1, pady=2, sticky="ew")
        self.add_tooltip(
            self.update_group_button,
            "先选中列表中的一组，再点击，用当前两个测量框、名称、方向和角色替换该组设置。下一次分析将使用更新后的设置。",
        )
        self.load_group_button = ttk.Button(action_row, text="载入选中", command=self.load_selected_group, style="Compact.TButton")
        self.load_group_button.grid(row=0, column=0, padx=(0, 6), pady=2, sticky="ew")
        self.add_tooltip(
            self.load_group_button,
            "先选中列表中的一组，再点击，显示它的两个测量框并填回名称、方向和角色。修改后点击“更新选中”才会保存到这组。",
        )
        self.delete_group_button = ttk.Button(action_row, text="删除选中", command=self.delete_selected_group, style="Danger.TButton")
        self.delete_group_button.grid(row=1, column=1, pady=2, sticky="ew")
        self.add_tooltip(
            self.delete_group_button,
            "先选中一组，再点击并确认删除。这组将不再参加后续分析，已经保存的结果文件会保留。",
        )
        self.clear_rois_button = ttk.Button(action_row, text="清除当前 ROI", command=self.clear_current_rois, style="Danger.TButton")
        self.clear_rois_button.grid(row=1, column=0, padx=(0, 6), pady=2, sticky="ew")
        self.add_tooltip(
            self.clear_rois_button,
            "点击并确认后，清空当前正在画的两个测量框，便于重新绘制；列表中已保存的分组会保留。也可按 Esc。",
        )

        tree_frame = ttk.Frame(group_frame, style="Card.TFrame")
        tree_frame.grid(row=4, column=0, sticky="ew", pady=(6, 0))
        tree_frame.columnconfigure(0, weight=1)
        columns = ("name", "role", "selected", "actual", "L0", "dx", "dy", "roi1", "roi2")
        # 保留水平滚动，首屏优先显示按钮和图像，不让列表请求宽度撑爆窗口。
        self.group_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=3,
                                      selectmode="browse", displaycolumns=("name", "role", "actual", "L0", "selected", "dx", "dy", "roi1", "roi2"))
        for col in columns:
            self.group_tree.heading(col, text=GROUP_TREE_HEADING_TEXTS[col])
            self.group_tree.column(
                col,
                width=80 if col in ("name", "role", "actual", "L0") else GROUP_TREE_COLUMN_WIDTHS[col],
                minwidth=40,
                anchor="center",
                stretch=False,
            )
        self.group_tree.grid(row=0, column=0, sticky="ew")
        tree_scroll_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.group_tree.xview)
        self.group_tree.configure(xscrollcommand=tree_scroll_x.set)
        tree_scroll_x.grid(row=1, column=0, sticky="ew")
        self.add_tooltip(tree_scroll_x, "左右拖动查看分组表格中被隐藏的列，如测量框的位置和尺寸。")
        tree_scroll_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.group_tree.yview)
        tree_scroll_y.grid(row=0, column=1, sticky="ns")
        self.add_tooltip(tree_scroll_y, "上下拖动查看列表中更多已保存的测量组。")
        self.group_tree.configure(yscrollcommand=tree_scroll_y.set)
        self.group_tree.bind("<Double-1>", lambda event: self.load_selected_group())
        self.group_tree.bind("<<TreeviewSelect>>", lambda _event: self.update_workflow_action_states())
        self.group_tree.bind("<Button-3>", self._show_group_tree_context_menu)  # 右键菜单
        self.add_tooltip(
            self.group_tree,
            self._group_table_help,
        )

    def _build_image_section(self, parent):
        self.image_frame = ttk.Frame(parent, style="Card.TFrame", padding=10)
        self.image_frame.grid(row=0, column=0, sticky="nsew")
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.rowconfigure(1, weight=1)
        toolbar = ttk.Frame(self.image_frame, style="Card.TFrame")
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        toolbar.columnconfigure(5, weight=1)
        buttons = [
            ("btn_zoom_out", "−", lambda: self.zoom_image(1 / 1.25), 3, "点击缩小图片，看到更大的范围（Ctrl + 减号）。只改变显示大小。"),
            ("btn_zoom_in", "+", lambda: self.zoom_image(1.25), 3, "点击放大图片，便于看清纹理和画框（Ctrl + 加号）。超出窗口的部分可用滚动条查看。"),
            ("btn_fit", "适应窗口", self.fit_image_to_view, 8, "点击把整张原图缩放到窗口内（Ctrl + F）。窗口大小改变时会自动调整，测量框仍对应原图位置。"),
            ("btn_1to1", "1:1", self.show_image_1to1, 4, "点击按原始像素大小显示图片，一个图片像素对应一个显示像素。图片较大时，拖动滚动条查看边缘。"),
        ]
        for col, (attr, text, command, width, tip) in enumerate(buttons):
            button = ttk.Button(toolbar, text=text, command=command, style="Compact.TButton", width=width)
            button.grid(row=0, column=col, padx=(0, 4))
            setattr(self, attr, button)
            self.add_tooltip(button, tip)
        self.zoom_label_var = tk.StringVar(value="—")
        self.zoom_label = ttk.Label(toolbar, textvariable=self.zoom_label_var, style="Hint.TLabel", width=7, anchor="center")
        self.zoom_label.grid(row=0, column=4, padx=(2, 4))
        self.add_tooltip(self.zoom_label, "显示图片相对原始像素大小的比例。100% 是原始大小，200% 是放大到两倍；可用加减按钮或滚轮调整。")
        self.canvas = tk.Canvas(self.image_frame, bg="#111827", cursor="crosshair", highlightthickness=0,
                                width=400, height=260, xscrollincrement=1, yscrollincrement=1)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        scroll_x = ttk.Scrollbar(self.image_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        scroll_y = ttk.Scrollbar(self.image_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        scroll_x.grid(row=2, column=0, sticky="ew")
        scroll_y.grid(row=1, column=1, sticky="ns")
        self.add_tooltip(scroll_x, "左右拖动查看放大图片的左侧或右侧，画出的测量框仍按原图位置记录。")
        self.add_tooltip(scroll_y, "上下拖动查看放大图片的顶部或底部，便于检查窗口之外的测量位置。")
        self.canvas.configure(xscrollcommand=scroll_x.set, yscrollcommand=scroll_y.set)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)
        self.canvas.bind("<Configure>", self._on_image_canvas_configure, add="+")
        self.image_context_var = tk.StringVar(value="在参考帧上绘制 ROI；滚轮缩放，滚动条移动视图。")
        self.image_context_label = ttk.Label(self.image_frame, textvariable=self.image_context_var,
                                             style="Hint.TLabel", wraplength=600, justify=tk.LEFT)
        self.image_context_label.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.image_frame.bind("<Configure>", lambda event: self.image_context_label.configure(wraplength=max(240, event.width - 24)))
        self.add_tooltip(self.canvas, lambda _event: (
            "按住左键拖出不想计算的矩形区域，松开后会添加一个排除框。滚轮缩放图片；红色区域表示被排除的位置。"
            if self._drawing_mask_exclusion else
            "在参考图片上按住左键拖出要分析的矩形区域，松开后保存范围。滚轮缩放图片，滚动条移动视图。"
            if self.is_fullfield_mode() else
            f"在参考图片上按住左键拖出第 {self.current_roi_index} 个测量框，松开后保存。两个框画好后点“添加 ROI 组”；滚轮可缩放图片。"))
        self.add_tooltip(self.image_context_label, "显示当前图片名称、序号和尺寸，以及用来比较变形的参考图片。px 表示像素；绘制提示说明现在要画哪个测量框。")

    def _build_analysis_section(self, parent):
        self.analysis_frame = ttk.LabelFrame(parent, text="04  导出设置", padding=(10, 8))
        self.analysis_frame.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        self.analysis_frame.columnconfigure(0, weight=1)
        self.export_frame = ttk.Frame(self.analysis_frame, style="Card.TFrame")
        self.export_frame.grid(row=0, column=0, sticky="ew")
        self.export_frame.columnconfigure((0, 1), weight=1)
        preset_bar = ttk.Frame(self.export_frame, style="Card.TFrame")
        preset_bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        preset_bar.columnconfigure((0, 1, 2), weight=1, uniform="export")
        presets = [
            ("export_research_preset_button", "推荐", self._apply_research_preset, "点击自动勾选数值表格、伸缩曲线图片、质量摘要和参数记录，其余选项取消。开始分析后才会生成文件。"),
            ("export_quick_preset_button", "快速查看", self._apply_quick_view_preset, "点击只保留伸缩曲线图片和质量摘要，取消其他导出选项。适合先查看分析效果，开始分析后才会生成文件。"),
            ("export_all_preset_button", "完整导出", self._apply_all_export_preset, "点击勾选所有导出内容，分析时会保存更多文件。包括 Origin 绘图软件的工程文件，需本机装有 OriginPro 和相应连接组件。"),
        ]
        for col, (attr, text, command, tip) in enumerate(presets):
            button = ttk.Button(preset_bar, text=text, command=command, style="Compact.TButton")
            button.grid(row=0, column=col, sticky="ew", padx=(0, 4 if col < 2 else 0))
            setattr(self, attr, button)
            self.add_tooltip(button, tip)
        options = [
            ("Origin TXT", self.export_origin_txt, "勾选后保存可导入 Origin 或表格软件的文本数据：图片序号、相对初始距离的伸缩比例、按距离比取对数的真应变。无需安装 Origin。"),
            ("应变 PNG", self.export_engineering_png, "勾选后，给每组保存一张伸缩曲线图片。纵轴是距离变化除以初始距离，例如 0.01 表示伸长 1%。PNG 是常见图片格式。"),
            ("QC 摘要", self.export_qc_summary, "QC 表示质量检查。勾选后保存哪些图片通过或未通过检查、纹理相似程度等摘要，便于判断结果是否需要复查。"),
            ("参数记录", self.export_parameters, "勾选后把本次使用的分析设置和通过检查的数量保存为文本，方便以后核对或重复分析。"),
            ("完整 CSV", self.export_full_csv, "勾选后保存详细表格，包括每张图片上的测量框位置、伸缩值和检查结果。CSV 是可用 Excel 等表格软件打开的文本表格。"),
            ("相关系数 PNG", self.export_corr_plot, "勾选后保存纹理相似程度随图片序号变化的曲线图。分数越接近 1 越相似，突然下降的位置值得复查。"),
            ("叠加图像", self.export_overlays, "勾选后按“高级设置”中的间隔保存带追踪框的原图，用来检查框是否始终跟在同一部位。"),
            ("论文图表包", self.export_publication_figures, "勾选后额外保存每英寸 600 个点的高清图片，以及放大后仍清晰的矢量图，便于排版或编辑。会增加文件数量。"),
            ("Origin OPJU", self.export_origin_opju, "勾选后生成可在 Origin 绘图软件中继续编辑的工程文件。需要 OriginPro 2021 或更新版本及 originpro 连接组件；缺少时会报告生成失败。"),
        ]
        self.export_checkbuttons = []
        for idx, (text, variable, tip) in enumerate(options):
            check = ttk.Checkbutton(self.export_frame, text=text, variable=variable)
            check.grid(row=idx // 2 + 1, column=idx % 2, sticky="w", pady=2)
            self.export_checkbuttons.append(check)
            self.add_tooltip(check, tip)
        self.fullfield_export_info_frame = ttk.Frame(self.analysis_frame, style="Card.TFrame")
        self.fullfield_export_info_frame.grid(row=1, column=0, sticky="ew")
        self.fullfield_export_info_frame.columnconfigure(0, weight=1)
        self.fullfield_export_info_label = ttk.Label(
            self.fullfield_export_info_frame,
            text="每张算出有效结果的图片都会保存数值表和移动、伸缩图。\n移动量以像素计；算不出的位置留空，不当作零。",
            style="Hint.TLabel", justify=tk.LEFT, wraplength=360)
        self.fullfield_export_info_label.grid(row=0, column=0, sticky="ew")
        self.add_tooltip(self.fullfield_export_info_label, "全场分析会自动保存各测量点的数值表和结果图片。NaN 表示这里没有算出有效数值，不能按零移动或零伸缩理解。")
        self.fullfield_export_overlays_checkbutton = ttk.Checkbutton(
            self.fullfield_export_info_frame, text="额外导出 Exx 叠加图", variable=self.export_overlays)
        self.fullfield_export_overlays_checkbutton.grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.add_tooltip(self.fullfield_export_overlays_checkbutton, "勾选后，额外把左右方向的伸缩值（Exx）用颜色画在对应的分析图片上，方便核对位置；分析结束时保存为图片。")
        self.export_hint_label = ttk.Label(self.analysis_frame, style="Hint.TLabel", justify=tk.LEFT, wraplength=360)
        self.export_hint_label.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.add_tooltip(self.export_hint_label, "导出选项决定下一次分析保存哪些文件。选好后点击“开始分析”，完成后可用顶部“打开输出”查看文件。")

    def _build_run_section(self, parent, row=2):
        self.run_frame = ttk.Frame(parent, style="Card.TFrame", padding=(12, 6))
        self.run_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.run_frame.columnconfigure(0, weight=1)
        self.run_state_var = tk.StringVar(value="待加载")
        self.run_state_label = ttk.Label(self.run_frame, textvariable=self.run_state_var, style="Badge.TLabel")
        self.run_state_label.grid(row=0, column=0, sticky="w")
        self.add_tooltip(self.run_state_label, "显示当前是否可以分析、是否正在计算或需要补充设置。具体下一步操作在下方；失败后可按提示修正并重试。")
        ttk.Label(self.run_frame, text="Ctrl + Enter", style="Hint.TLabel").grid(row=0, column=1, sticky="e")
        self.workflow_hint_label = ttk.Label(self.run_frame, textvariable=self.workflow_hint_var,
                                             style="Hint.TLabel", wraplength=360, justify=tk.LEFT)
        self.workflow_hint_label.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 6))
        self.add_tooltip(self.workflow_hint_label, "这里告诉你下一步需要做什么。出现质量提醒时，点击“质量与日志”查看具体图片或参数需要检查的原因。")
        self.start_button = ttk.Button(self.run_frame, text="开始分析", command=self.start_processing, style="Primary.TButton")
        self.start_button.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.add_tooltip(self.start_button, "点击按当前图片范围、测量框和参数开始计算（Ctrl + Enter）。完成后显示结果图，并把选定文件保存到输出文件夹。")
        self.progress_value = tk.DoubleVar(value=0)
        self.progress_percent_var = tk.StringVar(value="0%")
        self.progress_value.trace_add("write", lambda *_args: self.progress_percent_var.set(f"{self.progress_value.get():.0f}%"))
        self.progress = ttk.Progressbar(self.run_frame, orient=tk.HORIZONTAL, mode="determinate", variable=self.progress_value)
        self.progress.grid(row=3, column=0, sticky="ew", pady=(10, 6), padx=(0, 8))
        self.add_tooltip(self.progress, "显示本次计算已处理的比例。计算后还会保存文件并更新图表，请等状态显示“处理完成”后再进行下一次分析。")
        ttk.Label(self.run_frame, textvariable=self.progress_percent_var, style="Hint.TLabel").grid(row=3, column=1, sticky="e")
        self.status_var = tk.StringVar(value="未加载图像")
        self.preview_scale_var = tk.StringVar(value="")
        self.status_label = ttk.Label(self.run_frame, textvariable=self.status_var, style="Hint.TLabel",
                                       wraplength=360, justify=tk.LEFT)
        self.status_label.grid(row=4, column=0, columnspan=2, sticky="ew")
        self.add_tooltip(self.status_label, "显示最近的操作、计算进度或错误原因。需要查看更完整的记录时，点击“质量与日志”。")

    def _build_quality_section(self, parent):
        parent.rowconfigure(0, weight=0)
        parent.rowconfigure(1, weight=1)
        qc_frame = ttk.Frame(parent, style="Card.TFrame", padding=(0, 2))
        qc_frame.grid(row=0, column=0, sticky="ew")
        qc_frame.columnconfigure(1, weight=1)
        ttk.Label(qc_frame, text="质量概览", style="Key.TLabel").grid(row=0, column=0, sticky="nw", padx=(0, 10))
        self.qc_overview_label = ttk.Label(qc_frame, textvariable=self.qc_overview_var,
                                           style="Hint.TLabel", wraplength=500, justify=tk.LEFT)
        self.qc_overview_label.grid(row=0, column=1, sticky="ew")
        self.add_tooltip(self.qc_overview_label, "分析后显示多少张图片、多少个测量点通过了检查。找到纹理和算出伸缩是两项检查；未通过的位置留空，数值仍需结合原图核对。")
        self.quality_notebook = ttk.Notebook(parent)
        self.quality_notebook.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.preflight_page = ttk.Frame(self.quality_notebook, style="Card.TFrame", padding=8)
        self.log_page = ttk.Frame(self.quality_notebook, style="Card.TFrame", padding=8)
        for page in (self.preflight_page, self.log_page):
            page.columnconfigure(0, weight=1)
            page.rowconfigure(0, weight=1)
        self.quality_notebook.add(self.preflight_page, text="运行前检查")
        self.quality_notebook.add(self.log_page, text="运行日志")
        self.add_tooltip(self.quality_notebook, "点击“运行前检查”查看开始分析需要补充什么；点击“运行日志”查看加载、计算、保存和错误的详细记录。两项均可滚动查看和复制。")
        self.preflight_summary_label = tk.Text(self.preflight_page, height=5, width=30, wrap=tk.WORD,
                                               font=self.ui_base_font, relief=tk.FLAT, bg=self.panel_bg,
                                               fg=self.text_color, padx=10, pady=8, state=tk.DISABLED)
        self.preflight_summary_label.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(self.preflight_page, orient=tk.VERTICAL, command=self.preflight_summary_label.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.preflight_summary_label.configure(yscrollcommand=scrollbar.set)
        self.add_tooltip(self.preflight_summary_label, "这里列出开始分析前的检查。“阻止”项需要先修正；“警告”项可继续，但需检查原因。可拖动选中文字，再按 Ctrl + C 复制。")
        self.add_tooltip(scrollbar, "上下拖动查看其他分析前检查项目和具体原因。")
        self.preflight_summary_var.trace_add("write", self._update_preflight_text)
        self.log_text = tk.Text(self.log_page, width=30, height=4, wrap=tk.WORD, bg=self.panel_bg,
                                fg=self.text_color, font=self.ui_log_font, relief=tk.FLAT,
                                padx=10, pady=8, state=tk.DISABLED)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_scroll = ttk.Scrollbar(self.log_page, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=self.log_scroll.set)
        self.add_tooltip(self.log_text, "这里按发生顺序记录加载、计算、保存和错误信息。内容不能修改；可拖动选中文字，再按 Ctrl + C 复制，便于核查问题。")
        self.add_tooltip(self.log_scroll, "上下拖动查看更早或更新的运行记录。")
        parent.bind("<Configure>", lambda event: self._resize_quality_labels(event.width))

    def _update_progress(self, value):
        self.progress.config(value=value)
        if hasattr(self, "progress_value"):
            self.progress_value.set(value)

    def _resize_quality_labels(self, width):
        self.qc_overview_label.configure(wraplength=max(220, width - 120))

    def _update_preflight_text(self, *_args):
        text = self.preflight_summary_label
        previous_view = text.yview()[0]
        text.configure(state=tk.NORMAL)
        text.delete("1.0", tk.END)
        text.tag_configure("block", foreground=self.warning_color)
        text.tag_configure("warn", foreground=self.key_color)
        for line in self.preflight_summary_var.get().splitlines():
            tag = "block" if line.startswith("[阻止]") else "warn" if line.startswith("[警告]") else "ok"
            text.insert(tk.END, line + "\n", (tag,))
        text.configure(state=tk.DISABLED)
        text.yview_moveto(previous_view)

    def _build_results_section(self, parent):
        self.viewer_frame = ttk.Frame(parent, style="Card.TFrame", padding=12)
        self.viewer_frame.grid(row=0, column=0, sticky="nsew")
        self.viewer_frame.columnconfigure(0, weight=1)
        self.viewer_frame.rowconfigure(2, weight=1)
        self.viewer_placeholder = ttk.Label(
            self.viewer_frame, text="尚无分析结果\n\n完成分析后，在此查看应变曲线或全场位移与应变。\n结果图支持缩放、平移和导出。",
            style="Hint.TLabel", anchor="center", justify=tk.CENTER, wraplength=440)
        self.viewer_placeholder.grid(row=2, column=0, sticky="nsew")
        viewer_btns = ttk.Frame(self.viewer_frame, style="Card.TFrame")
        viewer_btns.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        viewer_btns.columnconfigure(0, weight=1)
        ttk.Label(viewer_btns, text="分析结果", style="StepTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.viewer_export_btn = ttk.Button(viewer_btns, text="导出当前图", command=self.export_viewer_figure,
                                             style="Secondary.TButton", state=tk.DISABLED)
        self.viewer_export_btn.grid(row=0, column=1, padx=(0, 6))
        self.add_tooltip(self.viewer_export_btn, "点击选择文件名和位置，把当前显示的结果图保存为图片或 PDF 文档。图中的坐标轴、颜色刻度和当前查看范围会一起保存。")
        self.viewer_clear_btn = ttk.Button(viewer_btns, text="清除预览", command=self.clear_viewer,
                                            style="Compact.TButton", state=tk.DISABLED)
        self.viewer_clear_btn.grid(row=0, column=2)
        self.add_tooltip(self.viewer_clear_btn, "点击清空这里的结果图，图像页面恢复原图。已保存的结果文件保留；要再次显示结果，可重新分析。")
        self.viewer_content_frame = ttk.Frame(self.viewer_frame, style="Card.TFrame")
        self.viewer_content_frame.grid(row=2, column=0, sticky="nsew")
        self.viewer_content_frame.columnconfigure(0, weight=1)
        self.viewer_content_frame.rowconfigure(0, weight=1)
        self.viewer_content_frame.grid_remove()

    def sync_strain_mode_from_display(self, event=None):
        label = self.strain_mode_display.get()
        self.strain_mode.set(STRAIN_MODE_LABEL_TO_VALUE.get(label, "auto"))

    def sync_strain_mode_display(self):
        value = self.strain_mode.get()
        self.strain_mode_display.set(STRAIN_MODE_VALUE_TO_LABEL.get(value, "自动判断"))

    def sync_roi_role_from_display(self, event=None):
        label = self.roi_role_display.get()
        self.roi_role.set(ROI_ROLE_LABEL_TO_VALUE.get(label, "none"))

    def sync_roi_role_display(self):
        value = normalize_roi_role(self.roi_role.get())
        self.roi_role.set(value)
        self.roi_role_display.set(ROI_ROLE_VALUE_TO_LABEL.get(value, "普通"))

    def apply_tracking_preset(self, event=None):
        name = self.tracking_preset.get()
        preset = TRACKING_PRESETS.get(name)
        if not preset:
            return

        self._applying_preset = True
        try:
            self.search_radius.set(preset["search_radius"])
            self.hard_corr.set(preset["hard_corr"])
            self.soft_corr.set(preset["soft_corr"])
            self.max_frame_strain_jump.set(preset["max_frame_strain_jump"])
            self.fb_tolerance_px.set(preset["fb_tolerance_px"])
        finally:
            self._applying_preset = False

        self.preset_status_var.set(f"当前追踪模式：{name}")
        if hasattr(self, "status_var"):
            self.status_var.set(f"当前追踪模式：{name}")

    def mark_tracking_custom(self, *args):
        if getattr(self, "_applying_preset", False):
            return
        if not hasattr(self, "preset_status_var"):
            return
        if self.tracking_preset.get() != "自定义":
            self.tracking_preset.set("自定义")
        self.preset_status_var.set("当前追踪模式：自定义")

    def toggle_advanced_settings(self):
        if self.advanced_visible.get():
            self.advanced_frame.grid_remove()
            self.advanced_visible.set(False)
            self.advanced_toggle_btn.config(text="显示高级设置")
            self.pixel_size_label.grid_remove()
            self.pixel_size_entry.grid_remove()
            self.preset_status_label.grid_remove()
        else:
            self.advanced_frame.grid()
            self.advanced_visible.set(True)
            self.advanced_toggle_btn.config(text="隐藏高级设置")
            self.pixel_size_label.grid()
            self.pixel_size_entry.grid()
            self.preset_status_label.grid()

    def show_usage_notice(self):
        messagebox.showinfo("About / Citation / Usage Notice", USAGE_NOTICE)

    # ---------- 日志和文件 ----------

    def log(self, msg):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, str(msg) + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)
        self.root.update_idletasks()

    def log_user_error(self, context, exc, details=None):
        message = str(exc).strip() or exc.__class__.__name__
        self.log(f"{context}失败：{message}")
        has_console = sys.stderr is not None or sys.stdout is not None
        if has_console:
            self.log("请检查输入路径、图像文件、ROI 和参数设置；详细调试信息已输出到控制台。")
            if details:
                stream = sys.stderr if sys.stderr is not None else sys.stdout
                print(details, file=stream)
            elif sys.stderr is not None:
                traceback.print_exc()
        else:
            self.log("请检查输入路径、图像文件、ROI 和参数设置。")

    def show_completion_and_open_output_folder(self, done_msg, output_dir):
        messagebox.showinfo("完成", done_msg)
        self.remember_recent_paths(
            image_dir=self.image_folder.get().strip() or None,
            output_dir=output_dir,
        )
        try:
            open_output_folder(output_dir)
        except Exception as exc:
            self.log(f"无法自动打开结果目录：{exc}")

    def clear_sequence_dependent_state(self):
        self.roi1 = None
        self.roi2 = None
        self.field_roi = None
        self.dic_mask_exclusions = []
        self._drawing_mask_exclusion = False
        self._canvas_mask_overlay_rgb = None
        self.dic_mask_path.set("")
        self.dic_mask_mode.set("矩形 ROI")
        self.roi1_reference_frame_1based = None
        self.roi2_reference_frame_1based = None
        self.field_roi_reference_frame_1based = None
        self.dic_last_field = None
        self.dic_last_image = None
        self.dic_last_frame_1based = None
        self.dic_last_filename = None
        self.dic_last_reference_frame_1based = None
        self.dic_last_reference_filename = None
        self.current_roi_index = 1
        self.drag_start = None
        self.temp_rect_id = None
        self.roi_groups.clear()
        self.next_group_idx = 1
        self.group_name_var.set("")
        self.refresh_group_tree()
        self.refresh_current_roi_summary()
        self.last_qc_summary = None
        if hasattr(self, "qc_overview_var"):
            self.qc_overview_var.set("分析完成后显示 QC 总览。")
        if hasattr(self, "clear_viewer"):
            self.clear_viewer(keep_placeholder=True)

    def select_image_folder(self):
        folder = filedialog.askdirectory(title="选择图像序列文件夹")
        if folder:
            self._capture_selection = None
            self._capture_prepared = None
            self.image_folder.set(folder)
            if not str(self.output_folder.get() or "").strip():
                default_name = (
                    "ezDIC_fullfield_output"
                    if self.is_fullfield_mode()
                    else "ezDIC_extensometer_output"
                )
                self.output_folder.set(os.path.join(folder, default_name))
            self.remember_recent_paths(image_dir=folder, output_dir=self.output_folder.get())
            self.log(f"图像文件夹：{folder}")

    def _choose_capture_experiment(self, *, live=False):
        folder = filedialog.askdirectory(parent=self.root, title="选择 StrainTrace Capture 实验")
        if not folder:
            return
        try:
            from straintrace_capture import ExperimentReader
            from straintrace_capture.handoff import launch_analysis

            with ExperimentReader(folder) as reader:
                cameras = [row["device"] for row in reader.cameras()]
            camera = None
            if len(cameras) > 1 and not live:
                options = "\n".join(f"{item['model']} · {item['id']}" for item in cameras)
                camera = simpledialog.askstring("选择相机", f"输入相机 ID\n{options}", parent=self.root,
                                               initialvalue=cameras[0]["id"])
                if not camera:
                    return
            if live:
                launch_analysis(folder, camera_id=camera, live=True)
            else:
                self.open_capture_experiment(folder, camera)
        except Exception as exc:
            messagebox.showerror("无法打开采集实验", str(exc), parent=self.root)

    def _open_capture_workbench(self):
        from straintrace_capture.handoff import launch_capture

        selection = getattr(self, "_capture_selection", None)
        try:
            launch_capture(selection[0] if selection else None)
        except OSError as exc:
            messagebox.showerror("无法启动 Capture", str(exc), parent=self.root)

    def open_capture_experiment(self, folder, camera_id=None, reference_sequence=None):
        from straintrace_capture import CaptureError
        from straintrace_capture.handoff import analysis_paths, analysis_snapshot, derive_analysis_inputs, write_snapshot

        if self.is_processing or self._completion_pending:
            messagebox.showinfo("正在处理", "当前分析完成后可打开采集实验。", parent=self.root)
            return
        try:
            binding = analysis_snapshot(folder, camera_id, reference_sequence)
        except CaptureError as exc:
            if exc.code != "CHOOSE_REFERENCE" or reference_sequence is not None:
                raise
            reference_sequence = simpledialog.askinteger("选择参考图像", "输入已提交参考帧的采集序号",
                                                         minvalue=1, parent=self.root)
            if reference_sequence is None:
                return
            binding = analysis_snapshot(folder, camera_id, reference_sequence)
        root = binding["experiment_root"]
        prepared = None
        try:
            analysis_paths(binding)
        except CaptureError as exc:
            if exc.code not in {"BAYER_DERIVATION_REQUIRED", "ALPHA_DERIVATION_REQUIRED"}:
                raise
            if not messagebox.askyesno("转换分析输入", "按相机像素格式生成独立的亮度图像序列并载入分析？", parent=self.root):
                return
            from uuid import uuid4

            derived = Path(root) / "analysis-inputs" / str(uuid4())
            prepared = derive_analysis_inputs(binding, derived, debayer=True, drop_alpha=True)
            write_snapshot(prepared, derived / "capture_binding.json")
        self._capture_selection = (root, binding["camera_id"], binding["frames"][0]["sequence"])
        self._capture_prepared = prepared
        self.image_folder.set(root)
        self.output_folder.set(str(Path(root) / "analysis" / "offline"))
        self.load_first_image()
        self.log(f"Capture 实验 {binding['experiment_id']} · {binding['camera_id']}")

    def select_output_folder(self):
        folder = filedialog.askdirectory(title="选择输出文件夹")
        if folder:
            self.output_folder.set(folder)
            self.remember_recent_paths(output_dir=folder)
            self.log(f"输出文件夹：{folder}")

    def load_first_image(self):
        """
        加载图像序列，并显示当前 preview_frame_1based 指定的帧。
        保留函数名是为了兼容前面按钮调用。
        """
        if self.is_processing or self._completion_pending:
            messagebox.showinfo("正在处理", "当前分析仍在处理或收尾，请完成后再刷新图像序列。")
            return
        folder = self.image_folder.get().strip()
        if not folder:
            messagebox.showwarning("缺少文件夹", "请先选择图像文件夹。")
            return

        selection = getattr(self, "_capture_selection", None)
        if selection and Path(folder).resolve() == Path(selection[0]):
            from straintrace_capture.handoff import analysis_paths, analysis_snapshot

            try:
                binding = getattr(self, "_capture_prepared", None) or analysis_snapshot(*selection)
                new_image_paths = analysis_paths(binding)
            except Exception as exc:
                messagebox.showerror("无法读取采集图像", str(exc), parent=self.root)
                return
        else:
            new_image_paths = collect_images(folder)
        if not new_image_paths:
            messagebox.showerror("未找到图像", "该文件夹中没有找到 tif/tiff/png/jpg/bmp 图像。")
            return

        preview, preview_err = self._safe_int_var(self.preview_frame_1based)
        start, start_err = self._safe_int_var(self.start_frame_1based)
        end, end_err = self._safe_int_var(self.end_frame_1based)
        if preview_err or start_err or end_err:
            bad = []
            if preview_err:
                bad.append("预览帧")
            if start_err:
                bad.append("起始帧")
            if end_err:
                bad.append("结束帧")
            message = f"{'、'.join(bad)}必须是整数，未加载新序列。"
            messagebox.showerror("加载失败", message)
            self.log(f"加载图像序列失败：{message}")
            return

        old_state = {
            "image_paths": list(self.image_paths),
            "loaded_image_folder": self.loaded_image_folder,
            "loaded_image_sequence_fingerprint": self.loaded_image_sequence_fingerprint,
            "preview_raw": self.preview_raw,
            "preview_img8": self.preview_img8,
            "reference_raw": self.reference_raw,
            "reference_img8": self.reference_img8,
            "reference_cache_key": self._reference_cache_key,
            "current_fullres_img8": self.current_fullres_img8,
            "canvas_mask_overlay_rgb": self._canvas_mask_overlay_rgb,
            "display_img": self.display_img,
            "display_scale": self.display_scale,
            "photo": self.photo,
            "preview": preview,
            "start": start,
            "end": end,
            "current_preview_index": self.current_preview_index,
        }

        n = len(new_image_paths)
        folder_key = os.path.normcase(os.path.abspath(folder))
        new_fingerprint = image_sequence_fingerprint(new_image_paths)
        sequence_changed = (
            self.loaded_image_folder is not None
            and (
                self.loaded_image_folder != folder_key
                or self.loaded_image_sequence_fingerprint != new_fingerprint
            )
        )

        # 如果用户还没设置范围，默认 1 到最后一帧。
        if end <= 1:
            self.end_frame_1based.set(n)

        self.image_paths = new_image_paths
        self._invalidate_reference_cache()
        # Preview is display-only and may be brought into range for a valid
        # first render.  Start/end remain untouched so out-of-range input is
        # surfaced by preflight instead of silently clamped.
        preview = max(1, min(preview, n))
        self.preview_frame_1based.set(preview)

        try:
            self.load_preview_frame(preview - 1)
            if sequence_changed:
                self.clear_sequence_dependent_state()
            self.loaded_image_folder = folder_key
            self.loaded_image_sequence_fingerprint = new_fingerprint
            if 1 <= start <= n:
                self._get_reference_frame_images(start - 1)
            self.show_image()
            self.log(f"找到 {n} 张图像。")
            self.log(f"当前预览：第 {self.current_preview_index + 1} 帧 / 共 {n} 帧")
            self.remember_recent_paths(image_dir=folder, output_dir=self.output_folder.get().strip() or None)
            self.update_workflow_action_states()
        except Exception as exc:
            self.image_paths = old_state["image_paths"]
            self.loaded_image_folder = old_state["loaded_image_folder"]
            self.loaded_image_sequence_fingerprint = old_state["loaded_image_sequence_fingerprint"]
            self.preview_raw = old_state["preview_raw"]
            self.preview_img8 = old_state["preview_img8"]
            self.reference_raw = old_state["reference_raw"]
            self.reference_img8 = old_state["reference_img8"]
            self._reference_cache_key = old_state["reference_cache_key"]
            self.current_fullres_img8 = old_state["current_fullres_img8"]
            self._canvas_mask_overlay_rgb = old_state["canvas_mask_overlay_rgb"]
            self.display_img = old_state["display_img"]
            self.display_scale = old_state["display_scale"]
            self.photo = old_state["photo"]
            self.preview_frame_1based.set(old_state["preview"])
            self.start_frame_1based.set(old_state["start"])
            self.end_frame_1based.set(old_state["end"])
            self.current_preview_index = old_state["current_preview_index"]
            self.show_image()
            messagebox.showerror("加载失败", str(exc))
            self.log_user_error("加载图像序列", exc)
            self.update_workflow_action_states()

    def load_preview_frame(self, index0):
        if not self.image_paths:
            raise RuntimeError("请先加载图像序列。")

        n = len(self.image_paths)
        index0 = int(max(0, min(index0, n - 1)))

        self.current_preview_index = index0
        self._show_image_workspace()
        self.preview_frame_1based.set(index0 + 1)

        path = self.image_paths[index0]
        self.preview_raw = read_gray_image(path)
        self.preview_img8 = normalize_to_uint8(self.preview_raw)
        self.current_fullres_img8 = self.preview_img8.copy()
        self._canvas_mask_overlay_rgb = None
        self._canvas_shows_field_overlay = False

        # Keep the cached reference frame strictly bound to start_idx; only
        # refresh it here when the user is previewing the current start frame.
        start_val, start_err = self._safe_int_var(self.start_frame_1based)
        if start_err is None and start_val - 1 == index0:
            self.reference_raw = self.preview_raw
            self.reference_img8 = self.preview_img8
            self._reference_cache_key = (
                int(index0),
                str(path),
                getattr(self, "loaded_image_sequence_fingerprint", None),
            )

        self.display_img = None
        self.display_scale = 1.0
        self.zoom_factor = 1.0
        self.auto_fit_enabled = True
        self.root.update_idletasks()
        self._rescale_display_to_current_size()
        if self.display_img is None:
            self.display_img, self.display_scale = get_display_image(self.preview_img8, max_w=1280, max_h=820)
            self.zoom_factor = self.display_scale
            self.show_image()
        self._update_zoom_label()
        self._update_image_toolbar_state()

        # 绑定一次 resize 监听（只绑一次）
        self._bind_image_resize_handler()
        try:
            self.root.after_idle(self._rescale_display_to_current_size)
        except tk.TclError:
            pass
        start, _ = self._safe_int_var(self.start_frame_1based)
        end, _ = self._safe_int_var(self.end_frame_1based)
        range_text = f"{start}–{end}" if start is not None and end is not None else "输入待校验"
        self.status_var.set(
            f"预览第 {index0 + 1}/{n} 帧：{os.path.basename(path)} | 分析范围 {range_text}"
        )
        self._update_preview_scale_label()
        self.log(f"已显示第 {index0 + 1} 帧：{path}")
        self.log(f"图像尺寸：{self.preview_img8.shape[1]} × {self.preview_img8.shape[0]} px")

    def go_to_preview_frame(self):
        if not self.image_paths:
            self.load_first_image()
            return

        try:
            idx = self.get_int_setting(self.preview_frame_1based, "预览帧") - 1
            self.load_preview_frame(idx)
        except Exception as exc:
            messagebox.showerror("预览失败", str(exc))
            self.log_user_error("预览帧", exc)

    def step_preview_frame(self, step):
        if not self.image_paths:
            self.load_first_image()
            return

        try:
            self.load_preview_frame(self.current_preview_index + int(step))
        except Exception as exc:
            messagebox.showerror("预览失败", str(exc))
            self.log_user_error("预览帧", exc)

    def clear_groups_if_start_changes(self, new_start_1based):
        old_start, old_start_err = self._safe_int_var(self.start_frame_1based)
        if old_start_err is None and new_start_1based == old_start:
            return True

        has_reference_state = bool(
            self.roi_groups
            or self.roi1 is not None
            or self.roi2 is not None
            or self.field_roi is not None
        )
        if has_reference_state:
            msg = (
                f"当前已有 {len(self.roi_groups)} 组 ROI 或正在编辑的 ROI。\n\n"
                f"这些 ROI 应该是在当前起始/参考帧第 {old_start if old_start is not None else '未知'} 帧上定义的。\n"
                f"如果把起始/参考帧改为第 {new_start_1based} 帧，已有 ROI 模板可能不再对应。\n\n"
                f"建议清空已有 ROI、结果和 QC，并在新参考帧上重画。是否清空并修改起始帧？"
            )
            if not messagebox.askyesno("修改起始/参考帧", msg):
                return False
            self.clear_sequence_dependent_state()
            self.log("由于修改起始/参考帧，已清空不兼容的 ROI、结果和 QC。")
        return True

    def set_start_to_current(self):
        if not self.image_paths:
            messagebox.showwarning("未加载图像", "请先加载图像序列。")
            return

        new_start = self.current_preview_index + 1
        if not self.clear_groups_if_start_changes(new_start):
            return

        self.start_frame_1based.set(new_start)
        self._get_reference_frame_images(new_start - 1)
        end, end_err = self._safe_int_var(self.end_frame_1based)
        if end_err is not None or end < new_start:
            self.end_frame_1based.set(len(self.image_paths))
        self.log(f"已将第 {new_start} 帧设为分析起始/参考帧。请在这张图上画 ROI。")
        end, _ = self._safe_int_var(self.end_frame_1based)
        self.status_var.set(f"起始/参考帧 = {new_start}；结束帧 = {end if end is not None else '待校验'}")

    def set_end_to_current(self):
        if not self.image_paths:
            messagebox.showwarning("未加载图像", "请先加载图像序列。")
            return

        end = self.current_preview_index + 1
        self.end_frame_1based.set(end)
        self.log(f"已将第 {end} 帧设为分析结束帧。")
        start, _ = self._safe_int_var(self.start_frame_1based)
        self.status_var.set(f"起始/参考帧 = {start if start is not None else '待校验'}；结束帧 = {end}")

    def get_analysis_indices(self):
        if not self.image_paths:
            raise RuntimeError("请先加载图像序列。")

        n = len(self.image_paths)
        s = self.get_int_setting(self.start_frame_1based, "起始帧") - 1
        e = self.get_int_setting(self.end_frame_1based, "结束帧") - 1

        if s < 0 or s >= n or e < 0 or e >= n:
            raise RuntimeError(f"分析范围必须位于 1 到 {n} 帧之间：start={s + 1}, end={e + 1}")

        if e < s:
            raise RuntimeError(
                f"分析结束帧不能早于起始帧：start={s + 1}, end={e + 1}"
            )

        return s, e

    def update_workflow_action_states(self):
        """根据当前工作流进度控制关键操作按钮，避免用户在不可运行状态下误点。"""
        has_sequence = bool(self.image_paths) and self.first_img8 is not None
        has_groups = bool(self.roi_groups)
        has_field_roi = self.field_roi is not None
        has_current_roi = self.field_roi is not None if self.is_fullfield_mode() else (self.roi1 is not None or self.roi2 is not None)
        is_processing = getattr(self, "is_processing", False)
        completion_pending = getattr(self, "_completion_pending", False)
        is_ff = self.is_fullfield_mode()
        preflight_items = self.refresh_preflight_panel() if hasattr(self, "preflight_summary_var") else []
        blocking_items = [item for item in preflight_items if item["level"] == "block"]
        warning_items = [item for item in preflight_items if item["level"] == "warn"]
        busy = is_processing or completion_pending
        self._set_run_controls_locked(busy)
        self.refresh_current_roi_summary()
        if hasattr(self, "dic_field_summary_var"):
            roi = self.field_roi
            if roi is None:
                self.dic_field_summary_var.set("尚未绘制全场 ROI。")
            else:
                x, y, w, h = roi
                self.dic_field_summary_var.set(f"ROI {w}×{h} px @ ({x},{y})")

        can_switch_mode = not is_processing and not completion_pending
        for attr in ("mode_extensometer_radio", "mode_fullfield_radio"):
            if hasattr(self, attr):
                getattr(self, attr).config(state=tk.NORMAL if can_switch_mode else tk.DISABLED)

        if hasattr(self, "start_button"):
            ready = has_field_roi if is_ff else has_groups
            can_start = has_sequence and ready and not blocking_items and not is_processing and not completion_pending
            self.start_button.config(state=tk.NORMAL if can_start else tk.DISABLED)
            self.start_button.configure(text="正在分析…" if is_processing else "正在收尾…" if completion_pending else "开始分析")

        has_pair = self.roi1 is not None and self.roi2 is not None
        selected_group = bool(self.group_tree.selection()) if hasattr(self, "group_tree") else has_groups
        for attr in ("roi1_button", "roi2_button", "draw_field_roi_button", "show_preview_button", "set_start_button", "set_end_button"):
            if hasattr(self, attr):
                getattr(self, attr).configure(state=tk.NORMAL if has_sequence and not busy else tk.DISABLED)
        for attr in ("align_x_button", "align_y_button", "add_group_button"):
            if hasattr(self, attr):
                getattr(self, attr).configure(state=tk.NORMAL if has_sequence and has_pair and not busy else tk.DISABLED)
        for attr, idx in (("roi1_button", 1), ("roi2_button", 2)):
            if hasattr(self, attr):
                getattr(self, attr).configure(style="Secondary.TButton" if self.current_roi_index == idx else "Compact.TButton")
        if hasattr(self, "prev_frame_button"):
            self.prev_frame_button.configure(state=tk.NORMAL if has_sequence and self.current_preview_index > 0 and not busy else tk.DISABLED)
            self.next_frame_button.configure(state=tk.NORMAL if has_sequence and self.current_preview_index < len(self.image_paths) - 1 and not busy else tk.DISABLED)

        if hasattr(self, "clear_rois_button"):
            can_clear = has_current_roi and not busy
            self.clear_rois_button.config(state=tk.NORMAL if can_clear else tk.DISABLED)

        can_manage_groups = has_groups and selected_group and not busy
        for attr in ("load_group_button", "update_group_button", "delete_group_button"):
            if hasattr(self, attr):
                can_use = can_manage_groups and (has_pair if attr == "update_group_button" else True)
                getattr(self, attr).config(state=tk.NORMAL if can_use else tk.DISABLED)

        if hasattr(self, "workflow_hint_var"):
            if is_processing:
                hint = "正在计算；可在“质量与日志”查看详情。"
            elif completion_pending:
                hint = "正在保存结果并更新预览，请稍候。"
            elif blocking_items:
                first = blocking_items[0]
                hint = f"{first['label']}：{first['message']}"
            elif not has_sequence:
                hint = "请选择图像文件夹并加载序列。"
            elif is_ff and not has_field_roi:
                hint = "请在参考帧绘制全场 ROI。"
            elif not is_ff and not has_groups:
                hint = "请绘制 ROI 1 和 ROI 2，并添加 ROI 组。"
            elif warning_items:
                hint = f"可分析；请复核{warning_items[0]['label']}。详情见“质量与日志”。"
            else:
                hint = "准备就绪，分析后自动显示结果。"
            self.workflow_hint_var.set(hint)

        if hasattr(self, "run_state_var"):
            failed = self.status_var.get().startswith("分析失败")
            if is_processing:
                state = "分析中"
            elif completion_pending:
                state = "正在收尾"
            elif not has_sequence:
                state = "待加载"
            elif failed:
                state = "分析失败 · 可重试"
            elif blocking_items:
                state = "待设置"
            elif warning_items:
                state = "可分析 · 有提示"
            else:
                state = "准备就绪"
            self.run_state_var.set(state)
            style = "Error.TLabel" if failed and not busy else "Ready.TLabel" if not busy and has_sequence and not blocking_items else "Badge.TLabel"
            self.run_state_label.configure(style=style)
        self._update_image_context()

        self._update_image_toolbar_state()

    def _set_run_controls_locked(self, locked):
        """Freeze editable inputs while an immutable run snapshot is executing."""
        if not hasattr(self, "controls_panel"):
            return
        previous = getattr(self, "_run_controls_locked", False)
        if previous == locked:
            return
        self._run_controls_locked = locked
        if locked:
            self._run_control_states = []
            def freeze(widget):
                if isinstance(widget, (ttk.Entry, ttk.Combobox, ttk.Button, ttk.Checkbutton, ttk.Radiobutton, ttk.Treeview)):
                    self._run_control_states.append((widget, widget.instate(["disabled"])))
                    widget.state(["disabled"])
                for child in widget.winfo_children():
                    freeze(child)
            freeze(self.project_frame)
            freeze(self.controls_panel)
        else:
            for widget, was_disabled in self._run_control_states:
                if widget.winfo_exists():
                    widget.state(["disabled"] if was_disabled else ["!disabled"])
            self._run_control_states = []

    # ---------- 对齐与组管理 ----------

    def auto_choose_alignment_axis(self):
        """
        返回 'x' 或 'y'。
        x 表示水平对齐，使用 x 方向标距；
        y 表示垂直对齐，使用 y 方向标距。
        """
        selected = self.strain_mode.get()
        if selected in ("x", "y"):
            return selected

        if self.roi1 is None or self.roi2 is None:
            return "x"

        dx, dy, _ = roi_separation(self.roi1, self.roi2)
        return "x" if dx >= dy else "y"

    def align_current_pair(self, axis, set_mode=True):
        if self.first_img8 is None:
            messagebox.showwarning("未加载图像", "请先加载第一张图。")
            return
        if self.roi1 is None or self.roi2 is None:
            messagebox.showwarning("缺少 ROI", "请先画 ROI 1 和 ROI 2。")
            return

        c1x, c1y = rect_center(self.roi1)

        if axis == "x":
            self.roi2 = move_rect_center(self.roi2, new_cy=c1y, img_shape=self.first_img8.shape)
            if set_mode:
                self.strain_mode.set("x")
                self.sync_strain_mode_display()
            self.log("已水平对齐：ROI2 中心 y 已强制等于 ROI1 中心 y；建议使用横向应变。")
        elif axis == "y":
            self.roi2 = move_rect_center(self.roi2, new_cx=c1x, img_shape=self.first_img8.shape)
            if set_mode:
                self.strain_mode.set("y")
                self.sync_strain_mode_display()
            self.log("已垂直对齐：ROI2 中心 x 已强制等于 ROI1 中心 x；建议使用纵向应变。")
        else:
            raise ValueError("axis must be x or y")

        dx, dy, dist = roi_separation(self.roi1, self.roi2)
        self.log(f"对齐后距离：dx={dx:.3f}px, dy={dy:.3f}px, distance={dist:.3f}px")
        self.log_roi_texture("当前 ROI 1", self.roi1)
        self.log_roi_texture("当前 ROI 2", self.roi2)
        self.show_image()

    def make_group_from_current(self, name=None):
        if self.first_img8 is None:
            raise RuntimeError("请先加载第一张图。")
        if self.roi1 is None or self.roi2 is None:
            raise RuntimeError("请先画 ROI 1 和 ROI 2。")

        # 安全限制：ROI 必须在分析起始/参考帧上定义。
        start_1based = self.get_int_setting(self.start_frame_1based, "起始帧")
        current_1based = int(self.current_preview_index + 1)
        if current_1based != start_1based:
            msg = (
                f"当前 ROI 是在第 {current_1based} 帧上画的，"
                f"但分析起始/参考帧是第 {start_1based} 帧。\n\n"
                f"为了保证模板和初始 ROI 一致，请先显示第 {start_1based} 帧再画 ROI，"
                f"或者点击“设为起始/参考”。"
            )
            raise RuntimeError(msg)
        for label, reference_frame in (
            ("ROI1", self.roi1_reference_frame_1based),
            ("ROI2", self.roi2_reference_frame_1based),
        ):
            if reference_frame is not None and int(reference_frame) != start_1based:
                raise RuntimeError(
                    f"{label} 在第 {reference_frame} 帧定义，但当前起始/参考帧为第 {start_1based} 帧，请重画。"
                )

        selected = self.strain_mode.get()
        actual = resolve_strain_mode(self.roi1, self.roi2, selected)
        dx, dy, dist = roi_separation(self.roi1, self.roi2)
        L0 = length_between(self.roi1, self.roi2, actual)

        if name is None or not str(name).strip():
            name = f"G{self.next_group_idx:02d}"
            self.next_group_idx += 1

        name = safe_name(name)

        group = {
            "name": name,
            "roi1": tuple(self.roi1),
            "roi2": tuple(self.roi2),
            "role": normalize_roi_role(self.roi_role.get()),
            "selected_mode": selected,
            "actual_mode": actual,
            "dx0": dx,
            "dy0": dy,
            "dist0": dist,
            "L0": L0,
            "reference_frame_1based": current_1based,
            "roi1_reference_frame_1based": self.roi1_reference_frame_1based or current_1based,
            "roi2_reference_frame_1based": self.roi2_reference_frame_1based or current_1based,
        }

        return group

    def check_group_warnings(self, group):
        messages = []

        if group["L0"] <= 0 or not np.isfinite(group["L0"]):
            messages.append(
                "L0 无效（≤0），两个 ROI 中心重合或计算错误，无法计算应变。请重新绘制。"
            )
        elif group["L0"] < 50:
            messages.append(
                f"L0 偏小：{group['L0']:.1f}px，线性应变误差会被放大。"
            )

        m1 = roi_texture_metrics(self.first_img8, group["roi1"])
        m2 = roi_texture_metrics(self.first_img8, group["roi2"])

        ok1 = texture_is_ok(
            m1,
            self.min_texture_std.get(),
            self.min_texture_contrast.get(),
            self.max_saturated_frac.get(),
        )
        ok2 = texture_is_ok(
            m2,
            self.min_texture_std.get(),
            self.min_texture_contrast.get(),
            self.max_saturated_frac.get(),
        )

        if not ok1:
            messages.append(
                f"ROI1 质量：偏弱，建议增大 ROI 或选择更独特散斑。std={m1['std_gray']:.1f}, P95-P5={m1['contrast_p95_p5']:.1f}。"
            )
        if not ok2:
            messages.append(
                f"ROI2 质量：偏弱，建议增大 ROI 或选择更独特散斑。std={m2['std_gray']:.1f}, P95-P5={m2['contrast_p95_p5']:.1f}。"
            )

        return messages

    def add_current_group(self):
        try:
            group = self.make_group_from_current(self.group_name_var.get())
        except Exception as exc:
            messagebox.showerror("无法添加 ROI 组", str(exc))
            return

        if any(g["name"] == group["name"] for g in self.roi_groups):
            messagebox.showerror("组名重复", f"已经存在组名：{group['name']}")
            return

        warnings = self.check_group_warnings(group)
        if warnings:
            text = "\n".join(warnings) + "\n\n是否仍然添加该组？"
            if not messagebox.askyesno("ROI 组警告", text):
                return

        self.roi_groups.append(group)
        self.refresh_group_tree()
        self.log_group_info("已添加", group)
        self.group_name_var.set("")
        self.show_image()

    def update_selected_group(self):
        selected_iid = self.get_selected_group_iid()
        if selected_iid is None:
            messagebox.showwarning("未选择组", "请先在列表中选择一组。")
            return

        edited_name = self.group_name_var.get().strip()
        target_name = edited_name if edited_name else self.roi_groups[selected_iid]["name"]
        try:
            new_group = self.make_group_from_current(target_name)
        except Exception as exc:
            messagebox.showerror("无法更新 ROI 组", str(exc))
            return

        if any(i != selected_iid and g["name"] == new_group["name"] for i, g in enumerate(self.roi_groups)):
            messagebox.showerror("组名重复", f"已经存在组名：{new_group['name']}")
            return

        warnings = self.check_group_warnings(new_group)
        if warnings:
            text = "\n".join(warnings) + "\n\n是否仍然更新该组？"
            if not messagebox.askyesno("ROI 组警告", text):
                return

        self.roi_groups[selected_iid] = new_group
        self.group_name_var.set(new_group["name"])
        self.refresh_group_tree()
        self.log_group_info("已更新", new_group)
        self.show_image()

    def get_selected_group_iid(self):
        sel = self.group_tree.selection()
        if not sel:
            return None
        try:
            return int(sel[0])
        except Exception:
            return None

    def load_selected_group(self):
        idx = self.get_selected_group_iid()
        if idx is None:
            messagebox.showwarning("未选择组", "请先在列表中选择一组。")
            return
        if idx < 0 or idx >= len(self.roi_groups):
            return

        group = self.roi_groups[idx]
        self.roi1 = tuple(group["roi1"])
        self.roi2 = tuple(group["roi2"])
        self.roi1_reference_frame_1based = group.get(
            "roi1_reference_frame_1based", group.get("reference_frame_1based")
        )
        self.roi2_reference_frame_1based = group.get(
            "roi2_reference_frame_1based", group.get("reference_frame_1based")
        )
        self.strain_mode.set(group["selected_mode"])
        self.sync_strain_mode_display()
        self.roi_role.set(normalize_roi_role(group.get("role", "none")))
        self.sync_roi_role_display()
        self.group_name_var.set(group["name"])
        self.log_group_info("已载入", group)
        self.show_image()
        self.update_workflow_action_states()

    def delete_selected_group(self):
        idx = self.get_selected_group_iid()
        if idx is None:
            messagebox.showwarning("未选择组", "请先在列表中选择一组。")
            return
        if idx < 0 or idx >= len(self.roi_groups):
            return

        group = self.roi_groups[idx]
        if not messagebox.askyesno("删除 ROI 组", f"确定删除 {group['name']} 吗？"):
            return

        del self.roi_groups[idx]
        self.refresh_group_tree()
        self.log(f"已删除组：{group['name']}")
        self.show_image()

    def _show_group_tree_context_menu(self, event):
        """为 ROI 组列表提供右键快捷菜单，提升多组操作效率。"""
        iid = self.group_tree.identify_row(event.y)
        if not iid:
            return
        try:
            idx = int(iid)
        except ValueError:
            return
        if idx < 0 or idx >= len(self.roi_groups):
            return

        # 临时选中该行
        self.group_tree.selection_set(iid)
        self.update_workflow_action_states()

        previous_menu = getattr(self, "_group_context_menu", None)
        if previous_menu is not None:
            try:
                previous_menu.destroy()
            except tk.TclError:
                pass
            self._group_context_menu = None

        menu = tk.Menu(self.group_tree, tearoff=0)
        self._group_context_menu = menu
        menu.add_command(label="载入选中组", command=self.load_selected_group, state=self.load_group_button.cget("state"))
        menu.add_command(label="更新选中组", command=self.update_selected_group, state=self.update_group_button.cget("state"))
        menu.add_separator()
        menu.add_command(label="删除选中组", command=self.delete_selected_group, state=self.delete_group_button.cget("state"))
        actions = {0: self.load_group_button, 1: self.update_group_button, 3: self.delete_group_button}
        self.add_tooltip(menu, lambda _event: actions[menu.index("active")]._tooltip.resolve_text()
                         if menu.index("active") in actions else "")
        menu.bind("<<MenuSelect>>", menu._tooltip.schedule, add="+")

        def _cleanup_menu(_event=None, target=menu):
            if getattr(self, "_group_context_menu", None) is target:
                self._group_context_menu = None
            try:
                if target.winfo_exists():
                    target.destroy()
            except tk.TclError:
                pass

        menu.bind("<Unmap>", lambda _event: self.root.after_idle(_cleanup_menu), add="+")
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            try:
                menu.grab_release()
            except tk.TclError:
                pass

    def refresh_group_tree(self):
        self.group_tree.delete(*self.group_tree.get_children())
        for idx, g in enumerate(self.roi_groups):
            selected_label = STRAIN_MODE_VALUE_TO_LABEL.get(g.get("selected_mode"), g.get("selected_mode", ""))
            actual_label = STRAIN_MODE_VALUE_TO_LABEL.get(g.get("actual_mode"), g.get("actual_mode", ""))
            values = (
                g["name"],
                ROI_ROLE_VALUE_TO_LABEL.get(normalize_roi_role(g.get("role", "none")), "普通"),
                selected_label,
                actual_label,
                f"{g['L0']:.1f} px",
                f"{g['dx0']:.1f} px",
                f"{g['dy0']:.1f} px",
                self._format_roi_for_tree(g["roi1"]),
                self._format_roi_for_tree(g["roi2"]),
            )
            self.group_tree.insert("", "end", iid=str(idx), values=values)
        self.update_workflow_action_states()

    @staticmethod
    def _format_roi_for_tree(rect):
        x, y, w, h = (int(round(value)) for value in rect)
        return f"({x}, {y}, {w}, {h}) px"

    def log_group_info(self, prefix, group):
        self.log(
            f"{prefix}组 {group['name']}: role={normalize_roi_role(group.get('role', 'none'))}, "
            f"selected={group['selected_mode']}, "
            f"actual={group['actual_mode']}, L0={group['L0']:.3f}px, "
            f"dx={group['dx0']:.3f}px, dy={group['dy0']:.3f}px, "
            f"ROI1={group['roi1']}, ROI2={group['roi2']}"
        )

    def log_roi_texture(self, name, rect):
        img8 = self.preview_img8
        if img8 is None or rect is None:
            return

        metrics = roi_texture_metrics(img8, rect)
        min_std, min_contrast, max_sat = self._safe_texture_thresholds()
        ok = texture_is_ok(metrics, min_std, min_contrast, max_sat)

        quality = "良好" if ok else "偏弱，建议增大 ROI 或选择更独特散斑"
        msg = (
            f"{name} 质量：{quality}。"
            f"std={metrics['std_gray']:.2f}, P95-P5={metrics['contrast_p95_p5']:.2f}, "
            f"low={metrics['low_frac']:.3f}, high={metrics['high_frac']:.3f}"
        )
        self.log(msg)

    def _core_texture_metrics_and_code(self, rect, image=None):
        """Run the shared texture contract for a GUI preflight ROI on the true reference frame."""
        if image is None:
            _ref_raw, image = self._get_reference_frame_images()
            if image is None and not self.image_paths:
                image = self.preview_img8
        if image is None or rect is None:
            return None, "INVALID_TEXTURE"
        metrics = roi_texture_metrics(image, rect)
        min_std, min_contrast, max_sat = self._safe_texture_thresholds()
        code = texture_failure_code(
            metrics,
            min_std,
            min_contrast,
            max_sat,
            _core.DEFAULT_TEXTURE_MIN_STRUCTURE_RATIO,
        )
        return metrics, code

    def _raise_on_ambiguous_texture(self, rois, image=None):
        """Block a scientific run for an ambiguous near-1D texture only."""
        for label, rect in rois:
            _metrics, code = self._core_texture_metrics_and_code(rect, image=image)
            if code == "AMBIGUOUS_TEXTURE":
                raise RuntimeError(f"AMBIGUOUS_TEXTURE：{label} 的二维纹理不可辨识，请更换或增大 ROI。")

    # ---------- 处理前检查 ----------

    def validate_before_processing(self):
        if not self.image_paths:
            raise RuntimeError("请先加载第一张图。")
        if self.is_fullfield_mode():
            return self.validate_fullfield_before_processing()
        if not self.roi_groups:
            raise RuntimeError("请先至少添加一组 ROI。")
        if not self.output_folder.get().strip():
            raise RuntimeError("请设置输出文件夹。")
        output_path = Path(self.output_folder.get().strip())
        if output_path.exists() and not output_path.is_dir():
            raise RuntimeError(f"输出路径已存在但不是文件夹：{output_path}")
        if not any(
            var.get()
            for var in [
                self.export_origin_txt,
                self.export_origin_opju,
                self.export_engineering_png,
                self.export_publication_figures,
                self.export_qc_summary,
                self.export_full_csv,
                self.export_corr_plot,
                self.export_overlays,
                self.export_parameters,
            ]
        ):
            raise RuntimeError("请至少选择一种导出内容。")

        start_idx, end_idx = self.get_analysis_indices()
        if end_idx <= start_idx:
            raise RuntimeError("分析范围至少应包含两帧。")

        expected_ref = start_idx + 1
        missing_ref = []
        mismatched_ref = []
        invalid_l0 = []
        for group in self.roi_groups:
            gname = str(group.get("name", "未命名"))
            ref = group.get("reference_frame_1based")
            if ref is None:
                missing_ref.append(gname)
            else:
                try:
                    ref_int = int(ref)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise RuntimeError(f"ROI 组 {gname} 的参考帧记录必须是整数。") from exc
                if ref_int != expected_ref:
                    mismatched_ref.append(f"{gname}(第 {ref_int} 帧)")
            try:
                l0_val = float(group.get("L0", np.nan))
            except (TypeError, ValueError):
                l0_val = float("nan")
            if not math.isfinite(l0_val) or l0_val <= 0:
                invalid_l0.append(gname)

        if missing_ref:
            raise RuntimeError(
                f"ROI 组 {', '.join(missing_ref)} 缺少参考帧记录，请在当前起始/参考帧（第 {expected_ref} 帧）重画并更新。"
            )
        if mismatched_ref:
            raise RuntimeError(
                f"ROI 组 {', '.join(mismatched_ref)} 与当前起始/参考帧（第 {expected_ref} 帧）不一致，请载入后重画或恢复参考帧。"
            )
        if invalid_l0:
            raise RuntimeError(
                f"ROI 组 {', '.join(invalid_l0)} 的初始标距 L0 无效（必须为大于 0 的有限数值），请重新绘制并更新。"
            )

        search_radius = self.get_int_setting(self.search_radius, "搜索半径")
        overlay_every = self.get_int_setting(self.overlay_every, "overlay 保存间隔")
        hard_corr = self.get_float_setting(self.hard_corr, "硬相关阈值")
        soft_corr = self.get_float_setting(self.soft_corr, "软相关下限")
        template_alpha = self.get_float_setting(self.template_alpha, "模板跟随系数")
        fb_tolerance = self.get_float_setting(self.fb_tolerance_px, "FB 容差")
        min_texture_std = self.get_float_setting(self.min_texture_std, "最小灰度标准差")
        min_texture_contrast = self.get_float_setting(self.min_texture_contrast, "最小 P95-P5 对比度")
        max_saturated_frac = self.get_float_setting(self.max_saturated_frac, "最大近黑/近白比例")

        if search_radius <= 0:
            raise RuntimeError("搜索半径必须 > 0。")
        if self.export_overlays.get() and overlay_every <= 0:
            raise RuntimeError("overlay 保存间隔必须 > 0。")
        if not (-1 <= hard_corr <= 1):
            raise RuntimeError("硬相关阈值应在 -1 到 1 之间。")
        if not (-1 <= soft_corr <= 1):
            raise RuntimeError("软相关下限应在 -1 到 1 之间。")
        if soft_corr > hard_corr:
            raise RuntimeError("软相关下限不能高于硬相关阈值。")
        if not (0 <= template_alpha <= 1):
            raise RuntimeError("模板跟随系数应在 0 到 1 之间。")
        if not math.isfinite(fb_tolerance) or fb_tolerance <= 0:
            raise RuntimeError("FB 容差必须是大于 0 的有限数字。")
        if min_texture_std < 0:
            raise RuntimeError("最小灰度标准差不能为负。")
        if min_texture_contrast < 0:
            raise RuntimeError("最小 P95-P5 对比度不能为负。")
        if not (0 <= max_saturated_frac <= 1):
            raise RuntimeError("最大近黑/近白比例应在 0 到 1 之间。")

        if self.max_frame_strain_jump.get().strip():
            try:
                jump = float(self.max_frame_strain_jump.get().strip())
            except ValueError:
                raise RuntimeError("单帧应变突变上限必须是数字，或者留空禁用。")
            if not math.isfinite(jump) or jump <= 0:
                raise RuntimeError("单帧应变突变上限必须是大于 0 的有限数字，或者留空禁用。")

        if self.pixel_size_mm.get().strip():
            try:
                pix = float(self.pixel_size_mm.get().strip())
            except ValueError:
                raise RuntimeError("像素尺寸必须是数字，或者留空。")
            if not math.isfinite(pix) or pix <= 0:
                raise RuntimeError("像素尺寸必须是大于 0 的有限数字，或者留空。")

        _ref_raw, ref_img8 = self._get_reference_frame_images(start_idx)
        if ref_img8 is None:
            raise RuntimeError("无法读取当前起始/参考帧图像。")
        for group in self.roi_groups:
            for label, rect in (("ROI1", group.get("roi1")), ("ROI2", group.get("roi2"))):
                if rect is None or not rect_is_inside_image(rect, ref_img8.shape):
                    raise RuntimeError(f"ROI 组 {group.get('name', '未命名')} 的 {label} 超出参考图像范围。")
        self._raise_on_ambiguous_texture(
            [
                (f"{group['name']} ROI1", group.get("roi1"))
                for group in self.roi_groups
            ]
            + [
                (f"{group['name']} ROI2", group.get("roi2"))
                for group in self.roi_groups
            ],
            image=ref_img8,
        )

        poisson_enabled = validate_poisson_role_groups(self.roi_groups)
        if poisson_enabled:
            axial_groups, transverse_groups = get_poisson_role_groups(self.roi_groups)
            axial = axial_groups[0]
            transverse = transverse_groups[0]
            if axial.get("actual_mode") == transverse.get("actual_mode"):
                msg = (
                    "泊松比通常需要一组拉伸方向 ROI 和一组横向收缩 ROI。\n\n"
                    f"当前两组 actual_mode 都是 {axial.get('actual_mode')}：\n"
                    f"拉伸方向：{axial.get('name')}\n"
                    f"横向方向：{transverse.get('name')}\n\n"
                    "这可能说明方向选择或 ROI 对齐不合适。是否仍然继续？"
                )
                if not messagebox.askyesno("泊松比方向警告", msg):
                    raise RuntimeError("用户取消：泊松比 ROI 方向可能不合适。")

        very_small = [g for g in self.roi_groups if g["L0"] < 50]
        if very_small:
            names = ", ".join(g["name"] for g in very_small)
            msg = (
                f"以下 ROI 组的 L0 < 50 px：{names}\n\n"
                "这通常说明应变方向选错，或两个 ROI 太近。\n"
                "是否仍然继续？"
            )
            if not messagebox.askyesno("L0 过小警告", msg):
                raise RuntimeError("用户取消：某些 ROI 组 L0 太小。")

    def validate_fullfield_before_processing(self):
        start_idx, end_idx = self.get_analysis_indices()
        ref_raw, ref_img8 = self._get_reference_frame_images(start_idx)
        if ref_raw is None or ref_img8 is None:
            raise RuntimeError("请先加载参考图像。")
        snapshot = {
            "output_dir": self.output_folder.get().strip(),
            "image_paths": list(self.image_paths),
            "start_idx": start_idx,
            "end_idx": end_idx,
            "reference_frame_1based": start_idx + 1,
            "field_roi": self.field_roi,
            "field_roi_reference_frame_1based": self.field_roi_reference_frame_1based,
            "normalization": compute_reference_normalization(ref_raw),
            "min_texture_std": 8.0,
            "min_texture_contrast": 25.0,
            "max_saturated_frac": 0.20,
            "dic_subset_size": self.get_int_setting(self.dic_subset_size, "子集尺寸"),
            "dic_step": self.get_int_setting(self.dic_step, "步长"),
            "dic_solver": str(self.dic_solver.get()),
            "dic_strain_window": self.get_int_setting(self.dic_strain_window, "应变窗口"),
            "dic_smooth_sigma": self.get_float_setting(self.dic_smooth_sigma, "高斯平滑 σ"),
            "dic_search_radius": self.get_int_setting(self.dic_search_radius, "DIC 搜索半径"),
            "dic_zncc_min": self.get_float_setting(self.dic_zncc_min, "ZNCC 下限"),
            "dic_pyramid_levels": self.get_int_setting(self.dic_pyramid_levels, "金字塔层数"),
            "dic_pyramid_scale": self.get_float_setting(self.dic_pyramid_scale, "金字塔缩放"),
            "strain_degree": self.get_int_setting(self.dic_strain_degree, "拟合阶次"),
            "robust_strain": bool(self.dic_robust_strain.get()),
            "reject_nonconverged": bool(self.dic_reject_nonconverged.get()),
            "outlier_threshold_px": self.get_float_setting(self.dic_outlier_threshold, "异常位移阈值"),
            "mask": self.dic_mask_settings(),
            "display": self.dic_display_options(),
        }
        return validate_fullfield_snapshot(snapshot, check_sequence_dimensions=False)

    def build_processing_settings(self):
        start_idx, end_idx = self.get_analysis_indices()
        is_ff = self.is_fullfield_mode()
        if is_ff:
            # Hidden 1-D controls are deliberately not read in full-field
            # mode.  Their stale/invalid values must not poison a valid 2-D
            # settings snapshot.
            search_radius_base = 180
            hard_corr = 0.55
            soft_corr = 0.35
            enable_adaptive = True
            use_prev_frame_template = False
            template_alpha = 0.70
            max_frame_jump = None
            enable_fb_check = True
            fb_tolerance = 12.0
            pixel_size_mm = None
            overlay_every = 5
            min_texture_std = 8.0
            min_texture_contrast = 25.0
            max_saturated_frac = 0.20
        else:
            max_frame_jump = None
            if self.max_frame_strain_jump.get().strip():
                max_frame_jump = float(self.max_frame_strain_jump.get().strip())

            pixel_size_mm = None
            if self.pixel_size_mm.get().strip():
                pixel_size_mm = float(self.pixel_size_mm.get().strip())
            search_radius_base = self.get_int_setting(self.search_radius, "搜索半径")
            hard_corr = self.get_float_setting(self.hard_corr, "硬相关阈值")
            soft_corr = self.get_float_setting(self.soft_corr, "软相关下限")
            enable_adaptive = bool(self.enable_adaptive.get())
            use_prev_frame_template = bool(self.use_prev_frame_template.get())
            template_alpha = self.get_float_setting(self.template_alpha, "模板跟随系数")
            enable_fb_check = bool(self.enable_fb_check.get())
            fb_tolerance = self.get_float_setting(self.fb_tolerance_px, "FB 容差")
            overlay_every = self.get_int_setting(self.overlay_every, "overlay 保存间隔")
            min_texture_std = self.get_float_setting(self.min_texture_std, "最小灰度标准差")
            min_texture_contrast = self.get_float_setting(self.min_texture_contrast, "最小 P95-P5 对比度")
            max_saturated_frac = self.get_float_setting(self.max_saturated_frac, "最大近黑/近白比例")

        if is_ff:
            export_origin_txt = False
            export_origin_opju = False
            export_engineering_png = False
            export_publication_figures = False
            export_qc_summary = False
            export_full_csv = False
            export_corr_plot = False
            export_parameters = False
            export_overlays = bool(self.export_overlays.get())
            tracking_preset = "fullfield"
        else:
            export_origin_txt = bool(self.export_origin_txt.get())
            export_origin_opju = bool(self.export_origin_opju.get())
            export_engineering_png = bool(self.export_engineering_png.get())
            export_publication_figures = bool(self.export_publication_figures.get())
            export_qc_summary = bool(self.export_qc_summary.get())
            export_full_csv = bool(self.export_full_csv.get())
            export_corr_plot = bool(self.export_corr_plot.get())
            export_overlays = bool(self.export_overlays.get())
            export_parameters = bool(self.export_parameters.get())
            tracking_preset = self.tracking_preset.get()

        field_roi = tuple(self.field_roi) if self.field_roi is not None else None
        reference_raw, _reference_img8 = self._get_reference_frame_images(start_idx)
        reference_normalization = (
            compute_reference_normalization(reference_raw) if reference_raw is not None else None
        )
        return {
            "output_dir": Path(self.output_folder.get().strip()),
            "image_paths": list(self.image_paths),
            "roi_groups": [dict(group) for group in self.roi_groups],
            "start_idx": start_idx,
            "end_idx": end_idx,
            "search_radius_base": search_radius_base,
            "hard_corr": hard_corr,
            "soft_corr": soft_corr,
            "enable_adaptive": enable_adaptive,
            "use_prev_frame_template": use_prev_frame_template,
            "template_policy": "experimental_follow" if use_prev_frame_template else "fixed_reference",
            "template_alpha": template_alpha,
            "max_frame_jump": max_frame_jump,
            "enable_fb_check": enable_fb_check,
            "fb_tolerance": fb_tolerance,
            "pixel_size_mm": pixel_size_mm,
            "overlay_every": overlay_every,
            "export_origin_txt": export_origin_txt,
            "export_origin_opju": export_origin_opju,
            "export_engineering_png": export_engineering_png,
            "export_publication_figures": export_publication_figures,
            "export_qc_summary": export_qc_summary,
            "export_full_csv": export_full_csv,
            "export_corr_plot": export_corr_plot,
            "export_overlays": export_overlays,
            "export_parameters": export_parameters,
            "image_folder": self.image_folder.get(),
            "tracking_preset": tracking_preset,
            "min_texture_std": min_texture_std,
            "min_texture_contrast": min_texture_contrast,
            "max_saturated_frac": max_saturated_frac,
            "analysis_mode": str(self.analysis_mode.get()),
            "field_roi": field_roi,
            "field_roi_reference_frame_1based": self.field_roi_reference_frame_1based,
            "reference_frame_1based": start_idx + 1,
            "reference_filename": Path(self.image_paths[start_idx]).name,
            "image_sequence_fingerprint": image_sequence_fingerprint(self.image_paths),
            "normalization": reference_normalization,
            "dic_subset_size": int(self.dic_subset_size.get()) if is_ff else 21,
            "dic_step": int(self.dic_step.get()) if is_ff else 5,
            "dic_solver": str(self.dic_solver.get()),
            "dic_strain_window": int(self.dic_strain_window.get()) if is_ff else 5,
            "dic_smooth_sigma": float(self.dic_smooth_sigma.get()) if is_ff else 0.0,
            "dic_search_radius": int(self.dic_search_radius.get()) if is_ff else 20,
            "dic_zncc_min": float(self.dic_zncc_min.get()) if is_ff else 0.75,
            "dic_pyramid_levels": int(self.dic_pyramid_levels.get()) if is_ff else 1,
            "dic_pyramid_scale": float(self.dic_pyramid_scale.get()) if is_ff else 0.5,
            "strain_degree": int(self.dic_strain_degree.get()) if is_ff else 2,
            "robust_strain": bool(self.dic_robust_strain.get()) if is_ff else True,
            "reject_nonconverged": bool(self.dic_reject_nonconverged.get()) if is_ff else False,
            "outlier_threshold_px": float(self.dic_outlier_threshold.get()) if is_ff else 1.0,
            "mask": self.dic_mask_settings() if is_ff else {"mode": "none"},
            "display": self.dic_display_options() if is_ff else {},
        }

    def post_to_ui(self, callback, run_token=None):
        if run_token is None:
            run_token = getattr(self._worker_context, "run_token", None)
        if run_token is not None:
            original_callback = callback

            def guarded_callback():
                with self._state_lock_ctx():
                    if self._active_run_token != run_token:
                        return
                original_callback()

            callback = guarded_callback
        self.ui_queue.put(callback)
        return True

    def start_ui_queue_polling(self):
        if getattr(self, "_closing", False):
            return
        try:
            if not self.root.winfo_exists():
                return
            self._ui_poll_after_id = self.root.after(50, self.drain_ui_queue)
        except (RuntimeError, tk.TclError):
            self._ui_poll_after_id = None

    def drain_ui_queue(self):
        self._ui_poll_after_id = None
        if getattr(self, "_closing", False):
            return
        try:
            while True:
                try:
                    callback = self.ui_queue.get_nowait()
                except queue.Empty:
                    break
                if getattr(self, "_closing", False):
                    break
                try:
                    callback()
                except Exception:
                    if sys.stderr is not None:
                        traceback.print_exc()
        finally:
            if not getattr(self, "_closing", False):
                self.start_ui_queue_polling()

    # ---------- 批量处理 ----------

    def _finalize_processing_state(self, run_token):
        """Release the completion gate only after all queued run callbacks drain."""
        with self._state_lock_ctx():
            if run_token is not None and self._active_run_token != run_token:
                return
            self._completion_pending = False
        self.update_workflow_action_states()

    def _mirror_failed_fullfield_runs(self, output_root, pre_existing_names, *, specific_failed_dir=None):
        """Safely mirror newly created failed runs into ``output_root/dic/_failed_runs``."""
        output_root = Path(output_root)
        failed_root = output_root / "_failed_runs"
        dic_root = output_root / "dic"
        legacy_failed_root = dic_root / "_failed_runs"

        def assert_safe(path, *, allow_missing_leaf):
            _core._assert_no_reparse_components(path, allow_missing_leaf=allow_missing_leaf)

        def assert_contained(path, root, *, code):
            _core._assert_contained_path(path, root, code=code)

        try:
            assert_safe(output_root, allow_missing_leaf=False)
            assert_safe(failed_root, allow_missing_leaf=True)
            assert_contained(failed_root, output_root, code="FAILED_MIRROR_OUTSIDE_ROOT")
            if not failed_root.exists() or not failed_root.is_dir():
                return
            assert_safe(dic_root, allow_missing_leaf=True)
            assert_contained(dic_root, output_root, code="FAILED_MIRROR_OUTSIDE_ROOT")
            assert_safe(legacy_failed_root, allow_missing_leaf=True)
            assert_contained(legacy_failed_root, output_root, code="FAILED_MIRROR_OUTSIDE_ROOT")

            with _core._operation_os_lock(output_root):
                with _core._transaction_lock(output_root):
                    assert_safe(output_root, allow_missing_leaf=False)
                    assert_safe(failed_root, allow_missing_leaf=False)
                    assert_contained(failed_root, output_root, code="FAILED_MIRROR_OUTSIDE_ROOT")
                    assert_safe(dic_root, allow_missing_leaf=True)
                    assert_contained(dic_root, output_root, code="FAILED_MIRROR_OUTSIDE_ROOT")
                    assert_safe(legacy_failed_root, allow_missing_leaf=True)
                    assert_contained(legacy_failed_root, output_root, code="FAILED_MIRROR_OUTSIDE_ROOT")
                    if dic_root.exists() and not dic_root.is_dir():
                        raise RuntimeError(f"失败归档镜像失败：{dic_root} 不是文件夹。")
                    if legacy_failed_root.exists() and not legacy_failed_root.is_dir():
                        raise RuntimeError(f"失败归档镜像失败：{legacy_failed_root} 不是文件夹。")

                    candidates = []
                    seen_names = set()
                    if specific_failed_dir is not None:
                        spec = Path(specific_failed_dir)
                        if spec.parent == failed_root and spec.exists():
                            candidates.append(spec)
                            seen_names.add(spec.name)
                    for path in sorted(failed_root.iterdir(), key=lambda item: item.name):
                        if path.name in pre_existing_names or path.name in seen_names:
                            continue
                        candidates.append(path)
                        seen_names.add(path.name)

                    for failed_dir in candidates:
                        assert_safe(failed_dir, allow_missing_leaf=False)
                        assert_contained(failed_dir, failed_root, code="FAILED_MIRROR_OUTSIDE_ROOT")
                        if not failed_dir.is_dir():
                            continue
                        # Verify the entire tree contains no symlinks/reparse points before writing.
                        files_to_copy = []
                        dirs_to_create = []
                        for dirpath, dirnames, filenames in os.walk(failed_dir, followlinks=False):
                            current = Path(dirpath)
                            assert_safe(current, allow_missing_leaf=False)
                            assert_contained(current, failed_dir, code="FAILED_MIRROR_OUTSIDE_ROOT")
                            for dirname in dirnames:
                                sub = current / dirname
                                assert_safe(sub, allow_missing_leaf=False)
                                assert_contained(sub, failed_dir, code="FAILED_MIRROR_OUTSIDE_ROOT")
                                dirs_to_create.append(sub.relative_to(failed_dir))
                            for filename in filenames:
                                src_file = current / filename
                                assert_safe(src_file, allow_missing_leaf=False)
                                assert_contained(src_file, failed_dir, code="FAILED_MIRROR_OUTSIDE_ROOT")
                                if src_file.is_file():
                                    files_to_copy.append((src_file, src_file.relative_to(failed_dir)))

                        target = legacy_failed_root / failed_dir.name
                        assert_safe(target, allow_missing_leaf=True)
                        assert_contained(target, legacy_failed_root, code="FAILED_MIRROR_OUTSIDE_ROOT")
                        if target.exists() or target.is_symlink():
                            continue
                        legacy_failed_root.mkdir(parents=True, exist_ok=True)
                        assert_safe(legacy_failed_root, allow_missing_leaf=False)
                        assert_contained(legacy_failed_root, output_root, code="FAILED_MIRROR_OUTSIDE_ROOT")
                        target.mkdir()
                        for rel_dir in dirs_to_create:
                            dst_dir = target / rel_dir
                            assert_contained(dst_dir, target, code="FAILED_MIRROR_OUTSIDE_ROOT")
                            dst_dir.mkdir(parents=True, exist_ok=True)
                        for src_file, rel_file in files_to_copy:
                            dst_file = target / rel_file
                            assert_contained(dst_file, target, code="FAILED_MIRROR_OUTSIDE_ROOT")
                            shutil.copy2(src_file, dst_file)
        except Exception as mirror_exc:
            if hasattr(self, "post_to_ui") and hasattr(self, "log"):
                self.post_to_ui(lambda m=f"全场 DIC 失败归档镜像已跳过：{mirror_exc}": self.log(m))

    def start_processing(self):
        if hasattr(self, "control_notebook"):
            selected_tab = self.control_notebook.select()
            if hasattr(self, "agent_page") and selected_tab == str(self.agent_page):
                self.agent_page.run_project()
                return
            if hasattr(self, "campaign_page") and selected_tab == str(self.campaign_page):
                if hasattr(self, "status_var"):
                    self.status_var.set("当前位于“批次项目”页，请使用本页内的批次运行按钮。")
                return
        if getattr(getattr(self, "agent_page", None), "_busy", False):
            messagebox.showinfo("正在处理", "自动分析任务仍在运行，请等待完成后再启动手动分析。")
            return
        if getattr(getattr(self, "campaign_page", None), "_busy", False):
            messagebox.showinfo("正在处理", "批次项目任务仍在运行，请等待完成后再启动手动分析。")
            return
        with self._state_lock_ctx():
            processing = bool(self.is_processing)
            pending = bool(self._completion_pending)
        if processing:
            messagebox.showinfo("正在处理", "程序正在处理，请等待当前任务完成。")
            return
        if pending:
            messagebox.showinfo("正在收尾", "上一轮分析正在收尾，请等待完成后再开始。")
            return

        try:
            self.validate_before_processing()
            settings = self.build_processing_settings()
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return

        with self._state_lock_ctx():
            self._run_generation += 1
            run_token = self._run_generation
            settings["_run_token"] = run_token
            self._active_run_token = run_token
            self._completion_pending = False
            self.is_processing = True
        self._update_progress(0)
        if settings.get("analysis_mode") == ANALYSIS_MODE_FULLFIELD:
            self.status_var.set("正在分析：全场 2D DIC。")
            self.log("开始全场 2D DIC：IC-GN/IC-LM 匹配 subset 网格。")
            worker = self.process_fullfield_thread
        else:
            self.status_var.set("正在分析：开始追踪 ROI 组。")
            self.log("开始分析：正在追踪各 ROI 组，请稍候。")
            worker = self.process_images_thread
        self.update_workflow_action_states()

        thread = threading.Thread(target=worker, args=(settings,), daemon=True)
        with self._state_lock_ctx():
            self._worker_thread = thread
        thread.start()

    def process_fullfield_thread(self, settings):
        run_token = settings.get("_run_token")
        self._worker_context.run_token = run_token
        try:
            self.process_fullfield(settings)
        except Exception as exc:
            message = str(exc)
            details = traceback.format_exc()

            def _report_failure(m=message, e=exc, d=details):
                self.status_var.set("分析失败，详情见“质量与日志”。")
                self.log_user_error("全场 DIC", e, d)
                messagebox.showerror("处理失败", m)

            self.post_to_ui(_report_failure)
        finally:
            with self._state_lock_ctx():
                if run_token is not None:
                    self._completion_pending = True
                self.is_processing = False
                if getattr(self, "_worker_thread", None) is threading.current_thread():
                    self._worker_thread = None
            if run_token is not None:
                self.post_to_ui(lambda token=run_token: self._finalize_processing_state(token))
            else:
                self.post_to_ui(lambda: self.update_workflow_action_states())
            self._worker_context.run_token = None

    def process_fullfield(self, settings):
        # The numerical/export engine is Tk-free; this adapter only translates
        # progress and final state back onto the UI thread.  Keep the legacy
        # body below as a compatibility fallback for old callers that
        # explicitly request it.
        if not settings.get("_legacy_direct_processing", False):
            # The worker must consume only the frozen settings snapshot and
            # image files.  In particular, do not call the GUI method here:
            # it reads Tk variables and is only safe on the main thread.
            snapshot_keys = (
                "image_paths",
                "start_idx",
                "end_idx",
                "reference_frame_1based",
                "field_roi",
                "field_roi_reference_frame_1based",
                "dic_subset_size",
                "dic_step",
                "dic_solver",
                "dic_strain_window",
                "dic_smooth_sigma",
                "dic_search_radius",
                "dic_zncc_min",
            )
            output_root = Path(settings["output_dir"])
            legacy_manifest = output_root / "run_manifest.json"
            needs_dim_check = (
                (not legacy_manifest.exists() and os.path.lexists(output_root / "dic"))
                or (validate_image_sequence_dimensions is not _core.validate_image_sequence_dimensions)
            )
            if (
                threading.current_thread() is threading.main_thread()
                and not all(key in settings for key in snapshot_keys)
            ):
                # Preserve the old direct-adapter extension point for callers
                # that supplied an intentionally minimal fake settings object.
                # Real GUI runs always provide a complete snapshot, and every
                # worker-thread invocation takes the pure path above.
                self.validate_fullfield_before_processing()
            else:
                validate_fullfield_snapshot(settings, check_sequence_dimensions=needs_dim_check)
            engine_settings = dict(settings)
            # ``__file__`` may point into a PyInstaller archive.  Resolve the
            # real auditable copies under ``_MEIPASS/sources`` instead of
            # passing a path that cannot be hashed or verified.
            engine_settings["_code_paths"] = [
                str(path) for path in _core.resolve_code_paths(include_gui=True)
            ]
            engine_settings["_gui_adapter"] = True
            # A pre-v0.2 GUI run has no ownership ledger.  Preserve its
            # established exact-name migration behaviour once, before the new
            # transactional engine starts; every v0.2 run is governed solely
            # by the manifest ledger.  Similar user names (for example
            # ``frame_0002_u.txt``) are not selected by this allowlist.
            if not legacy_manifest.exists():
                # The pure snapshot preflight above is complete before any
                # legacy output is migrated.  The new core then owns all
                # subsequent staging and publication.
                archive_previous_fullfield_outputs(output_root / "dic")
            # GUI preflight historically permits low/saturated texture so the
            # user can inspect a structured invalid-frame result.  Headless
            # callers remain fail-closed on this contract; this flag is scoped
            # to the adapter and is recorded in the run configuration.
            engine_settings["_allow_low_texture_preflight"] = True

            pre_existing_failed = set()
            failed_root = output_root / "_failed_runs"
            try:
                _core._assert_no_reparse_components(output_root, allow_missing_leaf=True)
                _core._assert_no_reparse_components(failed_root, allow_missing_leaf=True)
                _core._assert_contained_path(failed_root, output_root, code="FAILED_MIRROR_OUTSIDE_ROOT")
                if failed_root.exists() and failed_root.is_dir():
                    pre_existing_failed = {p.name for p in failed_root.iterdir()}
            except Exception:
                pre_existing_failed = set()

            def progress(event):
                if isinstance(event, dict):
                    value = 100.0 * float(event.get("fraction", 0.0))
                    frame = event.get("frame_global_1based")
                    state = event.get("status", "")
                else:
                    value = 100.0 * float(event)
                    frame = None
                    state = ""
                self.post_to_ui(lambda v=value: self._update_progress(v))
                if frame is not None:
                    self.post_to_ui(lambda f=frame, s=state: self.status_var.set(f"正在处理全场 DIC：第 {f} 帧 {s}"))

            try:
                result = _core.run_fullfield_sequence(engine_settings, progress_callback=progress)
            except Exception:
                self._mirror_failed_fullfield_runs(output_root, pre_existing_failed)
                raise
            if not result.get("scientific_ok", False):
                manifest_path = result.get("manifest_path")
                specific_dir = Path(manifest_path).parent if manifest_path else None
                self._mirror_failed_fullfield_runs(
                    output_root,
                    pre_existing_failed,
                    specific_failed_dir=specific_dir,
                )
                raise RuntimeError("全场 DIC 科学有效性门未通过（无有效应变/有限应变）：" + ", ".join(result.get("errors", [])))
            last_field = result.get("last_field")
            if last_field is None:
                raise RuntimeError("全场 DIC 没有可展示的有效结果。")
            last_image = result.get("last_image")
            frames = result.get("frames", [])
            valid_count = sum(1 for item in frames if item.get("status") == "scientific_valid")
            total_count = len(frames)

            def finish():
                self.dic_last_field = last_field
                self.dic_last_image = np.asarray(last_image).copy() if last_image is not None else None
                self.dic_last_frame_1based = last_field.get("frame_global_1based")
                self.dic_last_filename = last_field.get("frame_filename")
                self.dic_last_reference_frame_1based = result.get("manifest", {}).get("reference_frame_1based", settings.get("reference_frame_1based", settings.get("start_idx", 0) + 1))
                self.dic_last_reference_filename = result.get("manifest", {}).get("reference_filename")
                self._update_progress(100)
                self.status_var.set(f"处理完成，全场 DIC：{valid_count}/{total_count} 帧科学有效。")
                if hasattr(self, "qc_overview_var"):
                    self.qc_overview_var.set(self.format_fullfield_qc_overview(frames))
                self.log(f"全场 DIC manifest 已验证：{result['manifest_path']}")
                try:
                    self.show_field_viewer(
                        last_field,
                        image=self.dic_last_image,
                        frame_1based=self.dic_last_frame_1based,
                        filename=self.dic_last_filename,
                        reference_frame_1based=self.dic_last_reference_frame_1based,
                        reference_filename=self.dic_last_reference_filename,
                    )
                except Exception as viewer_exc:
                    self.log(f"全场结果预览窗口初始化失败（不影响已验证文件）：{viewer_exc}")
                self.show_completion_and_open_output_folder(
                    f"全场 2D DIC 完成：{valid_count}/{total_count} 个变形帧有效。结果已写入 {settings['output_dir']}",
                    settings["output_dir"],
                )

            self.post_to_ui(finish)
            return result

        raise RuntimeError("不支持已废弃的 _legacy_direct_processing 直接执行路径，请使用标准核心引擎。")

    def process_images_thread(self, settings):
        run_token = settings.get("_run_token")
        self._worker_context.run_token = run_token
        try:
            self.process_images(settings)
        except Exception as exc:
            message = str(exc)
            details = traceback.format_exc()

            def _report_failure(m=message, e=exc, d=details):
                self.status_var.set("分析失败，详情见“质量与日志”。")
                self.log_user_error("批量处理", e, d)
                messagebox.showerror("处理失败", m)

            self.post_to_ui(_report_failure)
        finally:
            with self._state_lock_ctx():
                if run_token is not None:
                    self._completion_pending = True
                self.is_processing = False
                if getattr(self, "_worker_thread", None) is threading.current_thread():
                    self._worker_thread = None
            if run_token is not None:
                self.post_to_ui(lambda token=run_token: self._finalize_processing_state(token))
            else:
                self.post_to_ui(lambda: self.update_workflow_action_states())
            self._worker_context.run_token = None

    def init_group_states(self, first_img8, groups=None, texture_settings=None):
        if groups is None:
            groups = self.roi_groups
        if texture_settings is None:
            texture_settings = {
                "min_texture_std": self.min_texture_std.get(),
                "min_texture_contrast": self.min_texture_contrast.get(),
                "max_saturated_frac": self.max_saturated_frac.get(),
            }
        return [
            initialize_extensometer_group_state(
                first_img8,
                group,
                min_texture_std=texture_settings.get("min_texture_std", 8.0),
                min_texture_contrast=texture_settings.get("min_texture_contrast", 25.0),
                max_saturated_frac=texture_settings.get("max_saturated_frac", 0.20),
                min_structure_ratio=_core.DEFAULT_TEXTURE_MIN_STRUCTURE_RATIO,
            )
            for group in groups
        ]

    def process_one_group_one_frame(
        self,
        state,
        img8,
        frame_idx,
        filename,
        params,
    ):
        # Canonical numerical state machine lives in the Tk-free core.
        return track_extensometer_group_frame(
            state,
            img8,
            frame_idx,
            filename,
            params,
        )

    def process_images(self, settings=None):
        if settings is None:
            settings = self.build_processing_settings()

        if not settings.get("_legacy_direct_processing", False):
            image_paths = settings.get("image_paths", [])
            start_idx = int(settings.get("start_idx", 0))
            end_idx = int(settings.get("end_idx", len(image_paths) - 1))
            # Core validates sequence dimensions before staging; only run the
            # module-level hook directly when a caller/test replaced it.
            if validate_image_sequence_dimensions is not _core.validate_image_sequence_dimensions:
                validate_image_sequence_dimensions(image_paths[start_idx:end_idx + 1])
            engine_settings = dict(settings)
            # Use the same source resolver as the full-field adapter; this is
            # required for frozen GUI runs where both module ``__file__``
            # values can be archive-style/non-filesystem paths.
            engine_settings["_code_paths"] = [
                str(path) for path in _core.resolve_code_paths(include_gui=True)
            ]
            engine_settings["_gui_adapter"] = True

            def progress(event):
                if isinstance(event, dict):
                    value = 100.0 * float(event.get("fraction", 0.0))
                    frame = event.get("frame_global_1based")
                else:
                    value = 100.0 * float(event)
                    frame = None
                self.post_to_ui(lambda v=value: self._update_progress(v))
                if frame is not None:
                    self.post_to_ui(lambda f=frame: self.status_var.set(f"正在批量追踪：第 {f} 帧"))

            result = _core.run_extensometer_sequence(engine_settings, progress_callback=progress)
            if not result.get("scientific_ok", False):
                raise RuntimeError("虚拟引伸计科学有效性门未通过：" + ", ".join(result.get("errors", [])))
            df = result.get("dataframe")
            summary = result.get("summary")
            output_dir = Path(settings["output_dir"])
            if df is None or summary is None:
                raise RuntimeError("引擎未返回可展示的结果表。")
            written_paths = [Path(path) for path in result.get("outputs", [])]
            qc_level = summary.get("overall", {}).get("qc_level", "Unknown")
            n_rejected = summary.get("overall", {}).get("rejected_frames", 0)
            n_adaptive = summary.get("overall", {}).get("adaptive_accepted_frames", 0)
            path_log = "输出文件：\n" + "\n".join(str(path) for path in written_paths)
            done_msg = (
                "处理完成。\n"
                f"核心结果已保存到: {output_dir / 'core'}\n"
                f"QC 状态：{qc_level}\n"
                f"拒绝帧数：{n_rejected}\n"
                f"自适应接受帧数：{n_adaptive}\n"
                f"manifest 已验证：{result['manifest_path']}"
            )
            if settings.get("export_origin_txt") and (output_dir / "core" / "strain_mean_groups.txt").is_file():
                done_msg += "\n平均应变文件: core\\strain_mean_groups.txt\n平均应变列也已写入: core\\strain_all_groups.txt"

            def _finish_extensometer():
                self._update_progress(100)
                self.status_var.set(f"处理完成，QC 状态：{qc_level}")
                self.update_qc_overview(summary)
                self.log(done_msg + "\n" + path_log)
                if not df.empty and settings.get("roi_groups"):
                    try:
                        self.show_results_viewer(df, settings["roi_groups"])
                    except Exception as viewer_exc:
                        self.log(f"结果预览窗口初始化失败（不影响已验证文件）：{viewer_exc}")
                self.show_completion_and_open_output_folder(done_msg, output_dir)

            self.post_to_ui(_finish_extensometer)
            return result

        raise RuntimeError("不支持已废弃的 _legacy_direct_processing 直接执行路径，请使用标准核心引擎。")

    # ========== 导出预设 ==========

    def _apply_research_preset(self):
        """科研推荐组合：核心 TXT + PNG + QC + 参数（最常用）"""
        self.export_origin_txt.set(True)
        self.export_engineering_png.set(True)
        self.export_qc_summary.set(True)
        self.export_parameters.set(True)
        self.export_full_csv.set(False)
        self.export_corr_plot.set(False)
        self.export_overlays.set(False)
        self.export_publication_figures.set(False)
        self.export_origin_opju.set(False)
        self.log("已应用「科研推荐」导出预设")

    def _apply_quick_view_preset(self):
        """快速查看：只保留最常用的可视化结果"""
        self.export_origin_txt.set(False)
        self.export_engineering_png.set(True)
        self.export_qc_summary.set(True)
        self.export_parameters.set(False)
        self.export_full_csv.set(False)
        self.export_corr_plot.set(False)
        self.export_overlays.set(False)
        self.export_publication_figures.set(False)
        self.export_origin_opju.set(False)
        self.log("已应用「快速查看导出」预设")

    def _apply_all_export_preset(self):
        """全部勾选（用于最完整的存档）"""
        for var in [
            self.export_origin_txt, self.export_origin_opju,
            self.export_engineering_png, self.export_qc_summary,
            self.export_full_csv, self.export_corr_plot,
            self.export_overlays, self.export_parameters,
            self.export_publication_figures,
        ]:
            var.set(True)
        self.log("已应用「全量复核导出」预设（注意文件会较多）")

    def run(self):
        self.root.mainloop()


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(prog="straintrace-gui")
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--camera")
    parser.add_argument("--reference-sequence", type=int)
    parser.add_argument("--live", action="store_true")
    args, _ = parser.parse_known_args(argv)
    if args.live:
        if not args.capture:
            parser.error("--live requires --capture")
        from straintrace_capture.live_gui import run_live

        return run_live(args.capture, args.camera, args.reference_sequence)
    enable_windows_dpi_awareness()
    root = tk.Tk()
    app = MultiROIGUI(root)
    if args.capture:
        try:
            app.open_capture_experiment(args.capture, args.camera, args.reference_sequence)
        except Exception as exc:
            messagebox.showerror("无法打开采集实验", str(exc), parent=root)
    app.run()


if __name__ == "__main__":
    main()
