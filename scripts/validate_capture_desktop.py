"""Exercise the Windows Capture and live Analysis controls and save desktop screenshots."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def screenshot(root, output: Path, name: str):
    from PIL import ImageGrab
    from straintrace_capture.viewport import refresh_window_surface

    root.lift()
    root.update_idletasks()
    refresh_window_surface(root)
    if sys.platform == "win32":
        rendered = ImageGrab.grab(window=int(root.wm_frame(), 0))
    else:
        x, y = root.winfo_rootx(), root.winfo_rooty()
        rendered = ImageGrab.grab(bbox=(x, y, x + root.winfo_width(), y + root.winfo_height()))
    rendered.save(output / f"{name}.png")


def trigger_when(root, check, action, timeout=15):
    started = time.monotonic()
    def probe():
        if check():
            action()
        elif time.monotonic() - started > timeout:
            raise TimeoutError(f"timed out waiting for {action.__name__}")
        else:
            root.after(120, probe)
    root.after(120, probe)


def live_child(root, output):
    from straintrace_capture import ExperimentReader
    from straintrace_capture.live_gui import LiveWorkbench

    run = output / "experiment"
    root.geometry("1170x750+72+73")
    app = LiveWorkbench(root, run, "sim:0")
    root.geometry("1170x750+72+73")
    app.rois["Gauge ROI 1"].set("273,113,33,33")
    app.rois["Gauge ROI 2"].set("273,298,33,33")
    app.strain_mode.set("y")

    def saved_count():
        with ExperimentReader(run) as read:
            return len(read.frames(camera_id="sim:0"))

    observed = {}

    def pause_then_resume():
        app.pause_button.invoke()
        assert app._paused
        observed["receipts_at_pause"] = saved_count()
        trigger_when(root, lambda: saved_count() >= observed["receipts_at_pause"] + 4, resume)

    def resume():
        observed["receipts_at_resume"] = saved_count()
        app.pause_button.invoke()
        trigger_when(root, lambda: len(app._data) >= 10, finish)

    def finish():
        assert any(strain is not None for _, strain in app._data)
        root.attributes("-topmost", True)
        screenshot(root, output, "capture-live-analysis")
        observed["preview_points"] = len(app._data)
        (output / "live-evidence.json").write_text(json.dumps(observed, indent=2) + "\n", encoding="utf-8")
        root.destroy()

    def start():
        app.start_button.invoke()
        trigger_when(root, lambda: len(app._data) >= 3, pause_then_resume)
    trigger_when(root, lambda: app.view._image is not None, start)


def capture_parent(root, output):
    from straintrace_capture import CaptureConfig, ExperimentReader
    from straintrace_capture.gui import CaptureWorkbench, demo_config
    from straintrace_capture.storage import TERMINAL_STATES

    data = demo_config()
    data["phases"][1] = {"mode": "continuous", "duration_s": 90.0}
    settings = {"width": 640, "height": 480, "pixel_format": "Mono12", "motion_px": 0.04,
                "strain_per_frame": 0.00012, "fps": 12.0}
    data["cameras"] = [{"id": f"sim:{index}", "settings": settings} for index in range(2)]
    run = output / "experiment"
    app = CaptureWorkbench(root, CaptureConfig.from_dict(data))
    root.geometry("1230x770+55+65")
    app.output_path.set(str(run))
    child = None
    confirmed_preview = {}

    def preview():
        app.preview_button.invoke()
        trigger_when(root, lambda: app.preview and all(app.preview.latest_image(camera) is not None for camera in app.views),
                     inspect)

    def inspect():
        for camera in app.views:
            confirmed_preview[camera] = app.preview.latest_image(camera).pixels.shape
        focus = app.views["sim:0"]
        canvas = focus.canvas
        zoom, left, top = focus._project
        x, y = round(left + 155 * zoom), round(top + 98 * zoom)
        end_x, end_y = round(left + 355 * zoom), round(top + 265 * zoom)
        canvas.event_generate("<ButtonPress-1>", x=x, y=y, state=1)
        canvas.event_generate("<B1-Motion>", x=end_x, y=end_y, state=257)
        canvas.event_generate("<ButtonRelease-1>", x=end_x, y=end_y, state=1)
        assert focus.roi is not None
        canvas.event_generate("<Motion>", x=end_x, y=end_y)
        px, py = focus._probe_coordinate
        assert focus._caption.cget("text") == f"({px},{py}) {focus._image.pixels[py, px]}"
        canvas.event_generate("<Leave>")
        app.guide.set("Crosshair")
        app.apply_preview_options()
        trigger_when(root, lambda: "Local texture" in app.quality_text.get(), record)

    def record():
        screenshot(root, output, "capture-preview")
        root.geometry("1020x665+60+76")
        app.tabs.select(app.plan_panel)
        root.after(1200, compact)

    def compact():
        assert all(view._caption.winfo_width() >= view._caption.winfo_reqwidth() for view in app.views.values())
        screenshot(root, output, "capture-compact")
        root.geometry("1230x770+55+65")
        app.record_button.invoke()
        trigger_when(root, lambda: app.session and app.session.status()["cameras"]["sim:0"]["saved"] >= 8, launch)

    def launch():
        nonlocal child
        screenshot(root, output, "capture-recording")
        logs = (output / "live-application.log").open("w", encoding="utf-8")
        try:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--child", str(output)],
                                     stdout=logs, stderr=logs)
        finally:
            logs.close()
        root.attributes("-topmost", False)
        trigger_when(root, lambda: child.poll() is not None, finish_recording, timeout=45)

    def finish_recording():
        app.stop_button.invoke()
        trigger_when(root, lambda: app.session.status()["state"] in TERMINAL_STATES, review, timeout=15)

    def review():
        if child.returncode:
            raise RuntimeError((output / "live-application.log").read_text(encoding="utf-8"))
        root.attributes("-topmost", True)
        app.tabs.select(app.review_panel)
        app.browse_frame(latest=True)
        expected = app.session.status()
        proof = {"geometry": confirmed_preview, "status": expected, "live": child.returncode}
        with ExperimentReader(run) as reader:
            proof["verification"] = reader.verify()
            assert proof["verification"]["state"] == "stopped"
            assert proof["verification"]["ok"] and not proof["verification"]["reported_camera_losses"]
        screenshot(root, output, "capture-review")
        (output / "capture-evidence.json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
        app.close()

    trigger_when(root, lambda: getattr(app, "_providers", None), preview)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new QA directory")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not args.child:
        output.mkdir(parents=True, exist_ok=False)

    import tkinter as tk
    from ezdic_gui.theme import enable_windows_dpi_awareness

    enable_windows_dpi_awareness()
    root = tk.Tk()
    root.attributes("-topmost", True)
    failure = []

    def failed(exc, value, tb):
        traceback.print_exception(exc, value, tb)
        failure.append(str(value))
        root.destroy()

    root.report_callback_exception = failed
    (live_child if args.child else capture_parent)(root, output)
    root.mainloop()
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
