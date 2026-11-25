"""Analysis desktop mode for following a Capture ledger."""

from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from ezdic_gui.theme import enable_windows_dpi_awareness

from .cameras.base import Image
from .gui import ACCENT, BG, BORDER, ENTRY, FG, MUTED, set_theme
from .handoff import launch_analysis, launch_capture
from .image_inputs import read_analysis_image
from .live import LiveAnalysis
from .storage import ExperimentReader
from .viewport import ImageViewport, repaint_after_resize


def _rectangle(text):
    return tuple(int(value.strip()) for value in text.split(",")) if text.strip() else None


class LiveWorkbench:
    def __init__(self, root, experiment, camera=None, reference_sequence=None):
        self.root, self.experiment = root, Path(experiment).resolve()
        root.title("StrainTrace Analysis | Live")
        root.geometry("1160x760")
        root.minsize(860, 620)
        set_theme(root)
        repaint_after_resize(root)
        self.camera = tk.StringVar(value=camera or "")
        self.reference = tk.StringVar(value=str(reference_sequence) if reference_sequence else "")
        self.rois = {key: tk.StringVar() for key in ("Gauge ROI 1", "Gauge ROI 2", "2D field ROI")}
        self.strain_mode = tk.StringVar(value="auto")
        self.debayer = tk.BooleanVar()
        self.draw_target = tk.StringVar(value="Gauge ROI 1")
        self.state = tk.StringVar(value="Waiting for committed images")
        self.result = tk.StringVar(value="Select two gauge regions or a 2D field region on the committed reference.")
        self.progress = tk.StringVar(value="")
        self._paused = False
        self._working = False
        self._jobs = queue.SimpleQueue()
        self._analyzer = self._display_key = None
        self._load_alignment = None
        self._last_sequence = None
        self._data = deque(maxlen=500)
        self._build()
        self.refresh()
        root.after(250, self._tick)

    def _build(self):
        layout = ttk.Frame(self.root, padding=14)
        layout.pack(fill=tk.BOTH, expand=True)
        layout.columnconfigure(0, weight=1)
        layout.rowconfigure(1, weight=1)
        banner = ttk.Frame(layout)
        banner.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        banner.columnconfigure(0, weight=1)
        ttk.Label(banner, text="StrainTrace Analysis", style="CaptureTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Button(banner, text="View Capture experiment ↗", command=lambda: launch_capture(self.experiment)).grid(
            row=0, column=1, padx=5)
        ttk.Button(banner, text="Open offline Analysis ↗", command=self.offline).grid(row=0, column=2)
        ttk.Label(banner, text=str(self.experiment), style="CaptureMuted.TLabel").grid(row=1, column=0, columnspan=3, sticky="w")
        dock = tk.PanedWindow(layout, orient=tk.HORIZONTAL, bg=BG, sashwidth=8, borderwidth=0)
        dock.grid(row=1, column=0, sticky="nsew")
        view_panel = ttk.Frame(dock)
        data_panel = ttk.Frame(dock)
        dock.add(view_panel, minsize=420, stretch="always")
        dock.add(data_panel, minsize=355, width=420, stretch="never")
        data_panel.columnconfigure(0, weight=1)
        data_panel.rowconfigure(3, weight=1)
        self.view = ImageViewport(view_panel, "analysis", "Committed reference", lambda _camera: None, self._draw_region)
        self.view.pack(fill=tk.BOTH, expand=True)
        view_actions = ttk.Frame(view_panel)
        view_actions.pack(fill=tk.X, pady=(7, 0))
        ttk.Button(view_actions, text="Fit", width=3, style="Small.TButton", command=self.view.fit).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(view_actions, text="1:1", width=3, style="Small.TButton", command=self.view.one_to_one).pack(side=tk.LEFT)
        ttk.Label(view_actions, text="Shift-drag assigns", style="CaptureMuted.TLabel").pack(side=tk.LEFT, padx=(10, 6))
        ttk.Combobox(view_actions, textvariable=self.draw_target, values=tuple(self.rois), state="readonly", width=15).pack(
            side=tk.LEFT)
        form = ttk.Frame(data_panel, padding=(2, 0, 1, 0))
        form.grid(row=0, column=0, sticky="ew")
        form.columnconfigure(1, weight=1)
        ttk.Label(form, text="Camera").grid(row=0, column=0, sticky="w", pady=3)
        self.options = ttk.Combobox(form, textvariable=self.camera, state="readonly")
        self.options.grid(row=0, column=1, sticky="ew", padx=(9, 5))
        self.options.bind("<<ComboboxSelected>>", self._selection_changed)
        ttk.Button(form, text="Refresh", width=7, style="Small.TButton", command=self.refresh).grid(row=0, column=2)
        ttk.Label(form, text="Reference sequence").grid(row=1, column=0, sticky="w", pady=3)
        reference_entry = ttk.Entry(form, textvariable=self.reference)
        reference_entry.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(9, 0))
        reference_entry.bind("<Return>", self._selection_changed)
        self.inputs = [self.options, reference_entry]
        for line, (name, variable) in enumerate(self.rois.items(), 2):
            ttk.Label(form, text=name).grid(row=line, column=0, sticky="w")
            entry = ttk.Entry(form, textvariable=variable)
            entry.grid(row=line, column=1, columnspan=2, sticky="ew", pady=2, padx=(9, 0))
            self.inputs.append(entry)
        ttk.Label(form, text="Gauge axis").grid(row=5, column=0, sticky="w")
        axis = ttk.Combobox(form, textvariable=self.strain_mode, values=("auto", "x", "y", "distance"), state="readonly")
        axis.grid(row=5, column=1, columnspan=2, sticky="ew", pady=3, padx=(9, 0))
        self.inputs.append(axis)
        bayer = ttk.Checkbutton(form, text="Convert Bayer to luminance", variable=self.debayer)
        bayer.grid(row=6, column=0, columnspan=3, sticky="w", pady=3)
        self.inputs.append(bayer)

        actions = ttk.Frame(data_panel)
        actions.grid(row=1, column=0, sticky="ew", pady=8)
        self.start_button = ttk.Button(actions, text="Follow", width=6, style="Record.TButton", command=self.start)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        self.pause_button = ttk.Button(actions, text="Pause", width=6, command=self.pause, style="Small.TButton")
        self.pause_button.pack(side=tk.LEFT, padx=(0, 5))
        self.end_button = ttk.Button(actions, text="Reset", width=5, command=self.reset, style="Small.TButton")
        self.end_button.pack(side=tk.LEFT)
        self.mechanics_button = ttk.Button(actions, text="Mechanics…", width=11, command=self.mechanics, style="Small.TButton")
        self.mechanics_button.pack(side=tk.RIGHT)

        ttk.Label(data_panel, text="Engineering strain (%)", style="CaptureSection.TLabel").grid(row=2, column=0, sticky="w", pady=(3, 5))
        self.plot = tk.Canvas(data_panel, bg=ENTRY, highlightthickness=1, highlightbackground=BORDER, height=150)
        self.plot.grid(row=3, column=0, sticky="nsew")
        self.plot.bind("<Configure>", lambda _event: self.draw_plot())
        ttk.Label(data_panel, text="Time since reference receipt (s)", style="CaptureMuted.TLabel").grid(row=4, column=0, sticky="e", pady=(4, 2))
        ttk.Label(data_panel, textvariable=self.result, justify=tk.LEFT, wraplength=345).grid(row=5, column=0, sticky="ew", pady=(5, 0))
        footer = ttk.Frame(layout)
        footer.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(footer, text="Provisional live results", foreground=ACCENT).pack(side=tk.LEFT, padx=(0, 15))
        ttk.Label(footer, textvariable=self.state).pack(side=tk.LEFT)
        ttk.Label(footer, textvariable=self.progress, style="CaptureMuted.TLabel").pack(side=tk.RIGHT)
        self._controls()

    def _controls(self):
        active = self._analyzer is not None
        self.start_button.state(["disabled" if active else "!disabled"])
        self.pause_button.state(["!disabled" if active else "disabled"])
        self.pause_button.configure(text="Resume" if self._paused else "Pause")
        self.end_button.state(["!disabled" if active and not self._working else "disabled"])
        self.mechanics_button.state(["disabled" if active else "!disabled"])
        for item in self.inputs:
            item.state(["disabled" if active else "!disabled"])

    def refresh(self):
        if self._analyzer is not None:
            return
        try:
            with ExperimentReader(self.experiment) as reader:
                identities = [item["device"]["id"] for item in reader.cameras()]
            self.options.configure(values=identities)
            if not self.camera.get() and len(identities) == 1:
                self.camera.set(identities[0])
            self._display_reference()
        except Exception as exc:
            self.state.set(str(exc) if (self.experiment / "capture.sqlite3").is_file() else "Waiting for the Capture ledger")

    def _selection_changed(self, _event=None):
        self._display_key = None
        self._display_reference()

    def _display_reference(self):
        if self._working or self._analyzer or not self.camera.get() or self._display_key:
            return
        try:
            sequence = int(self.reference.get()) if self.reference.get().strip() else None
            with ExperimentReader(self.experiment) as reader:
                frames = reader.reference_frames(self.camera.get(), sequence=sequence)
                if not frames:
                    return
                if len(frames) > 1:
                    self.state.set("Choose a committed reference sequence.")
                    return
                row = frames[0]
                path = reader.image_path(row)
            pixels = read_analysis_image(path, row, debayer=row["color_model"] == "bayer")
            self._update_image(row, pixels)
            self._display_key = row["frame_id"]
            self.state.set(f"Reference sequence {row['sequence']}")
        except Exception as exc:
            self.state.set(str(exc))

    def _update_image(self, row, pixels, *, display_only=False):
        timestamps = row["timestamps"]
        label = "Bayer luminance" if row.get("image_conversion") == "derived_bayer_luminance" else row.get("pixel_format", "Camera luminance")
        self.view.image(Image(
            pixels, "DisplayRGB8" if display_only else label,
            pixels.dtype.itemsize * 8 if display_only else row.get("significant_bits", pixels.dtype.itemsize * 8),
            "rgb" if pixels.ndim == 3 else "mono",
            timestamps["received_utc"]["value"], timestamps["received_monotonic"]["value"],
        ))

    def _draw_region(self, _camera, region):
        if not self._analyzer:
            key = self.draw_target.get()
            self.rois[key].set(", ".join(map(str, region)))
            if key == "Gauge ROI 1":
                self.draw_target.set("Gauge ROI 2")

    def start(self):
        try:
            first, second, field = (_rectangle(var.get()) for var in self.rois.values())
            if (first is None) != (second is None):
                raise ValueError("Both gauge regions are required.")
            groups = [{"name": "G01", "roi1": first, "roi2": second, "strain_mode": self.strain_mode.get()}] if first else []
            sequence = int(self.reference.get()) if self.reference.get().strip() else None
            self._analyzer = LiveAnalysis(self.experiment, self.camera.get() or None, reference_sequence=sequence,
                                          gauges=groups, field_roi=field, load_alignment=self._load_alignment,
                                          debayer_luminance=self.debayer.get())
            self._paused = False
            self.state.set("Following committed frames")
            self._controls()
        except Exception as exc:
            messagebox.showerror("Live Analysis", str(exc), parent=self.root)

    def pause(self):
        self._paused = not self._paused
        self.state.set("Paused" if self._paused else "Following committed frames")
        self._controls()

    def reset(self):
        if not self._working and self._analyzer:
            self._analyzer.close()
            self._analyzer = None
            self._data.clear()
            self._last_sequence = None
            self._paused = False
            self._display_key = None
            self._controls()
            self.draw_plot()
            self.refresh()

    def offline(self):
        try:
            sequence = int(self.reference.get()) if self.reference.get().strip() else None
            launch_analysis(self.experiment, camera_id=self.camera.get() or None, reference_sequence=sequence)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Offline Analysis", str(exc), parent=self.root)

    def mechanics(self):
        if name := filedialog.askopenfilename(parent=self.root, filetypes=[("Mechanics alignment JSON", "*.json")]):
            try:
                from .synchronization import load_alignment_settings

                path = Path(name)
                data = json.loads(path.read_text(encoding="utf-8-sig"))
                if "mechanics" in data:
                    data = data["mechanics"]
                channel = load_alignment_settings(data, base_dir=path.parent)
                channel.refresh()
                self._load_alignment = channel
                self.state.set(f"Mechanics loaded · {channel.basis}")
            except Exception as exc:
                messagebox.showerror("Live mechanics", str(exc), parent=self.root)

    def _tick(self):
        if not self._jobs.empty():
            row, image, legend, error = self._jobs.get_nowait()
            self._working = False
            if error:
                self._paused = True
                self.state.set(str(error))
            elif row:
                self._render_result(row, image, legend)
            self._controls()
        analyzer = self._analyzer
        if analyzer and not self._paused and not self._working:
            self._working = True
            def poll():
                try:
                    row = analyzer.poll()
                    image = analyzer.image
                    legend = None
                    quality = row["field_quality"] if row else None
                    if quality and quality.get("scientific_ok"):
                        core = analyzer._core
                        image8 = core.normalize_with_bounds(image, analyzer._norm)
                        image = core.overlay_dic_field_on_image(image8, analyzer.field, component="Eyy")
                        lo, hi, scale = core.dic_color_limits(analyzer.field, "Eyy")
                        legend = f"Eyy color range  {lo / scale:.3%} … {hi / scale:.3%}"
                    self._jobs.put((row, image, legend, None))
                except Exception as exc:
                    self._jobs.put((None, None, None, exc))
            threading.Thread(target=poll, name="live-analysis-read", daemon=True).start()
            self._controls()
        elif not analyzer and not self._display_key:
            self.refresh()
        self.root.after(400, self._tick)

    def _render_result(self, row, image, legend):
        gauge = row["gauges"][0] if row["gauges"] else None
        quality = row["field_quality"]
        self.view.set_title(f"Committed sequence {row['sequence']}" + (" · Eyy" if quality and quality.get("scientific_ok") else ""))
        self._update_image(row, image, display_only=legend is not None)
        self.state.set("Paused" if self._paused else f"Capture {self._analyzer.experiment_state}")
        self.progress.set(f"{row['processed']} analyzed  ·  {row['skipped_committed_frames']} skipped")
        if gauge:
            strain = gauge["engineering_strain"]
            if self._last_sequence is not None and row["sequence"] != self._last_sequence + 1:
                self._data.append((row["elapsed_receive_s"], None))
            self._last_sequence = row["sequence"]
            self._data.append((row["elapsed_receive_s"], None if strain is None else strain * 100))
            result = f"{gauge['name']}  ·  {gauge['quality_status']}"
            if strain is not None:
                result += f"  ·  {strain:.4%}"
        else:
            result = ""
        if quality:
            result += ("\n" if result else "") + (
                f"2D DIC  ·  {quality['strain_valid_fraction']:.0%} strain-valid\n{legend}" if legend
                else f"2D DIC quality gate  ·  {', '.join(quality['scientific_reasons'])}"
            )
        if mechanics := row.get("mechanics"):
            support = mechanics.get("force_support", {}).get("status", mechanics.get("error", "Unavailable"))
            force = mechanics.get("force_N")
            result += f"\nForce  {force:.4g} N  ·  {mechanics['time_basis']}" if force is not None else f"\nForce  ·  {support}"
        self.result.set(result)
        self.draw_plot()

    def draw_plot(self):
        canvas = self.plot
        canvas.delete("all")
        w, h = canvas.winfo_width(), canvas.winfo_height()
        data = list(self._data)
        finite = [value for _, value in data if value is not None]
        if not finite:
            canvas.create_text(w / 2, h / 2, text="Awaiting valid gauge strain", fill=MUTED)
            return
        x0, x1 = data[0][0], max(data[-1][0], data[0][0] + .1)
        y0, y1 = min(finite), max(finite)
        pad = max(.005, (y1 - y0) * .12)
        y0, y1 = y0 - pad, y1 + pad
        left, right, top, bottom = 53, w - 15, 15, h - 25
        for index in range(4):
            y = top + index * (bottom - top) / 3
            t = y1 - index * (y1 - y0) / 3
            canvas.create_line(left, y, right, y, fill=BORDER)
            canvas.create_text(left - 4, y, text=f"{t:.3f}", anchor="e", fill=FG)
        for index in range(4):
            x = left + index * (right - left) / 3
            t = x0 + index * (x1 - x0) / 3
            canvas.create_text(x, bottom + 10, text=f"{t:.1f}", fill=FG)
        previous = None
        for time_value, value in data:
            if value is None:
                previous = None
                continue
            x = left + (time_value - x0) / (x1 - x0) * (right - left)
            y = bottom - (value - y0) / (y1 - y0) * (bottom - top)
            if previous:
                canvas.create_line(*previous, x, y, fill=ACCENT, width=2)
            canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=ACCENT, outline="")
            previous = x, y


def run_live(experiment, camera=None, reference_sequence=None):
    enable_windows_dpi_awareness()
    root = tk.Tk()
    LiveWorkbench(root, experiment, camera, reference_sequence)
    root.mainloop()
    return 0
