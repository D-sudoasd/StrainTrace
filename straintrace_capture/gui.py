"""Independent StrainTrace Capture desktop workbench."""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter.font import Font

from ezdic_gui.theme import enable_windows_dpi_awareness

from . import CameraRegistry, CaptureConfig, CaptureError, CaptureSession, ExperimentReader
from .calibration import calibrate, read_profile
from .cameras.base import Image
from .config import Phase
from .control import CaptureControl
from .handoff import analysis_snapshot, launch_analysis, write_snapshot
from .image_inputs import decode_samples
from .preview import PreviewSession, image_quality
from .storage import TERMINAL_STATES
from .viewport import ImageViewport, repaint_after_resize


BG, CARD, ENTRY = "#121c25", "#1a2a36", "#111e29"
FG, MUTED, ACCENT, BORDER = "#eef4f7", "#b4c7d1", "#66dec0", "#375060"
GOOD, WARN, BAD = "#a6edce", "#ffda87", "#ffacab"


def set_theme(root):
    root.configure(bg=BG)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", font=("Segoe UI", 10), foreground=FG, background=BG)
    style.configure("CaptureCard.TFrame", background=CARD)
    style.configure("CaptureCard.TLabel", background=CARD, foreground=FG)
    style.configure("CaptureMuted.TLabel", foreground=MUTED, background=BG)
    style.configure("CaptureMono.TLabel", font=("Consolas", 9), foreground=MUTED, background=CARD)
    style.configure("CaptureTitle.TLabel", font=("Segoe UI Semibold", 23), foreground=FG)
    style.configure("CaptureSection.TLabel", font=("Segoe UI Semibold", 12), foreground=FG)
    style.configure("CaptureSelected.TLabel", font=("Segoe UI Semibold", 10), foreground=ACCENT, background=CARD)
    style.configure("TButton", padding=(11, 6), bordercolor=BORDER, lightcolor=CARD, darkcolor=CARD, background=CARD)
    style.map("TButton", background=[("pressed", "#304857"), ("active", "#283d4a")],
              foreground=[("disabled", "#889ba6")])
    style.configure("Small.TButton", padding=(7, 4))
    style.configure("TCheckbutton", background=BG, indicatorcolor=ENTRY, foreground=FG)
    style.map("TCheckbutton", background=[("disabled", BG), ("active", CARD)],
              foreground=[("disabled", MUTED)], indicatorcolor=[("selected", ACCENT), ("disabled", BORDER)])
    style.configure("Record.TButton", font=("Segoe UI Semibold", 10), padding=(17, 8),
                    foreground="#112932", background=ACCENT, lightcolor=ACCENT, darkcolor=ACCENT, bordercolor=ACCENT)
    style.map("Record.TButton", background=[("disabled", "#375060"), ("pressed", "#48bd9f"), ("active", "#a4f0dc")],
              foreground=[("disabled", MUTED)])
    style.configure("TEntry", fieldbackground=ENTRY, bordercolor=BORDER, insertcolor=FG, padding=6)
    style.configure("TCombobox", fieldbackground=ENTRY, arrowcolor=FG, bordercolor=BORDER, padding=4)
    style.map("TCombobox", fieldbackground=[("readonly", ENTRY)], selectbackground=[("readonly", ENTRY)],
              selectforeground=[("readonly", FG)])
    style.configure("TNotebook", background=BG, bordercolor=BORDER)
    style.configure("TNotebook.Tab", background=CARD, foreground=MUTED, padding=(11, 8))
    style.map("TNotebook.Tab", background=[("selected", ENTRY), ("active", "#263b49")],
              foreground=[("selected", ACCENT)])
    row_font = Font(root=root, family="Segoe UI", size=9)
    style.configure("Treeview", background=ENTRY, fieldbackground=ENTRY, foreground=FG, bordercolor=BORDER,
                    rowheight=max(26, row_font.metrics("linespace") + 6), font=("Segoe UI", 9))
    style.map("Treeview", background=[("selected", "#325968")], foreground=[("selected", "#ffffff")])
    style.configure("Treeview.Heading", background=CARD, foreground=MUTED, padding=(4, 4), bordercolor=BORDER)
    root.option_add("*TCombobox*Listbox.background", ENTRY)
    root.option_add("*TCombobox*Listbox.foreground", FG)


def demo_config() -> dict:
    return {
        "config_version": 1, "name": "Speckle rehearsal",
        "cameras": [{"id": "sim:0", "settings": {"width": 640, "height": 480, "pixel_format": "Mono12"}}],
        "phases": [
            {"mode": "single", "role": "reference"},
            {"mode": "timed", "role": "test", "fps": 12, "duration_s": 8},
        ],
    }


class CaptureWorkbench:
    def __init__(self, root, config=None, experiment=None):
        self.root = root
        root.title("StrainTrace Capture")
        root.geometry("1280x815")
        root.minsize(1000, 650)
        set_theme(root)
        repaint_after_resize(root)
        self.config = {}
        self.inventory = {}
        self.views = {}
        self.active_camera = None
        self.experiment = None
        self.session = self.preview = self.control = None
        self.review_image = None
        self._calibration_profiles = {}
        self._jobs = queue.SimpleQueue()
        self._quality_work = False
        self._quality_due = 0.
        self._releasing_preview = False
        self._review_cursor = 0
        self._review_rows = []
        self._closing = False
        self.name = tk.StringVar()
        self.output_path = tk.StringVar(value=self._new_path())
        self.state = tk.StringVar(value="READY")
        self.activity = tk.StringVar(value="Select a camera to prepare the next acquisition.")
        self.counter = tk.StringVar()
        self.provider_text = tk.StringVar()
        self.quality_text = tk.StringVar(value="Start preview or select a committed image.")
        self.footer_roi = tk.StringVar(value="Shift-drag to inspect a region")
        self.external = tk.BooleanVar()
        self.guide = tk.StringVar(value="Off")
        self.palette = tk.StringVar(value="Native")
        self.review_text = tk.StringVar(value="Open an experiment or record a new one.")
        self._build()
        self._set_config(config or CaptureConfig.from_dict(demo_config()))
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(90, self._tick)
        if experiment:
            self.open_experiment(experiment)
        self.discover()

    def _new_path(self, parent=None):
        folder = Path(parent) if parent else Path.home() / "Documents" / "StrainTrace Capture"
        return str(folder / time.strftime("Capture-%Y%m%d-%H%M%S"))

    def _build(self):
        layout = ttk.Frame(self.root, padding=(15, 12))
        layout.pack(fill=tk.BOTH, expand=True)
        layout.columnconfigure(0, weight=1)
        layout.rowconfigure(2, weight=1)

        title = ttk.Frame(layout)
        title.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        title.columnconfigure(1, weight=1)
        ttk.Label(title, text="StrainTrace Capture", style="CaptureTitle.TLabel").grid(row=0, column=0, sticky="w")
        menu = ttk.Frame(title)
        menu.grid(row=0, column=1, sticky="e")
        for label, command in (
            ("Load plan", self.load_plan), ("Save plan", self.save_plan), ("Open run…", self.open_experiment),
        ):
            ttk.Button(menu, text=label, command=command, width=10, style="Small.TButton").pack(side=tk.LEFT, padx=(8, 0))

        experiment = ttk.Frame(layout)
        experiment.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        experiment.columnconfigure(3, weight=1)
        ttk.Label(experiment, text="Experiment").grid(row=0, column=0, sticky="w", padx=(0, 7))
        ttk.Entry(experiment, textvariable=self.name, width=23).grid(row=0, column=1, sticky="ew", padx=(0, 14))
        ttk.Label(experiment, text="New output folder").grid(row=0, column=2, sticky="w", padx=(0, 7))
        ttk.Entry(experiment, textvariable=self.output_path).grid(row=0, column=3, sticky="ew", padx=(0, 7))
        ttk.Button(experiment, text="Choose parent…", command=self.choose_output).grid(row=0, column=4)

        dock = tk.PanedWindow(layout, orient=tk.HORIZONTAL, background=BG, borderwidth=0, sashwidth=8,
                             sashrelief=tk.FLAT, opaqueresize=True)
        dock.grid(row=2, column=0, sticky="nsew")
        left = ttk.Frame(dock)
        center = ttk.Frame(dock)
        right = ttk.Frame(dock)
        dock.add(left, minsize=190, width=225, stretch="never")
        dock.add(center, minsize=395, stretch="always")
        dock.add(right, minsize=295, width=340, stretch="never")
        self._device_panel(left)
        self._preview_panel(center)
        self._detail_panel(right)
        self._build_footer(layout)

    def _device_panel(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        header = ttk.Frame(parent)
        header.grid(row=0, column=0, sticky="ew", pady=(4, 8))
        ttk.Label(header, text="Devices", style="CaptureSection.TLabel").pack(side=tk.LEFT)
        self.refresh_button = ttk.Button(header, text="Refresh", width=7, style="Small.TButton", command=self.discover)
        self.refresh_button.pack(side=tk.RIGHT)
        self.devices = ttk.Treeview(parent, columns=("source",), height=7, selectmode="extended", show="tree headings")
        self.devices.heading("#0", text="Camera")
        self.devices.heading("source", text="Source")
        self.devices.column("#0", width=132, minwidth=88, stretch=True)
        self.devices.column("source", width=65, minwidth=55, stretch=False)
        self.devices.grid(row=1, column=0, sticky="nsew")
        self.devices.bind("<<TreeviewSelect>>", self._camera_selection)
        ttk.Label(parent, textvariable=self.provider_text, wraplength=195, style="CaptureMuted.TLabel").grid(
            row=2, column=0, sticky="ew", padx=2, pady=(8, 4))
        ttk.Button(parent, text="Backend status…", command=self.show_providers, style="Small.TButton").grid(
            row=3, column=0, sticky="ew", pady=(0, 5))
        self.settings_button = ttk.Button(parent, text="Camera settings…", command=self.edit_camera, style="Small.TButton")
        self.settings_button.grid(row=4, column=0, sticky="ew", pady=(0, 5))
        self.cti_button = ttk.Button(parent, text="GenTL producers…", command=self.choose_cti, style="Small.TButton")
        self.cti_button.grid(row=5, column=0, sticky="ew", pady=(0, 5))
        self.preview_button = ttk.Button(parent, text="Connect & preview", command=self.toggle_preview, style="Small.TButton")
        self.preview_button.grid(row=6, column=0, sticky="ew")

    def _preview_panel(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        header = ttk.Frame(parent)
        header.grid(row=0, column=0, sticky="ew", pady=(4, 8))
        ttk.Label(header, text="Preview", style="CaptureSection.TLabel").pack(side=tk.LEFT, padx=(0, 5))
        guides = ttk.Combobox(header, textvariable=self.guide, values=("Off", "Crosshair", "Grid"),
                             width=7, state="readonly")
        guides.pack(side=tk.RIGHT, padx=(5, 0))
        guides.bind("<<ComboboxSelected>>", self.apply_preview_options)
        palette = ttk.Combobox(header, textvariable=self.palette, values=("Native", "Inferno"), width=6, state="readonly")
        palette.pack(side=tk.RIGHT, padx=(5, 0))
        palette.bind("<<ComboboxSelected>>", self.apply_preview_options)
        ttk.Button(header, text="1:1", width=3, style="Small.TButton",
                   command=lambda: self._viewport_action("one_to_one")).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(header, text="Fit", width=3, style="Small.TButton",
                   command=lambda: self._viewport_action("fit")).pack(side=tk.RIGHT)
        self.preview_area = ttk.Frame(parent)
        self.preview_area.grid(row=1, column=0, sticky="nsew")
        self.empty_preview = ttk.Label(self.preview_area, text="No camera selected", anchor="center",
                                       style="CaptureCard.TLabel")
        self.empty_preview.grid(row=0, column=0, sticky="nsew")
        ttk.Label(parent, textvariable=self.footer_roi, style="CaptureMuted.TLabel").grid(
            row=2, column=0, sticky="ew", pady=(7, 0))

    def _detail_panel(self, parent):
        self.tabs = ttk.Notebook(parent)
        self.tabs.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self.plan_panel = ttk.Frame(self.tabs, padding=10)
        self.quality_panel = ttk.Frame(self.tabs, padding=10)
        self.review_panel = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(self.plan_panel, text="Plan")
        self.tabs.add(self.quality_panel, text="Image quality")
        self.tabs.add(self.review_panel, text="Review")
        self.tabs.bind("<<NotebookTabChanged>>", lambda _event: setattr(self, "_quality_due", 0.))
        self._build_plan()
        self._build_quality()
        self._build_review()

    def _build_plan(self):
        panel = self.plan_panel
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)
        ttk.Label(panel, text="Acquisition phases", style="CaptureSection.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.phases = ttk.Treeview(panel, columns=("role", "limits"), show="tree headings", height=6, selectmode="browse")
        self.phases.heading("#0", text="Mode")
        self.phases.heading("role", text="Role")
        self.phases.heading("limits", text="Limit")
        self.phases.column("#0", width=104, minwidth=75)
        self.phases.column("role", width=88, minwidth=62)
        self.phases.column("limits", width=107, minwidth=80)
        self.phases.grid(row=1, column=0, sticky="nsew")
        self.phases.bind("<Double-Button-1>", lambda _event: self.edit_phase())
        actions = ttk.Frame(panel)
        actions.grid(row=2, column=0, sticky="ew", pady=(7, 9))
        self.plan_buttons = []
        for text, action in (
            ("Add", lambda: self.edit_phase(add=True)), ("Edit", self.edit_phase), ("Remove", self.remove_phase),
            ("↑", lambda: self.move_phase(-1)), ("↓", lambda: self.move_phase(1)),
        ):
            width = len(text) + int(len(text) > 1)
            button = ttk.Button(actions, text=text, width=width, style="Small.TButton", command=action)
            button.pack(side=tk.LEFT, padx=(0, 3))
            self.plan_buttons.append(button)
        self.external_box = ttk.Checkbutton(panel, text="Enable external commands", variable=self.external)
        self.external_box.grid(row=3, column=0, sticky="w", pady=(2, 4))
        ttk.Label(panel, text="Use a manual or event phase for operator and DAQ signals.",
                  style="CaptureMuted.TLabel", wraplength=260).grid(row=4, column=0, sticky="w", pady=(0, 7))

    def _build_quality(self):
        panel = self.quality_panel
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(2, weight=1)
        ttk.Label(panel, text="Intensity histogram", style="CaptureSection.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.hist = tk.Canvas(panel, background=ENTRY, height=105, highlightthickness=1, highlightbackground=BORDER)
        self.hist.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        detail = ttk.Frame(panel)
        detail.grid(row=2, column=0, sticky="nsew")
        text = tk.Text(detail, height=1, width=1, bg=BG, fg=FG, font=("Segoe UI", 9), borderwidth=0,
                       highlightthickness=0, wrap="word", state="disabled")
        scrollbar = ttk.Scrollbar(detail, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(fill=tk.BOTH, expand=True)

        def update_text(*_args):
            scroll = text.yview()[0]
            text.configure(state="normal")
            text.delete("1.0", tk.END)
            text.insert("1.0", self.quality_text.get())
            text.configure(state="disabled")
            text.yview_moveto(scroll)

        self.quality_text.trace_add("write", update_text)
        update_text()
        self.hist.bind("<Configure>", lambda _event: setattr(self, "_quality_due", 0.))
        region_buttons = ttk.Frame(panel)
        region_buttons.grid(row=3, column=0, sticky="ew", pady=(7, 0))
        for label, command in (("Set ROI", self.set_region), ("Clear", self.clear_region), ("Copy", self.copy_region)):
            ttk.Button(region_buttons, text=label, command=command, style="Small.TButton", width=6).pack(side=tk.LEFT, padx=(0, 5))

    def _build_review(self):
        panel = self.review_panel
        ttk.Label(panel, text="Committed images", style="CaptureSection.TLabel").pack(anchor="w", pady=(0, 8))
        ttk.Label(panel, textvariable=self.review_text, justify=tk.LEFT, wraplength=253).pack(fill=tk.X, pady=(0, 10))
        row = ttk.Frame(panel)
        row.pack(fill=tk.X)
        for label, command in (("Previous", lambda: self.browse_frame(-1)), ("Next", lambda: self.browse_frame(1)),
                               ("Latest", lambda: self.browse_frame(latest=True))):
            ttk.Button(row, text=label, command=command, style="Small.TButton", width=8).pack(side=tk.LEFT, padx=(0, 3))
        for label, command in (
            ("Verify files and ledger", self.verify), ("Open in Analysis ↗", lambda: self.open_analysis(False)),
            ("Open Live Analysis ↗", lambda: self.open_analysis(True)),
        ):
            ttk.Button(panel, text=label, command=command).pack(fill=tk.X, pady=(6, 0))
        button = ttk.Menubutton(panel, text="Lens calibration")
        actions = tk.Menu(button, tearoff=0)
        for label, command in (("Fit committed checkerboards…", self.fit_calibration),
                               ("Attach profile to next capture…", self.load_calibration),
                               ("Toggle corrected preview", self.toggle_calibration),
                               ("Clear profiles from plan", self.clear_calibration)):
            actions.add_command(label=label, command=command)
        button.configure(menu=actions)
        button.pack(fill=tk.X, pady=(8, 0))

    def _build_footer(self, layout):
        bottom = ttk.Frame(layout)
        bottom.grid(row=3, column=0, sticky="ew", pady=(13, 0))
        bottom.columnconfigure(1, weight=1)
        self.state_label = ttk.Label(bottom, textvariable=self.state, font=("Segoe UI Semibold", 11), width=13)
        self.state_label.grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 8))
        for index, variable in enumerate((self.activity, self.counter)):
            status = ttk.Label(bottom, textvariable=variable, style="CaptureMuted.TLabel" if index else "TLabel",
                               font=("Segoe UI", 9), width=1)
            status.grid(row=index, column=1, sticky="ew")
        self.trigger_button = ttk.Button(bottom, text="Send trigger", command=self.trigger)
        self.stop_button = ttk.Button(bottom, text="Stop & save", command=self.stop)
        self.record_button = ttk.Button(bottom, text="Start acquisition", style="Record.TButton", command=self.start)
        for identifier, button in enumerate((self.trigger_button, self.stop_button, self.record_button), 2):
            button.grid(row=0, column=identifier, rowspan=2, padx=(6, 0), sticky="ns")

    def _set_config(self, config):
        self.config = config.to_dict()
        self._calibration_profiles = {}
        self.name.set(config.name)
        for device in config.cameras:
            self.inventory.setdefault(device.id, {"id": device.id, "model": device.id, "backend": device.id.split(":")[0]})
        self._populate_devices([device.id for device in config.cameras])
        self._populate_phases()

    def _registry(self):
        return CameraRegistry(cti_files=self.config.get("cti_files", []))

    def _read_config(self, *, phases=None):
        data = dict(self.config)
        data["name"] = self.name.get()
        if phases is not None:
            data["phases"] = phases
        saved = {device["id"]: device["settings"] for device in data["cameras"]}
        selected = self.devices.selection()
        data["cameras"] = [{"id": identity, "settings": saved.get(identity, {})} for identity in selected]
        return CaptureConfig.from_dict(data)

    def _busy(self):
        return self.session is not None and self.session.status()["state"] not in TERMINAL_STATES

    def _job(self, action, report, failure=None):
        def work():
            try:
                self._jobs.put((report, action()))
            except Exception as exc:
                self._jobs.put((failure or self._error, exc))
        threading.Thread(target=work, name="capture-desktop-task", daemon=True).start()

    def _error(self, exc):
        self.activity.set(str(exc))
        messagebox.showerror("Capture", str(exc), parent=self.root)

    def discover(self):
        if self._busy() or self.preview:
            return
        self.refresh_button.state(["disabled"])
        self.provider_text.set("Discovering devices…")
        registry = self._registry()
        self._job(registry.discover, self._discovered)

    def _discovered(self, result):
        selected = self.devices.selection()
        configured = {entry["id"] for entry in self.config["cameras"]}
        retained = configured | set(selected)
        self.inventory = {key: value for key, value in self.inventory.items() if key in retained}
        self.inventory.update({item["id"]: item for item in result["devices"]})
        self._populate_devices(selected)
        for identity, view in self.views.items():
            view.set_title(self._view_title(identity))
        providers = result["providers"]
        self.provider_text.set("   ".join(f"{name} {state.get('count', '·')}" for name, state in providers.items()))
        self._providers = providers
        self.refresh_button.state(["!disabled"])
        self.activity.set(f"{len(result['devices'])} devices · Ctrl-click to multi-select")

    def show_providers(self):
        providers = getattr(self, "_providers", {})
        if providers:
            text = "\n\n".join(f"{name} · {state['state']}\n{state.get('message', str(state.get('count', 0)) + ' cameras')}"
                               for name, state in providers.items())
            messagebox.showinfo("Camera backends", text, parent=self.root)

    def _populate_devices(self, selected):
        self.devices.delete(*self.devices.get_children())
        for identity, item in self.inventory.items():
            title = (item.get("serial") or item["model"]) if item["backend"] == "sim" else item["model"]
            self.devices.insert("", "end", iid=identity, text=title, values=(item["backend"],))
        self.devices.selection_set(*selected)
        self._camera_selection()

    def _camera_selection(self, _event=None):
        if self._busy() or self.preview:
            return
        selected = self.devices.selection()
        if selected:
            self._show_views(selected)

    def _show_views(self, identities):
        if list(self.views) == list(identities):
            for identity in identities:
                self.views[identity].set_title(self._view_title(identity))
            return
        self.empty_preview.grid_remove()
        for view in self.views.values():
            view.destroy()
        self.views.clear()
        cols = 2 if len(identities) > 1 else 1
        for idx in range(2):
            self.preview_area.columnconfigure(idx, weight=int(idx < cols), uniform="view-cols" if idx < cols else "")
        for idx in range(max(getattr(self, "_view_rows", 1), (len(identities) + cols - 1) // cols)):
            self.preview_area.rowconfigure(idx, weight=int(idx < (len(identities) + cols - 1) // cols))
        self._view_rows = (len(identities) + cols - 1) // cols
        for idx, identity in enumerate(identities):
            pane = ImageViewport(self.preview_area, identity, self._view_title(identity), self.select_view, self._inspect_region)
            pane.grid(row=idx // cols, column=idx % cols, sticky="nsew", padx=3, pady=3)
            self.views[identity] = pane
        self.select_view(next(iter(self.views), None))
        if not self.views:
            self.empty_preview.grid(row=0, column=0, sticky="nsew")

    def _view_title(self, identity):
        item = self.inventory.get(identity, {})
        serial = f" · {item['serial']}" if item.get("serial") else ""
        name = "Speckle" if item.get("backend") == "sim" else item.get("model", identity)
        return f"{name}{serial}"

    def select_view(self, identity):
        if identity != self.active_camera:
            self.active_camera = identity
            self._quality_due = 0.
            if self.review_image is not None and not self.preview and not self._busy():
                self._review_rows = []
        for key, view in self.views.items():
            view.set_active(key == identity)

    def _inspect_region(self, camera, roi):
        self.select_view(camera)
        self.footer_roi.set(f"Inspection region  {roi[0]}, {roi[1]}   {roi[2]} × {roi[3]} px")
        self.tabs.select(self.quality_panel)
        self._quality_due = 0.

    def clear_region(self):
        if view := self.views.get(self.active_camera):
            view.roi = None
            view.draw()
            self.footer_roi.set("Shift-drag to inspect a region")
            self._quality_due = 0.

    def set_region(self):
        if not (view := self.views.get(self.active_camera)) or (image := self._image(self.active_camera)) is None:
            self.activity.set("Preview or open a frame before choosing its inspection ROI.")
            return
        if view.corrected:
            self.activity.set("Select the raw viewport before entering image coordinates.")
            return
        answer = simpledialog.askstring("Inspection ROI", "Raw image coordinates in pixels (x, y, width, height)",
                                        parent=self.root)
        if not answer:
            return
        try:
            x, y, width, height = map(int, answer.split(","))
            ih, iw = image.pixels.shape[:2]
            if min(x, y) < 0 or min(width, height) < 1 or x + width > iw or y + height > ih:
                raise ValueError
        except ValueError:
            self._error(CaptureError("ROI_OUT_OF_BOUNDS", "enter four integer coordinates inside the raw frame"))
            return
        view.roi = (x, y, width, height)
        view.draw()
        self._inspect_region(self.active_camera, view.roi)

    def copy_region(self):
        view = self.views.get(self.active_camera)
        if not view or not view.roi:
            self.activity.set("Draw an inspection region before copying coordinates.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(", ".join(map(str, view.roi)))
        self.activity.set("Inspection coordinates copied in raw image pixels.")

    def _viewport_action(self, function):
        if view := self.views.get(self.active_camera):
            getattr(view, function)()

    def apply_preview_options(self, _event=None):
        for view in self.views.values():
            view.guide = self.guide.get()
            view.palette = "inferno" if self.palette.get() == "Inferno" else "native"
            view.image(self._image(view.camera_id))
            view.draw()

    def _populate_phases(self, index=0):
        self.phases.delete(*self.phases.get_children())
        for idx, row in enumerate(self.config["phases"]):
            limits = []
            if row.get("fps"):
                limits.append(f"{row['fps']:g} Hz")
            if count := row.get("frame_count"):
                limits.append(f"{count} frame" + ("s" if count != 1 else ""))
            if row.get("duration_s"):
                limits.append(f"{row['duration_s']:g} s")
            if row.get("post_frames"):
                limits.append(f"{row.get('pre_frames', 0)}+{row['post_frames']}")
            self.phases.insert("", "end", iid=str(idx), text=row["mode"].capitalize(),
                               values=(row["role"].capitalize(), " / ".join(limits) or "Until stopped"))
        if self.phases.get_children():
            self.phases.selection_set(str(min(index, len(self.config["phases"]) - 1)))

    def _selected_phase(self):
        return int(self.phases.selection()[0]) if self.phases.selection() else None

    def edit_phase(self, *, add=False):
        if self._busy():
            return
        index = None if add else self._selected_phase()
        if index is None and not add:
            return
        phase = {"mode": "timed", "role": "test", "fps": 10., "frame_count": 20} if add else self.config["phases"][index]
        PhaseDialog(self.root, phase, lambda value: self._replace_phase(index, value))

    def _replace_phase(self, index, phase):
        if index is None:
            self.config["phases"].append(phase)
            index = len(self.config["phases"]) - 1
        else:
            self.config["phases"][index] = phase
        self._populate_phases(index)

    def remove_phase(self):
        if len(self.config["phases"]) <= 1:
            self.activity.set("An acquisition plan needs at least one phase.")
            return
        if (index := self._selected_phase()) is not None:
            self.config["phases"].pop(index)
            self._populate_phases(index)

    def move_phase(self, offset):
        index = self._selected_phase()
        rows = self.config["phases"]
        if index is not None and 0 <= index + offset < len(rows):
            rows.insert(index + offset, rows.pop(index))
            self._populate_phases(index + offset)

    def choose_output(self):
        if folder := filedialog.askdirectory(parent=self.root, title="Choose parent for a new experiment"):
            self.output_path.set(self._new_path(folder))

    def choose_cti(self):
        paths = filedialog.askopenfilenames(parent=self.root, title="Add trusted vendor GenTL producers",
                                            filetypes=[("GenTL producer", "*.cti")])
        if paths:
            self.config["cti_files"] = list(paths)
            self.discover()

    def load_plan(self):
        if self._busy() or self.preview:
            self.activity.set("Disconnect preview or let acquisition finish before changing the plan.")
            return
        if path := filedialog.askopenfilename(parent=self.root, filetypes=[("Capture JSON", "*.json")]):
            try:
                self._set_config(CaptureConfig.load(path))
                self.discover()
            except Exception as exc:
                self._error(exc)

    def save_plan(self):
        try:
            config = self._read_config()
            path = filedialog.asksaveasfilename(parent=self.root, defaultextension=".json",
                                                filetypes=[("Capture JSON", "*.json")])
            if path:
                Path(path).write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                self.activity.set("Acquisition plan saved.")
        except Exception as exc:
            self._error(exc)

    def edit_camera(self):
        selected = self.devices.selection()
        if not selected:
            self.activity.set("Select a camera to view its settings.")
            return
        identity = self.active_camera if self.active_camera in selected else selected[0]
        registry = self._registry()
        settings = next((item["settings"] for item in self.config["cameras"] if item["id"] == identity), {})

        def features():
            with registry.camera(identity) as camera:
                camera.set_features(settings)
                return camera.device, camera.features()

        self._after_preview(lambda: self._job(features, lambda data: self._camera_dialog(data, registry, settings)))

    def _camera_dialog(self, data, registry, settings):
        device, features = data
        def accept(changes):
            def apply():
                with registry.camera(device.id) as camera:
                    camera.set_features(changes)
                return changes
            self._job(apply, lambda applied: self._set_camera_settings(device.id, applied))
        FeatureDialog(self.root, device, features, settings, accept)

    def _set_camera_settings(self, identity, settings):
        self.config["cameras"] = [item for item in self.config["cameras"] if item["id"] != identity]
        self.config["cameras"].append({"id": identity, "settings": settings})
        self.activity.set(f"Verified device accepted settings for {identity}.")

    def toggle_preview(self):
        if self._releasing_preview:
            return
        if self.preview:
            self._after_preview(lambda: self.activity.set("Preview disconnected."))
            return
        try:
            config = self._read_config()
            self.session = None
            self.review_image = None
            self.counter.set("")
            self._show_views([item.id for item in config.cameras])
            self.preview = PreviewSession(config.cameras, self._registry())
            self.preview.start()
            self.activity.set("Connecting cameras for preview…")
        except Exception as exc:
            self._error(exc)

    def _after_preview(self, action):
        if self._releasing_preview:
            return
        if self.preview is None:
            action()
            return
        self._releasing_preview = True
        self.preview.stop()
        self.activity.set("Releasing preview devices…")

        def ready():
            if self.preview is not None and not self.preview.stopped:
                self.root.after(80, ready)
            else:
                self.preview = None
                self._releasing_preview = False
                if not self._closing:
                    action()
        self.root.after(50, ready)

    def start(self):
        if self._busy() or self._releasing_preview:
            return
        try:
            config = self._read_config()
            output = self.output_path.get().strip()
            if not output:
                raise CaptureError("OUTPUT_REQUIRED", "choose a new output folder")
            self._after_preview(lambda: self._start_capture(config, output))
        except Exception as exc:
            self._error(exc)

    def _start_capture(self, config, output):
        try:
            session = CaptureSession(config, output)
            session.start()
            self.session = session
            self.experiment = session.output
            self.review_image = None
            self._review_rows = []
            if self.external.get():
                control = CaptureControl(session)
                control.__enter__()
                self.control = control
            self.activity.set("Opening capture devices…")
        except Exception as exc:
            self._error(exc)

    def trigger(self):
        try:
            self.session.request_trigger()
            self.activity.set("Operator trigger requested.")
        except Exception as exc:
            self._error(exc)

    def stop(self):
        if self.session:
            self.session.request_stop()

    def open_experiment(self, folder=None):
        if self._busy() or self.preview:
            self.activity.set("Stop the current session or preview before opening another experiment.")
            return
        folder = folder or filedialog.askdirectory(parent=self.root, title="Open a Capture experiment")
        if not folder:
            return
        try:
            with ExperimentReader(folder) as reader:
                camera_rows = reader.cameras()
                experiment = reader.experiment
                self.experiment = reader.root
            self.session = None
            self._review_rows = []
            self.review_image = None
            self.inventory.update({row["device"]["id"]: row["device"] for row in camera_rows})
            self._show_views([row["device"]["id"] for row in camera_rows])
            self.tabs.select(self.review_panel)
            self.review_text.set(f"{experiment['config']['name']}\n{experiment['state'].capitalize()}")
            self.browse_frame(latest=True)
        except Exception as exc:
            self._error(exc)

    def browse_frame(self, direction=0, *, latest=False):
        if not self.experiment or not self.active_camera:
            self.activity.set("Select an experiment and a camera.")
            return
        try:
            with ExperimentReader(self.experiment) as reader:
                self._review_rows = reader.frames(camera_id=self.active_camera)
                if not self._review_rows:
                    self.review_text.set("This camera has no committed frames yet.")
                    return
                total = len(self._review_rows)
                self._review_cursor = total - 1 if latest else min(total - 1, max(0, self._review_cursor + direction))
                row = self._review_rows[self._review_cursor]
                data = decode_samples(reader.image_path(row), color_model=row["color_model"])
                color = {"rgb": "bgr", "rgba": "bgra"}.get(row["color_model"], row["color_model"])
                self.review_image = (self.active_camera, Image(
                    data, row["pixel_format"], row["significant_bits"], color,
                    row["timestamps"]["received_utc"]["value"], row["timestamps"]["received_monotonic"]["value"],
                ))
            self.review_text.set(f"{self.active_camera}\n{self._review_cursor + 1} / {total}  ·  sequence {row['sequence']}\n"
                                 f"{row['role'].capitalize()}  ·  {row['pixel_format']}  ·  phase {row['phase']}")
        except Exception as exc:
            self._error(exc)

    def verify(self):
        if not self.experiment:
            self.activity.set("Choose an experiment before verifying files.")
            return
        path = self.experiment
        def action():
            with ExperimentReader(path) as reader:
                return reader.verify()
        def report(data):
            status = "Complete" if data["complete"] else ("Check data" if data["state"] == "completed" else data["state"].capitalize())
            self.review_text.set(f"{status}  ·  {data['committed_frames']} committed\n"
                                 f"{len(data['issues'])} ledger or file issues\n"
                                 f"{data['scheduled_misses']} schedule misses")
            self.activity.set("Committed files were checked against the experiment ledger.")
            if data["issues"]:
                messagebox.showwarning("Integrity issues", "\n".join(str(issue) for issue in data["issues"][:8]),
                                       parent=self.root)
        self._job(action, report)

    def fit_calibration(self):
        if not self.experiment or not self.active_camera:
            self.activity.set("Choose a committed calibration experiment.")
            return
        camera_ids = [self.active_camera]
        peers = [camera for camera in self.views if camera != self.active_camera]
        if peers and messagebox.askyesno("Stereo calibration", "Fit a stereo model with this camera as the left camera?",
                                         parent=self.root):
            peer = simpledialog.askstring("Stereo calibration", "Right camera ID\n" + "\n".join(peers),
                                          initialvalue=peers[0], parent=self.root)
            if peer is None:
                return
            camera_ids.append(peer)
        board = simpledialog.askstring("Checkerboard target", "Inner columns, inner rows, measured square size in mm",
                                       initialvalue="9, 6, 20", parent=self.root)
        if not board:
            return
        try:
            columns, rows, size = board.split(",")
            params = dict(inner_columns=int(columns), inner_rows=int(rows), square_mm=float(size))
        except ValueError:
            self._error(CaptureError("BOARD_FORMAT", "enter two integer corner counts and the measured square size"))
            return
        output = filedialog.asksaveasfilename(parent=self.root, defaultextension=".json", filetypes=[("Lens profile", "*.json")])
        if not output:
            return
        root = self.experiment
        def run():
            profile = calibrate(root, camera_ids, **params)
            write_snapshot(profile, output)
            return profile
        def finished(profile):
            self._calibration_profiles.update(profile["cameras"])
            rms = ", ".join(f"{name} {data['reprojection_rms_px']:.3f} px" for name, data in profile["cameras"].items())
            self.review_text.set(f"Calibration saved\nReprojection RMS\n{rms}")
        self._job(run, finished)

    def load_calibration(self):
        if path := filedialog.askopenfilename(parent=self.root, filetypes=[("Lens profile", "*.json")]):
            try:
                profile = read_profile(path)
                self._calibration_profiles.update(profile["cameras"])
                if path not in self.config["calibration_files"]:
                    self.config["calibration_files"].append(path)
                self.activity.set("Lens profile will be copied with the next acquisition.")
            except Exception as exc:
                self._error(exc)

    def clear_calibration(self):
        self.config["calibration_files"] = []
        self._calibration_profiles.clear()
        for view in self.views.values():
            view.show_calibration(None)
        self.activity.set("Calibration profiles removed from the next acquisition plan.")

    def toggle_calibration(self):
        view = self.views.get(self.active_camera)
        if not view:
            self.activity.set("Select a camera for corrected preview.")
            return
        if view.corrected:
            view.show_calibration(None)
            return
        try:
            if not self._calibration_profiles:
                for path in self.config["calibration_files"]:
                    self._calibration_profiles.update(read_profile(path)["cameras"])
            if self.experiment and self.active_camera not in self._calibration_profiles:
                with ExperimentReader(self.experiment) as reader:
                    for record in reader.experiment.get("calibrations", ()):
                        self._calibration_profiles.update(read_profile(reader.root / record["path"])["cameras"])
            model = self._calibration_profiles.get(self.active_camera)
            if not model:
                raise CaptureError("CALIBRATION_UNAVAILABLE", "load or fit a lens profile for this camera first")
            if self.preview:
                geometry = self.preview.status().get(self.active_camera, {}).get("features", {})
            elif self.experiment:
                with ExperimentReader(self.experiment) as reader:
                    camera, = [item for item in reader.cameras() if item["device"]["id"] == self.active_camera]
                    geometry = {name: feature["value"] for name, feature in camera["features"].items()}
            else:
                geometry = {}
            for key, expected in model.get("sensor_geometry", {}).items():
                if expected is not None and geometry.get(key) != expected:
                    raise CaptureError("CALIBRATION_GEOMETRY_MISMATCH", f"cannot match calibration setting {key}={expected}")
            view.show_calibration(model)
            self.activity.set("The viewport uses the lens profile. Saved TIFFs retain the camera pixels.")
        except Exception as exc:
            self._error(exc)

    def open_analysis(self, live):
        if not self.experiment or not self.active_camera:
            self.activity.set("Select an experiment and its camera before opening Analysis.")
            return
        try:
            selection = None
            if not live:
                try:
                    analysis_snapshot(self.experiment, self.active_camera)
                except CaptureError as exc:
                    if exc.code != "CHOOSE_REFERENCE":
                        raise
                    selection = simpledialog.askinteger("Choose reference", "Committed reference sequence",
                                                        minvalue=1, parent=self.root)
                    if selection is None:
                        return
            launch_analysis(self.experiment, camera_id=self.active_camera, live=live, reference_sequence=selection)
            self.activity.set("Analysis opened in its own process.")
        except Exception as exc:
            self._error(exc)

    def _image(self, camera):
        if self.preview:
            return self.preview.latest_image(camera)
        if self.review_image and camera == self.review_image[0] and self.tabs.select() == str(self.review_panel):
            return self.review_image[1]
        return self.session.latest_image(camera) if self.session else None

    def _tick_quality(self):
        view = self.views.get(self.active_camera)
        if not view or self.tabs.select() != str(self.quality_panel) or self._quality_work:
            return
        image = self._image(self.active_camera)
        now = time.monotonic()
        if image is None or now < self._quality_due:
            return
        self._quality_due = now + .65
        self._quality_work = True
        roi = view.roi
        self._job(lambda: image_quality(image, roi), self._quality_result, self._quality_failed)

    def _quality_failed(self, exc):
        self._quality_work = False
        self.quality_text.set(str(exc))

    def _quality_result(self, result):
        self._quality_work = False
        bins = result["histogram"]
        self.hist.delete("all")
        w, h = self.hist.winfo_width(), self.hist.winfo_height() - 19
        peak = max(1, max(bins))
        for index, count in enumerate(bins):
            self.hist.create_rectangle(index * w / len(bins), h, (index + 1) * w / len(bins),
                                       h - count / peak * (h - 5), fill="#67cdb9", outline="")
        self.hist.create_text(4, h + 9, text="0", anchor="w", fill=MUTED)
        self.hist.create_text(w - 4, h + 9, text=str(result["code_max"]), anchor="e", fill=MUTED)
        lo, med, hi = result["percentiles_p5_p50_p95"]
        text = (f"P5 / P50 / P95\n{lo:.0f} / {med:.0f} / {hi:.0f}\n"
                f"Near black    {result['dark_fraction']:.1%}\n"
                f"Near full scale    {result['bright_fraction']:.1%}\n"
                f"Focus response    {result['focus_laplacian_variance']:.1f}")
        if max(result["dark_fraction"], result["bright_fraction"]) >= .2:
            text += "\nCheck exposure and illumination."
        if texture := result["texture"]:
            text += (f"\n\nLocal texture\nContrast (P95−P5)    {texture['contrast_p95_p5']:.1f}"
                     f"\nGradient ratio    {texture['structure_tensor_ratio']:.3f}"
                     f"\n{texture['failure_code'] or 'Core texture preflight passed'}")
        self.quality_text.set(text)

    def _tick(self):
        while not self._jobs.empty():
            send, value = self._jobs.get_nowait()
            send(value)
        status = self.session.status() if self.session else None
        active = status is not None and status["state"] not in TERMINAL_STATES
        if status:
            stage = status["state"].upper()
            if active and status["mode"] in ("manual", "event", "hardware"):
                stage = "ARMED"
            if status["error"]:
                self.activity.set(status["error"]["message"])
            elif active and status["phase"] and status["mode"]:
                self.activity.set(f"Phase {status['phase']}  ·  {status['mode']}")
            elif self.state.get() not in {"COMPLETED", "STOPPED"} and not active:
                self.activity.set("Acquisition ended. Images are available in Review.")
            self.state.set(stage)
            samples = status["cameras"]
            saved = sum(value["saved"] for value in samples.values())
            received = sum(value["received"] for value in samples.values())
            losses = sum(sum(value[key] for key in ("native_counter_gaps", "native_counter_duplicates", "app_unsaved",
                                                   "hardware_identity_missing", "reported_driver_skips", "transport_errors"))
                         for value in samples.values())
            if not active and stage == "COMPLETED" and (losses or status["scheduled_misses"] or status["short_pre_windows"]):
                self.state.set("CHECK DATA")
                self.activity.set("Review missed requests and camera or storage discrepancies.")
            free = status["disk_free_bytes"]
            disk = "disk unknown" if free is None else f"{free / 2**30:.0f} GiB free"
            self.counter.set(f"{saved}/{received} saved · Q {status['queue_size']}/{status['queue_capacity']} · "
                             f"{status['disk_bytes_per_s'] / 2**20:.1f} MiB/s · {disk} · {losses} gaps")
            if not active and self.control:
                self.control.__exit__(None, None, None)
                self.control = None
        elif self.preview:
            states = self.preview.status()
            errors = [f"{key} {value['error']}" for key, value in states.items() if value["error"]]
            self.activity.set(errors[0] if errors else "    ".join(f"{key}  {value['state']}" for key, value in states.items()))
            self.state.set("PREVIEW")
        else:
            self.state.set("READY")
        stage = self.state.get()
        self.state_label.configure(foreground=BAD if stage == "FAILED" else (WARN if stage in ("ARMED", "CHECK DATA") else GOOD))
        self.preview_button.configure(text="Disconnect preview" if self.preview else "Connect & preview")
        for button in (self.settings_button, self.cti_button):
            button.state(["disabled" if active or self.preview else "!disabled"])
        for button in (*self.plan_buttons, self.preview_button, self.record_button, self.external_box):
            button.state(["disabled" if active or self._releasing_preview else "!disabled"])
        self.stop_button.state(["!disabled" if active else "disabled"])
        self.trigger_button.state(["!disabled" if active and status["mode"] in ("manual", "event") else "disabled"])
        for camera, view in self.views.items():
            view.image(self._image(camera))
        self._tick_quality()
        if self._closing and not active and not self.preview:
            self.root.destroy()
            return
        self.root.after(90, self._tick)

    def close(self):
        if self._busy():
            if not messagebox.askyesno("Stop acquisition?", "Stop acquisition, save queued frames and close Capture?",
                                       parent=self.root):
                return
            self.stop()
        self._closing = True
        self._after_preview(lambda: None)


class PhaseDialog:
    _fields = {
        "frame_count": ("Frame count per camera", int), "duration_s": ("Maximum duration (s)", float),
        "fps": ("Scheduled rate (Hz)", float), "frame_timeout_s": ("Frame timeout (s)", float),
        "hardware_source": ("Physical trigger input", str), "pre_frames": ("Frames before event", int),
        "post_frames": ("Frames after event", int),
    }
    _settings = {
        "single": (), "timed": ("fps", "frame_count", "duration_s"), "continuous": ("frame_count", "duration_s"),
        "manual": ("frame_count", "duration_s"), "hardware": ("hardware_source", "frame_count", "duration_s"),
        "event": ("fps", "pre_frames", "post_frames", "duration_s"),
    }

    def __init__(self, parent, phase, accept):
        self.top = tk.Toplevel(parent, bg=BG)
        self.top.title("Acquisition phase")
        self.top.transient(parent)
        self.top.grab_set()
        self.accept = accept
        self.hardware_pairing = phase.get("hardware_pairing")
        self.mode = tk.StringVar(value=phase["mode"])
        self.role = tk.StringVar(value=phase["role"])
        self.values = {key: tk.StringVar(value=json.dumps(phase[key]) if isinstance(phase.get(key), dict) else str(phase.get(key, "")))
                       for key in self._fields}
        panel = ttk.Frame(self.top, padding=15)
        panel.pack(fill=tk.BOTH, expand=True)
        panel.columnconfigure(1, weight=1)
        for index, (label, var, options) in enumerate((
            ("Mode", self.mode, tuple(self._settings)), ("Image role", self.role, ("test", "reference", "calibration")),
        )):
            ttk.Label(panel, text=label).grid(row=index, column=0, sticky="w", padx=(0, 15), pady=5)
            menu = ttk.Combobox(panel, textvariable=var, values=options, state="readonly", width=19)
            menu.grid(row=index, column=1, sticky="ew")
            if index == 0:
                menu.bind("<<ComboboxSelected>>", self.show_fields)
        self.form = ttk.Frame(panel)
        self.form.grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Button(panel, text="Apply phase", command=self.save, style="Record.TButton").grid(
            row=3, column=1, sticky="e", pady=(9, 2))
        self.top.bind("<Escape>", lambda _event: self.top.destroy())
        self.show_fields()

    def show_fields(self, _event=None):
        for widget in self.form.winfo_children():
            widget.destroy()
        for line, key in enumerate((*self._settings[self.mode.get()], "frame_timeout_s")):
            ttk.Label(self.form, text=self._fields[key][0]).grid(row=line, column=0, sticky="w", padx=(0, 15))
            ttk.Entry(self.form, textvariable=self.values[key], width=20).grid(row=line, column=1, sticky="ew", pady=5)
        self.form.columnconfigure(1, weight=1)

    def save(self):
        try:
            mode = self.mode.get()
            data = {"mode": mode, "role": self.role.get()}
            if mode == "hardware" and self.hardware_pairing is not None:
                data["hardware_pairing"] = self.hardware_pairing
            for key in (*self._settings[mode], "frame_timeout_s"):
                if value := self.values[key].get().strip():
                    data[key] = json.loads(value) if key == "hardware_source" and value.startswith("{") else self._fields[key][1](value)
            validated = Phase.from_dict(data).to_dict()
            self.accept(validated)
            self.top.destroy()
        except (CaptureError, ValueError) as exc:
            messagebox.showerror("Acquisition phase", str(exc), parent=self.top)


class FeatureDialog:
    def __init__(self, parent, device, features, settings, accept):
        self.top = tk.Toplevel(parent, bg=BG)
        self.top.title(f"Camera settings · {device.id}")
        self.top.geometry("600x560")
        self.top.minsize(490, 390)
        self.top.transient(parent)
        self.top.grab_set()
        self.features, self.settings, self.accept = features, settings, accept
        self.fields = {}
        header = ttk.Frame(self.top, padding=13)
        header.pack(fill=tk.X)
        ttk.Label(header, text=device.model, style="CaptureSection.TLabel").pack(anchor="w")
        ttk.Label(header, text=f"{device.id}  ·  {device.support}", style="CaptureMuted.TLabel").pack(anchor="w", pady=(4, 0))
        middle = ttk.Frame(self.top, padding=(12, 1, 12, 1))
        middle.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(middle, bg=BG, highlightthickness=0, width=560)
        scrollbar = ttk.Scrollbar(middle, orient="vertical", command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)
        content = ttk.Frame(canvas)
        anchor = canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(anchor, width=event.width))
        self.top.bind("<MouseWheel>", lambda event: canvas.yview_scroll(-int(event.delta / 120), "units"))
        content.columnconfigure(1, weight=1)
        ordered = sorted(features, key=lambda key: (key not in {"width", "height", "offset_x", "offset_y", "pixel_format",
                                                               "exposure_us", "fps", "gain_db"}, key))
        for index, key in enumerate(ordered):
            feature = features[key]
            baseline = settings.get(key, feature.value)
            var = tk.StringVar(value="" if baseline is None else str(baseline))
            self.fields[key] = var, var.get()
            ttk.Label(content, text=key.replace("_", " ").capitalize()).grid(row=index, column=0, sticky="w", padx=(0, 8))
            choices = feature.choices or (("False", "True") if type(feature.value) is bool else ())
            if choices and feature.writable:
                widget = ttk.Combobox(content, textvariable=var, values=choices, state="readonly", width=14)
            else:
                widget = ttk.Entry(content, textvariable=var, width=16, state="normal" if feature.writable else "readonly")
            widget.grid(row=index, column=1, sticky="ew", pady=3)
            constraints = f"{feature.minimum:g}…{feature.maximum:g}" if feature.minimum is not None and feature.maximum is not None else ""
            unit = feature.unit or ""
            suffix = "Read only" if not feature.writable else f"{constraints} {unit}".strip()
            ttk.Label(content, text=suffix, style="CaptureMuted.TLabel").grid(row=index, column=2, sticky="w", padx=(9, 0))
        ttk.Button(self.top, text="Apply camera settings", command=self.save, style="Record.TButton").pack(anchor="e", padx=14, pady=12)
        self.top.bind("<Escape>", lambda _event: self.top.destroy())

    def save(self):
        try:
            changes = dict(self.settings)
            for name, (variable, initial) in self.fields.items():
                feature = self.features[name]
                text = variable.get().strip()
                if not feature.writable or text == initial:
                    continue
                if not text:
                    changes.pop(name, None)
                    continue
                original = feature.value
                if type(original) is bool:
                    value = text == "True"
                elif type(original) is int:
                    value = int(text)
                elif type(original) is float or feature.unit in ("Hz", "dB", "us"):
                    value = float(text)
                else:
                    value = text
                feature.validate(value)
                changes[name] = value
            self.accept(changes)
            self.top.destroy()
        except (CaptureError, ValueError) as exc:
            messagebox.showerror("Camera settings", str(exc), parent=self.top)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="straintrace-capture-gui")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--experiment", type=Path)
    args = parser.parse_args(argv)
    enable_windows_dpi_awareness()
    root = tk.Tk()
    CaptureWorkbench(root, CaptureConfig.load(args.config) if args.config else None, args.experiment)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
