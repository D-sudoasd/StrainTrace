"""Hover/F1 tooltips, combobox choice help, and Matplotlib navigation toolbar."""

import os
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

_WIN32_MONITOR_USER32 = None
_WIN32_MONITOR_INFO_CLS = None
_WIN32_POINT_CLS = None


def _get_win32_monitor_api():
    """Initialize Win32 monitor bounds bindings once without mutating ctypes.windll."""
    global _WIN32_MONITOR_USER32, _WIN32_MONITOR_INFO_CLS, _WIN32_POINT_CLS
    if _WIN32_MONITOR_USER32 is None and os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class _MonitorInfo(ctypes.Structure):
            _fields_ = [
                ("size", wintypes.DWORD),
                ("monitor", wintypes.RECT),
                ("work", wintypes.RECT),
                ("flags", wintypes.DWORD),
            ]

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.MonitorFromPoint.argtypes = [wintypes.POINT, wintypes.DWORD]
        user32.MonitorFromPoint.restype = wintypes.HANDLE
        user32.GetMonitorInfoW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_MonitorInfo)]
        user32.GetMonitorInfoW.restype = wintypes.BOOL
        _WIN32_MONITOR_INFO_CLS = _MonitorInfo
        _WIN32_POINT_CLS = wintypes.POINT
        _WIN32_MONITOR_USER32 = user32
    return _WIN32_MONITOR_USER32, _WIN32_MONITOR_INFO_CLS, _WIN32_POINT_CLS


class ToolTip:
    def __init__(self, widget, text, wraplength=360, delay_ms=450):
        self.widget = widget
        self.owner = widget.winfo_toplevel()
        self.text = text
        self.wraplength = wraplength
        self.delay_ms = delay_ms
        self.after_id = None
        self.tip_window = None
        self.event = None
        self.pending_text = None
        self.shown_text = None

        self.widget._tooltip_text = text if isinstance(text, str) else ""
        self.widget._tooltip = self
        self.widget.bind("<Enter>", self.schedule, add="+")
        self.widget.bind("<Motion>", self.schedule, add="+")
        self.widget.bind("<Leave>", self.hide, add="+")
        self.widget.bind("<ButtonPress>", self.hide, add="+")
        self.widget.bind("<FocusOut>", self.hide, add="+")
        self.widget.bind("<Unmap>", self.hide, add="+")
        self.widget.bind("<Destroy>", self.dispose, add="+")
        self.widget.bind("<F1>", self.show_help, add="+")
        self.widget.bind("<Escape>", self.dismiss, add="+")

    def resolve_text(self, event=None):
        # The registered _tooltip_text describes the whole control. Hovering a
        # table header or candidate must not replace that general help; the
        # active context is held separately in pending_text / shown_text.
        return self.text(event) if callable(self.text) else self.text

    def schedule(self, event=None):
        text = self.resolve_text(event)
        self.event = event
        if not text:
            self.hide()
            return
        if text == self.shown_text or (self.after_id is not None and text == self.pending_text):
            return
        self.hide()
        self.event = event
        self.pending_text = text
        self.after_id = self.owner.after(self.delay_ms, self.show)

    def show_help(self, event=None):
        self.hide()
        self.event = event
        self.show()
        return "break"

    def dismiss(self, event=None):
        visible = self.tip_window is not None
        self.hide()
        if visible:
            return "break"

    def unschedule(self):
        if self.after_id is not None:
            try:
                self.owner.after_cancel(self.after_id)
            except tk.TclError:
                pass
            self.after_id = None

    def screen_bounds(self, x, y):
        if os.name == "nt":
            import ctypes

            user32, monitor_info_cls, point_cls = _get_win32_monitor_api()
            if user32 is not None and monitor_info_cls is not None and point_cls is not None:
                info = monitor_info_cls(size=ctypes.sizeof(monitor_info_cls))
                monitor = user32.MonitorFromPoint(point_cls(int(x), int(y)), 2)
                if monitor and user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
                    rect = info.work
                    return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top
        return 0, 0, self.widget.winfo_screenwidth(), self.widget.winfo_screenheight()

    def show(self):
        self.after_id = None
        text = self.resolve_text(self.event)
        if self.tip_window is not None or not text:
            return

        try:
            if not self.widget.winfo_ismapped():
                return
            wx = self.widget.winfo_rootx()
            wy = self.widget.winfo_rooty()
            wh = self.widget.winfo_height()
            pointer = self.event is not None and getattr(self.event, "keysym", "??") in ("??", "")
            if pointer and getattr(self.event, "type", None) == tk.EventType.VirtualEvent:
                anchor_x, anchor_y = self.widget.winfo_pointerxy()
            else:
                anchor_x = self.event.x_root if pointer else wx
                anchor_y = self.event.y_root if pointer else wy + wh
            screen_x, screen_y, screen_w, screen_h = self.screen_bounds(anchor_x, anchor_y)

            self.tip_window = tk.Toplevel(self.widget)
            self.tip_window.withdraw()
            self.tip_window.wm_overrideredirect(True)
            self.tip_window.wm_attributes("-topmost", True)
            label = ttk.Label(
                self.tip_window, text=text, justify=tk.LEFT,
                wraplength=min(self.wraplength, screen_w - 40),
                background="#fff8dc", foreground="#1f2937",
                relief=tk.SOLID, borderwidth=1, padding=(8, 5),
            )
            label.pack()
            self.tip_window.update_idletasks()
            width = self.tip_window.winfo_reqwidth()
            height = self.tip_window.winfo_reqheight()
            region = getattr(self.event, "help_region", None)
            if region is not None:
                region_x, region_width = region
                x = region_x + region_width + 8
                if x + width > screen_x + screen_w - 8:
                    x = region_x - width - 8
                y = anchor_y
            else:
                x, y = anchor_x + 16, anchor_y + 20
            x = min(max(screen_x + 8, x), screen_x + screen_w - width - 8)
            if y + height > screen_y + screen_h - 8:
                y = anchor_y - height - 12
            y = min(max(screen_y + 8, y), screen_y + screen_h - height - 8)
            self.tip_window.wm_geometry(f"+{x}+{y}")
            self.tip_window.deiconify()
            self.shown_text = text
        except tk.TclError:
            self.hide()
            return

    def hide(self, event=None):
        self.unschedule()
        self.pending_text = None
        self.shown_text = None
        if self.tip_window is not None:
            try:
                self.tip_window.destroy()
            except tk.TclError:
                pass
            self.tip_window = None

    def dispose(self, event=None):
        self.hide()
        # Do not retain destroyed controls or their app through event objects
        # and dynamic text closures until a later worker-thread collection.
        self.event = None
        self.text = ""
        self.widget._tooltip_text = ""


class ChoiceToolTip(ToolTip):
    """Explain both the selected value and each native combobox list item."""

    def __init__(self, widget, text, choices):
        self.summary = text
        self.choices = {str(value): explanation for value, explanation in choices.items()}
        self.popup_choice = None
        self.listbox = None
        self._disposed = False
        super().__init__(widget, self.choice_text)
        widget._tooltip_choices = self.choices
        self.previous_postcommand = widget.cget("postcommand")
        widget.configure(postcommand=self.prepare_choices)
        self.motion_command = widget.register(self.on_choice_motion)
        self.selection_command = widget.register(self.on_choice_selection)
        self.hide_command = widget.register(self.hide_choices)

    def choice_text(self, event=None):
        if self.popup_choice is not None:
            return f"{self.popup_choice}：{self.choices[self.popup_choice]}\n点击或按回车选择。"
        summary = self.summary(event) if callable(self.summary) else self.summary
        value = str(self.widget.get())
        return f"{summary}\n当前 {value}：{self.choices[value]}" if value in self.choices else summary

    def prepare_choices(self):
        if self.previous_postcommand:
            self.widget.tk.call(self.previous_postcommand)
        popup = self.widget.tk.call("ttk::combobox::PopdownWindow", str(self.widget))
        listbox = f"{popup}.f.l"
        if self.listbox == listbox:
            return
        self.listbox = listbox
        for sequence, script in (
            ("<Motion>", f"{self.motion_command} %y %X %Y"),
            ("<Map>", self.selection_command),
            ("<KeyRelease>", self.selection_command),
            ("<<ListboxSelect>>", self.selection_command),
            ("<Leave>", self.hide_command),
            ("<Unmap>", self.hide_command),
            ("<ButtonPress>", self.hide_command),
            ("<FocusOut>", self.hide_command),
        ):
            self.widget.tk.call("bind", listbox, sequence, "+" + script)

    def on_choice_motion(self, y, x_root, y_root):
        if not self.listbox or self._disposed:
            return
        try:
            index = int(self.widget.tk.call(self.listbox, "nearest", int(y)))
            box = self.widget.tk.call(self.listbox, "bbox", index)
        except (tk.TclError, ValueError):
            self.hide_choices()
            return
        if not box or not int(box[1]) <= int(y) < int(box[1]) + int(box[3]):
            self.hide_choices()
            return
        self.schedule_choice(index, x_root, y_root)

    def on_choice_selection(self):
        if not self.listbox or self._disposed:
            return
        try:
            selected = self.widget.tk.call(self.listbox, "curselection")
        except tk.TclError:
            return
        if selected:
            index = int(selected[0])
            box = self.widget.tk.call(self.listbox, "bbox", index)
            if box:
                x = int(self.widget.tk.call("winfo", "rootx", self.listbox))
                y = int(self.widget.tk.call("winfo", "rooty", self.listbox)) + int(box[1])
                self.schedule_choice(index, x, y)

    def schedule_choice(self, index, x_root, y_root):
        values = self.widget.cget("values")
        value = str(values[index])
        if value not in self.choices:
            self.hide_choices()
            return
        self.popup_choice = value
        event = tk.Event()
        event.x_root, event.y_root = int(x_root), int(y_root)
        event.help_region = (int(self.widget.tk.call("winfo", "rootx", self.listbox)),
                             int(self.widget.tk.call("winfo", "width", self.listbox)))
        self.schedule(event)

    def hide_choices(self):
        self.popup_choice = None
        self.hide()

    def dispose(self, event=None):
        self._disposed = True
        super().dispose(event)
        for attr in ("motion_command", "selection_command", "hide_command"):
            cmd = getattr(self, attr, None)
            if cmd:
                try:
                    self.widget.deletecommand(cmd)
                except tk.TclError:
                    tcl_cmds = getattr(self.widget, "_tclCommands", None)
                    if isinstance(tcl_cmds, list) and cmd in tcl_cmds:
                        tcl_cmds.remove(cmd)
                setattr(self, attr, None)
        self.listbox = None
        self.popup_choice = None
        self.summary = ""
        self.previous_postcommand = ""


class HelpNavigationToolbar(NavigationToolbar2Tk):
    # Use the same delayed help as the rest of the app, without duplicate
    # immediate English tooltips from Matplotlib's default toolbar.
    toolitems = tuple((name, None, icon, action)
                      for name, _tip, icon, action in NavigationToolbar2Tk.toolitems)

    def configure_subplots(self, *args):
        if hasattr(self, "subplot_tool"):
            self._subplot_window.deiconify()
            self._subplot_window.lift()
            return
        from matplotlib.widgets import SubplotTool

        # Keep the embedded app and this dialog in one Tcl interpreter.
        # Matplotlib's standalone manager would create another Tk root.
        window = tk.Toplevel(self)
        self._subplot_window = window
        figure = Figure(figsize=(6, 3), dpi=100)
        figure.subplots_adjust(top=0.9)
        canvas = FigureCanvasTkAgg(figure, master=window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.subplot_tool = SubplotTool(self.canvas.figure, figure)
        window.protocol("WM_DELETE_WINDOW", self.close_subplots)
        self.subplot_help(self.subplot_tool, window)
        canvas.draw()
        return self.subplot_tool

    def close_subplots(self):
        if not hasattr(self, "subplot_tool"):
            return
        canvas = self.subplot_tool.figure.canvas
        for attr in ("_idle_draw_id", "_event_loop_id"):
            job = getattr(canvas, attr, None)
            if job is not None:
                canvas.get_tk_widget().after_cancel(job)
                setattr(canvas, attr, None)
        self._subplot_window.destroy()
        canvas._tkphoto = None
        canvas.figure.set_canvas(None)
        del self.subplot_tool
        self._subplot_window = None
