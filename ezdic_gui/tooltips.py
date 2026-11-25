"""Hover/F1 tooltips, combobox choice help, and Matplotlib navigation toolbar."""

import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib import rcParams
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
    def __init__(self, widget, text, wraplength=360, delay_ms=380, *, title=None):
        self.widget = widget
        self.owner = widget.winfo_toplevel()
        self.text = text
        self.title = title
        self.wraplength = wraplength
        self.delay_ms = delay_ms
        self.after_id = None
        self.refresh_id = None
        self.tip_window = None
        self.event = None
        self.pending_text = None
        self.shown_text = None
        self._keyboard_open = False

        self.widget._tooltip_text = text if isinstance(text, str) else ""
        self.widget._tooltip = self
        self.widget.bind("<Enter>", self.schedule, add="+")
        self.widget.bind("<Motion>", self.schedule, add="+")
        self.widget.bind("<Leave>", self.leave, add="+")
        self.widget.bind("<ButtonPress>", self.hide, add="+")
        self.widget.bind("<MouseWheel>", self.hide, add="+")
        self.widget.bind("<KeyPress>", self.hide, add="+")
        self.widget.bind("<FocusOut>", self.hide, add="+")
        self.widget.bind("<Unmap>", self.hide, add="+")
        self.widget.bind("<Destroy>", self.dispose, add="+")
        self.widget.bind("<F1>", self.show_help, add="+")
        self.widget.bind("<Shift-F1>", self._pin_hovered_help, add="+")
        self.widget.bind("<Escape>", self.dismiss, add="+")
        if not getattr(self.owner, "_pointer_help_bound", False):
            self.owner.bind("<F1>", self._help_for_window, add="+")
            self.owner.bind("<Shift-F1>", self._pin_hovered_help, add="+")
            self.owner.bind("<ButtonPress>", self._clear_pinned_help, add="+")
            self.owner._pointer_help_bound = True

    def resolve_text(self, event=None):
        # The registered _tooltip_text describes the whole control. Hovering a
        # table header or candidate must not replace that general help; the
        # active context is held separately in pending_text / shown_text.
        return self.text(event) if callable(self.text) else self.text

    def schedule(self, event=None):
        active = getattr(self.owner, "_active_tooltip", None)
        if self._keyboard_open or (active is not None and active._keyboard_open):
            return
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
        self._keyboard_open = True
        self.event = event
        self.show()
        if self.tip_window is not None:
            return "break"
        self._keyboard_open = False

    def leave(self, event=None):
        if not self._keyboard_open:
            self.hide()

    def dismiss(self, event=None):
        visible = self.tip_window is not None
        self.hide()
        if visible:
            return "break"

    @staticmethod
    def dismiss_window(window):
        tip = getattr(window, "_active_tooltip", None)
        return tip.dismiss() if tip else None

    @staticmethod
    def _clear_pinned_help(event):
        active = getattr(event.widget.winfo_toplevel(), "_active_tooltip", None)
        if active is not None and active._keyboard_open:
            active.hide()

    @staticmethod
    def _pin_hovered_help(event):
        active = getattr(event.widget.winfo_toplevel(), "_active_tooltip", None)
        if active is not None and not active._keyboard_open:
            return active.show_help(active.event)
        return ToolTip._help_for_window(event)

    @staticmethod
    def _help_for_window(key_event):
        window = key_event.widget.winfo_toplevel()
        x, y = window.winfo_pointerxy()
        try:
            widget = window.winfo_containing(x, y)
        except (KeyError, tk.TclError):
            return None
        while widget is not None and widget.winfo_toplevel() == window:
            tip = getattr(widget, "_tooltip", None)
            if tip is not None:
                event = tk.Event()
                event.type = tk.EventType.Motion
                event.x_root, event.y_root = x, y
                event.x, event.y = x - widget.winfo_rootx(), y - widget.winfo_rooty()
                return tip.show_help(event)
            widget = widget.master
        return None

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

            previous = getattr(self.owner, "_active_tooltip", None)
            if previous is not None and previous._keyboard_open and not self._keyboard_open:
                return
            if previous is not None and previous is not self:
                previous.hide()
            self.tip_window = tk.Toplevel(self.widget)
            self.tip_window.withdraw()
            self.tip_window.wm_overrideredirect(True)
            self.tip_window.wm_attributes("-topmost", True)
            self._anchor = (anchor_x, anchor_y, screen_x, screen_y, screen_w, screen_h)
            width = round(self.wraplength * self.widget.winfo_fpixels("1i") / 96)
            self._build_card(text, min(width, screen_w - 40))
            self._place()
            self.tip_window.deiconify()
            self.shown_text = text
            self.owner._active_tooltip = self
            if callable(self.text) or hasattr(self.widget, "instate"):
                self.refresh_id = self.owner.after(300, self._refresh_text)
        except tk.TclError:
            self.hide()

    def _place(self):
        anchor_x, anchor_y, screen_x, screen_y, screen_w, screen_h = self._anchor
        window = self.tip_window
        window.update_idletasks()
        width, height = window.winfo_reqwidth(), window.winfo_reqheight()
        region = getattr(self.event, "help_region", None)
        if region is not None:
            region_x, region_width = region
            x, y = region_x + region_width + 8, anchor_y
            if x + width > screen_x + screen_w - 8:
                x = region_x - width - 8
        else:
            x, y = anchor_x + 16, anchor_y + 20
        x = max(screen_x + 8, min(x, screen_x + screen_w - width - 8))
        if y + height > screen_y + screen_h - 8:
            y = anchor_y - height - 12
        y = max(screen_y + 8, min(y, screen_y + screen_h - height - 8))
        window.wm_geometry(f"+{x}+{y}")

    def _refresh_text(self):
        self.refresh_id = None
        try:
            text = self.resolve_text(self.event)
            if not text:
                self.hide()
                return
            heading = self._heading_text()
            rearrange = heading != self._heading.cget("text")
            if rearrange:
                self._update_heading(heading)
            if text != self.shown_text:
                self._body.configure(text=text)
                self.shown_text = text
                rearrange = True
            if rearrange:
                self._place()
            self.refresh_id = self.owner.after(300, self._refresh_text)
        except tk.TclError:
            self.hide()

    def _heading_text(self):
        title = self.title
        if title is None and "text" in self.widget.keys() and not (
            "textvariable" in self.widget.keys() and self.widget.cget("textvariable")
        ):
            caption = str(self.widget.cget("text"))
            title = caption if 1 < len(caption) <= 20 and "\n" not in caption else ""
        if hasattr(self.widget, "instate") and self.widget.instate(["disabled"]):
            return f"{title} · 当前不可用" if title else "当前不可用"
        return title or ""

    def _update_heading(self, heading):
        self._heading.configure(text=heading)
        if heading:
            self._heading.pack(before=self._body, anchor="w", pady=(0, 3))
        else:
            self._heading.pack_forget()

    def _build_card(self, text, wraplength):
        style = ttk.Style(self.widget)
        background = style.lookup("Help.TLabel", "background") or "#f0f6ff"
        foreground = style.lookup("Help.TLabel", "foreground") or "#18263c"
        border = style.lookup("Help.TLabel", "bordercolor") or "#a8bedc"
        frame = tk.Frame(self.tip_window, background=background,
                         highlightbackground=border, highlightthickness=1, padx=11, pady=8)
        frame.pack(fill=tk.BOTH, expand=True)
        self._heading = ttk.Label(frame, style="HelpTitle.TLabel", background=background,
                                  foreground=foreground, wraplength=wraplength)
        self._body = ttk.Label(frame, text=text, style="Help.TLabel", background=background, foreground=foreground,
                               justify=tk.LEFT, wraplength=wraplength)
        self._body.pack(anchor="w")
        self._update_heading(self._heading_text())
        controls = "Esc 收起" if self._keyboard_open or getattr(self.event, "help_region", None) else "Shift+F1 固定 · Esc 收起"
        ttk.Label(frame, text=controls, style="HelpKey.TLabel", background=background).pack(anchor="w", pady=(5, 0))

    def hide(self, event=None):
        self.unschedule()
        if self.refresh_id is not None:
            try:
                self.owner.after_cancel(self.refresh_id)
            except tk.TclError:
                pass
            self.refresh_id = None
        if getattr(self.owner, "_active_tooltip", None) is self:
            self.owner._active_tooltip = None
        self._keyboard_open = False
        self.pending_text = None
        self.shown_text = None
        if self.tip_window is not None:
            try:
                self.tip_window.destroy()
            except tk.TclError:
                pass
            self.tip_window = None
            self._body = None
            self._heading = None

    def dispose(self, event=None):
        self.hide()
        # Do not retain destroyed controls or their app through event objects
        # and dynamic text closures until a later worker-thread collection.
        self.event = None
        self.text = ""
        self.widget._tooltip_text = ""


class ChoiceToolTip(ToolTip):
    """Explain both the selected value and each native combobox list item."""

    def __init__(self, widget, text, choices, *, title=None):
        self.summary = text
        self.popup_choice = None
        self.listbox = None
        self._disposed = False
        super().__init__(widget, self.choice_text, title=title)
        self.set_choices(choices)
        self.previous_postcommand = widget.cget("postcommand")
        widget.configure(postcommand=self.prepare_choices)
        self.motion_command = widget.register(self.on_choice_motion)
        self.selection_command = widget.register(self.on_choice_selection)
        self.help_command = widget.register(self.show_choice_help)
        self.dismiss_command = widget.register(self.dismiss)
        self.hide_command = widget.register(self.hide_choices)

    def set_choices(self, choices):
        self.hide_choices()
        self.choices = {str(value): explanation for value, explanation in choices.items()}
        self.widget._tooltip_choices = self.choices

    def choice_text(self, event=None):
        if self.popup_choice in self.choices:
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
            *((f"<KeyRelease-{key}>", self.selection_command) for key in ("Up", "Down", "Home", "End", "Prior", "Next")),
            ("<F1>", f"{self.help_command}; break"),
            ("<Escape>", f'if {{[{self.dismiss_command}] eq "break"}} {{break}}'),
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

    def show_choice_help(self):
        self.on_choice_selection()
        if self.popup_choice in self.choices:
            self.unschedule()
            self.show()

    def schedule_choice(self, index, x_root, y_root):
        values = self.widget.cget("values")
        value = str(values[index]) if 0 <= index < len(values) else None
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
        for attr in ("motion_command", "selection_command", "help_command", "dismiss_command", "hide_command"):
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

    def set_message(self, message):
        super().set_message({"pan/zoom": "平移与缩放", "zoom rect": "框选放大"}.get(message, message))

    def save_figure(self, *args):
        types = {f"{extension.upper()} 文件": extension for extension in self.canvas.get_supported_filetypes()}
        selected = tk.StringVar(self, f"{self.canvas.get_default_filetype().upper()} 文件")
        directory = os.path.expanduser(rcParams["savefig.directory"])
        filename = filedialog.asksaveasfilename(
            parent=self.winfo_toplevel(), title="保存结果图", initialdir=directory,
            initialfile=Path(self.canvas.get_default_filename()).stem,
            filetypes=[(label, f"*.{ext}") for label, ext in types.items()], typevariable=selected,
        )
        if not filename:
            return None
        if directory:
            rcParams["savefig.directory"] = str(Path(filename).parent)
        extension = None if Path(filename).suffix else types.get(selected.get())
        try:
            self.canvas.figure.savefig(filename, format=extension)
        except Exception as exc:
            messagebox.showerror("无法保存图表", str(exc), parent=self.winfo_toplevel())
            return None
        return filename

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
