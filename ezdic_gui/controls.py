"""Display labels and contextual help shared by the desktop workflows."""

import tkinter as tk
from tkinter import ttk


def localized_combobox(parent, variable, labels, **options):
    """Show translated labels while keeping stable values in the project variable."""
    display = tk.StringVar(parent)
    choices = tuple(labels.values())
    box = ttk.Combobox(parent, textvariable=display, values=choices, state="readonly", **options)
    values = tuple(labels)

    def refresh(*_args):
        display.set(labels.get(variable.get(), ""))

    def select(_event):
        index = box.current()
        if index >= 0:
            displayed = tuple(box.cget("values"))
            selected = values[index] if displayed == choices else next((key for key in values if labels[key] == displayed[index]), None)
            if selected is not None and variable.get() != selected:
                variable.set(selected)

    refresh()
    trace = variable.trace_add("write", refresh)

    def remove_trace(_event):
        if any(name == trace for _modes, name in variable.trace_info()):
            variable.trace_remove("write", trace)

    box.bind("<<ComboboxSelected>>", select)
    box.bind("<Destroy>", remove_trace)
    return box


def notebook_help(notebook, descriptions, event=None):
    try:
        target = "current" if event is None or getattr(event, "keysym", "") == "F1" else f"@{event.x},{event.y}"
        index = notebook.index(target)
    except tk.TclError:
        return ""
    return descriptions[index] if 0 <= index < len(descriptions) else ""


def focus_canvas_child(widget, canvas, content):
    """Scroll a canvas-embedded form to the focused child without moving focus."""
    if not widget.winfo_ismapped() or not canvas.winfo_ismapped():
        return
    bounds = canvas.bbox("all")
    if not bounds or bounds[3] <= bounds[1]:
        return
    top = canvas.canvasy(0)
    height = canvas.winfo_height()
    start = widget.winfo_rooty() - content.winfo_rooty() + bounds[1]
    end = start + widget.winfo_height()
    if start < top or widget.winfo_height() > height:
        target = start - 4
    elif end > top + height:
        target = end - height + 4
    else:
        return
    canvas.yview_moveto(max(0, target - bounds[1]) / (bounds[3] - bounds[1]))


def wrap_label_on_resize(label, inset=3):
    def wrap(event):
        width = max(1, event.width - inset)
        if int(label.cget("wraplength") or 0) != width:
            label.configure(wraplength=width)

    label.bind("<Configure>", wrap, add="+")


class ScrollableForm(ttk.Frame):
    """Put a form's fields in a vertically scrolling area with keyboard access."""

    def __init__(self, parent, *, fill_height=False, **options):
        super().__init__(parent, **options)
        self._fill_height = fill_height
        self._focus_after = None
        background = ttk.Style(self).lookup("TFrame", "background")
        self.canvas = tk.Canvas(self, background=background, highlightthickness=0, width=1, height=1, takefocus=True)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.configure(yscrollcommand=self._set_scrollbar if fill_height else self.scrollbar.set)
        self.body = ttk.Frame(self.canvas)
        self._slot = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind("<Configure>", self._update_extent)
        self.canvas.bind("<Configure>", self._resize_content)
        self.bind("<Destroy>", self._cancel_focus_reveal, add="+")
        for key, direction in (("<Prior>", -1), ("<Next>", 1)):
            self.canvas.bind(key, lambda _event, pages=direction: self.canvas.yview_scroll(pages, "pages"))

    def _resize_content(self, event):
        self.canvas.itemconfigure(self._slot, width=event.width)
        self._update_extent()
        if self._fill_height:
            self._defer_focus_reveal()

    def _update_extent(self, _event=None):
        if self._fill_height:
            height = max(self.body.winfo_reqheight(), self.canvas.winfo_height())
            if int(self.canvas.itemcget(self._slot, "height")) != height:
                self.canvas.itemconfigure(self._slot, height=height)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _set_scrollbar(self, first, last):
        self.scrollbar.set(first, last)
        needs_scroll = float(first) > 0 or float(last) < 1
        if needs_scroll and not self.scrollbar.winfo_manager():
            self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y, before=self.canvas)
        elif not needs_scroll and self.scrollbar.winfo_manager():
            self.scrollbar.pack_forget()

    def _defer_focus_reveal(self, _event=None):
        if self._focus_after is None:
            self._focus_after = self.after_idle(self._reveal_focused_control)

    def _reveal_focused_control(self):
        self._focus_after = None
        child = self.focus_get()
        if child is not None and str(child).startswith(f"{self.body}."):
            focus_canvas_child(child, self.canvas, self.body)

    def _cancel_focus_reveal(self, event):
        if event.widget is self and self._focus_after is not None:
            self.after_cancel(self._focus_after)
            self._focus_after = None

    def _wheel(self, event):
        if event.num in (4, 5):
            units = -1 if event.num == 4 else 1
        elif not event.delta:
            return None
        else:
            units = -int(event.delta / 120) or (-1 if event.delta > 0 else 1)
        self.canvas.yview_scroll(units, "units")
        return "break"

    def bind_controls(self, *, keep_wheel=()):
        def bind(widget):
            if not getattr(widget, "_scrollable_form_bound", False):
                if not isinstance(widget, keep_wheel):
                    for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                        widget.bind(sequence, self._wheel, add="+")
                if self._fill_height and widget not in (self.canvas, self.body, self.scrollbar):
                    widget.bind("<Configure>", self._update_extent, add="+")
                reveal = self._defer_focus_reveal if self._fill_height else (
                    lambda _event: focus_canvas_child(widget, self.canvas, self.body)
                )
                widget.bind("<FocusIn>", reveal, add="+")
                widget._scrollable_form_bound = True
            for child in widget.winfo_children():
                bind(child)

        bind(self.canvas)
        bind(self.scrollbar)
