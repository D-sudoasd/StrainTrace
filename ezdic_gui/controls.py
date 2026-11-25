"""Display labels and contextual help shared by the desktop workflows."""

import tkinter as tk
from tkinter import ttk


def localized_combobox(parent, variable, labels, **options):
    """Show translated labels while keeping stable values in the project variable."""
    display = tk.StringVar(parent)
    box = ttk.Combobox(parent, textvariable=display, values=tuple(labels.values()), state="readonly", **options)
    values = tuple(labels)

    def refresh(*_args):
        display.set(labels.get(variable.get(), ""))

    def select(_event):
        index = box.current()
        if index >= 0 and variable.get() != values[index]:
            variable.set(values[index])

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
