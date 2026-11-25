"""Layout helpers for the Capture workbench."""


def wrap_toolbar(parent, leading, commands, *, gap=8, collapse_heading=False):
    """Keep command widths when the heading and commands cannot share a row."""
    parent.columnconfigure(0, weight=1)
    leading.grid(row=0, column=0, sticky="w")
    commands.grid(row=0, column=1, sticky="e", padx=(gap, 0))
    stacked = False

    def adjust(_event=None):
        nonlocal stacked
        width = parent.winfo_width()
        if width <= 1:
            return
        needs_stack = width < leading.winfo_reqwidth() + commands.winfo_reqwidth() + gap
        if needs_stack == stacked:
            return
        stacked = needs_stack
        if stacked:
            if collapse_heading:
                leading.grid_remove()
            commands.grid(row=0 if collapse_heading else 1, column=0, columnspan=2, sticky="e", padx=0,
                          pady=0 if collapse_heading else (3, 0))
        else:
            leading.grid()
            commands.grid(row=0, column=1, columnspan=1, sticky="e", padx=(gap, 0), pady=0)

    for widget in (parent, leading, commands):
        widget.bind("<Configure>", adjust, add="+")


def fit_line(text, font, width):
    """Use an ellipsis if the available width cannot display the full line."""
    if font.measure(text) <= width:
        return text
    mark = "…"
    if font.measure(mark) > width:
        return ""
    low, high = 0, len(text)
    while high - low > 1:
        middle = (low + high) // 2
        if font.measure(text[:middle] + mark) <= width:
            low = middle
        else:
            high = middle
    return text[:low].rstrip() + mark
