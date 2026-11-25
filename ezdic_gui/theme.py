"""UI typography, DPI scaling, light/dark color palettes, and ttk styling."""

import os
import tkinter as tk
from tkinter import font as tkfont, ttk

UI_FONT_FAMILY = "Microsoft YaHei UI"
UI_FONT_FALLBACKS = (
    UI_FONT_FAMILY, "Microsoft YaHei", "PingFang SC", "Noto Sans SC",
    "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei",
    "Heiti SC", "SimHei", "SimSun", "song ti", "fangsong ti",
)
UI_BASE_FONT_SIZE = 10
UI_TITLE_FONT_SIZE = 12
UI_PRIMARY_FONT_SIZE = 11
UI_STEP_FONT_SIZE = 11
UI_LOG_FONT_SIZE = 10
UI_TREE_ROWHEIGHT = 32
UI_VIEWER_TICK_FONT_SIZE = 9
UI_VIEWER_LEGEND_FONT_SIZE = 9
UI_VIEWER_LABEL_FONT_SIZE = 10
UI_VIEWER_TITLE_FONT_SIZE = 11
UI_MIN_TK_SCALING = 1.20
UI_MAX_TK_SCALING = 2.50
UI_SCALE_ENV_VAR = "EZDIC_UI_SCALE"
RECENT_CONFIG_ENV_VAR = "EZDIC_RECENT_CONFIG"
RECENT_CONFIG_FILENAME = "recent_paths.json"


def choose_ui_font_family(root):
    """Use an installed Chinese font instead of Tk's missing-family fallback."""
    available = {name.casefold(): name for name in tkfont.families(root)}
    for name in UI_FONT_FALLBACKS:
        if name.casefold() in available:
            return available[name.casefold()]
    return tkfont.nametofont("TkDefaultFont", root=root).actual("family")


def enable_windows_dpi_awareness():
    """Make the Windows process DPI aware before Tk creates the first window."""
    if os.name != "nt":
        return
    try:
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
            return
        except Exception:
            pass
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
    except Exception:
        pass


class ThemeMixin:
    """Window geometry, DPI scaling, font metrics, ttk styles, and dark-mode toggle."""

    def configure_initial_window(self):
        self.ui_scaling = self.configure_tk_scaling()
        self.configure_ui_metrics()

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        width = min(650, screen_w - 80)
        height = min(940, screen_h - 100)
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(620, 680)

    def configure_tk_scaling(self):
        try:
            platform_scaling = float(self.root.tk.call("tk", "scaling"))
        except Exception:
            platform_scaling = 1.0

        env_value = os.environ.get(UI_SCALE_ENV_VAR)
        if env_value:
            try:
                target_scaling = float(env_value)
            except ValueError:
                target_scaling = platform_scaling
        else:
            target_scaling = max(platform_scaling, UI_MIN_TK_SCALING)

        target_scaling = min(max(target_scaling, 1.0), UI_MAX_TK_SCALING)
        try:
            self.root.tk.call("tk", "scaling", target_scaling)
            return float(self.root.tk.call("tk", "scaling"))
        except Exception:
            return platform_scaling

    def configure_ui_metrics(self):
        self.ui_font_family = choose_ui_font_family(self.root)
        self.ui_base_font = (self.ui_font_family, UI_BASE_FONT_SIZE)
        self.ui_title_font = (self.ui_font_family, UI_TITLE_FONT_SIZE, "bold")
        self.ui_primary_font = (self.ui_font_family, UI_PRIMARY_FONT_SIZE, "bold")
        self.ui_step_font = (self.ui_font_family, UI_STEP_FONT_SIZE, "bold")
        self.ui_log_font = (self.ui_font_family, UI_LOG_FONT_SIZE)
        self.canvas_group_font = (self.ui_font_family, UI_BASE_FONT_SIZE, "bold")
        self.canvas_current_roi_font = (self.ui_font_family, UI_TITLE_FONT_SIZE, "bold")
        self.viewer_tick_font_size = UI_VIEWER_TICK_FONT_SIZE
        self.viewer_legend_font_size = UI_VIEWER_LEGEND_FONT_SIZE
        self.viewer_label_font_size = UI_VIEWER_LABEL_FONT_SIZE
        self.viewer_title_font_size = UI_VIEWER_TITLE_FONT_SIZE

    def configure_final_window_limits(self):
        self.root.update_idletasks()
        # Requested widths of hidden tables must not dictate the window size.
        scale = max(1.0, self.ui_scaling / (120 / 72))
        control_scale = max(1.0, self.ui_scaling / (96 / 72))
        self.root.minsize(round(620 * control_scale), round(680 * scale))
        self.visual_window.minsize(round(740 * scale), round(580 * scale))
        self.arrange_windows()

    def _on_control_window_configure(self, event=None):
        """Return space to scrollable settings in short or high-DPI windows."""
        if event is not None and event.widget is not self.root:
            return
        height = event.height if event is not None else self.root.winfo_height()
        scale = max(1.0, self.ui_scaling / (120 / 72))
        compact = height < round(800 * scale)
        if compact == self._compact_control_layout:
            return
        self._compact_control_layout = compact
        self.main_frame.configure(padding=(16, 8 if compact else 12, 16, 8 if compact else 12))
        self.app_header.grid_configure(pady=(0, 8 if compact else 12))
        self.project_frame.configure(padding=(12, 6 if compact else 8))
        self.control_workspace.grid_configure(pady=(6 if compact else 12, 0))
        self.workflow_guide_frame.configure(padding=(12, 6 if compact else 10))
        self.run_frame.configure(padding=(12, 4 if compact else 6))
        self.run_frame.grid_configure(pady=(6 if compact else 10, 0))
        self._on_control_tab_changed()

    def configure_ui_style(self):
        self.style = ttk.Style(self.root)
        try:
            if "clam" in self.style.theme_names() and self.style.theme_use() != "clam":
                self.style.theme_use("clam")
        except tk.TclError:
            pass

        self._apply_color_palette()

        base_font = self.ui_base_font
        primary_font = self.ui_primary_font
        step_font = self.ui_step_font

        for font_name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            try:
                tkfont.nametofont(font_name).configure(family=self.ui_font_family, size=UI_BASE_FONT_SIZE)
            except tk.TclError:
                pass

        self.root.configure(background=self.ui_bg)
        self.style.configure(".", font=base_font, background=self.card_bg, foreground=self.text_color,
                             bordercolor=self.border_color, lightcolor=self.card_bg, darkcolor=self.border_color)
        self.style.configure("App.TFrame", background=self.ui_bg)
        self.style.configure("Panel.TFrame", background=self.panel_bg)
        self.style.configure("Card.TFrame", background=self.card_bg)
        self.style.configure("TLabel", background=self.card_bg, font=base_font, foreground=self.text_color)
        self.style.configure("Hint.TLabel", background=self.card_bg, font=base_font, foreground=self.muted_color)
        self.style.configure("Key.TLabel", background=self.card_bg, font=step_font, foreground=self.key_color)
        self.style.configure("Warning.TLabel", background=self.card_bg, font=step_font, foreground=self.warning_color)
        self.style.configure("StepTitle.TLabel", background=self.card_bg, font=step_font, foreground=self.text_color)
        self.style.configure("TLabelframe", background=self.card_bg, bordercolor=self.border_color, relief="solid")
        self.style.configure("TLabelframe.Label", background=self.card_bg, foreground=self.text_color, font=step_font)
        self.style.configure("Brand.TLabel", background=self.ui_bg, foreground=self.text_color,
                             font=(self.ui_font_family, 18, "bold"))
        self.style.configure("AppHint.TLabel", background=self.ui_bg, foreground=self.muted_color, font=base_font)
        help_bg, help_fg, help_border = (
            ("#263750", "#f1f5f9", "#6386b2") if self.dark_mode.get() else ("#f0f6ff", "#18263c", "#a8bedc")
        )
        self.style.configure("Help.TLabel", background=help_bg, foreground=help_fg, bordercolor=help_border)
        self.style.configure("HelpTitle.TLabel", font=step_font)
        self.style.configure("HelpKey.TLabel", foreground=self.muted_color, font=(self.ui_font_family, 9))
        self.style.configure("NextStep.TLabel", background=self.panel_bg, foreground=self.key_color,
                             font=base_font, padding=(10, 7))
        self.style.configure("Badge.TLabel", background=self.panel_bg, foreground=self.key_color,
                             font=base_font, padding=(10, 4))
        self.style.configure("Ready.TLabel", background=self.success_bg, foreground=self.success_color,
                             font=step_font, padding=(10, 4))
        self.style.configure("Error.TLabel", background=self.card_bg, foreground=self.warning_color,
                             font=step_font, padding=(10, 4))
        self.style.configure("TButton", font=base_font, padding=(10, 6), background=self.card_bg, width=0,
                             foreground=self.text_color, borderwidth=1, focusthickness=2,
                             focuscolor=self.primary_color)
        self.style.map("TButton", background=[("disabled", self.panel_bg), ("pressed", self.border_color),
                                              ("active", self.hover_bg)],
                       foreground=[("disabled", self.muted_color)],
                       bordercolor=[("focus", self.primary_color)])
        self.style.configure("Compact.TButton", font=base_font, padding=(8, 4))
        self.style.configure("Primary.TButton", font=primary_font, padding=(14, 8), foreground="#ffffff", background=self.primary_color)
        self.style.configure("Secondary.TButton", font=base_font, padding=(9, 5), foreground=self.primary_text, background=self.card_bg)
        self.style.configure("Danger.TButton", font=base_font, padding=(8, 5), foreground=self.warning_color, background=self.card_bg)
        self.style.map(
            "Primary.TButton",
            foreground=[("disabled", self.muted_color), ("active", "#ffffff")],
            background=[("disabled", self.panel_bg), ("pressed", self.primary_active), ("active", self.primary_active)],
        )
        self.style.map(
            "Danger.TButton",
            foreground=[("disabled", self.muted_color), ("active", self.warning_color)],
            background=[("active", self.hover_bg), ("pressed", self.panel_bg)],
        )
        for style_name in ("TEntry", "TCombobox"):
            self.style.configure(style_name, font=base_font, padding=(6, 4), fieldbackground=self.card_bg,
                                 foreground=self.text_color, insertcolor=self.text_color,
                                 selectbackground=self.primary_color, selectforeground="#ffffff")
            self.style.map(style_name, fieldbackground=[("disabled", self.panel_bg), ("readonly", self.card_bg)],
                           foreground=[("disabled", self.muted_color), ("readonly", self.text_color)],
                           bordercolor=[("focus", self.primary_color)])
        self.root.option_add("*TCombobox*Listbox.background", self.card_bg)
        self.root.option_add("*TCombobox*Listbox.foreground", self.text_color)
        self.root.option_add("*TCombobox*Listbox.selectBackground", self.primary_color)
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        for style_name in ("TCheckbutton", "TRadiobutton"):
            self.style.configure(style_name, background=self.card_bg, foreground=self.text_color, padding=(2, 3))
            self.style.map(style_name, background=[("active", self.card_bg)],
                           foreground=[("disabled", self.muted_color)],
                           indicatorbackground=[("selected", self.primary_color), ("!selected", self.card_bg)])
        self.style.configure("Mode.TRadiobutton", padding=(10, 7), font=step_font, relief="flat")
        self.style.layout("Mode.TRadiobutton", [("Radiobutton.padding", {"sticky": "nswe", "children": [
            ("Radiobutton.focus", {"sticky": "nswe", "children": [("Radiobutton.label", {"sticky": "nswe"})]})
        ]})])
        self.style.map("Mode.TRadiobutton", background=[("selected", self.primary_color), ("active", self.hover_bg)],
                       foreground=[("disabled", self.muted_color), ("selected", "#ffffff")])
        self.style.configure("Treeview", font=base_font, rowheight=UI_TREE_ROWHEIGHT, background=self.card_bg,
                             fieldbackground=self.card_bg, foreground=self.text_color)
        self.style.map("Treeview", background=[("selected", self.primary_color)], foreground=[("selected", "#ffffff")])
        self.style.configure("Treeview.Heading", font=step_font, background=self.panel_bg, foreground=self.text_color,
                             padding=(6, 5), relief="flat")
        self.style.map("Treeview.Heading", background=[("active", self.hover_bg)])
        self.style.configure("TNotebook", background=self.ui_bg, borderwidth=0, tabmargins=(0, 0, 0, 6))
        self.style.configure("TNotebook.Tab", background=self.ui_bg, foreground=self.muted_color,
                             padding=(18, 8), font=step_font)
        self.style.map("TNotebook.Tab", background=[("selected", self.card_bg), ("active", self.hover_bg)],
                       foreground=[("selected", self.primary_text)])
        self.style.configure("Detail.TNotebook.Tab", padding=(10, 6), font=base_font)
        self.style.configure("Horizontal.TProgressbar", background=self.primary_color, troughcolor=self.panel_bg,
                             borderwidth=0, thickness=5)
        self.style.configure("TScrollbar", background=self.panel_bg, troughcolor=self.ui_bg,
                             arrowcolor=self.muted_color, borderwidth=0, arrowsize=12)
        self.style.map("TScrollbar", background=[("active", self.border_color)])

    def _apply_color_palette(self):
        """集中管理浅色/暗色配色，便于后续完整暗色模式切换。"""
        if self.dark_mode.get():
            # 暗色主题（实验室长时间使用更护眼）
            self.ui_bg = "#101827"
            self.panel_bg = "#243247"
            self.card_bg = "#192538"
            self.border_color = "#3b4c64"
            self.hover_bg = "#2c3d56"
            self.primary_color = "#2563eb"
            self.primary_active = "#1d4ed8"
            self.warning_color = "#f87171"
            self.key_color = "#e0f2fe"
            self.text_color = "#e2e8f0"
            self.muted_color = "#b0bfd2"
            self.success_color = "#86efac"
            self.success_bg = "#163b35"
        else:
            # 浅色主题（默认）
            self.ui_bg = "#f1f4f8"
            self.panel_bg = "#edf2f8"
            self.card_bg = "#ffffff"
            self.border_color = "#d5deea"
            self.hover_bg = "#e8eff9"
            self.primary_color = "#1d5dcc"
            self.primary_active = "#164ba8"
            self.warning_color = "#b91c1c"
            self.key_color = "#0f3f6e"
            self.text_color = "#0f172a"
            self.muted_color = "#475569"
            self.success_color = "#166534"
            self.success_bg = "#e5f4eb"
        self.primary_text = "#a8ccff" if self.dark_mode.get() else self.primary_color

    def toggle_dark_mode(self):
        """切换暗色/浅色模式，并尝试刷新结果预览图的配色。"""
        is_dark = self.dark_mode.get()
        self.dark_mode.set(not is_dark)
        self._apply_color_palette()

        self.configure_ui_style()
        self.root.configure(background=self.ui_bg)
        self.visual_window.configure(background=self.ui_bg)

        for attr in ("log_text", "preflight_summary_label"):
            widget = getattr(self, attr, None)
            if isinstance(widget, tk.Text):
                widget.configure(bg=self.panel_bg, fg=self.text_color, insertbackground=self.text_color)
        if hasattr(self, "preflight_summary_label"):
            self._update_preflight_text()
        if hasattr(self, "controls_canvas"):
            self.controls_canvas.configure(bg=self.card_bg, highlightbackground=self.border_color)
        if hasattr(self, "campaign_result_canvas"):
            self.campaign_result_canvas.configure(background=self.card_bg)
        if hasattr(self, "agent_page") and hasattr(self.agent_page, "apply_theme"):
            self.agent_page.apply_theme()
        if hasattr(self, "campaign_page"):
            if hasattr(self.campaign_page, "apply_theme"):
                self.campaign_page.apply_theme()
            elif hasattr(self.campaign_page, "_draw_result_curve"):
                try:
                    self.campaign_page._draw_result_curve()
                except Exception as exc:
                    self.log(f"刷新批次曲线主题失败：{exc}")
        self._draw_image_empty_state()

        # 刷新结果预览器（matplotlib）
        if (hasattr(self, "results_df") and self.results_df is not None) or getattr(self, "dic_last_field", None) is not None:
            try:
                self._refresh_viewer_for_dark_mode()
            except Exception as exc:
                self.log(f"刷新暗色主题预览图失败：{exc}")

        self.root.update_idletasks()
        if hasattr(self, "dark_mode_btn"):
            self.dark_mode_btn.configure(text="浅色模式" if self.dark_mode.get() else "暗色模式")
        self.log("已切换显示模式。" if self.dark_mode.get() else "已恢复默认显示模式。")

    def _refresh_viewer_for_dark_mode(self):
        """当暗色模式切换时，重新绘制当前结果预览图以匹配新主题。"""
        if getattr(self, "_viewer_kind", "extensometer") == "fullfield" and getattr(self, "dic_last_field", None) is not None:
            self._rebuild_field_viewer_plot()
            return
        if not hasattr(self, "results_df") or self.results_df is None:
            return
        if hasattr(self, "viewer_figure") and self.viewer_figure is not None:
            try:
                self._rebuild_viewer_plot()
            except Exception as exc:
                self.log(f"重绘结果预览图失败：{exc}")
