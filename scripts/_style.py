"""Shared visual system for the motor-cortex (M1) figure set.

One place for the palette, the matplotlib chrome, and the small helpers every figure
reuses, so the figures read as one system. The palette is the data-viz skill's validated
categorical default (worst adjacent CVD deltaE 37.7, well clear of the >=12 target);
colour is assigned by the *role a quantity plays*, in a fixed order, never cycled.

Roles (fixed hues, used identically across every figure):

* ``DIRECT``    the direct neuromechanical (beta-carrier) term -- the specific signal.
* ``OSMOTIC``   the osmotic / ECS envelope mechanism.
* ``VASCULAR``  the neurovascular / CBV (fUS) envelope mechanism.
* ``FLOOR``     the through-skull detection floor (a limit, not a signal): status-red.
* ``ISO`` / ``DIR``   the isotropic (monopole) vs directional (deviatoric) split of the
  direct term.
* ``PEDESTAL``  the incoherent 1/sqrt(N) noise pedestal.

Text always wears an ink token (primary/secondary/muted), never a series colour.
"""

from __future__ import annotations

# --- The palette (data-viz skill default, light surface) ------------------------------
SURFACE = "#fcfcfb"      # chart surface
PLANE = "#f4f3f0"        # panel plane behind the surface
INK = "#0b0b0b"          # primary ink
INK_2 = "#52514e"        # secondary ink
MUTED = "#898781"        # axis labels / de-emphasis
GRID = "#e1e0d9"         # hairline gridline
AXIS = "#c3c2b7"         # baseline / axis spine

# Categorical slots, assigned by role (fixed order 1,2,3,6,5,8 of the skill palette).
BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
RED = "#e34948"
VIOLET = "#4a3aa7"
ORANGE = "#eb6834"
GREEN = "#008300"

# Role -> hue. Change a role here and every figure follows.
DIRECT = BLUE            # direct neuromechanical (the signal)
ISO = BLUE              # isotropic monopole share of the direct term
DIR = VIOLET            # directional (deviatoric) share of the direct term
OSMOTIC = YELLOW        # osmotic / ECS envelope
VASCULAR = AQUA         # neurovascular / CBV (fUS) envelope
PEDESTAL = MUTED        # incoherent noise pedestal
FLOOR = RED             # detection floor (a limit): status colour, always labelled
GOOD = GREEN            # a passing / detectable state

# The M1 accent used for titles / operating-point stars.
ACCENT = "#1f3b73"


def apply_style() -> None:
    """Set the shared matplotlib rcParams. Call once at the top of each figure build."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.titlelocation": "left",
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlepad": 8,
        "axes.labelcolor": INK_2,
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "text.color": INK,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        "font.size": 9.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "figure.dpi": 110,
    })


def suptitle(fig, title: str, subtitle: str | None = None) -> None:
    """A left-aligned bold title with an optional muted subtitle line beneath it."""
    fig.text(0.008, 0.982, title, ha="left", va="top",
             fontsize=15, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.008, 0.945, subtitle, ha="left", va="top",
                 fontsize=10, color=INK_2)


def nm(value_m: float, places: int = 3) -> str:
    """Format a metre displacement as a nanometre string (e.g. '0.270 nm')."""
    return f"{value_m * 1e9:.{places}f} nm"


def db(value: float, places: int = 1) -> str:
    """Signed dB string (e.g. '-59.8 dB')."""
    return f"{value:+.{places}f} dB"


def tidy(ax, *, xgrid: bool = True, ygrid: bool = True) -> None:
    """Selective grid: keep it recessive and only where it aids reading."""
    ax.grid(False)
    if xgrid:
        ax.grid(axis="x", color=GRID, lw=0.8)
    if ygrid:
        ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(length=0)


def footer(fig, text: str) -> None:
    """A muted provenance line at the very bottom -- every figure is model-derived."""
    fig.text(0.008, 0.008, text, ha="left", va="bottom", fontsize=7.5, color=MUTED)


def caption(ax, text: str, *, loc: str = "lower right", color: str | None = None) -> None:
    """A small in-axes explanatory note (the 'read this' line)."""
    ha = "right" if "right" in loc else "left"
    va = "bottom" if "lower" in loc else "top"
    x = 0.98 if ha == "right" else 0.02
    y = 0.03 if va == "bottom" else 0.97
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=8, color=color or INK_2, wrap=True)


def save(fig, output, dpi: int = 150) -> None:
    """Save with the shared surface and a tight box; print the path."""
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {output.resolve() if hasattr(output, 'resolve') else output}")
