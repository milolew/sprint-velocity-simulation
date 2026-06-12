"""Streamlit UI for the sprint simulation.

A prettier front-end for ``model.TeamModel``: the same parameters and the same
five charts as the Solara app (``app.py``), plus an animated view of the team.

Each developer is a node placed around a circle. A node has two colour
channels:

* the **outer ring** encodes *where* the dev works today — office vs home;
* the **inner fill** encodes *what* the dev is doing — idle / working / blocked
  / helping.

When one dev is helping another, an **edge** is drawn between the requester and
the helper (solid = co-located/sync session, dashed = remote/async session).

The model is run once to completion, every tick is snapshotted, and the result
is replayed via Plotly's native play/pause controls and time slider — no
Streamlit reruns while scrubbing.

Run with:
    streamlit run app_streamlit.py
"""

import math

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

from model import TeamModel

# --------------------------------------------------------------------------- #
# Visual encoding                                                             #
# --------------------------------------------------------------------------- #

# Inner fill = activity state. Colours match the StatePlot series in app.py.
STATE_FILL = {
    "idle":    "#7f7f7f",  # tab:gray
    "working": "#2ca02c",  # tab:green
    "blocked": "#d62728",  # tab:red
    "helping": "#ff7f0e",  # tab:orange
}

# Outer ring = location (the two "colour states").
LOCATION_RING = {
    "office": "#1f77b4",  # blue
    "home":   "#9467bd",  # purple
}

# Help-session edges, coloured by channel.
EDGE_STYLE = {
    "sync":  dict(color="#2ca02c", width=3, dash="solid"),   # co-located, fast
    "async": dict(color="#9467bd", width=2, dash="dash"),    # remote, slow
}


# --------------------------------------------------------------------------- #
# Parameters — mirror the sliders in app.py (label, default, min, max, step).  #
# A trailing ``int`` flag marks integer-valued knobs.                          #
# --------------------------------------------------------------------------- #

PARAM_GROUPS = {
    "Headline": [
        ("remote_share", "Share of remote work", 0.5, 0.0, 1.0, 0.05, False),
    ],
    "Team heterogeneity": [
        ("mean_solo_skill", "Mean solo skill", 0.5, 0.0, 1.0, 0.05, False),
        ("solo_skill_spread", "Solo-skill spread", 0.2, 0.0, 0.5, 0.05, False),
    ],
    "Process": [
        ("block_prob", "Per-tick blocker probability", 0.02, 0.0, 0.10, 0.005, False),
        ("sync_help_mean", "Sync help mean (ticks)", 1, 0, 5, 1, True),
        ("async_help_mean", "Async help mean (ticks)", 5, 1, 20, 1, True),
    ],
    "Learning": [
        ("learning_rate", "Learning rate per event", 0.0123, 0.0, 0.05, 0.001, False),
        ("skill_cap", "Skill ceiling", 0.90, 0.5, 1.0, 0.05, False),
        ("sync_help_weight", "Sync-help learning weight", 1.5, 0.0, 3.0, 0.1, False),
        ("async_help_weight", "Async-help learning weight", 1.0, 0.0, 3.0, 0.1, False),
    ],
    "Turnover": [
        ("annual_attrition", "Annual attrition rate", 0.12, 0.0, 0.30, 0.01, False),
        ("junior_mean_skill", "Junior hire skill (frac of cap)", 0.40, 0.0, 0.8, 0.05, False),
        ("remote_attrition_coef", "Remote attrition coefficient", 0.0, -1.0, 1.0, 0.1, False),
    ],
    "Structural": [
        ("n_devs", "Number of developers", 5, 2, 10, 1, True),
        ("sprint_length", "Sprint length (ticks)", 320, 32, 640, 32, True),
        ("n_sprints", "Number of sprints", 24, 1, 52, 1, True),
        ("sp_scaling_k", "SP to ticks scaling k", 4, 1, 16, 1, True),
    ],
    "Reproducibility": [
        ("seed", "RNG seed", 42, 0, 10_000, 1, True),
    ],
}


def render_sidebar():
    """Draw the parameter sliders and return a kwargs dict for ``TeamModel``."""
    params = {}
    st.sidebar.header("Parameters")
    for group, specs in PARAM_GROUPS.items():
        with st.sidebar.expander(group, expanded=(group == "Headline")):
            for key, label, default, lo, hi, step, is_int in specs:
                if is_int:
                    params[key] = st.slider(
                        label, int(lo), int(hi), int(default), int(step), key=key
                    )
                else:
                    params[key] = st.slider(
                        label, float(lo), float(hi), float(default), float(step), key=key
                    )
    return params


# --------------------------------------------------------------------------- #
# Simulation run + per-tick snapshots                                          #
# --------------------------------------------------------------------------- #

def _snapshot(model):
    """Capture the displayable state of every dev at the current tick.

    Positions are derived from a dev's index in ``model.devs``; the list keeps a
    constant length (turnover retires and immediately replaces), so index 0..n-1
    is a stable set of slots even as individual devs come and go.
    """
    devs = list(model.devs)
    index_of = {d: i for i, d in enumerate(devs)}
    edges = []
    for d in devs:
        # Only the requester carries `.helper`, so each active session yields
        # exactly one edge (requester -> helper).
        if d.helper is not None and d.helper in index_of:
            edges.append((index_of[d], index_of[d.helper], d.help_channel or "async"))
    return {
        "tick": model.tick,
        "states": [d.state for d in devs],
        "locations": [d.location for d in devs],
        "skills": [round(d.solo_skill, 3) for d in devs],
        "edges": edges,
    }


@st.cache_data(show_spinner="Running simulation…")
def run_simulation(params, max_frames):
    """Run the model to completion, snapshotting (sub-sampled) ticks.

    Returns ``(frames, df, n_devs)``. Cached on the parameter set so scrubbing
    the time slider never re-runs the model.
    """
    model = TeamModel(**params)
    horizon = model.horizon_ticks
    stride = max(1, horizon // max_frames)

    frames = [_snapshot(model)]  # tick 0
    for _ in range(horizon):
        model.step()
        if model.tick % stride == 0 or model.tick == horizon:
            frames.append(_snapshot(model))

    df = model.datacollector.get_model_vars_dataframe()
    return frames, df, params["n_devs"]


# --------------------------------------------------------------------------- #
# Animated network figure                                                      #
# --------------------------------------------------------------------------- #

def _circle_positions(n):
    """Evenly spaced points on the unit circle, starting at the top."""
    xs, ys = [], []
    for i in range(n):
        angle = math.pi / 2 - 2 * math.pi * i / n
        xs.append(math.cos(angle))
        ys.append(math.sin(angle))
    return xs, ys


def _edge_coords(frame, xs, ys, channel):
    """Flatten the given channel's edges into None-separated line segments."""
    ex, ey = [], []
    for i, j, ch in frame["edges"]:
        if ch == channel:
            ex += [xs[i], xs[j], None]
            ey += [ys[i], ys[j], None]
    return ex, ey


def _node_visuals(frame):
    """Per-node fill colours, ring colours and hover text for one frame."""
    fills, rings, text = [], [], []
    for idx, (state, loc, skill) in enumerate(
        zip(frame["states"], frame["locations"], frame["skills"])
    ):
        fills.append(STATE_FILL.get(state, "#cccccc"))
        rings.append(LOCATION_RING.get(loc, "#000000"))
        text.append(
            f"Dev {idx}<br>state: {state}<br>location: {loc}<br>skill: {skill:.2f}"
        )
    return fills, rings, text


def _frame_traces(frame, xs, ys):
    """Build the three traces (sync edges, async edges, nodes) for a frame."""
    sx, sy = _edge_coords(frame, xs, ys, "sync")
    ax, ay = _edge_coords(frame, xs, ys, "async")
    fills, rings, text = _node_visuals(frame)

    sync_trace = go.Scatter(
        x=sx, y=sy, mode="lines", line=EDGE_STYLE["sync"],
        hoverinfo="skip", name="sync help", showlegend=False,
    )
    async_trace = go.Scatter(
        x=ax, y=ay, mode="lines", line=EDGE_STYLE["async"],
        hoverinfo="skip", name="async help", showlegend=False,
    )
    node_trace = go.Scatter(
        x=xs, y=ys, mode="markers+text",
        marker=dict(
            size=44, color=fills,
            line=dict(color=rings, width=6),
        ),
        text=[str(i) for i in range(len(xs))],
        textposition="middle center",
        textfont=dict(color="white", size=12),
        customdata=text, hovertemplate="%{customdata}<extra></extra>",
        showlegend=False,
    )
    return [sync_trace, async_trace, node_trace]


def build_network_figure(frames, n_devs):
    """A Plotly figure with native play/pause + time-slider over all frames."""
    xs, ys = _circle_positions(n_devs)

    base = _frame_traces(frames[0], xs, ys)
    plotly_frames = [
        go.Frame(data=_frame_traces(f, xs, ys), name=str(f["tick"]))
        for f in frames
    ]

    slider_steps = [
        dict(
            method="animate",
            label=str(f["tick"]),
            args=[[str(f["tick"])],
                  dict(mode="immediate",
                       frame=dict(duration=0, redraw=True),
                       transition=dict(duration=0))],
        )
        for f in frames
    ]

    play_args = [None, dict(mode="immediate",
                            frame=dict(duration=120, redraw=True),
                            transition=dict(duration=0),
                            fromcurrent=True)]
    pause_args = [[None], dict(mode="immediate",
                              frame=dict(duration=0, redraw=True),
                              transition=dict(duration=0))]

    fig = go.Figure(data=base, frames=plotly_frames)
    fig.update_layout(
        height=560,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="white",
        xaxis=dict(visible=False, range=[-1.35, 1.35], fixedrange=True),
        yaxis=dict(visible=False, range=[-1.35, 1.35], fixedrange=True,
                   scaleanchor="x", scaleratio=1),
        updatemenus=[dict(
            type="buttons", direction="left", showactive=False,
            x=0.0, y=1.12, xanchor="left", yanchor="top",
            buttons=[
                dict(label="▶ Play", method="animate", args=play_args),
                dict(label="⏸ Pause", method="animate", args=pause_args),
            ],
        )],
        sliders=[dict(
            active=0, x=0.0, y=0, len=1.0,
            currentvalue=dict(prefix="Tick: "),
            steps=slider_steps,
        )],
    )
    return fig


def render_legend():
    """Compact colour legend rendered as HTML chips."""
    def chip(color, label, ring=False):
        style = (
            f"display:inline-block;width:14px;height:14px;border-radius:50%;"
            f"margin-right:6px;vertical-align:middle;"
        )
        if ring:
            style += f"background:white;border:4px solid {color};"
        else:
            style += f"background:{color};"
        return f"<span style='{style}'></span>{label}"

    fills = "&nbsp;&nbsp;".join(
        chip(STATE_FILL[s], s) for s in ("idle", "working", "blocked", "helping")
    )
    rings = "&nbsp;&nbsp;".join(
        chip(LOCATION_RING[l], l, ring=True) for l in ("office", "home")
    )
    edges = (
        "<span style='border-bottom:3px solid #2ca02c;'>&nbsp;&nbsp;&nbsp;</span> sync help"
        "&nbsp;&nbsp;"
        "<span style='border-bottom:3px dashed #9467bd;'>&nbsp;&nbsp;&nbsp;</span> async help"
    )
    st.markdown(
        f"**Inner fill (state):** {fills}<br>"
        f"**Outer ring (location):** {rings}<br>"
        f"**Edges:** {edges}",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Charts — same five series/colours as app.py, drawn from the datacollector.   #
# --------------------------------------------------------------------------- #

CHART_SPECS = [
    {
        "title": "Per-sprint velocity (resets each sprint)",
        "ylabel": "Story points this sprint",
        "series": {"velocity": "tab:blue"},
    },
    {
        "title": "Mean waiting time per resolved blocker",
        "ylabel": "Wait (ticks)",
        "series": {"avg_wait": "tab:red"},
    },
    {
        "title": "Developer states over time",
        "ylabel": "Number of developers",
        "series": {
            "n_working": "tab:green",
            "n_blocked": "tab:red",
            "n_helping": "tab:orange",
            "n_idle":    "tab:gray",
        },
    },
    {
        "title": "Team skill over time (learning + churn)",
        "ylabel": "Solo skill",
        "series": {
            "mean_skill": "tab:purple",
            "min_skill":  "tab:gray",
            "max_skill":  "tab:cyan",
        },
    },
    {
        "title": "Attrition and tenure",
        "ylabel": "Count / mean tenure (ticks)",
        "series": {
            "attrition_count": "tab:brown",
            "mean_tenure":     "tab:olive",
        },
    },
]


def make_chart(df, spec):
    fig, ax = plt.subplots(figsize=(5, 3))
    for column, color in spec["series"].items():
        ax.plot(df.index, df[column], color=color, label=column)
    ax.set_title(spec["title"], fontsize=10)
    ax.set_xlabel("Tick")
    ax.set_ylabel(spec["ylabel"])
    if len(spec["series"]) > 1:
        ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Page                                                                         #
# --------------------------------------------------------------------------- #

def main():
    st.set_page_config(
        page_title="Hybrid-work policy simulation", layout="wide"
    )
    st.title("Hybrid-work policy: velocity, learning, and turnover")

    params = render_sidebar()
    st.sidebar.divider()
    max_frames = st.sidebar.slider(
        "Animation frames (sub-sampling)", 100, 1000, 300, 50,
        help="Upper bound on captured ticks; finer = smoother but heavier.",
    )

    frames, df, n_devs = run_simulation(params, max_frames)

    st.subheader("Team animation")
    render_legend()
    st.plotly_chart(build_network_figure(frames, n_devs), use_container_width=True)

    st.subheader("Metrics")
    cols = st.columns(2)
    for i, spec in enumerate(CHART_SPECS):
        with cols[i % 2]:
            st.pyplot(make_chart(df, spec))
            plt.close("all")


if __name__ == "__main__":
    main()
