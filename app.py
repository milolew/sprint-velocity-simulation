"""Solara UI for the sprint simulation.

Run with:
    solara run app.py

Requires Mesa 3.x with the visualisation extras:
    pip install "mesa[viz]"
"""

from mesa.visualization import SolaraViz, Slider, make_plot_component

from model import TeamModel


def velocity_post(ax):
    ax.set_title("Per-sprint velocity (resets each sprint)")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Story points this sprint")


def wait_post(ax):
    ax.set_title("Mean waiting time per resolved blocker")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Wait (ticks)")


def state_post(ax):
    ax.set_title("Developer states over time")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Number of developers")
    ax.legend(loc="upper right", fontsize="small")


def skill_post(ax):
    ax.set_title("Team skill over time (learning + churn)")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Solo skill")
    ax.legend(loc="lower right", fontsize="small")


def attrition_post(ax):
    ax.set_title("Attrition and tenure")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Count / mean tenure (ticks)")
    ax.legend(loc="upper left", fontsize="small")


VelocityPlot = make_plot_component({"velocity": "tab:blue"}, post_process=velocity_post)
WaitPlot = make_plot_component({"avg_wait": "tab:red"}, post_process=wait_post)
StatePlot = make_plot_component(
    {
        "n_working": "tab:green",
        "n_blocked": "tab:red",
        "n_helping": "tab:orange",
        "n_idle":    "tab:gray",
    },
    post_process=state_post,
)
SkillPlot = make_plot_component(
    {
        "mean_skill": "tab:purple",
        "min_skill":  "tab:gray",
        "max_skill":  "tab:cyan",
    },
    post_process=skill_post,
)
AttritionPlot = make_plot_component(
    {
        "attrition_count": "tab:brown",
        "mean_tenure":     "tab:olive",
    },
    post_process=attrition_post,
)


# Slider-driven model parameters. Slider signature: (label, value, min, max, step)
model_params = {
    # Headline parameter under study
    "remote_share":      Slider("Share of remote work", 0.5, 0.0, 1.0, 0.05),
    # Team heterogeneity
    "mean_solo_skill":   Slider("Mean solo skill", 0.5, 0.0, 1.0, 0.05),
    "solo_skill_spread": Slider("Solo-skill spread", 0.2, 0.0, 0.5, 0.05),
    # Process knobs
    "block_prob":        Slider("Per-tick blocker probability", 0.02, 0.0, 0.10, 0.005),
    "sync_help_mean":    Slider("Sync help mean (ticks)", 1, 0, 5, 1),
    "async_help_mean":   Slider("Async help mean (ticks)", 5, 1, 20, 1),
    # Learning (event-driven)
    "learning_rate":     Slider("Learning rate per event", 0.0123, 0.0, 0.05, 0.001),
    "skill_cap":         Slider("Skill ceiling", 0.90, 0.5, 1.0, 0.05),
    "sync_help_weight":  Slider("Sync-help learning weight", 1.5, 0.0, 3.0, 0.1),
    "async_help_weight": Slider("Async-help learning weight", 1.0, 0.0, 3.0, 0.1),
    # Turnover
    "annual_attrition":  Slider("Annual attrition rate", 0.12, 0.0, 0.30, 0.01),
    "junior_mean_skill": Slider("Junior hire skill (frac of cap)", 0.40, 0.0, 0.8, 0.05),
    "remote_attrition_coef": Slider("Remote attrition coefficient", 0.0, -1.0, 1.0, 0.1),
    # Structural
    "n_devs":            Slider("Number of developers", 5, 2, 10, 1),
    "sprint_length":     Slider("Sprint length (ticks)", 320, 32, 640, 32),
    "n_sprints":         Slider("Number of sprints", 24, 1, 52, 1),
    "sp_scaling_k":      Slider("SP to ticks scaling k", 4, 1, 16, 1),
    # Reproducibility
    "seed":              Slider("RNG seed", 42, 0, 10_000, 1),
}


page = SolaraViz(
    TeamModel(),
    components=[VelocityPlot, WaitPlot, StatePlot, SkillPlot, AttritionPlot],
    model_params=model_params,
    name="Hybrid-work policy: velocity, learning, and turnover",
)
page  # noqa: must be the last expression for Solara to pick up
