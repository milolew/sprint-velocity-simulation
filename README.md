# Sprint velocity simulation

Agent-based simulation of a small software-development team. It studies how the
share of remote work affects team velocity, both within a single sprint and over
a roughly one-year horizon in which developers learn from experience (their skill
at resolving blockers grows) and the team turns over (people leave and juniors are
hired and must learn). Co-located help resolves blockers faster and teaches more
than remote, asynchronous help.

Built against Mesa 3.x.

## Setup

```bash
pip install -r requirements.txt
```


## Run the interactive UI

Two front-ends are available over the same model.

**Streamlit** (charts + animated team view):

```bash
streamlit run app_streamlit.py
```

Developers are nodes around a circle. The outer ring colours location
(office/home), the inner fill colours activity (idle/working/blocked/helping),
and an edge appears between a blocked dev and whoever is helping them (solid =
co-located/sync, dashed = remote/async). The model runs once to completion and
replays via the play/pause controls and time slider; the same five charts as
the Solara app are shown below.

**Solara** (original charts-only UI):

```bash
solara run app.py
```

Solara opens a local server (default `http://localhost:8765`). 
