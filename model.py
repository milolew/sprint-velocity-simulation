"""Agent-based simulation of a small software-development team during one sprint.

Studies how the share of remote work affects team velocity, accounting for
heterogeneous individual ability to resolve blockers without help.

Built against Mesa 3.x.
"""

import math
from dataclasses import dataclass

from mesa import Agent, Model
from mesa.datacollection import DataCollector

TICKS_PER_DAY = 32  # eight-hour day at 15 min per tick
STORY_POINT_CHOICES = (1, 2, 3, 5)  # story-point sizes drawn for backlog tasks


@dataclass
class Task:
    """A unit of work circulating between backlog, in-progress, and done.

    Tasks are passive data, not Mesa agents: they are mutated by whichever
    developer currently owns them.
    """

    story_points: int
    work_remaining: int
    state: str = "awaiting"  # awaiting / in_progress / blocked / completed


class Dev(Agent):
    """A team member with a per-tick state machine.

    Top-level dispatch is on ``state``; while ``state == "blocked"`` there
    is a secondary dispatch on ``blocker_stage`` (solo / ask_next_tick /
    waiting_for_help).
    """

    def __init__(self, model, solo_skill):
        super().__init__(model)
        self.solo_skill = max(0.0, min(1.0, solo_skill))

        # Rolled at the start of every day
        self.location = "office"

        # Activity state
        self.state = "idle"
        self.current_task = None

        # Blocker bookkeeping (used when this dev is the requester)
        self.blocker_stage = None
        self.block_start_tick = None
        self.help_timer = None
        self.helper = None
        # Channel of the active help session ("sync"/"async"); set iff helper
        # is set, drives how much is learned from the resolution.
        self.help_channel = None

        # Helper bookkeeping (used when this dev is helping someone else)
        self.prev_state = None

        # Tenure: tick at which this dev joined (0 for the founding team).
        self.hired_at_tick = 0

    def step(self):
        """Run one tick of the per-developer state machine."""
        if self.state == "idle":
            self._step_idle()
        elif self.state == "working":
            self._step_working()
        elif self.state == "blocked":
            self._step_blocked()
        elif self.state == "helping":
            # Helper is locked in for the duration of the session and is
            # unavailable for own work. The session timer is decremented
            # by the requester only (single source of truth) so the
            # shared counter does not get ticked down twice per tick.
            pass

    def _step_idle(self):
        """Pull a ticket from the backlog if one is available."""
        if not self.model.backlog:
            return
        task = self.model.backlog.pop(0)
        task.state = "in_progress"
        self.current_task = task
        self.state = "working"

    def _step_working(self):
        """Make progress on the current task, then maybe hit a blocker."""
        task = self.current_task
        task.work_remaining -= 1

        if task.work_remaining <= 0:
            task.state = "completed"
            self.model.completed_tasks.append(task)
            self.model.velocity += task.story_points
            self._apply_learning(self.model.task_completion_weight)
            self.current_task = None
            self.state = "idle"
            return

        # Blockers can only arise while working
        if self.random.random() < self.model.block_prob:
            self.state = "blocked"
            self.blocker_stage = "solo"
            self.block_start_tick = self.model.tick
            task.state = "blocked"

    def _step_blocked(self):
        """Dispatch on the current blocker stage."""
        if self.blocker_stage == "solo":
            self._try_solo()
        elif self.blocker_stage == "ask_next_tick":
            self._try_find_helper()
        elif self.blocker_stage == "waiting_for_help":
            self._tick_help_session()

    def _try_solo(self):
        """Attempt to resolve the blocker alone; success prob = solo_skill."""
        if self.random.random() < self.solo_skill:
            self._unblock("solo")
        else:
            self.blocker_stage = "ask_next_tick"

    def _try_find_helper(self):
        """Find an available helper and start a help session, or wait."""
        # Helpers must be either working or idle (not blocked, not already helping)
        candidates = [
            d for d in self.model.devs
            if d is not self and d.state in ("working", "idle")
        ]
        if not candidates:
            # No fallback by design: this is the throughput limit. Wait, retry next tick.
            return

        # If I'm in office, prefer office colleagues so the session is sync.
        # If I'm remote, no preference: every contact is async anyway.
        if self.location == "office":
            in_office = [d for d in candidates if d.location == "office"]
            if in_office:
                candidates = in_office

        helper = self.random.choice(candidates)
        helper.prev_state = helper.state
        helper.state = "helping"
        self.helper = helper

        # Sync if both are in the office, async otherwise. The channel is
        # stored on the requester so the resolution knows how much was learned.
        if self.location == "office" and helper.location == "office":
            self.help_channel = "sync"
            duration = self._sample_duration(
                self.model.sync_help_mean, self.model.sync_help_jitter
            )
        else:
            self.help_channel = "async"
            duration = self._sample_duration(
                self.model.async_help_mean, self.model.async_help_jitter
            )

        # A 0-tick session resolves immediately; otherwise the timer is
        # decremented once per tick starting on the next tick, giving a
        # session of exactly `duration` ticks (no off-by-one).
        if duration <= 0:
            helper.state = helper.prev_state
            helper.prev_state = None
            self._unblock(self.help_channel)
            return

        self.help_timer = duration
        self.blocker_stage = "waiting_for_help"

    def _sample_duration(self, mean, jitter):
        """Uniform integer on [mean-jitter, mean+jitter], clipped at 0."""
        return max(0, self.random.randint(mean - jitter, mean + jitter))

    def _tick_help_session(self):
        """Decrement the session timer; resolve when it expires."""
        self.help_timer -= 1
        if self.help_timer <= 0:
            helper = self.helper
            helper.state = helper.prev_state
            helper.prev_state = None
            self._unblock(self.help_channel)

    def _unblock(self, resolution):
        """Bookkeeping when a blocker is resolved.

        ``resolution`` is the channel that resolved it ("solo" / "sync" /
        "async"); it selects the learning gain credited for the experience.
        """
        wait = self.model.tick - self.block_start_tick
        self.model.wait_times.append(wait)
        self.state = "working"
        self.current_task.state = "in_progress"
        self.blocker_stage = None
        self.block_start_tick = None
        self.help_timer = None
        self.helper = None
        self.help_channel = None
        self._apply_learning(self._weight_for(resolution))

    def _weight_for(self, resolution):
        """Map a resolution channel to its learning-gain weight."""
        weights = {
            "solo": self.model.solo_resolution_weight,
            "sync": self.model.sync_help_weight,
            "async": self.model.async_help_weight,
        }
        if resolution not in weights:
            raise ValueError(f"unknown resolution channel: {resolution!r}")
        return weights[resolution]

    def _apply_learning(self, weight):
        """Move solo_skill toward the ceiling by a bounded, deterministic step.

        Diminishing returns toward ``skill_cap`` (no RNG); monotonic and
        bounded above by the cap. A no-op when ``learning_rate == 0``.
        """
        gap = max(0.0, self.model.skill_cap - self.solo_skill)
        self.solo_skill += self.model.learning_rate * weight * gap


class TeamModel(Model):
    """A small dev team running through one sprint.

    Parameter under study:
        remote_share -- probability that a developer works from home on any
            given day (rolled independently each morning, per developer).

    Heterogeneity knobs:
        mean_solo_skill, solo_skill_spread -- solo skill drawn uniformly
            from [mean - spread, mean + spread] for each developer.

    Process knobs:
        block_prob -- per-tick blocker probability while working.
        sync_help_mean, sync_help_jitter -- in-office help duration (1 +/- 1).
        async_help_mean, async_help_jitter -- remote help duration (5 +/- 2).

    Structural:
        n_devs (5), sprint_length (320 ticks = 10 days x 32),
        sp_scaling_k (4 ticks per story point by default).
    """

    def __init__(
        self,
        remote_share=0.5,
        block_prob=0.02,
        n_devs=5,
        sprint_length=320,
        sp_scaling_k=4,
        mean_solo_skill=0.5,
        solo_skill_spread=0.2,
        sync_help_mean=1,
        sync_help_jitter=1,
        async_help_mean=5,
        async_help_jitter=2,
        backlog_size=500,
        # Horizon
        n_sprints=24,
        sprints_per_year=24,
        backlog_target=500,
        # Learning (event-driven). learning_rate is calibrated (not guessed):
        # see calibrate_learning.py — fitted so a junior closes ~90% of the gap
        # to skill_cap in ~12 sprints (~6 months), the literature ramp-up target.
        learning_rate=0.0123,
        skill_cap=0.90,
        task_completion_weight=0.4,
        solo_resolution_weight=1.0,
        sync_help_weight=1.5,
        async_help_weight=1.0,
        # Turnover
        annual_attrition=0.12,
        junior_mean_skill=0.40,    # fraction of skill_cap (hire skill = mean*cap)
        junior_skill_spread=0.10,  # fraction of skill_cap
        remote_attrition_coef=0.0,
        remote_baseline=0.5,
        junior_threshold=0.40,     # ABSOLUTE skill cutoff for the n_juniors metric
        seed=None,
    ):
        super().__init__(seed=seed)

        self.remote_share = remote_share
        self.block_prob = block_prob
        self.n_devs = n_devs
        self.sprint_length = sprint_length
        self.sp_scaling_k = sp_scaling_k
        self.sync_help_mean = sync_help_mean
        self.sync_help_jitter = sync_help_jitter
        self.async_help_mean = async_help_mean
        self.async_help_jitter = async_help_jitter

        # Horizon
        self.n_sprints = n_sprints
        self.sprints_per_year = sprints_per_year
        self.backlog_target = backlog_target

        # Learning
        self.learning_rate = learning_rate
        self.skill_cap = skill_cap
        self.task_completion_weight = task_completion_weight
        self.solo_resolution_weight = solo_resolution_weight
        self.sync_help_weight = sync_help_weight
        self.async_help_weight = async_help_weight

        # Turnover
        self.annual_attrition = annual_attrition
        self.junior_mean_skill = junior_mean_skill
        self.junior_skill_spread = junior_skill_spread
        self.remote_attrition_coef = remote_attrition_coef
        self.remote_baseline = remote_baseline
        self.junior_threshold = junior_threshold

        self._validate_params()

        self.tick = 0
        self.velocity = 0
        self.wait_times = []
        self.completed_tasks = []
        self.sprint_velocities = []
        self.attrition_count = 0

        # Total run length and per-sprint leave hazard (no RNG)
        self.horizon_ticks = sprint_length * n_sprints
        self.p_leave_base = 1 - (1 - annual_attrition) ** (1 / sprints_per_year)

        # Pre-generate a generous backlog so the team never starves
        self.backlog = []
        for _ in range(backlog_size):
            self.backlog.append(self._new_task())

        # Build the dev team with heterogeneous solo skills
        self.devs = []
        for _ in range(n_devs):
            skill = self.random.uniform(
                mean_solo_skill - solo_skill_spread,
                mean_solo_skill + solo_skill_spread,
            )
            self.devs.append(Dev(self, skill))

        # Roll locations for day 1 so the initial datacollector row reflects
        # real starting state rather than the constructor default.
        self._roll_locations()

        self.datacollector = DataCollector(
            model_reporters={
                "velocity": "velocity",
                "avg_wait": lambda m: (
                    sum(m.wait_times) / len(m.wait_times) if m.wait_times else 0.0
                ),
                "blockers_resolved": lambda m: len(m.wait_times),
                "tick": "tick",
                "day": lambda m: m.tick // TICKS_PER_DAY + 1,
                "sprint": lambda m: (
                    m.tick // m.sprint_length + 1 if m.sprint_length else 1
                ),
                "n_working": lambda m: sum(d.state == "working" for d in m.devs),
                "n_blocked": lambda m: sum(d.state == "blocked" for d in m.devs),
                "n_helping": lambda m: sum(d.state == "helping" for d in m.devs),
                "n_idle":    lambda m: sum(d.state == "idle"    for d in m.devs),
                # Capability & retention metrics
                "mean_skill": lambda m: (
                    sum(d.solo_skill for d in m.devs) / len(m.devs) if m.devs else 0.0
                ),
                "min_skill": lambda m: min(
                    (d.solo_skill for d in m.devs), default=0.0
                ),
                "max_skill": lambda m: max(
                    (d.solo_skill for d in m.devs), default=0.0
                ),
                "n_juniors": lambda m: sum(
                    d.solo_skill < m.junior_threshold for d in m.devs
                ),
                "attrition_count": "attrition_count",
                "mean_tenure": lambda m: (
                    sum(m.tick - d.hired_at_tick for d in m.devs) / len(m.devs)
                    if m.devs else 0.0
                ),
            }
        )
        self.datacollector.collect(self)
        self.running = True

    def _validate_params(self):
        """Fail fast on out-of-range knobs (validation at the boundary)."""
        if not 0.0 <= self.skill_cap <= 1.0:
            raise ValueError(f"skill_cap must be in [0,1], got {self.skill_cap}")
        if self.learning_rate < 0.0:
            raise ValueError(f"learning_rate must be >= 0, got {self.learning_rate}")
        if not 0.0 <= self.annual_attrition <= 1.0:
            raise ValueError(
                f"annual_attrition must be in [0,1], got {self.annual_attrition}"
            )
        weights = (
            self.task_completion_weight, self.solo_resolution_weight,
            self.sync_help_weight, self.async_help_weight,
        )
        if any(w < 0.0 for w in weights):
            raise ValueError("learning weights must be >= 0")
        if self.n_sprints < 1:
            raise ValueError(f"n_sprints must be >= 1, got {self.n_sprints}")
        if self.sprints_per_year < 1:
            raise ValueError(f"sprints_per_year must be >= 1, got {self.sprints_per_year}")
        if not math.isfinite(self.remote_attrition_coef):
            raise ValueError("remote_attrition_coef must be finite")
        if not 0.0 <= self.remote_baseline <= 1.0:
            raise ValueError(f"remote_baseline must be in [0,1], got {self.remote_baseline}")
        if self.junior_mean_skill < 0.0 or self.junior_skill_spread < 0.0:
            raise ValueError("junior skill mean/spread must be >= 0")

    def _new_task(self):
        """Build one backlog task with a random story-point size."""
        sp = self.random.choice(STORY_POINT_CHOICES)
        return Task(story_points=sp, work_remaining=sp * self.sp_scaling_k)

    def _roll_locations(self):
        """Assign each dev home/office for the day, drawn from remote_share."""
        for d in self.devs:
            d.location = (
                "home" if self.random.random() < self.remote_share else "office"
            )

    def _is_day_boundary(self):
        return self.tick > 0 and self.tick % TICKS_PER_DAY == 0

    def _is_sprint_boundary(self):
        # sprint_length>0 guard: defensive against div-by-zero if the
        # horizon stop-guard in step() is ever bypassed.
        return (
            self.sprint_length > 0
            and self.tick > 0
            and self.tick % self.sprint_length == 0
        )

    def _top_up_backlog(self):
        """Replenish the backlog up to the target floor for the new sprint."""
        while len(self.backlog) < self.backlog_target:
            self.backlog.append(self._new_task())

    def _close_sprint(self):
        """Record the live velocity, reset the per-sprint counter, and top up
        the backlog. The final sprint is recorded separately in step()'s tail."""
        self.sprint_velocities.append(self.velocity)
        self.velocity = 0
        self._top_up_backlog()

    def _p_leave(self):
        """Per-sprint leave probability with the signed remote modulation.

        remote_share shifts the base hazard around remote_baseline; the sign of
        remote_attrition_coef encodes the (contested) direction of remote work's
        effect on attrition. Dev-independent for now (no seniority conditioning).
        """
        raw = self.p_leave_base * (
            1 + self.remote_attrition_coef * (self.remote_share - self.remote_baseline)
        )
        return max(0.0, min(1.0, raw))

    def _is_settled(self, dev):
        """True when `dev` can leave cleanly (not mid help session).

        A helping dev or one waiting for help is deferred: it is skipped this
        boundary and re-rolls next boundary (keeps the RNG draw count constant
        and avoids unwinding a foreign help session).
        """
        if dev.state == "helping":
            return False
        if dev.state == "blocked" and dev.blocker_stage == "waiting_for_help":
            return False
        return True

    def _run_turnover_checks(self):
        """Resolve per-sprint turnover: decide all, then apply.

        Exactly one RNG draw per dev in a fixed order (constant regardless of
        how many leave), then settled leavers are retired and replaced.
        """
        p_leave = self._p_leave()
        leavers = [d for d in list(self.devs) if self.random.random() < p_leave]
        for dev in leavers:
            if self._is_settled(dev):
                self._retire_dev(dev)

    def _retire_dev(self, dev):
        """Remove a settled dev and immediately replace it with a junior hire.

        Any task the leaver held returns to the backlog with progress reset
        (partial work is lost — knowledge walks out the door). Both Mesa's
        AgentSet and our `devs` list are kept in sync.
        """
        if dev.current_task is not None:
            task = dev.current_task
            task.state = "awaiting"
            task.work_remaining = task.story_points * self.sp_scaling_k
            self.backlog.append(task)
            dev.current_task = None

        dev.remove()                 # deregister from Mesa's AgentSet
        self.devs.remove(dev)

        # Junior skill is a fraction of the team's skill ceiling.
        skill = self.random.uniform(
            (self.junior_mean_skill - self.junior_skill_spread) * self.skill_cap,
            (self.junior_mean_skill + self.junior_skill_spread) * self.skill_cap,
        )
        hire = Dev(self, skill)      # auto-registers in Mesa's AgentSet
        hire.hired_at_tick = self.tick
        self.devs.append(hire)
        self.attrition_count += 1

    def step(self):
        """Advance the simulation by one tick.

        Boundary ordering (a determinism contract): at the start of sprints
        2..N the previous sprint is closed (velocity recorded + reset, backlog
        topped up) and turnover is resolved *before* pass 1, so the RNG stream
        is stable. The final sprint is recorded in the tail at the stop.

        Activation is two-phase so that a help-seeker sees a stable view of
        who is available: every other dev has already taken their tick's
        action by the time recruitment happens. Without this, whether a
        recruited helper produced 1 or 0 ticks of work in the recruitment
        tick depended on the random iteration order.
        """
        if self.tick >= self.horizon_ticks:
            self.running = False
            return

        # Day 1's locations were rolled in __init__; re-roll at every
        # subsequent day boundary.
        if self._is_day_boundary():
            self._roll_locations()

        # Sprint boundary (start of sprints 2..N): close the prior sprint and
        # resolve turnover before any work happens this tick.
        if self._is_sprint_boundary():
            self._close_sprint()
            self._run_turnover_checks()

        # Pass 1: settle every non-recruitment action (work, idle pickup,
        # solo attempts, help-session ticks, helping no-ops).
        seekers = []
        for d in self.random.sample(self.devs, len(self.devs)):
            if d.state == "blocked" and d.blocker_stage == "ask_next_tick":
                seekers.append(d)
                continue
            d.step()

        # Pass 2: help-seekers act on the now-stable candidate pool.
        for d in self.random.sample(seekers, len(seekers)):
            d.step()

        self.tick += 1
        self.datacollector.collect(self)

        if self.tick >= self.horizon_ticks:
            # Record the final sprint's velocity (not reset — it is the run's
            # last; nothing follows) and stop.
            self.sprint_velocities.append(self.velocity)
            self.running = False
