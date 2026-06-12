"""Compute summary tables from experiment CSVs and print as markdown."""
import csv
from statistics import mean, stdev


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def cast(rows):
    out = []
    for r in rows:
        out.append({k: to_float(v) for k, v in r.items()})
    return out


def group_summary(rows, group_keys, metric):
    groups = {}
    for r in rows:
        k = tuple(r[gk] for gk in group_keys)
        groups.setdefault(k, []).append(r[metric])
    out = []
    for k in sorted(groups.keys()):
        vals = groups[k]
        m = mean(vals)
        sd = stdev(vals) if len(vals) > 1 else 0.0
        ci = 2.093 * sd / (len(vals) ** 0.5)
        out.append({**dict(zip(group_keys, k)), "n": len(vals),
                    "mean": m, "sd": sd, "ci95": ci,
                    "min": min(vals), "max": max(vals)})
    return out


def print_table(rows, title, key_cols, metric_label="velocity"):
    print(f"\n### {title}")
    headers = key_cols + ["n", f"mean {metric_label}", "sd", "95% CI", "min", "max"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        keys = [f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]) for c in key_cols]
        line = keys + [
            str(r["n"]),
            f"{r['mean']:.1f}",
            f"{r['sd']:.1f}",
            f"+/-{r['ci95']:.1f}",
            f"{r['min']:.0f}",
            f"{r['max']:.0f}",
        ]
        print("| " + " | ".join(line) + " |")


if __name__ == "__main__":
    # Exp 1
    r1 = cast(load("results/exp1_baseline.csv"))
    print("## Exp 1: baseline remote_share sweep (n_devs=10)")
    print_table(group_summary(r1, ["remote_share"], "velocity"),
                "Velocity", ["remote_share"])
    print_table(group_summary(r1, ["remote_share"], "avg_wait"),
                "Avg wait (ticks)", ["remote_share"], "avg_wait")
    print_table(group_summary(r1, ["remote_share"], "blockers_resolved"),
                "Blockers resolved", ["remote_share"], "blockers")

    # Exp 2
    r2 = cast(load("results/exp2_skill_interaction.csv"))
    print("\n## Exp 2: remote_share x mean_solo_skill (n_devs=10)")
    print_table(group_summary(r2, ["mean_solo_skill", "remote_share"], "velocity"),
                "Velocity", ["mean_solo_skill", "remote_share"])

    # Exp 3
    r3 = cast(load("results/exp3_blockprob_interaction.csv"))
    print("\n## Exp 3: remote_share x block_prob (n_devs=10)")
    print_table(group_summary(r3, ["block_prob", "remote_share"], "velocity"),
                "Velocity", ["block_prob", "remote_share"])
