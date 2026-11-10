"""Cenários determinísticos usados para gerar e conferir tests/golden_pre_core.json."""
import random


def planner_cases(n=80, seed=11):
    r = random.Random(seed)
    out = []
    for _ in range(n):
        B = [(r.uniform(0, 1080), r.uniform(900, 2300), r.uniform(-300, 300), r.uniform(-150, 1500))
             for _ in range(r.randint(0, 40))]
        kw = {"t0": r.choice((0.1, 0.15)),
              "prev_end": r.choice((None, [r.uniform(60, 1020), r.uniform(1650, 2150)])),
              "prev_target": r.choice((None, [r.uniform(60, 1020), r.uniform(1650, 2150)])),
              "enemies": [[r.uniform(0, 1080), r.uniform(300, 1500), 60.0] for _ in range(r.randint(0, 4))] or None}
        if r.random() < 0.3:
            kw.update(w_dps=0.0, sigma0=20.0, y_min=1800.0)
        out.append({"ship": [r.uniform(60, 1020), r.uniform(1650, 2150)], "bullets": B, "kw": kw})
    return out


def tracker_sequence(n=150, seed=12):
    """Rajadas de balas descendo (algumas somem e voltam) + uma nave andando."""
    r = random.Random(seed)
    shots = [[r.uniform(100, 980), r.uniform(300, 900), r.uniform(-150, 150), r.uniform(500, 1400)] for _ in range(25)]
    seq, t = [], 0.0
    for k in range(n):
        dt = r.uniform(0.04, 0.12)
        t += dt
        det = []
        for s in shots:
            s[0] += s[2] * dt
            s[1] += s[3] * dt
            if s[1] > 2340:
                s[:] = [r.uniform(100, 980), r.uniform(300, 600), r.uniform(-150, 150), r.uniform(500, 1400)]
            if r.random() > 0.1:                       # 10% de falha de detecção
                det.append([s[0] + r.uniform(-3, 3), s[1] + r.uniform(-3, 3)])
        seq.append({"t": t, "ship": [540 + 300 * (k % 40) / 40, 2000.0], "det": det})
    return seq
