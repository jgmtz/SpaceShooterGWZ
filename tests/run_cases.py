"""Roda os cenários de tests/cases.py com os módulos gwz.planner / gwz.evade fornecidos e devolve as saídas
em JSON. Usado para gerar o golden (com o código anterior ao core) e para conferir o código atual."""
from cases import planner_cases, tracker_sequence


def run(P, E):
    plans = []
    for c in planner_cases():
        kw = dict(c["kw"])
        for k in ("prev_end", "prev_target"):
            if kw[k] is not None:
                kw[k] = tuple(kw[k])
        tgt, clear, path = P.plan(tuple(c["ship"]), P.linear_predictor(c["bullets"]), **kw)
        plans.append({"tgt": list(tgt), "clear": clear, "path": [list(p) for p in path]})
    track = []
    prev, coast, t_prev = [], [], None
    for f in tracker_sequence():
        dt = f["t"] - t_prev if t_prev else 0
        vel = E.track_velocities(prev, [tuple(d) for d in f["det"]], dt, store_turn=True)
        coast = E.coast(prev, vel, coast, dt, tuple(f["ship"]))
        ep = E.enemy_points([(x, y, vx, vy) for x, y, vx, vy in vel[:3]], tuple(f["ship"]))
        track.append({"vel": [list(v) for v in vel], "coast": [list(c) for c in coast], "enemy": [list(p) for p in ep],
                      "ages": E.ages_of(vel)})
        prev, t_prev = vel, f["t"]
    return {"plans": plans, "track": track}
