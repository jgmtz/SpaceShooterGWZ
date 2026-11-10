"""Mede a qualidade da PREVISÃO de trajetória nos dados reais de partidas (logs/frames/*.jsonl), sem celular.
    python tools/track_eval.py [arquivos...]
Para cada bala de cada quadro, prevê a posição no quadro seguinte (posição + velocidade * dt) e mede a
distância até a bala real mais próxima. Compara o estimador antigo com o rastreador novo."""
import sys, os, glob, math
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gwz import evade

def frames(paths):
    for p in paths:
        seq = [__import__("json").loads(l) for l in open(p) if l.strip()]
        yield seq

def evaluate(paths, method):
    errs = []
    for seq in frames(paths):
        prev_raw, prev_est, prev_t = [], [], None
        for f in seq:
            cur = [(x, y) for x, y, *_ in f["b"]]
            dt = f["t"] - prev_t if prev_t else 0
            if dt > 0.3:
                prev_raw, prev_est, dt = [], [], 0
            est = evade.track_velocities(prev_est, cur, dt) if method == "novo" else \
                evade.estimate_velocities(prev_raw, cur, dt)
            # erro da previsão feita no quadro ANTERIOR para este
            if prev_est and cur and 0 < dt < 0.3:
                P = np.array(cur, float)
                for x, y, vx, vy in prev_est_used:
                    q = np.array([x + vx * dt, y + vy * dt])
                    if 0 < q[1] < 2340:
                        errs.append(float(np.min(np.hypot(P[:, 0] - q[0], P[:, 1] - q[1]))))
            prev_raw, prev_est, prev_t = cur, est, f["t"]
            prev_est_used = est
    e = np.array(errs)
    return len(e), float(np.median(e)), float((e < 30).mean()), float((e > 80).mean())

def evaluate_ahead(paths, method, ahead=0.4, smooth=0.5):
    """Previsão `ahead` s à frente (o que a esquiva usa): erro até a bala real mais próxima naquele quadro."""
    errs = []
    for seq in frames(paths):
        ests, prev, prev_t = [], [], None
        for f in seq:
            cur = [(x, y) for x, y, *_ in f["b"]]
            dt = f["t"] - prev_t if prev_t else 0
            if dt > 0.3:
                prev, dt = [], 0
            if method == "novo":
                est = evade.track_velocities(prev, cur, dt, smooth=smooth)
            else:
                est = evade.estimate_velocities([(x, y) for x, y, *_ in prev], cur, dt)
            ests.append((f["t"], est, cur)); prev, prev_t = est, f["t"]
        for k, (t, est, _) in enumerate(ests):
            fut = next((e for e in ests[k + 1:] if e[0] - t >= ahead), None)
            if fut is None or fut[0] - t > ahead + 0.1 or not fut[2]:
                continue
            P = np.array(fut[2], float); dt = fut[0] - t
            for x, y, vx, vy in est:
                q = (x + vx * dt, y + vy * dt)
                if 400 < q[1] < 2300:
                    errs.append(float(np.min(np.hypot(P[:, 0] - q[0], P[:, 1] - q[1]))))
    e = np.array(errs)
    return len(e), float(np.median(e)), float((e < 40).mean()), float((e > 120).mean())

if __name__ == "__main__":
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join("logs", "frames", "*.jsonl")))
    for m in ("antigo", "novo"):
        n, med, ok, bad = evaluate(paths, m)
        print(f"1 quadro  {m:7s} previsões={n}  erro mediano={med:5.1f}px  acerto<30px={100*ok:4.1f}%  erro>80px={100*bad:4.1f}%")
    for ahead in (0.3, 0.5):
        for m, sm in (("antigo", 0), ("novo", 0.0), ("novo", 0.5), ("novo", 0.75)):
            n, med, ok, bad = evaluate_ahead(paths, m, ahead, sm)
            lab = m if m == "antigo" else f"novo s={sm}"
            print(f"{ahead}s à frente {lab:12s} n={n} erro mediano={med:5.1f}px  <40px={100*ok:4.1f}%  >120px={100*bad:4.1f}%")
