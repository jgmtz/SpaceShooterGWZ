"""Balas que CURVAM: quanto a previsão em linha reta erra? (logs/frames/*.jsonl, sem celular)

    python tools/curve_eval.py [arquivos...]   (padrão: os 30 trechos mais recentes)

Segue cada bala quadro a quadro (casamento pela posição prevista), mede a curvatura (giro da direção em graus/s)
e o erro da previsão daqui a H s com 3 modelos: reto (velocidade constante), curva (giro constante) e
reto + margem. Uso: decidir se vale prever curvas na esquiva.
"""
import sys, os, glob, json, math
import numpy as np

H = 0.4

def tracks(seq):
    """Lista de trilhas [(t, x, y)] encadeando cada bala ao vizinho mais próximo da posição prevista."""
    live, done = [], []
    prev_t = None
    for f in seq:
        t = f["t"]; cur = [(b[0], b[1]) for b in f["b"]]
        if prev_t is not None and t <= prev_t:
            continue
        if prev_t is None or t - prev_t > 0.3:
            done += live; live = []
        used = set(); nxt = []
        for tr in live:
            (t1, x1, y1) = tr[-1]
            if len(tr) >= 2 and tr[-1][0] > tr[-2][0]:
                (t0, x0, y0) = tr[-2]; vx, vy = (x1 - x0) / (t1 - t0), (y1 - y0) / (t1 - t0)
            else:
                vx, vy = 0.0, 850.0
            px, py = x1 + vx * (t - t1), y1 + vy * (t - t1)
            best, bd = None, (45 + 0.4 * math.hypot(vx, vy) * (t - t1)) ** 2
            for i, (x, y) in enumerate(cur):
                d = (x - px) ** 2 + (y - py) ** 2
                if i not in used and d < bd:
                    best, bd = i, d
            if best is None:
                done.append(tr)
            else:
                used.add(best); tr.append((t,) + cur[best]); nxt.append(tr)
        nxt += [[(t, x, y)] for i, (x, y) in enumerate(cur) if i not in used]
        live, prev_t = nxt, t
    return [tr for tr in done + live if len(tr) >= 6]

def vel(tr, i, k=2):
    """Velocidade suavizada em i (diferença de k quadros)."""
    a, b = tr[max(0, i - k)], tr[i]
    dt = b[0] - a[0]
    return ((b[1] - a[1]) / dt, (b[2] - a[2]) / dt) if dt > 0 else (0.0, 0.0)

def main():
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join("logs", "frames", "*.jsonl")))[-30:]
    rows = []
    for p in paths:
        seq = [json.loads(l) for l in open(p) if l.strip()]
        for tr in tracks(seq):
            for i in range(4, len(tr)):
                t, x, y = tr[i]
                j = next((j for j in range(i + 1, len(tr)) if tr[j][0] - t >= H), None)
                if j is None:
                    break
                vx, vy = vel(tr, i); ux, uy = vel(tr, i - 2)
                dt0 = tr[i][0] - tr[i - 2][0]
                sp = math.hypot(vx, vy)
                if sp < 200 or dt0 <= 0:
                    continue
                w = (math.atan2(vy, vx) - math.atan2(uy, ux) + math.pi) % (2 * math.pi) - math.pi
                w /= dt0                                                     # rad/s
                h = tr[j][0] - t
                lin = (x + vx * h, y + vy * h)
                a = math.atan2(vy, vx) + w * h / 2; cur_ = (x + sp * h * math.cos(a), y + sp * h * math.sin(a))
                real = tr[j][1:]
                rows.append((abs(math.degrees(w)), math.dist(lin, real), math.dist(cur_, real)))
    R = np.array(rows)
    print(f"{len(R)} amostras de balas seguidas por >= {H}s em {len(paths)} trechos")
    for lo, hi in ((0, 30), (30, 90), (90, 180), (180, 1e9)):
        m = (R[:, 0] >= lo) & (R[:, 0] < hi)
        if m.sum():
            print(f"giro {lo:>3}-{hi if hi < 1e9 else '+':>4} graus/s: {100 * m.mean():4.1f}% das balas | erro reto mediana "
                  f"{np.median(R[m, 1]):5.0f} p90 {np.percentile(R[m, 1], 90):5.0f} | erro c/ curva mediana {np.median(R[m, 2]):5.0f} "
                  f"p90 {np.percentile(R[m, 2], 90):5.0f} px")

if __name__ == "__main__":
    main()
