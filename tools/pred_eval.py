"""Qual previsão de trajetória das balas erra menos? (2026-10-03: em ~50% dos contatos com a bala VISÍVEL a tempo,
a previsão linear do bot punha a bala a > 40 px de onde ela estava — tools/pred_check.py)

Verdade = trajetórias reconstruídas (tools/sim_play.py). Em cada ponto k de cada trajetória, cada previsor só
usa as observações ATÉ k (causal, como ao vivo) e prevê a posição em t_k + H; erro = distância até a posição real.
Só conta alvos na metade de baixo da tela (y > 1300 no instante previsto), onde a nave está.

    python tools/pred_eval.py [pasta ...]
"""
import sys, os, math, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
import sim_play as SP

HS = (0.25, 0.45, 0.7)
DEFAULT_V = np.array([0.0, 850.0])

def p_smooth(obs, h, a=0.5):
    """Atual (evade.track_velocities): velocidade com suavização exponencial, previsão reta."""
    if len(obs) < 2:
        return obs[-1, 1:] + DEFAULT_V * h
    v = (obs[1, 1:] - obs[0, 1:]) / (obs[1, 0] - obs[0, 0])
    for k in range(2, len(obs)):
        m = (obs[k, 1:] - obs[k - 1, 1:]) / (obs[k, 0] - obs[k - 1, 0])
        v = a * v + (1 - a) * m
    return obs[-1, 1:] + v * h

def p_lastn(obs, h, n=4):
    """Reta ajustada (mínimos quadrados) nas últimas n observações."""
    o = obs[-n:]
    if len(o) < 2:
        return o[-1, 1:] + DEFAULT_V * h
    t = o[:, 0] - o[-1, 0]
    A = np.c_[np.ones_like(t), t]
    cx = np.linalg.lstsq(A, o[:, 1], rcond=None)[0]; cy = np.linalg.lstsq(A, o[:, 2], rcond=None)[0]
    return np.array([cx[0] + cx[1] * h, cy[0] + cy[1] * h])

def p_quad(obs, h, n=5):
    """Parábola (aceleração constante) nas últimas n observações; cai p/ reta com poucas."""
    o = obs[-n:]
    if len(o) < 4:
        return p_lastn(obs, h, 4)
    t = o[:, 0] - o[-1, 0]
    A = np.c_[np.ones_like(t), t, t * t]
    cx = np.linalg.lstsq(A, o[:, 1], rcond=None)[0]; cy = np.linalg.lstsq(A, o[:, 2], rcond=None)[0]
    return np.array([cx[0] + cx[1] * h + cx[2] * h * h, cy[0] + cy[1] * h + cy[2] * h * h])

def p_turn(obs, h, n=5):
    """Giro constante: velocidade (módulo e taxa de giro) estimadas nas últimas n observações."""
    o = obs[-n:]
    if len(o) < 3:
        return p_lastn(obs, h, 4)
    v = np.diff(o[:, 1:], axis=0) / np.diff(o[:, 0])[:, None]
    ang = np.unwrap(np.arctan2(v[:, 1], v[:, 0])); tm = (o[1:, 0] + o[:-1, 0]) / 2
    w = np.polyfit(tm, ang, 1)[0] if len(v) >= 2 else 0.0
    w = float(np.clip(w, -4.0, 4.0))                # rad/s (limite contra ruído)
    sp = float(np.mean(np.hypot(v[:, 0], v[:, 1]))); a0 = ang[-1] + w * (o[-1, 0] - tm[-1])
    p = o[-1, 1:].astype(float).copy(); steps = 10
    for s in range(steps):
        a = a0 + w * h * (s + 0.5) / steps
        p += sp * np.array([math.cos(a), math.sin(a)]) * h / steps
    return p

PREDICTORS = {
    "atual (suav. 0.5)": p_smooth,
    "suav. 0.3": lambda o, h: p_smooth(o, h, 0.3),
    "suav. 0.7": lambda o, h: p_smooth(o, h, 0.7),
    "reta 3 pontos": lambda o, h: p_lastn(o, h, 3),
    "reta 4 pontos": lambda o, h: p_lastn(o, h, 4),
    "reta 6 pontos": lambda o, h: p_lastn(o, h, 6),
    "parábola 5": p_quad,
    "parábola 7": lambda o, h: p_quad(o, h, 7),
    "giro 4": lambda o, h: p_turn(o, h, 4),
    "giro 6": lambda o, h: p_turn(o, h, 6),
}

def main():
    dirs = sys.argv[1:] or [r"logs\watch\20261002_022540", r"logs\watch\20260930_224108", r"logs\watch\bot_neve_v"]
    errs = {(name, h): [] for name in PREDICTORS for h in HS}
    for d in dirs:
        play = list(np.load(os.path.join(d, "play.npy"), allow_pickle=True))
        tf = os.path.join(d, "times.npy")
        SP.TIMES = list(np.load(tf)) if os.path.exists(tf) else None; SP.STEP = 0.1112
        if SP.TIMES is not None:
            dts = np.diff(SP.TIMES); SP.STEP = float(np.median(dts[(dts > 0) & (dts < 0.5)]))
        ntr = 0
        for seg in SP.segments(play):
            S = SP.user_ship(play, seg)
            for tr, te in SP.clean_tracks(SP.build_tracks(play, seg), S):
                if len(tr) < 4:
                    continue
                ntr += 1
                for k in range(1, len(tr) - 1):
                    obs = tr[:k + 1]
                    for h in HS:
                        tt = tr[k, 0] + h
                        if tt > tr[-1, 0]:
                            continue                              # só compara com posição OBSERVADA (sem extrapolar)
                        j = np.searchsorted(tr[:, 0], tt)
                        a, b = tr[j - 1], tr[j]
                        real = a[1:] + (b[1:] - a[1:]) * (tt - a[0]) / (b[0] - a[0])
                        if real[1] < 1300:
                            continue
                        for name, fn in PREDICTORS.items():
                            errs[(name, h)].append(float(np.hypot(*(fn(obs, h) - real))))
        print(f"{d}: {ntr} trajetórias", flush=True)
    print(f"\n{'previsor':20s} " + "  ".join(f"H={h:.2f}s: mediana / p90 / >40px" for h in HS))
    for name in PREDICTORS:
        cols = []
        for h in HS:
            e = np.array(errs[(name, h)])
            cols.append(f"{np.median(e):5.1f} / {np.percentile(e, 90):5.0f} / {100 * np.mean(e > 40):4.1f}%")
        print(f"{name:20s} " + "     ".join(cols))
    print(f"(n = {len(errs[(next(iter(PREDICTORS)), HS[1])])} previsões a 0.45 s)")

if __name__ == "__main__":
    main()
