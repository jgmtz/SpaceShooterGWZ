"""Simulador da esquiva COM a "câmera lenta" do jogo (dedo solto = jogo mais lento).

Medido na gravação do usuário (2026-09-30, tools/analyze_play.py): balas a ~1010 px/s com o dedo na tela;
solto há <0.4 s ~945, 0.4-1 s ~628 (0.62x), >1 s ~289 (0.29x); ao recolocar o dedo volta ao normal em <0.4 s.
A nave NÃO anda com o dedo solto.

Modelo: o tempo das gravações é tempo de JOGO. A cada passo de jogo dg, o tempo REAL anda dg / ritmo. O bot
decide a cada 1/BOT_HZ s reais (vê as balas do instante de jogo atual) e o comando chega LAT s reais depois.
Assim, com o jogo lento, o atraso do bot "encolhe" em tempo de jogo — é aí que a técnica ajuda.

    python tools/sim_slow.py [video.mp4 ...]   (padrão: os mesmos vídeos do sim_evade)
"""
import sys, os, glob, math, collections
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from gwz import evade
import sim_evade as S

LAT = 0.15            # s REAIS até o jogo obedecer (toque/vídeo)
BOT_HZ = 16.0         # decisões por s REAL (loop ao vivo ~16-17 quadros/s)
SPEED = 1100.0        # px por s de JOGO da nave com o dedo na tela
HIT = S.HIT_TRUE
SUB = 120             # passos de simulação por s de jogo
INVULN = 1.0

def rate(up, since):
    """Ritmo do jogo (1 = normal) com o dedo solto há `since` s reais."""
    if not up:
        return 1.0
    if since < 0.4:
        return 1.0 - 0.15 * since / 0.4
    if since < 1.0:
        return 0.62
    return 0.29

def prepare(frames):
    """Velocidades rastreadas por quadro (como o bot faz)."""
    out, prev, pt = [], [], None
    for t, bl, *rest in frames:
        if bl is None:
            out.append((t, None)); prev, pt = [], None; continue
        vel = evade.track_velocities(prev, bl, t - pt if pt and t - pt < 0.3 else 0)
        out.append((t, np.array(vel, float).reshape(-1, 4))); prev, pt = vel, t
    return out

def simulate(F, policy, start=(540, 2000), storm_n=12, stay_r=40.0, safe_h=0.0):
    """policy: 'nunca' (dedo sempre na tela), 'parada_tempestade', 'parada_sempre'."""
    hits, last_hit, game_s, real_s, up_s = 0, -9.0, 0.0, 0.0, 0.0
    k = 0
    while k < len(F) - 1:
        # um trecho contínuo de partida
        if F[k][1] is None:
            k += 1; continue
        j = k
        while j + 1 < len(F) and F[j + 1][1] is not None:
            j += 1
        seg = F[k:j + 1]; k = j + 1
        if len(seg) < 3:
            continue
        ts = np.array([s[0] for s in seg])
        ship = np.array(start, float); target = tuple(start); prev = None
        up, since = False, 0.0
        queue = collections.deque()      # (t_real_aplicar, alvo, dedo_solto)
        active, active_up = tuple(start), False
        g, r_clock, next_dec = ts[0], 0.0, 0.0
        dg = 1.0 / SUB
        while g < ts[-1]:
            i = int(np.searchsorted(ts, g, side="right") - 1)
            B = seg[i][1]
            P = B[:, :2] + B[:, 2:4] * (g - ts[i]) if len(B) else np.zeros((0, 2))
            # decisão do bot (no relógio REAL)
            if r_clock >= next_dec:
                next_dec = r_clock + 1.0 / BOT_HZ
                vel = [tuple(b) for b in np.c_[P, B[:, 2:4]]] if len(B) else []
                seen = [b for b in vel if b[1] >= ship[1] - 750]
                tgt = evade.choose_target(tuple(ship), seen, t0=0.10, prev=prev)
                prev = tgt
                stay = math.hypot(tgt[0] - ship[0], tgt[1] - ship[1]) < stay_r
                want_up = policy != "nunca" and stay and (policy == "parada_sempre" or len(seen) >= storm_n)
                if want_up and safe_h > 0:
                    # só solta se FICAR PARADA é seguro por safe_h s (previsão): soltadas curtas não desaceleram o jogo
                    want_up = evade.risk_at(ship[0], ship[1], seen, horizon=safe_h, t0=0.0) < 1e-6 if seen else True
                queue.append((r_clock + LAT, tgt, want_up))
            while queue and queue[0][0] <= r_clock:
                _, active, a_up = queue.popleft()
                if a_up != up:
                    up, since = a_up, 0.0
                active_up = a_up
            rt = rate(up, since)
            # nave anda só com o dedo na tela
            if not up:
                d = np.array(active) - ship; nd = math.hypot(*d)
                if nd > 30:
                    ship += d * min(1.0, SPEED * dg / nd)
            # colisão
            if len(P) and g - last_hit > INVULN:
                if (((P - ship) ** 2).sum(1) < HIT ** 2).any():
                    hits += 1; last_hit = g
            g += dg
            real = dg / rt
            r_clock += real; since += real; real_s += real; game_s += dg
            if up:
                up_s += real
    return {"min_jogo": game_s / 60, "acertos": hits, "acertos/min_jogo": hits / max(1e-9, game_s / 60),
            "min_reais": real_s / 60, "%dedo_solto": 100 * up_s / max(1e-9, real_s)}

def main():
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join("logs", "watch", "*", "video.mp4")) +
                                   glob.glob(os.path.join("logs", "recordings", "*.mp4")))
    paths += [p for p in sorted(glob.glob(os.path.join("logs", "frames", "*.mp4"))) if os.path.exists(p + ".sim.npy")]
    frames = []
    for v in paths:
        if os.path.exists(v + ".sim.npy"):        # só vídeos já processados (a gravação com câmera lenta do usuário distorce o tempo)
            fr = S.load_cached(v); off = frames[-1][0] + 5.0 if frames else 0.0
            frames += [(t + off,) + tuple(rest) for t, *rest in fr]
    F = prepare(frames)
    print(f"{len(paths)} vídeos, {len(F)} quadros")
    starts = [(300, 2000), (540, 2000), (780, 2000)]
    if "--seguro" in sys.argv:
        for pol, sn, sr, sh in (("nunca", 12, 40, 0), ("parada_tempestade", 6, 80, 0), ("parada_tempestade", 6, 80, 0.8),
                                ("parada_sempre", 0, 80, 0.8), ("parada_tempestade", 6, 80, 1.2)):
            rs = [simulate(F, pol, start=s, storm_n=sn, stay_r=sr, safe_h=sh) for s in starts]
            avg = {k: np.mean([r[k] for r in rs]) for k in rs[0]}
            print(f"{pol:18s} balas>={sn:2d} parada<{sr}px seguro>={sh}s " + "  ".join(f"{k}={v:.2f}" for k, v in avg.items()), flush=True)
        return
    cfgs = [("nunca", 12, 40)] if "--sweep" not in sys.argv else []
    cfgs += [("parada_tempestade", 12, 40), ("parada_sempre", 0, 40)] if "--sweep" not in sys.argv else \
            [("parada_tempestade", 6, 40), ("parada_tempestade", 6, 80), ("parada_tempestade", 3, 80), ("parada_sempre", 0, 80)]
    for pol, sn, sr in cfgs:
        rs = [simulate(F, pol, start=s, storm_n=sn, stay_r=sr) for s in starts]
        avg = {k: np.mean([r[k] for r in rs]) for k in rs[0]}
        print(f"{pol:18s} balas>={sn:2d} parada<{sr:.0f}px " + "  ".join(f"{k}={v:.2f}" for k, v in avg.items()), flush=True)

if __name__ == "__main__":
    main()
