"""Simulador offline da esquiva (NÃO gasta vidas): balas reais das gravações + nave simulada.

    python tools/sim_evade.py [video.mp4 ...]      (padrão: todos os logs/watch/*/video.mp4)

Para cada quadro de partida da gravação: detecta as balas (visão real do bot), estima velocidades,
e a nave SIMULADA decide com evade.choose_target. A nave obedece com a latência e a velocidade medidas no
aparelho (pilot: ~0.15 s, ~1100 px/s, zona morta). Entre quadros as balas andam em linha reta (60 Hz).
Colisão = bala a menos de HIT_TRUE px do centro da nave; depois 1 s de invencibilidade.

Limitações: as balas miravam a nave do jogador que gravou, não a simulada; vídeo em meia resolução a
~10 quadros/s (o bot vê ~17/s em resolução cheia). Serve para COMPARAR configurações, não para prever
o número exato de acertos.
"""
import sys, os, glob, math, time, collections
import cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gwz import vision, evade, states

HIT_TRUE = 80.0        # hitbox "real" usada para contar acertos (medida em logs/hits)
LAT = 0.15             # s até a nave começar a obedecer
SPEED = 1100.0         # px/s da nave
SUB = 60               # passos de simulação por segundo
INVULN = 1.0

def load_frames(paths):
    """[(t, balas[(x,y,r,k)]), ...] só dos trechos de partida; cache em logs/sim_cache.npz."""
    out = []
    for v in paths:
        cap = cv2.VideoCapture(v)
        vfps = cap.get(cv2.CAP_PROP_FPS) or 10
        skip = max(1, round(vfps / 15))      # gravações do celular (30-60 fps) -> ~15 quadros/s, como o bot
        fps = vfps / skip
        i = -1
        while True:
            ok = cap.grab()
            if not ok:
                break
            i += 1
            if i % skip:
                continue
            ok, f = cap.retrieve()
            if not ok:
                break
            big = cv2.resize(f, (1080, 2340), interpolation=cv2.INTER_LINEAR)
            if states.gameplay_score(big) > 0.9 or vision.damage_flash(big) > 100:
                real = vision.detect_ship(big)
                bl = vision.detect_bullets(big, hud_top=350, thr=0.75, scale=0.5)
                if real is not None:   # tira o falso positivo do miolo da nave de quem gravou
                    bl = [b for b in bl if (b[0] - real[0]) ** 2 + (b[1] - real[1]) ** 2 > 30 ** 2]
                out.append((len(out) and out[-1][0] + 1.0 / fps or 0.0, bl, v, i))
            elif out and out[-1][1] is not None:
                out.append((out[-1][0] + 5.0, None, v, i))      # marca de corte entre trechos
        if out and out[-1][1] is not None:
            out.append((out[-1][0] + 5.0, None, v, i))          # corte entre vídeos diferentes
    return out

def load_cached(v):
    """Quadros de um vídeo, com cache ao lado (<video>.sim.npy) para não reprocessar."""
    c = v + ".sim.npy"
    if os.path.exists(c) and os.path.getmtime(c) >= os.path.getmtime(v):
        return list(np.load(c, allow_pickle=True))
    t = time.time(); fr = load_frames([v])
    np.save(c, np.array(fr, dtype=object), allow_pickle=True)
    print(f"{os.path.basename(v)}: {len(fr)} quadros de partida em {time.time() - t:.0f}s", flush=True)
    return fr

def load_enemies(v, frames):
    """Inimigos (gwz.enemies) de cada quadro de partida, cache <video>.en.npy. Devolve {índice: [(x,y,r)]}."""
    c = v + ".en.npy"
    if os.path.exists(c) and os.path.getmtime(c) >= os.path.getmtime(v):
        return np.load(c, allow_pickle=True).item()
    from gwz import enemies
    want = sorted({f[3] for f in frames if f[1] is not None and f[2] == v})
    cap, out, i, t0 = cv2.VideoCapture(v), {}, -1, time.time()
    for w in want:
        while i < w:
            cap.grab(); i += 1
        ok, f = cap.retrieve()
        if ok:
            big = cv2.resize(f, (1080, 2340), interpolation=cv2.INTER_LINEAR)
            out[w] = [(e["x"], e["y"], e["r"]) for e in enemies.detect_enemies(big)]
    np.save(c, np.array(out, dtype=object), allow_pickle=True)
    print(f"{os.path.basename(v)}: inimigos em {len(out)} quadros ({time.time() - t0:.0f}s)", flush=True)
    return out

def attach_enemies(frames):
    """Acrescenta a lista de inimigos a cada quadro: (t, balas, video, i, inimigos)."""
    cache = {}
    out = []
    for f in frames:
        if f[1] is None:
            out.append(tuple(f[:4]) + (None,)); continue
        if f[2] not in cache:
            cache[f[2]] = load_enemies(f[2], frames)
        out.append(tuple(f[:4]) + (cache[f[2]].get(f[3], []),))
    return out

def simulate(frames, choose, start=(540, 2000), deadband=15.0, roi=None, every=1):
    """Roda a nave simulada. choose(ship, vel, prev[, inimigos]) -> alvo. Retorna métricas.
    roi: a DECISÃO só vê balas até roi px acima da nave (como run.ROI_ABOVE); colisões usam todas.
    every: usa 1 de cada `every` quadros para decidir (imita o bot mais lento; ~15/every quadros/s).
    Com inimigos nos quadros (attach_enemies): mede contatos com inimigo e % do tempo MIRANDO
    (inimigo acima da nave, alinhado horizontalmente)."""
    if every > 1:
        frames = [f for i, f in enumerate(frames) if i % every == 0 or f[1] is None]
    ship = np.array(start, float); target = tuple(start)
    queue = collections.deque()            # (t_aplicar, alvo): latência do canal
    active = tuple(start)
    hits, last_hit, dist, flips, prev_side = 0, -9, 0.0, 0, 0
    prev_bl, prev_t, secs = [], None, 0.0
    contacts, last_contact, aim_secs = 0, -9, 0.0
    import inspect
    wants_enemies = len(inspect.signature(choose).parameters) >= 4
    for k in range(len(frames) - 1):
        t, bl, *rest = frames[k]
        t_next, bl_next, *_ = frames[k + 1]
        en = rest[2] if len(rest) >= 3 and rest[2] is not None else []
        if bl is None or bl_next is None:
            prev_bl, prev_t = [], None
            ship[:] = start; queue.clear(); active = tuple(start); continue
        dt = t_next - t
        vel = evade.track_velocities(prev_bl, bl, t - prev_t if prev_t and t - prev_t < 0.3 else 0, store_turn=True)
        prev_bl, prev_t = vel, t
        seen = vel if roi is None else [b for b in vel if b[1] >= ship[1] - roi]
        target = choose(tuple(ship), seen, target, en) if wants_enemies else choose(tuple(ship), seen, target)
        side = np.sign(round((target[0] - ship[0]) / 40))
        if side and prev_side and side != prev_side:
            flips += 1
        if side:
            prev_side = side
        queue.append((t + LAT, target))
        B = np.array([(x, y, vx, vy) for x, y, vx, vy in vel], float).reshape(-1, 4)
        steps = max(1, int(round(dt * SUB)))
        for s in range(steps):
            ts = t + dt * s / steps
            while queue and queue[0][0] <= ts:
                active = queue.popleft()[1]
            d = np.array(active) - ship
            nd = np.hypot(*d)
            if nd > deadband:
                mv = d * min(1.0, SPEED / SUB / nd)
                ship += mv; dist += np.hypot(*mv)
            if len(B):
                P = B[:, :2] + B[:, 2:] * (ts - t)
                if (np.hypot(P[:, 0] - ship[0], P[:, 1] - ship[1]) < HIT_TRUE).any() and ts - last_hit > INVULN:
                    hits += 1; last_hit = ts
            for ex, ey, er in en:
                if math.hypot(ex - ship[0], ey - ship[1]) < er + 60 and ts - last_contact > INVULN:
                    contacts += 1; last_contact = ts
        # mirando: algum inimigo ACIMA da nave e alinhado (as armas atiram para cima)
        if any(ey < ship[1] - 100 and abs(ex - ship[0]) < max(er, 60) for ex, ey, er in en):
            aim_secs += dt
        secs += dt
    return {"min": secs / 60, "acertos": hits, "acertos/min": hits / max(secs / 60, 1e-9),
            "trocas_de_lado/min": flips / max(secs / 60, 1e-9), "px_andados/s": dist / max(secs, 1e-9),
            "contatos_inimigo/min": contacts / max(secs / 60, 1e-9), "mirando_%": 100 * aim_secs / max(secs, 1e-9)}

def main():
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join("logs", "watch", "*", "video.mp4"))
                                   + glob.glob(os.path.join("logs", "recordings", "*.mp4")))
    frames = []
    for v in paths:
        fr = load_cached(v)
        off = frames[-1][0] + 5.0 if frames else 0.0          # emenda os vídeos numa linha do tempo só
        frames += [(t + off,) + tuple(rest) for t, *rest in fr]
    n_b = sum(len(f[1]) for f in frames if f[1] is not None)
    print(f"quadros: {len(frames)}, balas detectadas: {n_b}")
    import run
    from gwz import pilot
    configs = {
        "parada (controle)": lambda s, v, p: s,
        "esquiva atual": lambda s, v, p: evade.choose_target(s, v, t0=run.REACT, prev=p),
        "esquiva sem histerese": lambda s, v, p: evade.choose_target(s, v, t0=run.REACT),
    }
    starts = [(300, 2000), (420, 2000), (540, 2000), (660, 2000), (780, 2000)]   # média: 1 largada é ruidosa
    for name, fn in configs.items():
        rs = [simulate(frames, fn, start=st, deadband=pilot.DEADBAND) for st in starts]
        avg = {k: np.mean([r[k] for r in rs]) for k in rs[0]}
        print(f"{name:26s} " + "  ".join(f"{k}={v:.2f}" for k, v in avg.items()))

if __name__ == "__main__":
    main()
