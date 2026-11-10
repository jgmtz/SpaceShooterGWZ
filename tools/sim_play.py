"""Simulador FIEL com a gravação de uma partida do usuário (tools/watch.py + tools/extract_play.py).

    python tools/sim_play.py logs/watch/<pasta> [--calibrar] [--config nome ...]

Diferente do sim_evade (que julga a colisão com as MESMAS balas que o bot vê), aqui:
  * VERDADE: cada bala é ligada entre quadros usando passado E futuro (offline), vira trajetória contínua e é
    estendida em linha reta depois da última detecção (bala que sumiu atrás da nave do jogador continua
    existindo para a nave simulada). As colisões são julgadas contra essas trajetórias a 60 Hz.
  * PERCEPÇÃO: o bot simulado só vê as balas detectadas no quadro atual (~9 quadros/s da gravação), com o
    rastreador causal dele (evade.track_velocities), inimigos/jatos como no run.py, latência e velocidade
    medidas da nave.
  * HITBOX: calibrada pelos tiros do usuário (dano + "闪避" da esquiva do jogo = contato) x balas que passaram
    perto sem contato (--calibrar).
Limitação: as balas MIRADAS miravam a nave do usuário, não a simulada.
"""
import sys, os, json, math, time, collections
import cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gwz import evade, pilot

STEP = 0.1112          # s por quadro da gravação do watch.py (medido no alinhamento)
TIMES = None           # instante de cada quadro (vídeos do PRÓPRIO bot: ritmo irregular); None = i * STEP

def _t(i):
    return TIMES[i] if TIMES is not None else i * STEP
LAT = 0.13             # s do comando até a nave começar a obedecer (pilot.LATENCY ~0.1 + quadro)
SPEED = 1100.0         # px/s da nave pilotada
SUB = 60
INVULN = 0.6           # s sem novo contato após um contato (o jogo pisca a nave)

# ---------- cache: jatos, inimigos e texto de esquiva por quadro ----------
def load_aux(d, play):
    c = os.path.join(d, "aux.npy")
    if os.path.exists(c):
        return np.load(c, allow_pickle=True).item()
    from gwz import enemies
    T = cv2.cvtColor(cv2.imread(os.path.join("assets", "dodge_text_half.png")), cv2.COLOR_BGR2GRAY)
    cap = cv2.VideoCapture(os.path.join(d, "video.mp4"))
    out, i, t0 = {}, -1, time.time()
    while True:
        ok, f = cap.read()
        if not ok:
            break
        i += 1
        p = play[i]
        if not (p["gp"] > 0.9 or p["red"] > 100):
            continue
        big = cv2.resize(f, (1080, 2340), interpolation=cv2.INTER_LINEAR)
        dodge = float(cv2.matchTemplate(cv2.cvtColor(f[500:], cv2.COLOR_BGR2GRAY), T, cv2.TM_CCOEFF_NORMED).max())
        out[i] = {"jets": enemies.detect_jets(big),
                  "foes": [(e["x"], e["y"], e["r"]) for e in enemies.detect_enemies(big)],
                  "dodge": dodge}
        if i % 1000 == 0:
            print(f"aux {i}/{len(play)} ({time.time() - t0:.0f}s)", flush=True)
    np.save(c, np.array(out, dtype=object), allow_pickle=True)
    return out

# ---------- trechos de partida ----------
def segments(play):
    """Listas de índices consecutivos de quadros de partida (corta em lacunas > 3 quadros)."""
    segs, cur, last = [], [], -9
    # quadro de partida = HUD de partida (gp > 0.9; vídeo em meia resolução nunca passa de ~0.965) E a nave do
    # jogador vista a até ~1 s (tela de SKILL dá gp ~0.966 e não tem nave; flash vermelho só perto de partida)
    import bisect
    ship_idx = [p["i"] for p in play if p["ship"]]
    def near(idx, i, k=10):
        j = bisect.bisect_left(idx, i)
        return any(0 <= jj < len(idx) and abs(idx[jj] - i) <= k for jj in (j - 1, j))
    for p in play:
        if (p["gp"] > GP_MIN or p["red"] > 100) and near(ship_idx, p["i"]):
            if (p["i"] - last > 3 or (cur and _t(p["i"]) - _t(last) > 1.0)) and cur:
                segs.append(cur); cur = []
            cur.append(p["i"]); last = p["i"]
    if cur:
        segs.append(cur)
    return [s for s in segs if len(s) >= 20]

# ---------- verdade: trajetórias ----------
def build_tracks(play, seg):
    """Liga as detecções do trecho em trajetórias. Devolve lista de arrays (n,3) [t, x, y] (ordenados)."""
    tracks = []            # cada uma: lista de (t, x, y)
    active = []            # índices em tracks ainda vivos
    for i in seg:
        t = _t(i)
        pts = [tuple(b) for b in play[i]["b"]]
        used = set()
        pairs = []
        for k in active:
            tr = tracks[k]
            tl, xl, yl = tr[-1]
            if t - tl > 2.5 * STEP:
                continue
            if len(tr) >= 2:
                vx = (tr[-1][1] - tr[-2][1]) / (tr[-1][0] - tr[-2][0]); vy = (tr[-1][2] - tr[-2][2]) / (tr[-1][0] - tr[-2][0])
                gate = 55 + 0.08 * math.hypot(vx, vy) * (t - tl) / STEP
            else:
                vx, vy = 0.0, 900.0; gate = 140
            px, py = xl + vx * (t - tl), yl + vy * (t - tl)
            for j, (x, y) in enumerate(pts):
                dd = math.hypot(x - px, y - py)
                if dd < gate and (len(tr) >= 2 or y > yl - 20):      # 1ª ligação: a bala não sobe
                    pairs.append((dd, k, j))
        pairs.sort()
        done_k = set()
        for dd, k, j in pairs:
            if k in done_k or j in used:
                continue
            tracks[k].append((t, *pts[j])); done_k.add(k); used.add(j)
        for j, p in enumerate(pts):
            if j not in used:
                tracks.append([(t, *p)]); active.append(len(tracks) - 1)
        active = [k for k in active if t - tracks[k][-1][0] <= 2.5 * STEP]
    return [np.array(tr, float) for tr in tracks if len(tr) >= 2]

ABSORB_R = 220.0   # bala que some a menos disso da nave do usuário foi absorvida (escudo/alas) ou acertou: não estende
EXTEND_S = 0.35    # bala que some LONGE da nave (falha de detecção) continua em linha reta por até isso. 0.6 -> 0.35
                   # + VMAX_TRUTH (2026-10-02): com o detector novo, no chefe da neve o caminho REAL do usuário dava 109
                   # contatos (ele levou 11 tiros); balas somem sob os raios dele e ligações erradas davam 1500-2300
                   # px/s. Calibrado nas 2 gravações longas: neve 35 (pessimista), ilhas 13 (real 15).
VMAX_TRUTH = 1500.0
GP_MIN = 0.9
MIN_SPEED = 150.0  # trajetória quase parada = falso positivo (enfeite do cenário, sprite da própria nave)
GLUED_R = 90.0     # trajetória com distância mediana à nave de quem gravou menor que isso = sprite da própria nave

def clean_tracks(tracks, S):
    """Tira as paradas e marca até quando cada uma pode ser estendida (coluna extra: t_fim_permitido)."""
    out = []
    for tr in tracks:
        dur = tr[-1, 0] - tr[0, 0]
        if dur <= 0 or not MIN_SPEED <= math.hypot(*(tr[-1, 1:] - tr[0, 1:])) / dur <= VMAX_TRUTH:
            continue
        # trajetória GRUDADA na nave de quem gravou (cockpit/motor lido como disco, anda junto com a nave): falso.
        # (2026-10-02: na neve isso dava 109 "contatos" no caminho real do usuário, que levou 11 tiros)
        ds = [math.hypot(*(p[1:] - q)) for p in tr for q in [ship_at(S, p[0])] if q is not None]
        if ds and np.median(ds) < GLUED_R:
            continue
        s = ship_at(S, tr[-1, 0])
        near = s is not None and math.hypot(*(tr[-1, 1:] - s)) < ABSORB_R
        out.append((tr, tr[-1, 0] + (0.03 if near else EXTEND_S)))
    return out

def truth_positions(tracks, t):
    """Posições (n,2) de todas as balas verdadeiras no instante t (interpola; estende após a última detecção na
    velocidade final até o limite da trajetória; antes da 1ª, só meio quadro). tracks: [(arr, t_fim)] ou [arr]."""
    return truth_positions_ids(tracks, t)[1]

def truth_positions_ids(tracks, t):
    """(ids, posições) — ids = índice da trajetória (para alinhar a mesma bala em instantes diferentes)."""
    out, ids = [], []
    for n, item in enumerate(tracks):
        tr, t_end = item if isinstance(item, tuple) else (item, item[-1, 0] + EXTEND_S)
        t0, t1 = tr[0, 0], tr[-1, 0]
        if t < t0 - 0.5 * STEP or t > t_end:
            continue
        if t <= t0:
            v = (tr[1, 1:] - tr[0, 1:]) / (tr[1, 0] - tr[0, 0]); p = tr[0, 1:] + v * (t - t0)
        elif t >= t1:
            v = (tr[-1, 1:] - tr[-2, 1:]) / (tr[-1, 0] - tr[-2, 0]); p = tr[-1, 1:] + v * (t - t1)
            if p[1] > 2420 or p[0] < -60 or p[0] > 1140:
                continue
        else:
            k = np.searchsorted(tr[:, 0], t)
            a, b = tr[k - 1], tr[k]
            p = a[1:] + (b[1:] - a[1:]) * (t - a[0]) / (b[0] - a[0])
        out.append(p); ids.append(n)
    return np.array(ids, int), np.array(out, float).reshape(-1, 2)

class Truth:
    """Posições verdadeiras pré-calculadas a 60 Hz para um trecho (rápido de consultar)."""
    def __init__(self, tracks, t_lo, t_hi):
        self.t_lo = t_lo
        self.ts = np.arange(t_lo, t_hi, 1.0 / SUB)
        self.R = [truth_positions_ids(tracks, t) for t in self.ts]
        self.P = [r[1] for r in self.R]
    def at(self, t):
        k = int(round((t - self.t_lo) * SUB))
        return self.P[k] if 0 <= k < len(self.P) else np.zeros((0, 2))
    def at_ids(self, t):
        k = int(round((t - self.t_lo) * SUB))
        return self.R[k] if 0 <= k < len(self.R) else (np.zeros(0, int), np.zeros((0, 2)))

# ---------- calibração do hitbox ----------
def user_ship(play, seg):
    pts = [(_t(i), *play[i]["ship"]) for i in seg if play[i]["ship"]]
    return np.array(pts, float).reshape(-1, 3)

def ship_at(S, t):
    if not len(S):
        return None
    k = np.searchsorted(S[:, 0], t)
    if k == 0 or k >= len(S):
        return None
    a, b = S[k - 1], S[k]
    if b[0] - a[0] > 0.5:
        return None
    return a[1:] + (b[1:] - a[1:]) * (t - a[0]) / (b[0] - a[0])

def contact_events(play, aux, seg):
    """Instantes de contato do usuário: início do flash vermelho (dano) e do texto 闪避 (esquiva do jogo)."""
    ev, last = [], -9
    prev_red = prev_dg = False
    for i in seg:
        t = _t(i)
        red = play[i]["red"] > 100
        dg = aux.get(i, {}).get("dodge", 0) >= 0.72
        if ((red and not prev_red) or (dg and not prev_dg)) and t - last > 0.45:
            ev.append((t, "dano" if red and not prev_red else "esquiva")); last = t
        prev_red, prev_dg = red, dg
    return ev

def calibrate(d, play, aux):
    near_ev, near_free = [], []
    for seg in segments(play):
        tracks = build_tracks(play, seg)
        S = user_ship(play, seg)
        evs = contact_events(play, aux, seg)
        ev_t = [t for t, _ in evs]
        for t, kind in evs:
            best = 9999
            for dt in np.arange(-0.35, 0.05, 1 / 60):
                s = ship_at(S, t + dt)
                if s is None:
                    continue
                P = truth_positions(tracks, t + dt)
                if len(P):
                    best = min(best, float(np.hypot(*(P - s).T).min()))
            near_ev.append((best, kind))
        # instantes SEM contato (longe de qualquer evento): menor distância de bala no 0.35 s seguinte
        for i in seg[::3]:
            t = _t(i)
            if any(-0.6 < t - e < 1.0 for e in ev_t):
                continue
            best = 9999
            for dt in np.arange(0, 0.35, 1 / 60):
                s = ship_at(S, t + dt)
                if s is None:
                    continue
                P = truth_positions(tracks, t + dt)
                if len(P):
                    best = min(best, float(np.hypot(*(P - s).T).min()))
            if best < 9999:
                near_free.append(best)
    e = np.array([b for b, _ in near_ev if b < 9999]); f = np.array(near_free)
    print(f"contatos do usuário: {len(near_ev)} ({sum(k == 'dano' for _, k in near_ev)} dano, "
          f"{sum(k == 'esquiva' for _, k in near_ev)} esquiva do jogo)")
    print("menor distância bala-nave no contato (px):", np.round(np.percentile(e, [10, 25, 50, 75, 90]), 0) if len(e) else "-")
    print("sem contato, menor distância em 0.35 s (px): p1/p5/p10 =", np.round(np.percentile(f, [1, 5, 10]), 0))
    for r in (40, 50, 60, 70, 80, 90, 100):
        print(f"  raio {r}: contatos explicados {np.mean(e < r) * 100:4.0f}%   janelas sem contato com bala < raio "
              f"{np.mean(f < r) * 100:4.1f}%")

# ---------- simulação ----------
def run_sim(play, aux, cfg, starts=None, verbose=False):
    """cfg: dict com 'choose'(ship, vel, prev, foes, jets) -> alvo, 'hit_r', 'lat', 'speed'. Devolve métricas."""
    hit_r = cfg.get("hit_r", 60.0)
    lat, speed = cfg.get("lat", LAT), cfg.get("speed", SPEED)
    tot_s = 0.0; contacts = []; jets_c = 0; flips = 0; ship_hits = []; aim_s = 0.0; foe_s = 0.0
    for seg_no, seg in enumerate(segments(play)):
        t_lo, t_hi = _t(seg[0]), _t(seg[-1])
        S = user_ship(play, seg)
        truth = cfg.get("_truth_cache", {}).get(seg[0])
        if truth is None:
            truth = Truth(clean_tracks(build_tracks(play, seg), S), t_lo, t_hi + STEP)
            cfg.setdefault("_truth_cache", {})[seg[0]] = truth
        st = ship_at(S, t_lo + STEP) if len(S) > 2 else None
        ship = np.array(st if st is not None else (540.0, 2000.0), float)
        target = tuple(ship); active = tuple(ship); queue = collections.deque()
        prev_vel, prev_t, foes_tr, prev_foe_t = [], None, [], None
        last_hit, last_jet = -9, -9
        state = {}
        for a, b in zip(seg, seg[1:]):
            t, tn = _t(a), _t(b)
            if tn - t > 0.5:
                continue
            dt_ = t - prev_t if prev_t and t - prev_t < 0.3 else 0
            roi = cfg.get("roi", 750)
            near_roi = cfg.get("roi_alt")              # faixa longe só em quadros alternados (como no run.py)
            state["k"] = state.get("k", 0) + 1
            perc = cfg.get("perc", play)               # PERCEPÇÃO pode vir de outra extração (ex.: detector antigo)
            if near_roi and state["k"] % 2 and dt_ > 0:
                bl = [(x, y) for x, y in perc[a]["b"] if y >= ship[1] - near_roi]
                bl += [(x + vx * dt_, y + vy * dt_) for x, y, vx, vy in prev_vel
                       if y + vy * dt_ < ship[1] - near_roi and y + vy * dt_ >= ship[1] - roi]
            else:
                bl = [(x, y) for x, y in perc[a]["b"] if y >= ship[1] - roi]
            tracked = evade.track_velocities(prev_vel, bl, dt_, store_turn=True)
            # memória de balas sumidas, igual ao run.py (evade.coast); cfg["coast"] = (idade, perto, vmin) p/ testar
            co = cfg.get("coast", (None, None, None))
            coast_prev = evade.coast(prev_vel, tracked, state.get("coast", []), dt_, tuple(ship), *co) \
                if co is not False else []
            state["coast"] = coast_prev
            state["ages"] = evade.ages_of(tracked)       # idade de cada bala rastreada (planner.young_xsv)
            vel = tracked + [(x, y, vx, vy) for x, y, vx, vy, _ in coast_prev]
            prev_vel, prev_t = tracked, t
            ax = aux.get(a, {})
            foes = ax.get("foes", []); jets = ax.get("jets", [])
            us = play[a]["ship"]           # falso positivo em cima da nave de QUEM GRAVOU (ao vivo: evade.ENEMY_SELF)
            if us:
                foes = [f for f in foes if math.hypot(f[0] - us[0], f[1] - us[1]) > 120]
                jets = [j for j in jets if math.hypot(j[0] - us[0], j[1] - us[1]) > 120]
            fdt = t - prev_foe_t if prev_foe_t else 0
            foes_tr = evade.track_velocities(foes_tr, [(x, y) for x, y, _ in foes] + list(jets),
                                             fdt if 0 < fdt < 0.4 else 0)
            prev_foe_t = t
            state["t"], state["truth"] = t, truth        # só o ORÁCULO usa a verdade (limite superior)
            state["seg_no"], state["t_in_seg"] = seg_no, t - t_lo   # p/ políticas por onda (modo defensivo)
            skip = cfg.get("skip_every")                 # imita loop mais lento: 1 de cada N quadros sem decisão
            new = target if skip and state["k"] % skip == 0 else cfg["choose"](tuple(ship), vel, target, foes, foes_tr, state)
            if abs(new[0] - target[0]) > 120:
                flips += 1
            target = new
            queue.append((t + lat, target))
            steps = max(1, int(round((tn - t) * SUB)))
            for s in range(steps):
                ts = t + (tn - t) * s / steps
                while queue and queue[0][0] <= ts:
                    active = queue.popleft()[1]
                dv = np.array(active) - ship; nd = float(np.hypot(*dv))
                if nd > pilot.DEADBAND:
                    ship += dv * min(1.0, speed / SUB / nd)
                ship[0] = min(max(ship[0], 40.0), 1040.0)
                P = truth.at(ts)
                if len(P) and ts - last_hit > INVULN and (np.hypot(*(P - ship).T) < hit_r).any():
                    last_hit = ts; contacts.append((ts, tuple(ship), seg[0]))
                for jx, jy in jets:                 # colisão com jato/inimigo (posição do quadro)
                    if math.hypot(jx - ship[0], jy - ship[1]) < 85 and ts - last_jet > INVULN:
                        last_jet = ts; jets_c += 1; ship_hits.append((ts, a, "jato", (jx, jy), tuple(ship)))
                for fx, fy, fr in foes:
                    if math.hypot(fx - ship[0], fy - ship[1]) < fr + 50 and ts - last_jet > INVULN:
                        last_jet = ts; jets_c += 1; ship_hits.append((ts, a, f"inimigo r={fr:.0f}", (fx, fy), tuple(ship)))
            # NA MIRA: algum inimigo (barra de vida) acima e alinhado (armas principais atiram reto p/ cima)
            if any(fy < ship[1] - 150 and abs(fx - ship[0]) < 70 for fx, fy, fr in foes):
                aim_s += tn - t
            if foes and any(fy < ship[1] - 150 for fx, fy, fr in foes):
                foe_s += tn - t
            tot_s += tn - t
    m = tot_s / 60
    return {"min": m, "balas": len(contacts), "balas/min": len(contacts) / m, "naves": jets_c,
            "naves/min": jets_c / m, "trocas/min": flips / m, "mira_%": 100 * aim_s / max(foe_s, 1e-9),
            "_contacts": contacts, "_ship_hits": ship_hits}

# ---------- decisores ----------
def choose_current(ship, vel, prev, foes, foes_tr, state):
    """Mesma lógica do run.py (sem drones/laser): balas + inimigos rastreados como obstáculo móvel."""
    v = vel + evade.enemy_points(foes_tr, ship)
    if not vel and not foes_tr:
        return prev
    return evade.choose_target(ship, v, t0=0.10, prev=prev, shooters=foes)

CONFIGS = {"atual": choose_current}

def main():
    d = sys.argv[1]
    play = list(np.load(os.path.join(d, "play.npy"), allow_pickle=True))
    aux = load_aux(d, play)
    if "--calibrar" in sys.argv:
        calibrate(d, play, aux); return
    names = sys.argv[sys.argv.index("--config") + 1:] if "--config" in sys.argv else list(CONFIGS)
    hit_r = float(os.environ.get("HIT_TRUE", 60))
    for n in names:
        r = run_sim(play, aux, {"choose": CONFIGS[n], "hit_r": hit_r})
        print(f"{n:28s} " + "  ".join(f"{k}={v:.2f}" for k, v in r.items() if not k.startswith("_")), flush=True)

if __name__ == "__main__":
    main()





