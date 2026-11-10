"""Teste OFFLINE completo nas fases gravadas (pedido do usuário 2026-10-02: "só testar ao vivo quando estiver
redondinho"). Para cada gravação do modo observador (logs/watch/<pasta> com play.npy):

  VOCÊ      = caminho real do jogador no simulador (validação: deve dar perto dos tiros reais dele)
  ANTIGO    = detector antigo (play_old.npy, se existir) + esquiva antiga (evade.choose_target, ROI 750)
  NOVO      = detector atual (play.npy) + planejador espaço-tempo (como o run.py roda hoje)

Todos julgados pela MESMA verdade (trajetórias reconstruídas com o detector atual). Relata contatos (bala +
nave inimiga), contatos por ONDA (cada trecho de partida entre montagens = 1 onda) e a pior RAJADA (mais contatos
em 30 s; ao vivo 5 tiros em 10 s mataram a nave). Também a escala "decide a 6/s" (loop ao vivo mais lento).

    python tools/eval_all.py [pasta ...]
"""
import sys, os, glob, math, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import sim_play as SP
from gwz import evade, planner as PL

HIT_R = 40.0
BURST_S = 30.0

def choose_old(ship, vel, prev, foes, foes_tr, state):
    v = vel + evade.enemy_points(foes_tr, ship)
    if not v:
        return prev
    return evade.choose_target(ship, v, t0=0.10, prev=prev, shooters=foes)

def choose_new(ship, vel, prev, foes, foes_tr, state):
    v = vel + evade.enemy_points(foes_tr, ship)
    if not v:
        return prev
    tgt, best, path = PL.plan(ship, PL.linear_predictor(v), t0=0.15, prev_end=state.get("end"), prev_target=prev,
                              enemies=foes)          # ofensiva (W_DPS) igual ao run.py
    state["end"] = path[-1][1:]
    return tgt

def choose_t0(ship, vel, prev, foes, foes_tr, state, t0):
    v = vel + evade.enemy_points(foes_tr, ship)
    if not v:
        return prev
    tgt, best, path = PL.plan(ship, PL.linear_predictor(v), t0=t0, prev_end=state.get("end"), prev_target=prev)
    state["end"] = path[-1][1:]
    return tgt

def user_contacts(play):
    out = []
    for seg in SP.segments(play):
        S = SP.user_ship(play, seg)
        tr = SP.Truth(SP.clean_tracks(SP.build_tracks(play, seg), S), SP._t(seg[0]), SP._t(seg[-1]) + SP.STEP)
        last = -9
        for t in tr.ts:
            s = SP.ship_at(S, t)
            if s is None:
                continue
            P = tr.at(t)
            if len(P) and t - last > SP.INVULN and (np.hypot(*(P - s).T) < HIT_R).any():
                out.append((t, tuple(s), seg[0])); last = t
    return out

def summary(name, contacts, ships, play):
    segs = SP.segments(play)
    mins = sum(len(s) for s in segs) * SP.STEP / 60
    ts = sorted(c[0] for c in contacts)
    burst = max((sum(1 for u in ts if t <= u < t + BURST_S) for t in ts), default=0)
    per = {s[0]: 0 for s in segs}
    for c in contacts:
        per[c[2]] = per.get(c[2], 0) + 1
    waves = [per[s[0]] for s in segs if len(s) * SP.STEP > 15]       # trechos curtos (<15 s) não são onda
    return (f"{name:8s} contatos={len(contacts):3d} (+{ships} naves)  {len(contacts) / mins:.2f}/min  "
            f"pior rajada={burst} em {BURST_S:.0f}s  por onda={waves}")

def main():
    dirs = sys.argv[1:] or sorted(d for d in glob.glob(os.path.join("logs", "watch", "*")) if os.path.exists(os.path.join(d, "play.npy")))
    for d in dirs:
        play = list(np.load(os.path.join(d, "play.npy"), allow_pickle=True))
        sf = os.path.join(d, "step.txt")                 # gravações de tela convertidas (tools/extract_rec.py)
        SP.STEP = float(open(sf).read()) if os.path.exists(sf) else 0.1112
        tf = os.path.join(d, "times.npy")                # partida do PRÓPRIO bot (tools/extract_botrun.py)
        bot_run = os.path.exists(tf)
        SP.TIMES = list(np.load(tf)) if bot_run else None
        if bot_run:
            dts = np.diff(SP.TIMES); SP.STEP = float(np.median(dts[(dts > 0) & (dts < 0.5)]))
        aux = SP.load_aux(d, play)
        mins = sum(len(s) for s in SP.segments(play)) * SP.STEP / 60
        real = sum(1 for a, b in zip(play, play[1:]) if b["red"] > 100 and a["red"] <= 100)
        print(f"\n=== {d}: {mins:.1f} min de partida, {len(SP.segments(play))} trechos; tiros REAIS "
              f"(flash vermelho): {real}", flush=True)
        print(summary("BOT REAL" if bot_run else "VOCÊ", user_contacts(play), 0, play), flush=True)
        if bot_run:
            # qual LATÊNCIA reproduz o que aconteceu ao vivo? (o simulador assumia 0.13 s)
            # medido nos registros desta partida (2026-10-02): destino -> nave andar 40 px = 0.225 s (mediana);
            # velocidade com balas na tela ~920 px/s (o simulador usava 0.13 s e 1100 px/s)
            cache = {}
            for lat, spd in ((0.13, 1100), (0.20, 950), (0.30, 950)):
                for pt0 in sorted({0.15, lat}):
                    fn = lambda s, v, p, f, ft, st, pt0=pt0: choose_t0(s, v, p, f, ft, st, pt0)
                    r = SP.run_sim(play, aux, {"choose": fn, "hit_r": HIT_R, "roi": 1500, "lat": lat, "speed": spd,
                                               "_truth_cache": cache})
                    print(summary(f"lat{lat}/v{spd}/t0{pt0}", r["_contacts"], r["naves"], play), flush=True)
            continue
        cache = {}
        old_p = os.path.join(d, "play_old.npy")
        if os.path.exists(old_p):
            perc = list(np.load(old_p, allow_pickle=True))
            r = SP.run_sim(play, aux, {"choose": choose_old, "hit_r": HIT_R, "roi": 750, "perc": perc, "_truth_cache": cache})
            print(summary("ANTIGO", r["_contacts"], r["naves"], play), flush=True)
        for label, skip in (("NOVO", None), ("NOVO 6/s", 3)):
            r = SP.run_sim(play, aux, {"choose": choose_new, "hit_r": HIT_R, "roi": 1500, "skip_every": skip, "_truth_cache": cache})
            print(summary(label, r["_contacts"], r["naves"], play), flush=True)

if __name__ == "__main__":
    main()


