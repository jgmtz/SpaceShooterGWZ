"""Nos contatos em que a bala JÁ ERA VISÍVEL (categoria A do attack_origin), o bot previu a trajetória certa?
Compara, ~0.5 s antes do contato, a bala que o bot tinha no registro (.jsonl: posição + velocidade usada) com a
trajetória verdadeira da culpada: erro de velocidade e onde a previsão do bot punha a bala no instante do contato.

    python tools/pred_check.py logs/watch/bot_neve_v
"""
import sys, os, math, glob, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
import sim_play as SP, eval_all as EA

def main():
    d = sys.argv[1]
    play = list(np.load(os.path.join(d, "play.npy"), allow_pickle=True))
    SP.TIMES = list(np.load(os.path.join(d, "times.npy"))); dts = np.diff(SP.TIMES)
    SP.STEP = float(np.median(dts[(dts > 0) & (dts < 0.5)]))
    # registros do bot (velocidades que ele USOU), pelo instante; "t" do play já tem o deslocamento entre trechos
    logs = {}
    for f in sorted(glob.glob(os.path.join("logs", "frames", "*.jsonl"))):
        for l in open(f):
            if l.strip():
                r = json.loads(l); logs[round(r["t"], 3)] = r
    # instante do play -> registro original: o extract_botrun somou um deslocamento por trecho; reconstrói pelo "vf"
    rows = []
    for seg in SP.segments(play):
        S = SP.user_ship(play, seg)
        tracks = SP.clean_tracks(SP.build_tracks(play, seg), S)
        tr = SP.Truth(tracks, SP._t(seg[0]), SP._t(seg[-1]) + SP.STEP)
        last = -9
        for t in tr.ts:
            s = SP.ship_at(S, t)
            if s is None or t - last <= SP.INVULN:
                continue
            ids, P = tr.at_ids(t)
            if not len(P):
                continue
            dist = np.hypot(*(P - s).T)
            if dist.min() >= EA.HIT_R:
                continue
            last = t
            k = int(ids[int(np.argmin(dist))]); tk, _ = tracks[k]
            if t - tk[0, 0] < 0.5:
                continue                                  # só os de bala visível com tempo
            td = t - 0.45                                 # instante de decisão (~latência + meio horizonte)
            i = int(np.argmin(np.abs(np.array(SP.TIMES) - td)))
            truth_p = SP.truth_positions([(tk, tk[-1, 0] + SP.EXTEND_S)], SP._t(i))
            if not len(truth_p):
                continue
            tp = truth_p[0]
            # velocidade verdadeira (média) da culpada entre a decisão e o contato
            tc = SP.truth_positions([(tk, tk[-1, 0] + SP.EXTEND_S)], t)
            vt = (tc[0] - tp) / max(t - SP._t(i), 1e-3) if len(tc) else None
            # bala que o BOT tinha nesse quadro (registro "b" do play original não guarda vel.) -> usa a re-detecção
            # + o rastreador causal do simulador, igual ao ao vivo
            rows.append((t, SP._t(i), tp, vt, i))
    # rastreador causal reproduzido (mesma conta do run.py) para obter a velocidade que o bot teria
    from gwz import evade
    vel_at = {}
    prev, pt = [], None
    for idx in range(len(play)):
        t = SP._t(idx); dt_ = t - pt if pt and t - pt < 0.3 else 0
        cur = [(x, y) for x, y in play[idx]["b"]]
        prev = evade.track_velocities(prev, cur, dt_); pt = t
        vel_at[idx] = prev
    errs = []
    for t, td, tp, vt, i in rows:
        cand = vel_at.get(i, [])
        if not cand or vt is None:
            continue
        b = min(cand, key=lambda q: (q[0] - tp[0]) ** 2 + (q[1] - tp[1]) ** 2)
        dpos = math.hypot(b[0] - tp[0], b[1] - tp[1])
        if dpos > 60:
            errs.append(("NAO_ASSOCIADA", dpos, None, None)); continue
        dv = math.hypot(b[2] - vt[0], b[3] - vt[1])
        # onde a previsão LINEAR do bot punha a bala no instante do contato x onde ela estava
        miss = math.hypot(b[0] + b[2] * (t - td) - (tp[0] + vt[0] * (t - td)), b[1] + b[3] * (t - td) - (tp[1] + vt[1] * (t - td)))
        default = (b[2], b[3]) == evade.DEFAULT_V
        errs.append(("PADRAO" if default else "RASTREADA", dpos, dv, miss))
    import collections
    c = collections.Counter(e[0] for e in errs)
    print(f"{len(errs)} contatos com bala visível >= 0.5 s: " + ", ".join(f"{k} {v}" for k, v in c.items()))
    for kind in ("RASTREADA", "PADRAO"):
        sel = [e for e in errs if e[0] == kind]
        if sel:
            dv = np.array([e[2] for e in sel]); miss = np.array([e[3] for e in sel])
            print(f"  {kind}: erro de velocidade mediana {np.median(dv):.0f} px/s; erro da previsão no contato "
                  f"mediana {np.median(miss):.0f} px (p75 {np.percentile(miss, 75):.0f}); > 40 px (hitbox) em "
                  f"{100 * np.mean(miss > 40):.0f}%")

if __name__ == "__main__":
    main()
