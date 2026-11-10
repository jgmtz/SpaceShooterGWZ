"""De onde vieram os contatos do bot? (pergunta do usuário 2026-10-03: "um manual por nave aumentaria a esquiva?")

Para cada contato no caminho REAL do bot (partidas convertidas por tools/extract_botrun.py), acha a bala culpada na
verdade reconstruída (tools/sim_play.py) e classifica:
  A = bala vista >= LEAD_OK s antes do contato  -> o manual NÃO ajuda (havia tempo; problema de decisão/movimento)
  B = bala apareceu tarde, mas havia INIMIGO (barra de vida) a < ORIGIN_R px de onde ela nasceu -> ANTECIPÁVEL pelo
      atirador (o que um "manual por tipo de nave" poderia evitar)
  C = apareceu tarde, sem atirador visível perto da origem
  D = contato sem bala verdadeira perto (colisão/efeito não detectado)

    python tools/attack_origin.py [pasta ...]      (padrão: logs/watch/bot_*)
"""
import sys, os, glob, math, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
import sim_play as SP, eval_all as EA

LEAD_OK = 0.5
ORIGIN_R = 250.0

def classify(d):
    play = list(np.load(os.path.join(d, "play.npy"), allow_pickle=True))
    tf = os.path.join(d, "times.npy")
    SP.TIMES = list(np.load(tf)) if os.path.exists(tf) else None
    SP.STEP = 0.1112
    if SP.TIMES is not None:
        dts = np.diff(SP.TIMES); SP.STEP = float(np.median(dts[(dts > 0) & (dts < 0.5)]))
    aux = SP.load_aux(d, play)
    idx_at = {}   # instante -> índice do quadro (p/ achar inimigos no nascimento da bala)
    times = np.array([SP._t(i) for i in range(len(play))])
    cats = collections.Counter(); rows = []
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
            t_born, born = tk[0, 0], tk[0, 1:]
            lead = t - t_born
            if lead >= LEAD_OK:
                c = "A"
            else:
                i = int(np.argmin(np.abs(times - t_born)))
                foes = []
                for j in range(max(0, i - 3), min(len(play), i + 1)):     # inimigos vistos até ~0.3 s antes
                    us = play[j]["ship"]
                    foes += [f for f in aux.get(j, {}).get("foes", []) if not us or math.hypot(f[0] - us[0], f[1] - us[1]) > 120]
                near = [f for f in foes if math.hypot(f[0] - born[0], f[1] - born[1]) < ORIGIN_R]
                c = "B" if near else "C"
            cats[c] += 1
            rows.append((round(t, 1), c, round(lead, 2), tuple(np.round(born)), tuple(np.round(s))))
    return cats, rows

def main():
    dirs = sys.argv[1:] or sorted(glob.glob(os.path.join("logs", "watch", "bot_*")))
    tot = collections.Counter()
    for d in dirs:
        cats, rows = classify(d)
        tot += cats
        n = sum(cats.values())
        print(f"{d}: {n} contatos  " + "  ".join(f"{k}={cats[k]} ({100 * cats[k] / max(n, 1):.0f}%)" for k in "ABC"))
        for r in rows[:6]:
            print("    t=%s %s nasceu %.2fs antes em %s (nave %s)" % (r[0], r[1], r[2], r[3], r[4]))
    n = sum(tot.values())
    print(f"\nTOTAL {n} contatos: A (havia tempo) {tot['A']} = {100 * tot['A'] / max(n, 1):.0f}%   "
          f"B (tarde, atirador visível = antecipável) {tot['B']} = {100 * tot['B'] / max(n, 1):.0f}%   "
          f"C (tarde, sem atirador visto) {tot['C']} = {100 * tot['C'] / max(n, 1):.0f}%")

if __name__ == "__main__":
    main()
