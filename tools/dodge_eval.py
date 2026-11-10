"""PROJÉTEIS QUE ATINGIRAM A NAVE, incluindo os que a mecânica de ESQUIVA do jogo anulou (pedido do usuário
2026-10-01: "analise também os que passaram, mas o dodge do jogo livrou a nave do dano").

    python tools/dodge_eval.py [--desde HH:MM] [--ate HH:MM]     (padrão: todas as sessões)

Por trecho de partida (logs/frames/<ms>.jsonl + .mp4 em meia resolução, 1 a cada 3 quadros):
  - tiros com DANO: início do flash vermelho (campo "r" do registro);
  - ESQUIVAS DO JOGO: o texto "闪避" (template assets/dodge_text_half.png) aparecendo no vídeo;
  - total ATINGIDO = dano + esquivas (é o que a esquiva do bot deixou chegar na nave).
"""
import sys, os, glob, json, time
import cv2, numpy as np

TPL = cv2.cvtColor(cv2.imread(os.path.join(os.path.dirname(__file__), "..", "assets", "dodge_text_half.png")),
                   cv2.COLOR_BGR2GRAY)
THR = 0.72
GAP = 0.45            # s: o texto fica ~0.5 s na tela; eventos mais próximos que isso = o mesmo

def segment(jf):
    L = [json.loads(l) for l in open(jf) if l.strip()]
    if len(L) < 20:
        return None
    hits, last = [], -9
    for a, b in zip(L, L[1:]):
        if b["r"] and not a["r"] and b["t"] - last > 1.0:
            hits.append(b["t"]); last = b["t"]
    dodges = []
    mp4 = jf[:-6] + ".mp4"
    if os.path.exists(mp4):
        cap = cv2.VideoCapture(mp4); j = 0; lastd = -9
        # desde 2026-10-02 cada registro diz o nº do quadro no vídeo ("vf"); antes: 1 quadro a cada 3 registros
        vf = {f["vf"]: f["t"] for f in L if f.get("vf") is not None}
        while True:
            ok, f = cap.read()
            if not ok:
                break
            t = vf.get(j) if vf else (L[3 * j]["t"] if 3 * j < len(L) else None); j += 1
            if t is None:
                break
            g = cv2.cvtColor(f[500:], cv2.COLOR_BGR2GRAY)          # metade de baixo (perto da nave)
            if cv2.matchTemplate(g, TPL, cv2.TM_CCOEFF_NORMED).max() >= THR:
                if t - lastd > GAP:
                    dodges.append(t)
                lastd = t
    secs = sum(b["t"] - a["t"] for a, b in zip(L, L[1:]) if 0 < b["t"] - a["t"] < 0.5)
    return L[0]["t"], secs, hits, dodges

def hhmm(s):
    h, m = s.split(":"); return int(h) * 60 + int(m)

def main():
    a = sys.argv
    lo = hhmm(a[a.index("--desde") + 1]) if "--desde" in a else None
    hi = hhmm(a[a.index("--ate") + 1]) if "--ate" in a else None
    day = a[a.index("--dia") + 1] if "--dia" in a else None
    sys.path.insert(0, os.path.dirname(__file__))
    import hit_analysis as HA, collections
    causes_h, causes_d = collections.Counter(), collections.Counter()
    tot_s = tot_h = tot_d = 0
    for jf in sorted(glob.glob(os.path.join("logs", "frames", "*.jsonl"))):
        lt = time.localtime(int(os.path.basename(jf)[:10]))      # nome = instante do início (ms) -> filtra ANTES de decodificar
        mins = lt.tm_hour * 60 + lt.tm_min
        if (lo is not None and mins < lo) or (hi is not None and mins > hi) or (day and time.strftime("%d/%m", lt) != day):
            continue
        r = segment(jf)
        if not r:
            continue
        t0, secs, hits, dodges = r
        tot_s += secs; tot_h += len(hits); tot_d += len(dodges)
        # CAUSA de cada contato (mesma análise dos tiros, também nas esquivas do jogo)
        fr = HA.load(jf); ts = [f["t"] for f in fr]
        for kind, evs, cnt in (("h", hits, causes_h), ("d", dodges, causes_d)):
            for t in evs:
                i = int(np.searchsorted(ts, t))
                if 1 <= i < len(fr):
                    try:
                        cnt[HA.analyze_hit(fr, i)["cause"]] += 1
                    except Exception:
                        cnt["?"] += 1
    if tot_s == 0:
        print("nenhum trecho"); return
    m = tot_s / 60
    tot = tot_h + tot_d
    print(f"{m:.1f} min de jogo")
    print(f"  tiros com DANO:              {tot_h:4d}  ({tot_h / m:.2f}/min)")
    print(f"  anulados pela ESQUIVA do jogo: {tot_d:4d}  ({tot_d / m:.2f}/min)")
    print(f"  TOTAL que atingiu a nave:    {tot:4d}  ({tot / m:.2f}/min)  -> a esquiva do jogo salvou {100 * tot_d / max(1, tot):.0f}%")
    allc = causes_h + causes_d
    print("\nCAUSAS de TODOS os contatos (dano + esquiva do jogo):")
    for k, v in allc.most_common():
        print(f"  {k:16s} {v:4d}  {100 * v / max(1, sum(allc.values())):5.1f}%   (dano {causes_h[k]}, esquiva {causes_d[k]})")

if __name__ == "__main__":
    main()
