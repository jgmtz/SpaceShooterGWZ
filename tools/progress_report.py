"""Relatório de EVOLUÇÃO de todas as partidas gravadas do bot (logs/frames/*.jsonl), sem celular.

    python tools/progress_report.py            -> tabela por sessão + totais por causa de tiro

Agrupa os trechos em SESSÕES (intervalo > 20 min entre trechos = sessão nova). Por sessão: minutos de jogo,
tiros (início de flash de dano), tiros/min, % do tempo com a nave perdida, quadros/s, e as causas dos tiros
(tools/hit_analysis.py). Referência: partida do usuário gravada em 2026-09-30 (1.26 tiros/min).
"""
import sys, os, glob, json, time, collections
sys.path.insert(0, os.path.dirname(__file__))
import hit_analysis as HA

GAP = 20 * 60

def load(p):
    try:
        return [json.loads(l) for l in open(p) if l.strip()]
    except Exception:
        return []

def main():
    files = sorted(glob.glob(os.path.join("logs", "frames", "*.jsonl")))
    segs = []
    for f in files:
        L = load(f)
        if len(L) >= 20:
            segs.append((L[0]["t"], L[-1]["t"], f, L))
    sessions, cur = [], []
    for s in segs:
        if cur and s[0] - cur[-1][1] > GAP:
            sessions.append(cur); cur = []
        cur.append(s)
    if cur:
        sessions.append(cur)
    rows = []
    tot_causes = collections.Counter()
    for ses in sessions:
        secs = lost = frames = 0.0
        for _, _, f, L in ses:
            for a, b in zip(L, L[1:]):
                dt = b["t"] - a["t"]
                if 0 < dt < 0.5:
                    secs += dt; lost += dt * (a["v"] == 0); frames += 1
        if secs < 30:
            continue
        txt, hits = HA.report([f for _, _, f, _ in ses])
        c = collections.Counter(h["cause"] for h in hits)
        tot_causes.update(c)
        rows.append({"ini": time.strftime("%d/%m %H:%M", time.localtime(ses[0][0])), "min": secs / 60, "tiros": len(hits),
                     "tpm": len(hits) / (secs / 60), "perdida": 100 * lost / secs, "qps": frames / secs, "c": c})
    print(f"{'sessão':12s} {'min':>5s} {'tiros':>5s} {'tiros/min':>9s} {'nave perdida':>12s} {'q/s':>5s}  causas")
    for r in rows:
        cs = ", ".join(f"{k}={v}" for k, v in r["c"].most_common())
        print(f"{r['ini']:12s} {r['min']:5.1f} {r['tiros']:5d} {r['tpm']:9.2f} {r['perdida']:11.1f}% {r['qps']:5.1f}  {cs}")
    n = sum(tot_causes.values())
    print(f"\nTOTAL: {sum(r['min'] for r in rows):.0f} min de jogo, {n} tiros")
    for k, v in tot_causes.most_common():
        print(f"  {k:16s} {v:4d}  {100 * v / n:5.1f}%")
    if len(rows) >= 2:
        a, b = rows[0], rows[-1]
        print(f"\nEVOLUÇÃO 1ª -> última sessão: tiros/min {a['tpm']:.2f} -> {b['tpm']:.2f} "
              f"({100 * (b['tpm'] - a['tpm']) / a['tpm']:+.0f}%), nave perdida {a['perdida']:.1f}% -> {b['perdida']:.1f}%")

if __name__ == "__main__":
    main()
