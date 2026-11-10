"""Relatório de uma sessão ao vivo do bot (para comparar configurações, ex.: com e sem --mira).
    python tools/session_report.py <pasta_da_sessão>
A pasta deve ter: run.out (saída do run.py) e trace.csv (registro por quadro). Grava relatorio.txt nela."""
import sys, os, re, csv
import numpy as np

def report(folder):
    out = open(os.path.join(folder, "run.out"), encoding="utf-8", errors="replace").read()
    rows = list(csv.DictReader(open(os.path.join(folder, "trace.csv"))))
    t = np.array([float(r["t"]) for r in rows])
    sx = np.array([float(r["nave_x"]) for r in rows])
    tx = np.array([float(r["alvo_x"]) for r in rows])
    dano = np.array([int(r["dano"]) for r in rows])
    dt = np.diff(t)
    # tempo de partida = soma dos intervalos curtos (buracos > 1 s = fora da partida)
    play = float(dt[dt < 1.0].sum())
    hits = len(re.findall(r"DANO \(acerto", out))
    deaths = len(re.findall(r"-> DEFEATED", out))
    waves = len(re.findall(r"-> GRID", out))
    side = np.sign(np.round((tx - sx) / 40))
    flips = int(((side[1:] * side[:-1]) < 0).sum())
    edge = float(((sx < 150) | (sx > 930)).mean())
    mv = np.abs(np.diff(sx))[dt < 1.0]
    lines = [
        f"sessão: {os.path.basename(os.path.abspath(folder))}",
        f"tempo em partida: {play / 60:.1f} min   (quadros: {len(rows)}, {1 / np.median(dt[dt < 1.0]):.1f} quadros/s)",
        f"acertos levados: {hits}  ->  {hits / max(play / 60, 1e-9):.2f}/min",
        f"mortes: {deaths}   entradas na montagem (waves): {waves}",
        f"trocas de lado do destino: {flips / max(play / 60, 1e-9):.1f}/min",
        f"nave perto da borda (<150 px): {100 * edge:.0f}% do tempo",
        f"deslocamento médio da nave: {mv.mean() / np.median(dt[dt < 1.0]):.0f} px/s",
    ]
    txt = "\n".join(lines)
    open(os.path.join(folder, "relatorio.txt"), "w", encoding="utf-8").write(txt + "\n")
    return txt

if __name__ == "__main__":
    print(report(sys.argv[1]))
