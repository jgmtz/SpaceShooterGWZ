"""Como o JOGADOR desvia: cruza play.npy (tools/extract_play.py) com gestures.json (dedo na tela).

    python tools/analyze_play.py logs/watch/<pasta>

Responde: quantos tiros levou e em que situação (quantidade de balas, dedo na tela ou solto), quanto o jogo
desacelera com o dedo solto, onde o jogador deixa a nave, e se soltar o dedo nas "tempestades" ajudou.
"""
import sys, os, json, glob
import cv2, numpy as np

FPS = 10.0

def align(d, play, meta):
    """Tempo absoluto de cada quadro. O watch.py agenda o próximo quadro 0.1 s DEPOIS do fim do anterior (intervalo
    real ~0.11 s, erro acumula). 1º chute: relógio uniforme entre o fechamento do vídeo e `secs` antes; depois
    refina com os prints gNNN_before (tirados no toque) procurando o quadro igual numa janela de ±6 s, e
    interpola linearmente entre os pontos achados. Devolve (tempos[], pontos (quadro, t))."""
    vp = os.path.join(d, "video.mp4")
    cap = cv2.VideoCapture(vp)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (54, 117)).astype(np.float32))
    n = len(frames)
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 1200.0
    t_end = os.path.getmtime(vp)
    guess = t_end - secs + np.arange(n) * secs / n
    pts = []
    for g in meta:
        p = os.path.join(d, f"g{g['i']:03d}_before.png")
        if not os.path.exists(p):
            continue
        k0 = int(np.searchsorted(guess, g["t_down"]))
        w = range(max(0, k0 - 55), min(n, k0 + 55))
        if not len(w):
            continue
        s = cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY), (54, 117)).astype(np.float32)
        err = np.array([np.abs(frames[k] - s).mean() for k in w])
        k = w[int(np.argmin(err))]
        srt = np.sort(err)
        # só aceita casamento NÍTIDO (o melhor bem melhor que a média da janela) e só em tela de partida
        if err.min() < 6 and srt[0] < 0.6 * np.median(err) and (play[k]["gp"] > 0.9):
            pts.append((k, g["t_down"]))
    if len(pts) >= 2:
        ks, ts = zip(*sorted(pts))
        T = np.interp(np.arange(n), ks, ts, left=np.nan, right=np.nan)
        # fora dos pontos: extrapola com o passo médio
        step = (ts[-1] - ts[0]) / max(1, ks[-1] - ks[0])
        T[:ks[0]] = ts[0] - (ks[0] - np.arange(ks[0])) * step
        T[ks[-1] + 1:] = ts[-1] + (np.arange(ks[-1] + 1, n) - ks[-1]) * step
    else:
        T = guess
    return T, pts

def main():
    d = sys.argv[1]
    play = list(np.load(os.path.join(d, "play.npy"), allow_pickle=True))
    meta = json.load(open(os.path.join(d, "gestures.json")))
    T, pts = align(d, play, meta)
    if len(pts) >= 2:
        res = [T[k] - t for k, t in pts]
        print(f"alinhamento: {len(pts)} prints casados; passo real {(pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0]):.4f} s/quadro")
    else:
        print("alinhamento: só pelo relógio (poucos prints casados)")
    downs = [(g["t_down"], g["t_up"]) for g in meta]
    def finger(t):
        return any(a <= t <= b for a, b in downs)
    gp = np.array([r["gp"] > 0.9 or r["red"] > 100 for r in play])
    nb = np.array([len(r["b"]) for r in play])
    fd = np.array([finger(t) for t in T])
    red = np.array([r["red"] > 100 for r in play])
    hit = red & ~np.r_[False, red[:-1]]
    # um tiro = começo de flash, separados por >= 1 s
    hits, last = [], -9
    for k in np.nonzero(hit & gp)[0]:
        if T[k] - last > 1.0:
            hits.append(k); last = T[k]
    G = np.nonzero(gp)[0]
    print(f"partida: {len(G) / FPS / 60:.1f} min de jogo, {len(hits)} tiros levados ({len(hits) / (len(G) / FPS / 60):.2f}/min)")
    # tempestade = muitas balas na tela (limiar pelo percentil 80 das telas de jogo)
    thr = max(12, int(np.percentile(nb[G], 80)))
    storm = nb >= thr
    print(f"tempestade = >= {thr} balas na tela ({100 * storm[G].mean():.0f}% do tempo de jogo)")
    # nº de balas no tiro: máximo nos 1.0 s antes (o flash vermelho atrapalha a detecção no próprio quadro)
    def ctx(k):
        w = slice(max(0, k - 10), k)
        return int(nb[w].max() if k else 0), bool(fd[max(0, k - 3):k + 1].all()), bool((~fd[max(0, k - 10):k]).any())
    half = T[G[0]] + (T[G[-1]] - T[G[0]]) / 2
    for name, sel in (("1ª metade", lambda k: T[k] < half), ("2ª metade", lambda k: T[k] >= half)):
        hs = [k for k in hits if sel(k)]
        gs = [k for k in G if sel(k)]
        st = [k for k in gs if storm[k]]
        up_storm = np.mean([not fd[k] for k in st]) if st else float("nan")
        hs_storm = [k for k in hs if ctx(k)[0] >= thr]
        print(f"{name}: {len(gs) / FPS / 60:.1f} min, {len(hs)} tiros ({len(hs_storm)} em tempestade); "
              f"tempo de tempestade {len(st) / FPS:.0f} s, dedo SOLTO em {100 * up_storm:.0f}% dele; "
              f"tiros por min de tempestade {len(hs_storm) / max(1e-9, len(st) / FPS / 60):.2f}")
    # velocidade das balas: dedo na tela x solto (casamento pelo vizinho mais próximo ABAIXO, quadro seguinte)
    vs = {True: [], False: []}
    for k in G[:-1]:
        if not (gp[k + 1] and fd[k] == fd[k + 1]) or red[k] or red[k + 1]:
            continue
        A, B = np.array(play[k]["b"], float).reshape(-1, 2), np.array(play[k + 1]["b"], float).reshape(-1, 2)
        if len(A) < 2 or len(B) < 2:
            continue
        for x, y in A:
            dy = B[:, 1] - y; dx = np.abs(B[:, 0] - x)
            ok = (dy > 10) & (dy < 220) & (dx < 40)
            if ok.sum() == 1:
                vs[bool(fd[k])].append(dy[ok][0] * FPS)
    for k_, v in vs.items():
        if v:
            print(f"balas com dedo {'NA TELA' if k_ else 'SOLTO  '}: {len(v):5d} medidas, vy mediana {np.median(v):.0f} px/s")
    if vs[True] and vs[False]:
        print(f"-> fator de câmera lenta ~{np.median(vs[False]) / np.median(vs[True]):.2f}")
    # onde o jogador deixa a nave
    ys = [play[k]["ship"][1] for k in G if play[k]["ship"]]
    xs = [play[k]["ship"][0] for k in G if play[k]["ship"]]
    if ys:
        print(f"nave: y mediana {np.median(ys):.0f} (p10 {np.percentile(ys, 10):.0f}, p90 {np.percentile(ys, 90):.0f}); "
              f"x p10-p90 {np.percentile(xs, 10):.0f}-{np.percentile(xs, 90):.0f}")
    # quanto tempo o dedo fica solto em geral x em tempestade, e duração típica das soltas
    ups = [(a2 - b1) for (a1, b1), (a2, b2) in zip(downs, downs[1:]) if 0.1 < a2 - b1 < 10 and gp[np.searchsorted(T, b1)] if np.searchsorted(T, b1) < len(gp)]
    if ups:
        print(f"dedo solto durante a partida: {len(ups)} vezes, duração mediana {np.median(ups):.2f} s (p90 {np.percentile(ups, 90):.2f})")
    np.save(os.path.join(d, "analysis.npy"), np.array({"T": T, "thr": thr, "hits": hits}, dtype=object), allow_pickle=True)

if __name__ == "__main__":
    main()
