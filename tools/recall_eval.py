"""Mede, SEM marcação manual, quantas balas o detector deixa de ver (percepção tardia), em vídeos de partida.
    python tools/recall_eval.py [video.mp4 ...]
Ideia: se uma bala é vista nos quadros k e k+1 (trilha com velocidade), ela também existia no quadro k-1,
na posição prevista (voltando pela velocidade). Se essa posição está na tela e na faixa de busca e o
detector NÃO viu nada ali em k-1 -> foi uma PERDA. Recall = vistas / (vistas + perdas), e a altura média
em que cada trilha aparece pela 1ª vez (quanto mais alto, mais cedo o bot percebe)."""
import sys, os, glob, math
import cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gwz import vision

def frames_of(path, fps_want=15, max_frames=600, start_s=0):
    cap = cv2.VideoCapture(path)
    vfps = cap.get(cv2.CAP_PROP_FPS) or 15
    step = max(1, round(vfps / fps_want))
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    i, out = 0, []
    while len(out) < max_frames:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, f = cap.retrieve()
            if not ok:
                break
            if f.shape[1] != 1080:
                f = cv2.resize(f, (1080, 2340))
            out.append((i / vfps, f))
        i += 1
    return out

def evaluate(frames, detect, gate=45.0):
    """detect(img) -> [(x, y)]. Retorna (recall, n_testes, y mediano da 1ª aparição)."""
    dets = [(t, [(x, y) for x, y, *_ in detect(img)]) for t, img in frames]
    seen = miss = 0
    first_y, covered = [], []
    evaluate.covered = covered
    for k in range(1, len(dets) - 1):
        (t0, D0), (t1, D1), (t2, D2) = dets[k - 1], dets[k], dets[k + 1]
        dt1, dt2 = t1 - t0, t2 - t1
        if not (0 < dt1 < 0.2 and 0 < dt2 < 0.2):
            continue
        for x1, y1 in D1:
            # continua no quadro seguinte (movimento de bala: desce 300-1900 px/s)
            nxt = [(x2, y2) for x2, y2 in D2 if 300 * dt2 <= (y2 - y1) <= 1900 * dt2 and abs(x2 - x1) <= 900 * dt2]
            if not nxt:
                continue
            x2, y2 = min(nxt, key=lambda p: math.hypot(p[0] - x1, p[1] - y1))
            vx, vy = (x2 - x1) / dt2, (y2 - y1) / dt2
            px, py = x1 - vx * dt1, y1 - vy * dt1          # onde estava no quadro anterior
            if py < 420 or not (30 < px < 1050):
                continue                                   # fora da área útil (HUD / fora da tela)
            if any(math.hypot(a - px, b - py) < gate for a, b in D0):
                seen += 1
            else:
                miss += 1
                first_y.append(y1)
                # perda coberta pela MEMÓRIA de 1 quadro? (bala vista em k-2 onde a trilha previa)
                if k >= 2:
                    tm, Dm = dets[k - 2]
                    qx, qy = x1 - vx * (t1 - tm), y1 - vy * (t1 - tm)
                    if 0 < t1 - tm < 0.35 and any(math.hypot(a - qx, b - qy) < gate for a, b in Dm):
                        covered.append(1)
                    else:
                        covered.append(0)
    tot = seen + miss
    return (seen / tot if tot else float("nan")), tot, (float(np.median(first_y)) if first_y else float("nan"))

def current(img):
    return vision.detect_bullets(img, hud_top=350, thr=0.75, scale=0.5)

if __name__ == "__main__":
    vids = sys.argv[1:] or sorted(glob.glob(os.path.join("logs", "frames", "*.mp4")))
    frames = []
    for v in vids:
        frames += frames_of(v)
    r, n, fy = evaluate(frames, current)
    print(f"detector atual: recall={100*r:.1f}%  testes={n}  1ª aparição (y mediano das perdas)={fy:.0f}")
