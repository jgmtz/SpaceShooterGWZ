"""Converte uma GRAVAÇÃO DE TELA do celular (30-60 q/s, resolução cheia) no formato do simulador fiel
(tools/sim_play.py / tools/eval_all.py): logs/watch/rec_<nome>/ com play.npy, video.mp4 (meia resolução,
~9 q/s, mesmos quadros do play.npy) e step.txt (segundos por quadro).

    python tools/extract_rec.py <video.mp4> [nome]
"""
import sys, os, time
import cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gwz import vision, states

TARGET = 0.111     # ~ o passo das gravações do watch.py

def main():
    src = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(os.path.basename(src))[0][-30:]
    d = os.path.join("logs", "watch", "rec_" + name)
    os.makedirs(d, exist_ok=True)
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    skip = max(1, round(fps * TARGET))
    step = skip / fps
    vw = cv2.VideoWriter(os.path.join(d, "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 1 / step, (540, 1170))
    out, i, k, ship, t0 = [], -1, -1, None, time.time()
    while True:
        ok = cap.grab()
        if not ok:
            break
        i += 1
        if i % skip:
            continue
        ok, big = cap.retrieve()
        if not ok:
            break
        if big.shape[:2] != (2340, 1080):
            big = cv2.resize(big, (1080, 2340))
        k += 1
        vw.write(cv2.resize(big, (540, 1170), interpolation=cv2.INTER_AREA))
        gp = states.gameplay_score(big); red = vision.damage_flash(big)
        rec = {"i": k, "gp": round(float(gp), 3), "red": round(float(red), 1), "ship": None, "b": []}
        if gp > 0.9 or red > 100:
            s = vision.detect_ship(big, near=ship); ship = s or ship
            bl = vision.detect_bullets(big, hud_top=350, thr=0.75, scale=0.5)
            if s is not None:
                bl = [b for b in bl if (b[0] - s[0]) ** 2 + (b[1] - s[1]) ** 2 > 30 ** 2]
            rec["ship"] = s; rec["b"] = [(round(x), round(y)) for x, y, *_ in bl]
        out.append(rec)
        if k % 200 == 0:
            print(f"{k} quadros ({time.time() - t0:.0f}s)", flush=True)
    vw.release()
    np.save(os.path.join(d, "play.npy"), np.array(out, dtype=object), allow_pickle=True)
    open(os.path.join(d, "step.txt"), "w").write(f"{step:.6f}")
    print(f"fim: {len(out)} quadros, passo {step:.4f} s -> {d}", flush=True)

if __name__ == "__main__":
    main()
