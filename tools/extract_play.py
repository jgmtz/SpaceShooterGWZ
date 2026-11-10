"""Extrai de uma gravação do modo observador (tools/watch.py) TUDO que serve para aprender com o jogador:
por quadro -> tempo, partida?, flash de dano, nave, balas. Junta com gestures.json (dedo na tela ou não).

    python tools/extract_play.py logs/watch/<pasta>      -> <pasta>/play.npy  (lista de dicts)
"""
import sys, os, json, time
import cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gwz import vision, states

def main():
    d = sys.argv[1]
    cap = cv2.VideoCapture(os.path.join(d, "video.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    meta = json.load(open(os.path.join(d, "gestures.json")))
    out, i, t0, ship = [], -1, time.time(), None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        i += 1
        big = cv2.resize(f, (1080, 2340), interpolation=cv2.INTER_LINEAR)
        gp = states.gameplay_score(big)
        red = vision.damage_flash(big)
        rec = {"i": i, "gp": round(float(gp), 3), "red": round(float(red), 1), "ship": None, "b": []}
        if gp > 0.9 or red > 100:
            s = vision.detect_ship(big, near=ship)
            ship = s or ship
            bl = vision.detect_bullets(big, hud_top=350, thr=0.75, scale=0.5)
            if s is not None:
                bl = [b for b in bl if (b[0] - s[0]) ** 2 + (b[1] - s[1]) ** 2 > 30 ** 2]
            rec["ship"] = s
            rec["b"] = [(round(x), round(y)) for x, y, *_ in bl]
        out.append(rec)
        if i % 500 == 0:
            print(f"{i}/{n} quadros ({time.time() - t0:.0f}s)", flush=True)
    np.save(os.path.join(d, "play.npy"), np.array(out, dtype=object), allow_pickle=True)
    print(f"fim: {len(out)} quadros, fps do vídeo {fps:.1f}, {len(meta) if isinstance(meta, list) else '?'} gestos", flush=True)

if __name__ == "__main__":
    main()
