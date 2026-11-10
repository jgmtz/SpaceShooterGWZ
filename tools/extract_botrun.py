"""Converte uma PARTIDA DO PRÓPRIO BOT (logs/frames/<trecho>.mp4 + .jsonl com o campo "vf", desde 2026-10-02) no
formato do simulador fiel: logs/watch/bot_<nome>/ com play.npy (cada quadro com o INSTANTE real "t", a nave
que o bot viu e as balas re-detectadas pelo detector atual), video.mp4 (os quadros em sequência) e times.npy.

    python tools/extract_botrun.py <nome> <trecho1.jsonl> [<trecho2.jsonl> ...]
"""
import sys, os, json, time
import cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gwz import vision

def main():
    name, files = sys.argv[1], sys.argv[2:]
    d = os.path.join("logs", "watch", "bot_" + name)
    os.makedirs(d, exist_ok=True)
    vw = cv2.VideoWriter(os.path.join(d, "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (540, 1170))
    out, times, off, t_prev, t0 = [], [], 0.0, None, time.time()
    for jf in sorted(files):
        L = [json.loads(l) for l in open(jf) if l.strip()]
        byvf = {f["vf"]: f for f in L if f.get("vf") is not None}
        if not byvf:
            print("sem 'vf' (vídeo antigo, 1/3 dos quadros):", jf); continue
        cap = cv2.VideoCapture(jf[:-6] + ".mp4"); j = -1
        if t_prev is not None:
            off = t_prev + 5.0 - L[0]["t"]            # corte de 5 s entre trechos (o simulador separa)
        while True:
            ok, f = cap.read()
            if not ok:
                break
            j += 1
            r = byvf.get(j)
            if r is None:
                continue
            big = cv2.resize(f, (1080, 2340), interpolation=cv2.INTER_LINEAR)
            s = r["s"] if r.get("v") else None
            bl = vision.detect_bullets(big, hud_top=350, thr=0.75, scale=0.5)
            if s:
                bl = [b for b in bl if (b[0] - s[0]) ** 2 + (b[1] - s[1]) ** 2 > 60 ** 2]   # = run.py (ship_exclude)
            k = len(out)
            out.append({"i": k, "t": r["t"] + off, "gp": 1.0, "red": 200.0 if r.get("r") else 0.0,
                        "ship": tuple(s) if s else None, "b": [(round(x), round(y)) for x, y, *_ in bl],
                        "goal": r.get("g")})
            times.append(r["t"] + off)
            vw.write(f)
            if k % 500 == 0:
                print(f"{k} quadros ({time.time() - t0:.0f}s)", flush=True)
        t_prev = times[-1] if times else t_prev
    vw.release()
    np.save(os.path.join(d, "play.npy"), np.array(out, dtype=object), allow_pickle=True)
    np.save(os.path.join(d, "times.npy"), np.array(times))
    print(f"fim: {len(out)} quadros -> {d}", flush=True)

if __name__ == "__main__":
    main()
