"""Roda a visão numa gravação de tela e gera um vídeo com caixas de debug.
    python tools/replay_vision.py gravacao.mp4 saida_debug.mp4
"""
import sys, os, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gwz.vision import detect_threats, debug_draw

src, dst = sys.argv[1], sys.argv[2]
cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS) or 30
out = None; n = 0
while True:
    ok, fr = cap.read()
    if not ok: break
    n += 1
    if n % 3: continue                        # ~1 em cada 3 quadros
    ship, bl = detect_threats(fr)
    im = cv2.resize(debug_draw(fr, ship, bl), (540, 1170))
    cv2.putText(im, f"balas={len(bl)}", (10, 1150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    if out is None:
        out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps / 3, (540, 1170))
    out.write(im)
out.release(); print("ok ->", dst)
