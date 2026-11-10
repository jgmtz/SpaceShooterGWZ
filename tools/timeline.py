"""Roda o classificador numa gravação (1 amostra/s) e imprime a linha do tempo de estados + ações.
    python tools/timeline.py gravacao.mp4
"""
import sys, os, cv2, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gwz.bot import Bot
logging.basicConfig(level=logging.WARNING)
cap = cv2.VideoCapture(sys.argv[1]); fps = cap.get(cv2.CAP_PROP_FPS) or 30
step = int(round(fps)) if fps < 200 else 30
bot = Bot(dry=True); n = 0; prev = None; start = 0
def flush(t, s, a):
    print(f"{start:6.0f}s-{t:6.0f}s  {s:14s} {a}")
last_t = 0
while True:
    ok = cap.grab()
    if not ok: break
    n += 1
    if n % step: continue
    ok, fr = cap.retrieve()
    t = n / fps if fps < 200 else n / 30
    if fr.shape[0] != 2340:
        fr = cv2.resize(fr, (1080, 2340))
    s, r = bot.step(fr)
    if s != prev:
        if prev is not None: flush(t, prev, last_a)
        prev, start = s, t
        last_a = str(r)[:70]
    last_t = t
flush(last_t, prev, last_a)
