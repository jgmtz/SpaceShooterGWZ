"""Gera assets/grid/*.png a partir das gravações do modo observador (tools/watch.py).
Rode de novo ao acrescentar entradas em SOURCES (novas peças/níveis/dígitos)."""
import os, sys, cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gwz.grid import cell_center, PITCH

REC = os.path.join("logs", "watch", "20260929_005807")
OUT = os.path.join("assets", "grid")
M = 24   # margem cortada de cada lado da peça (tira borda da casa, mantém o desenho)

def piece_box(c0, r0, w, h):
    x, y = cell_center(c0, r0)
    x0, y0 = x - PITCH // 2 + M, y - PITCH // 2 + M
    return x0, y0, x0 + w * PITCH - 2 * M, y0 + h * PITCH - 2 * M

# nome -> (quadro, caixa)   (c0, r0 = casa superior esquerda da peça)
SOURCES = {
    "piece_beamer_1":        ("g007_before", piece_box(1, 2, 2, 1)),
    "piece_beamer_2":        ("g007_before", piece_box(2, 4, 2, 1)),
    "piece_missile_pod_1":   ("g007_before", piece_box(3, 2, 1, 2)),
    "piece_gatling_gun_1":   ("g007_before", piece_box(1, 3, 1, 2)),
    "piece_cash_purge_1":    ("g007_before", piece_box(2, 3, 1, 1)),
    "piece_shield_pulsar_2": ("g010_before", piece_box(1, 2, 2, 1)),
    "piece_shield_pulsar_1": ("g009_before", piece_box(1, 2, 2, 1)),   # balão de nome só encosta no topo
    "piece_tool_kit_1":      ("../../grid/live_0148_final", piece_box(2, 2, 2, 1)),   # 1º teste ao vivo do bot
    "piece_gatling_gun_2":   ("../../grid/live_merge", piece_box(1, 2, 1, 2)),        # merge feito pelo bot
    # piece_boost_fuel_1: nas gravações o balão de nome sempre cobre a peça na grade -> falta recorte limpo
    "cell_empty":            ("g020_after", None),
    "cell_locked":           ("g020_after", None),
    "lvl_1":                 ("g007_before", (574 - 20, 846 - 20, 574 + 20, 846 + 20)),
    "lvl_2":                 ("g007_before", (758 - 20, 1213 - 20, 758 + 20, 1213 + 20)),
}

def main():
    os.makedirs(OUT, exist_ok=True)
    for name, (f, box) in SOURCES.items():
        im = cv2.imread(os.path.join(REC, f + ".png"))
        if name == "cell_empty":
            x, y = cell_center(1, 4); box = (x - 60, y - 60, x + 60, y + 60)
        elif name == "cell_locked":
            x, y = cell_center(0, 0); box = (x - 60, y - 60, x + 60, y + 60)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        cv2.imwrite(os.path.join(OUT, name + ".png"), im[y0:y1, x0:x1])
    # bandeja: (tipo_nível, quadro, x aproximado do círculo) — rotulado olhando as gravações
    from gwz import grid_read as R
    from gwz.grid import PIECES
    TRAY = [("beamer_1", "g000_before", 528), ("cash_purge_1", "g000_before", 735),
            ("shield_pulsar_1", "g007_before", 570), ("boost_fuel_1", "g010_before", 735),
            ("beamer_2", "g020_before", 480), ("shield_pulsar_2", "g020_before", 769),
            ("gatling_gun_1", "g028_before", 188), ("missile_pod_1", "g026_before", 961)]
    TRAY += [("tool_kit_1", "../../grid/live_0144", 914)]      # print ao vivo de 2026-09-29 01:44
    for key, f, bx in TRAY:
        im = cv2.imread(os.path.join(REC, f + ".png"))
        b = min(R.find_badges(im, R.TRAY_BOX, 38), key=lambda p: abs(p[0] - bx))
        assert abs(b[0] - bx) < 25, (key, b)
        x0, y0, x1, y1 = R.tray_box(b, PIECES[key.rpartition("_")[0]]["shape"])
        # só o miolo (tira 15% de cada lado): peças coladas na bandeja invadiam as bordas do recorte
        mx, my = int((x1 - x0) * 0.15), int((y1 - y0) * 0.15)
        cv2.imwrite(os.path.join(OUT, f"tray_{key}.png"), im[y0 + my:y1 - my, x0 + mx:x1 - mx])
    # dígitos do dinheiro (HUD da montagem): "17" em g007, "0" em g000
    for f, digits in (("g007_before", "17"), ("g000_before", "0"), ("../../grid/live_0144", "15")):
        g = cv2.cvtColor(cv2.imread(os.path.join(REC, f + ".png"))[225:290, 860:1000], cv2.COLOR_BGR2GRAY)
        n, _, st, _ = cv2.connectedComponentsWithStats((g > 200).astype(np.uint8))
        boxes = sorted((x, y, w, h) for x, y, w, h, a in st[1:] if h >= 18 and a >= 40)
        assert len(boxes) == len(digits), (f, boxes)
        for d, (x, y, w, h) in zip(digits, boxes):
            cv2.imwrite(os.path.join(OUT, f"digit_{d}.png"), cv2.cvtColor(g[y:y + h, x:x + w], cv2.COLOR_GRAY2BGR))
    print("ok:", sorted(os.listdir(OUT)))

if __name__ == "__main__":
    main()
