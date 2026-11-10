"""EXPERIMENTAL — ainda não confiável (ver README). Calibração de cor por fase/estágio (substitui a suposição de cor fixa).
Ideia: em vez de fixar 'balas são vermelhas', o bot detecta o FORMATO (disco com
miolo claro + anel bem saturado, qualquer matiz) e agrupa as cores vistas nos
últimos quadros com um k-means simples em 1D (sobre o matiz/hue). Isso é o
"aprendizado simples" pedido: nada de rede neural, só um agrupamento estatístico
que se readapta sempre que a paleta da fase muda.
"""
import cv2
import numpy as np

def _white_blobs(bgr, hud_top):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    white = ((s < 90) & (v > 200)).astype(np.uint8) * 255
    white[:hud_top] = 0
    n, lab, stats, cents = cv2.connectedComponentsWithStats(white, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, hh, area = stats[i]
        if 12 <= area <= 500 and w <= 40 and hh <= 40:
            out.append((int(cents[i][0]), int(cents[i][1])))
    return out, h, s

def sample_ring_hues(bgr, hud_top=350, r0=15, r1=30, ship=None, hud_card=(0, 700, 260, 2340)):
    """Para cada 'miolo claro' candidato a bala, colhe o matiz do anel ao redor
    (só onde a saturação é alta o bastante para ser cor de bala, não água/grama).
    Exclui a vizinhança da nave (cockpit/turbina também têm miolo claro + anel saturado)
    e o cartão de recompensas do HUD esquerdo, que também confundem a amostra."""
    blobs, h, s = _white_blobs(bgr, hud_top)
    H, W = h.shape
    hues = []
    hx0, hy0, hx1, hy1 = hud_card
    for cx, cy in blobs:
        if hx0 <= cx <= hx1 and hy0 <= cy <= hy1:
            continue
        if ship is not None and (cx - ship[0]) ** 2 + (cy - ship[1]) ** 2 < 130 ** 2:
            continue
        x0, x1 = max(cx - r1, 0), min(cx + r1 + 1, W)
        y0, y1 = max(cy - r1, 0), min(cy + r1 + 1, H)
        if x1 - x0 < r1 or y1 - y0 < r1:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        ring = (d2 >= r0 * r0) & (d2 <= r1 * r1)
        seg_h, seg_s = h[y0:y1, x0:x1], s[y0:y1, x0:x1]
        sel = seg_h[ring & (seg_s > 130)]
        if len(sel) >= (ring.sum() * 0.5):     # anel precisa estar "cheio" de cor saturada
            hues.extend(sel.tolist())
    return np.array(hues, dtype=np.float32)

def _hue_dist(a, b):
    d = np.abs(a - b)
    return np.minimum(d, 180 - d)          # matiz é circular (0..179)

def kmeans_hue(hues, k=2, iters=20, seed=0):
    """k-means 1D simples em matiz circular. Devolve [(centro, contagem), ...]."""
    if len(hues) < 20:
        return []
    rng = np.random.default_rng(seed)
    centers = hues[rng.choice(len(hues), size=min(k, len(hues)), replace=False)]
    for _ in range(iters):
        d = np.stack([_hue_dist(hues, c) for c in centers], axis=1)
        assign = d.argmin(axis=1)
        new_centers = []
        for i in range(len(centers)):
            pts = hues[assign == i]
            if len(pts) == 0:
                new_centers.append(centers[i]); continue
            ang = pts / 180.0 * 2 * np.pi
            m = np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) / 2.0
            new_centers.append(m % 180)
        centers = np.array(new_centers, dtype=np.float32)
    d = np.stack([_hue_dist(hues, c) for c in centers], axis=1)
    assign = d.argmin(axis=1)
    return sorted(((float(centers[i]), int((assign == i).sum())) for i in range(len(centers))),
                  key=lambda t: -t[1])

class Palette:
    """Paleta ativa de cores de bala para a fase atual. Recalibrar ao trocar de fase
    (ex.: quando SKILL_SELECT aparece, ou a cada N segundos como salvaguarda)."""
    def __init__(self):
        self.clusters = []   # [(hue_centro, peso)]

    def calibrate(self, frames, hud_top=350, min_samples=40):
        hues = np.concatenate([sample_ring_hues(f, hud_top) for f in frames]) if frames else np.array([])
        if len(hues) < min_samples:
            return False              # poucas balas na tela; mantém a paleta anterior
        self.clusters = kmeans_hue(hues, k=2)
        return True

    def classify(self, hue):
        if not self.clusters:
            return "normal"           # sem calibração ainda: assume o único tipo padrão
        if len(self.clusters) == 1:
            return "normal"
        # o cluster mais comum = balas comuns; o outro = variante (ex. boss)
        d = [_hue_dist(np.array([hue]), c)[0] for c, _ in self.clusters]
        idx = int(np.argmin(d))
        return "normal" if idx == 0 else "variant"
