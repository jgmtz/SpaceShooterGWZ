"""Detecção de inimigos pelas BARRAS DE VIDA (iguais para todos os tipos de nave inimiga).

Ideia combinada com o usuário (2026-09-29): as naves inimigas são muito diferentes entre si, mas todas
têm embaixo uma barra de vida em ARCO: contorno PRETO fino + preenchimento rosa-avermelhado (H ~171-175)
proporcional à vida. Achando o arco preto (sempre presente, mesmo com a vida quase no fim) sabemos onde
está cada inimigo, sem template por tipo.

Validado em quadros das gravações do usuário (estrada e ilhas): acha caças verdes/brancos/roxos/amarelos/
azul-esverdeados, inclusive com vida quase zerada; ~17 ms por quadro (meia resolução).
Limitações conhecidas:
- drones pequenos sem barra (ex.: formação vertical da estrada) NÃO são detectados;
- CHEFES geram vários falsos positivos (partes do chefe/efeitos viram "barras");
- dois inimigos colados podem virar uma barra só, larga (para a esquiva = obstáculo largo, ok);
- fora de partida (alerta de chefe, escolha de skill, derrota) aparecem falsos positivos — só usar em GAMEPLAY.
NÃO está ligado na esquiva ainda (próximo passo: obstáculo no risco + preferência de ficar embaixo = mirar).
"""
import os
import cv2
import numpy as np

TOP, BOTTOM = 250, 2250        # abaixo do HUD de cima; acima da borda de baixo

# ---------- drones pequenos (kamikaze) ----------
# Pedido do usuário (2026-09-30): o bot morria muito para os drones pequenos cinza de nariz verde e rotores
# laranja que descem NA DIREÇÃO da nave. Não têm barra de vida (detect_enemies não acha). Morrem com um tiro:
# ficar embaixo deles os derruba antes de chegarem. Template + versões giradas (vêm em diagonal).
DRONE_SCALE = 0.4
DRONE_THR = 0.6
DRONE_ANGLES = (0, -30, 30, -60, 60)
_DRONE_T = []

def _drone_templates():
    if not _DRONE_T:
        import os
        t = cv2.imread(os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "drone_small.png"))
        t = cv2.resize(t, None, fx=DRONE_SCALE, fy=DRONE_SCALE, interpolation=cv2.INTER_AREA)
        h, w = t.shape[:2]
        side = int(np.hypot(h, w)) + 2
        for a in DRONE_ANGLES:
            pad = cv2.copyMakeBorder(t, (side - h) // 2, (side - h + 1) // 2, (side - w) // 2, (side - w + 1) // 2,
                                     cv2.BORDER_REPLICATE)
            M = cv2.getRotationMatrix2D((side / 2, side / 2), a, 1.0)
            r = cv2.warpAffine(pad, M, (side, side), borderMode=cv2.BORDER_REPLICATE)
            m = int(side * 0.18)                      # tira os cantos (fundo replicado)
            _DRONE_T.append(r[m:side - m, m:side - m])
    return _DRONE_T

def detect_drones(bgr, top=350, bottom=None, ship=None):
    """Drones pequenos: [(x, y)] entre top e bottom (px). Pelos 2 ROTORES (anel amarelo-alaranjado com centro
    escuro, ~35 px de diâmetro, par a ~60-100 px): funciona com o drone em qualquer ângulo — ao vivo eles
    chegam também DE LADO, colados na nave, e o template de frente não pegava (nem girado)."""
    bottom = bottom or bgr.shape[0]
    S = 0.5
    sm = cv2.resize(bgr[top:bottom], None, fx=S, fy=S, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
    ring = cv2.inRange(hsv, (12, 110, 140), (32, 255, 255))
    n, lab, st, cen = cv2.connectedComponentsWithStats(ring)
    rotors = []
    for i, ((x, y, w, h, a), (cx, cy)) in enumerate(zip(st[1:], cen[1:]), 1):
        if not (9 <= w <= 26 and 9 <= h <= 26 and 0.2 <= a / (w * h) <= 0.75):
            continue
        # centro escuro (é um anel, não um disco/moeda cheia)
        c = hsv[int(y + h * 0.35):int(y + h * 0.65) + 1, int(x + w * 0.35):int(x + w * 0.65) + 1, 2]
        if c.size == 0 or c.mean() > 110:
            continue
        rotors.append((cx / S, cy / S + top))
    out, used = [], set()
    for i, (ax, ay) in enumerate(rotors):
        for j, (bx, by) in enumerate(rotors):
            if j <= i or i in used or j in used:
                continue
            if 55 <= np.hypot(ax - bx, ay - by) <= 110:
                x, y = (ax + bx) / 2, (ay + by) / 2
                if x < 170 and y < 900:            # card de recompensas do HUD (moedas)
                    continue
                used.update((i, j)); out.append((x, y))
    return out

def detect_drones_template(bgr, top=350, bottom=None):
    """(antigo) Drones por template de frente — não pega drone de lado. Mantido para comparação."""
    bottom = bottom or bgr.shape[0]
    sm = cv2.resize(bgr[top:bottom], None, fx=DRONE_SCALE, fy=DRONE_SCALE, interpolation=cv2.INTER_AREA)
    k = np.ones((9, 9), np.uint8)
    cands = []
    for t in _drone_templates():
        if t.shape[0] >= sm.shape[0] or t.shape[1] >= sm.shape[1]:
            continue
        r = cv2.matchTemplate(sm, t, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.nonzero((r >= DRONE_THR) & (r >= cv2.dilate(r, k)))
        for x, y in zip(xs, ys):
            cands.append((float(r[y, x]), (x + t.shape[1] / 2) / DRONE_SCALE, (y + t.shape[0] / 2) / DRONE_SCALE + top))
    out = []
    for s, x, y in sorted(cands, reverse=True):
        if all((x - a) ** 2 + (y - b) ** 2 > 60 ** 2 for a, b in out):
            out.append((x, y))
    return out

def detect_enemies(bgr):
    """Lista de inimigos: dict(x, y, r, hp, bar=(x, y, w, h)). (x, y) = centro estimado do inimigo
    (acima da barra), r = raio estimado (~metade da largura da barra), hp = fração de vida (0..1)."""
    sm = cv2.resize(bgr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    out = []
    for x, y, w, h, fill in _bars(sm, TOP // 2, BOTTOM // 2):
        X, Y, Wd, Hd = x * 2, y * 2, w * 2, h * 2
        out.append({"x": X + Wd / 2, "y": Y - 0.6 * Wd, "r": 0.5 * Wd, "hp": fill, "bar": (X, Y, Wd, Hd)})
    return out

def _bars(im, top, bottom):
    """Arcos pretos finos e longos (escala 0.5). Devolve (x, y, w, h, fração rosa)."""
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    dark = cv2.inRange(hsv, (0, 0, 0), (179, 255, 50))
    pink = cv2.inRange(hsv, (160, 120, 110), (179, 255, 255)) | cv2.inRange(hsv, (0, 120, 110), (8, 255, 255))
    bar = dark | pink
    bar[:top] = 0
    bar[bottom:] = 0
    # só trechos HORIZONTAIS longos: sombras/manchas escuras caem fora
    bar = cv2.morphologyEx(bar, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (18, 1)))
    bar = cv2.morphologyEx(bar, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(bar)
    out = []
    for i, (x, y, w, h, a) in enumerate(st[1:], 1):
        thick = a / max(w, 1)
        if not (35 <= w <= 240 and h <= 17 and 2.5 <= thick <= 7.5):
            continue
        comp = lab[y:y + h, x:x + w] == i
        fill = float((pink[y:y + h, x:x + w] > 0)[comp].mean())
        box = hsv[max(0, y - 4):y + h + 4, x:x + w].reshape(-1, 3)
        green = float(((box[:, 0] >= 40) & (box[:, 0] <= 85) & (box[:, 1] > 120) & (box[:, 2] > 120)).mean())
        white = float(((box[:, 1] < 60) & (box[:, 2] > 215)).mean())
        if green > 0.08:             # barra de vida da PRÓPRIA nave (segmentos verdes)
            continue
        if white > 0.12:             # contorno preto de número de dano / miolo branco de balas
            continue
        if x + w < 85 and y < 450:   # card de recompensas do HUD (canto esquerdo)
            continue
        out.append((x, y, w, h, fill))
    return out


# ---- JATOS ROXOS (sem barra de vida) que descem em COLUNAS e passam na altura da nave: ao vivo 2026-10-01 a
#      nave morreu na 1ª onda batendo neles (-90 cada); o detector de barras de vida não os enxerga.
#      Cinza, 1/3 da resolução, sem máscara: ~5 ms (com máscara em cor: 344 ms), mesmos 6 jatos no quadro real.
_JET = {}
JET_THR = 0.62
JET_K = 0.33

JET_FILES = ("enemy_jet.png",    # roxo/branco (deserto)
             "enemy_jet2.png",   # branco/verde com miolo ciano, em coluna (fase da NEVE, 2026-10-02: colisão sem bala)
             "enemy_jet3.png",   # caça VERMELHO com estrelas brancas, descendo (FLORESTA, 2026-10-04: colisão sem bala)
             "enemy_jet4.png")   # o jato branco/verde da neve sobre ASFALTO (nível 19, estrada, 2026-10-06: coluna
                                 # subindo pela direita, nota 0.55-0.60 com o template da neve; 1 nova em 220 prints, real)

def detect_jets(bgr, top=350, bottom=None):
    """Jatos sem barra de vida que descem em coluna (templates JET_FILES). Devolve [(x, y)] (centro, tela)."""
    if "t" not in _JET:
        _JET["t"] = []
        for fn in JET_FILES:
            t = cv2.imread(os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", fn))
            if t is not None:
                _JET["t"].append(cv2.cvtColor(cv2.resize(t, None, fx=JET_K, fy=JET_K, interpolation=cv2.INTER_AREA),
                                              cv2.COLOR_BGR2GRAY))
    bottom = bgr.shape[0] if bottom is None else bottom
    reg = cv2.cvtColor(cv2.resize(bgr[top:bottom], None, fx=JET_K, fy=JET_K, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    out = []
    for t in _JET["t"]:
        if reg.shape[0] <= t.shape[0] or reg.shape[1] <= t.shape[1]:
            continue
        r = cv2.matchTemplate(reg, t, cv2.TM_CCOEFF_NORMED)
        pk = np.argwhere((r >= JET_THR) & (r >= cv2.dilate(r, np.ones((11, 11), np.uint8))))
        for y, x in pk:
            p = (float((x + t.shape[1] / 2) / JET_K), float((y + t.shape[0] / 2) / JET_K + top))
            if all(abs(p[0] - q[0]) > 30 or abs(p[1] - q[1]) > 30 for q in out):
                out.append(p)
    return out
