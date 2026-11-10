"""Leitura da tela de montagem (GRID): casas, peças (tipo + nível), bandeja e dinheiro.

Método (medido nas gravações de 2026-09-29, logs/watch/20260929_005807):
- Cada peça tem um círculo amarelo com o nível no canto inferior direito: ~52 px na grade e ~38 px
  na bandeja (lá as peças aparecem ~0.73x menores). Na grade o círculo fica em (+34, +36) do centro
  da casa INFERIOR DIREITA da peça.
- Tipo: template matching do recorte da peça contra assets/grid/<tipo>_<nível>.png (gerados das
  gravações). Sem template bom -> tipo "unknown" (o recorte vai para logs/grid/unknown/ p/ rotular).
- Dinheiro: dígitos brancos do HUD vs assets/grid/digit_<d>.png. Dígito sem template -> None
  (o bot então NÃO usa Refresh).
"""
import os, glob, time
import cv2, numpy as np
from .grid import COLS, ROWS, PITCH, X0, Y0, PIECES, cell_center

_A = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "grid")
TRAY_SCALE = 0.73
BADGE_OFF = (34, 36)                 # círculo em relação ao centro da casa inferior direita
CASH_BOX = (860, 225, 1000, 290)     # HUD do dinheiro na montagem
TRAY_BOX = (0, 1900, 1080, 2125)     # bandeja (abaixo do quadro "Temporary slots"; vai até a borda direita)
TYPE_THR = 0.75
DIGIT_THR = 0.80

_T = {}

def _lv(s):
    """Nível a partir do fim da chave do template (aceita variantes: "1b" -> 1)."""
    d = "".join(ch for ch in s if ch.isdigit())
    return int(d) if d else None
AUTO = os.path.join(_A, "auto")      # templates aprendidos sozinhos pelo bot (grid_exec.harvest)

def _tpls(prefix):
    if prefix not in _T:
        _T[prefix] = {}
        for d in (AUTO, _A):             # os feitos à mão (_A) têm prioridade sobre os automáticos
            for p in glob.glob(os.path.join(d, prefix + "*.png")):
                _T[prefix][os.path.basename(p)[len(prefix):-4]] = cv2.imread(p)
    return _T[prefix]

def piece_box(cells):
    """Caixa do template piece_* para uma peça que ocupa `cells` (mesma conta de make_grid_assets)."""
    cs = [c for c, _ in cells]; rs = [r for _, r in cells]
    x0 = X0 + min(cs) * PITCH - PITCH // 2 + MARGIN
    y0 = Y0 + min(rs) * PITCH - PITCH // 2 + MARGIN
    return x0, y0, X0 + max(cs) * PITCH + PITCH // 2 - MARGIN, Y0 + max(rs) * PITCH + PITCH // 2 - MARGIN

def learn_piece(bgr, kind, level, cells):
    """Salva o recorte da peça como template automático, se ainda não existir um para (tipo, nível)."""
    key = f"{kind}_{level}"
    if key in _tpls("piece_"):
        return False
    os.makedirs(AUTO, exist_ok=True)
    x0, y0, x1, y1 = piece_box(cells)
    cv2.imwrite(os.path.join(AUTO, f"piece_{key}.png"), bgr[y0:y1, x0:x1])
    _T.pop("piece_", None)
    return True

# ---------- círculos de nível ----------
def find_badges(bgr, box, size):
    """Centros (x, y) dos círculos amarelos de diâmetro ~size dentro de box=(x0,y0,x1,y1)."""
    x0, y0, x1, y1 = box
    hsv = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    # círculo: H 23-24, S ~205, V >= 245 (medido). Faixa estreita: o verde-amarelado do Shield nível 2
    # (H 32-47) e o laranja do Beamer nível 2 grudavam no círculo e deslocavam o centro.
    m = cv2.inRange(hsv, (20, 165, 225), (28, 245, 255))
    n, _, st, cen = cv2.connectedComponentsWithStats(m)
    v = hsv[:, :, 2]
    out = []
    for (x, y, w, h, a), (cx, cy) in zip(st[1:], cen[1:]):
        # altura pode vir cortada (círculo da bandeja parcialmente atrás dos botões)
        if not (0.8 * size <= w <= 1.2 * size and 0.6 * size <= h <= 1.2 * size and 0.4 <= a / (w * h) <= 0.85):
            continue
        # tem que ter o dígito ESCURO no meio (a luz amarela do Boost Fuel também é um disco amarelo)
        k = max(3, w // 6)
        mid = v[y + h // 2 - k:y + h // 2 + k, x + w // 2 - k:x + w // 2 + k]
        if mid.size and np.percentile(mid, 20) < 150:
            out.append((float(x + w / 2) + x0, float(y + h / 2) + y0) if h < 0.9 * w else (float(cx) + x0, float(cy) + y0))
    return out

def badge_level(bgr, c, size):
    """Nível = dígito escuro dentro do círculo, por template (assets/grid/lvl_<n>.png)."""
    # janela maior que o círculo + busca por translação: a peça às vezes cobre o topo do círculo e
    # desloca o centro calculado em ~6 px (visto no Shield Pulsar nível 2)
    tsz = int(round(size * 40 / 52))          # os templates lvl_ foram recortados com 40 px (círculo de 52)
    r = tsz // 2 + 10
    x, y = int(c[0]), int(c[1])
    crop = cv2.cvtColor(bgr[max(0, y - r):y + r, max(0, x - r):x + r], cv2.COLOR_BGR2GRAY)
    best = (0.0, None)
    for k, t in _tpls("lvl_").items():
        for f in (1.0, 0.88, 0.78):          # círculo encolhe junto com as peças da bandeja cheia
            ts = cv2.resize(cv2.cvtColor(t, cv2.COLOR_BGR2GRAY), (int(tsz * f), int(tsz * f)))
            if ts.shape[0] > crop.shape[0] or ts.shape[1] > crop.shape[1]:
                continue
            s = float(cv2.matchTemplate(crop, ts, cv2.TM_CCOEFF_NORMED).max())
            if s > best[0]:
                best = (s, int(k))
    return best[1] if best[0] >= 0.6 else None

# ---------- aparência (para merge sem depender do nome do tipo) ----------
PATCH = 56        # meia-largura (px, escala da grade) do miolo comparado

def _badge_w(bgr, c, size):
    """Largura real do círculo de nível (a bandeja encolhe peças e círculos quando está cheia)."""
    x, y, r = int(c[0]), int(c[1]), int(size)
    hsv = cv2.cvtColor(bgr[max(0, y - r):y + r, max(0, x - r):x + r], cv2.COLOR_BGR2HSV)
    xs = np.nonzero(cv2.inRange(hsv, (20, 165, 225), (28, 245, 255)).any(axis=0))[0]
    return float(xs.max() - xs.min() + 1) if len(xs) else float(size)

def appearance(bgr, badge, scale):
    """Miolo da casa onde fica o círculo, normalizado p/ a escala da grade (2*PATCH px). scale = tamanho
    da peça em relação à grade (1.0 na grade; ~0.53-0.73 na bandeja)."""
    cx, cy = badge[0] - BADGE_OFF[0] * scale, badge[1] - BADGE_OFF[1] * scale
    r = PATCH * scale
    x0, y0, x1, y1 = int(cx - r), int(cy - r), int(cx + r), int(cy + r)
    if x0 < 0 or y0 < 0 or x1 > bgr.shape[1] or y1 > bgr.shape[0]:
        return None
    return cv2.resize(bgr[y0:y1, x0:x1], (2 * PATCH, 2 * PATCH), interpolation=cv2.INTER_AREA)

def similar(a, b):
    """Semelhança (-1..1) entre duas aparências (correlação em cor, tolerante a pequeno deslocamento)."""
    if a is None or b is None:
        return -1.0
    m = 8
    return float(cv2.matchTemplate(a, b[m:-m, m:-m], cv2.TM_CCOEFF_NORMED).max())

# ---------- casas ----------
def cell_states(bgr):
    """{(col,row): 'locked' | 'empty' | 'occupied'} comparando com os templates de casa vazia / com X."""
    # Estatística de cor em vez de template: a casa vazia é quase lisa e a correlação oscilava
    # (0.70 ao vivo x 0.86-1.0 nas gravações). Medido: vazia = cinza BGR ~(95,88,84), desvio ~5;
    # com X = escura; ocupada = alto contraste.
    locked = _tpls("cell_")["locked"]
    out = {}
    for c in range(COLS):
        for r in range(ROWS):
            x, y = cell_center(c, r)
            mid = bgr[y - 50:y + 50, x - 50:x + 50]
            mean, std = float(mid.mean()), float(mid.std())
            chroma = float(np.abs(mid.astype(np.int16) - mid.mean(axis=2, keepdims=True)).mean())
            if std < 14 and 65 <= mean <= 140 and chroma < 12:
                out[(c, r)] = "empty"
                continue
            sl = float(cv2.matchTemplate(bgr[y - 70:y + 70, x - 70:x + 70], locked, cv2.TM_CCOEFF_NORMED).max())
            out[(c, r)] = "locked" if sl >= 0.7 else "occupied"
    return out

# ---------- tipo da peça ----------
def _identify(region, scale):
    """Melhor (tipo, nível_template, score) para o recorte region."""
    best = ("unknown", None, 0.0)
    for key, t in _tpls("piece_").items():
        if scale != 1.0:
            t = cv2.resize(t, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if t.shape[0] > region.shape[0] or t.shape[1] > region.shape[1]:
            continue
        s = float(cv2.matchTemplate(region, t, cv2.TM_CCOEFF_NORMED).max())
        if s > best[2]:
            kind, _, lv = key.rpartition("_")
            best = (kind, _lv(lv), s)
    return best if best[2] >= TYPE_THR else ("unknown", None, best[2])

MARGIN = 24   # mesma margem usada em tools/make_grid_assets.py (definida antes, usada em piece_box)
SLACK = 14    # px de folga na busca ancorada

def _identify_anchored(bgr, c, r):
    """Testa cada template NA posição que a peça ocuparia se sua casa inferior direita fosse (c, r).
    (Busca livre numa região larga confundia com a peça vizinha.)"""
    best = ("unknown", None, 0.0)
    for key, t in _tpls("piece_").items():
        kind, _, lv = key.rpartition("_")
        w, h = PIECES.get(kind, {}).get("shape", (1, 1))
        x0 = X0 + (c - w + 1) * PITCH - PITCH // 2 + MARGIN - SLACK
        y0 = Y0 + (r - h + 1) * PITCH - PITCH // 2 + MARGIN - SLACK
        if x0 < 0 or y0 < 0:
            continue
        region = bgr[y0:y0 + t.shape[0] + 2 * SLACK, x0:x0 + t.shape[1] + 2 * SLACK]
        if region.shape[0] < t.shape[0] or region.shape[1] < t.shape[1]:
            continue
        s = float(cv2.matchTemplate(region, t, cv2.TM_CCOEFF_NORMED).max())
        if s > best[2]:
            best = (kind, _lv(lv), s)
    return best if best[2] >= TYPE_THR else ("unknown", None, best[2])

def grid_pieces(bgr):
    """Peças na grade: [{type, level, cells:[(c,r)...], badge:(x,y), score}]."""
    out = []
    for b in find_badges(bgr, (max(0, X0 - PITCH // 2), Y0 - PITCH // 2, X0 + COLS * PITCH, Y0 + ROWS * PITCH), 52):
        c = round((b[0] - BADGE_OFF[0] - X0) / PITCH); r = round((b[1] - BADGE_OFF[1] - Y0) / PITCH)
        kind, tpl_lv, score = _identify_anchored(bgr, c, r)
        w, h = PIECES.get(kind, {}).get("shape", (1, 1))
        cells = [(c - i, r - j) for i in range(w) for j in range(h)]
        if kind == "unknown":
            _save_unknown("piece", bgr[max(0, int(b[1]) - 3 * PITCH):int(b[1]) + 30, max(0, int(b[0]) - 2 * PITCH):int(b[0]) + 30])
        # nível: o do TEMPLATE quando o tipo foi reconhecido (cada nível tem visual próprio: nv2 camuflado, nv3 azul);
        # a leitura do dígito só vale p/ peça desconhecida (antes o 3 era lido como 2 -> merge errado)
        lv = tpl_lv if kind != "unknown" and tpl_lv else badge_level(bgr, b, 52)
        out.append({"type": kind, "level": lv, "cells": cells, "badge": b, "score": round(score, 2),
                    "look": appearance(bgr, b, 1.0)})
    return out

TRAY_CELL = round(PITCH * TRAY_SCALE)   # ~134 px

def tray_box(badge, shape):
    """Caixa da peça na bandeja a partir do círculo (canto inferior direito). Medido: Shield (2x1) ocupa
    x -189..+31, y -89..+36 do círculo; Boost Fuel (1x2) x -85..+35, y -217..+13 (base atrás do botão)."""
    bx, by = int(badge[0]), int(badge[1])
    w, h = shape
    return bx - w * TRAY_CELL + 60, by - h * TRAY_CELL + 45, bx + 20, by + 12

def _tray_offset(shape):
    """Canto sup. esquerdo do template tray_* em relação ao círculo (mesma conta de make_grid_assets)."""
    x0, y0, x1, y1 = tray_box((0, 0), shape)
    return x0 + int((x1 - x0) * 0.15), y0 + int((y1 - y0) * 0.15)

TRAY_SCALES = (1.0, 0.9, 0.8, 0.72)   # a bandeja ENCOLHE as peças quando tem muitas (3 peças: Shield
                                      # ~220 px de largura; 4-5 peças: ~175 px)
WORK = 0.5                            # busca em meia resolução (720 ms -> dezenas de ms)

def _tray_candidates(bgr):
    """Busca livre (multi-escala) de cada template na faixa da bandeja; cada pico prevê onde estaria
    o círculo. (Recorte ancorado no círculo falhava com peças coladas e com a bandeja encolhida.)"""
    x0, y0, x1, y1 = TRAY_BOX
    oy = max(0, y0 - 260)
    band = cv2.resize(bgr[oy:y1, x0:x1], None, fx=WORK, fy=WORK, interpolation=cv2.INTER_AREA)
    cands = []
    for key, t in _tpls("tray_").items():
        kind, _, lv = key.rpartition("_")
        dx, dy = _tray_offset(PIECES.get(kind, {}).get("shape", (1, 1)))
        for s in TRAY_SCALES:
            ts = cv2.resize(t, None, fx=WORK * s, fy=WORK * s, interpolation=cv2.INTER_AREA)
            if ts.shape[0] > band.shape[0] or ts.shape[1] > band.shape[1]:
                continue
            res = cv2.matchTemplate(band, ts, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.nonzero((res >= TYPE_THR - 0.1) & (res >= cv2.dilate(res, np.ones((9, 9), np.uint8))))
            for x, y in zip(xs, ys):
                cands.append((float(res[y, x]), kind, _lv(lv), (x / WORK + x0 - dx * s, y / WORK + oy - dy * s)))
    return cands

def _identify_tray(cands, badge, tol=30):
    best = ("unknown", None, 0.0)
    for s, kind, lv, (bx, by) in cands:
        if abs(bx - badge[0]) <= tol and abs(by - badge[1]) <= tol and s > best[2]:
            best = (kind, lv, s)
    return best if best[2] >= TYPE_THR else ("unknown", None, best[2])

def tray_blobs(bgr):
    """Objetos na bandeja achados pela FORMA (partes claras sobre o fundo escuro, brilho ~35): pega peça cujo
    círculo de nível ficou atrás dos botões (peça comprida em pé — visto ao vivo 2026-10-01: o bot leu 'só 1
    arma' com 2 na bandeja). Devolve [(cx, cy, x0, y0, x1, y1)] em coordenadas da tela."""
    x0, y0, x1, y1 = 0, 1860, 1080, 2100     # para ANTES da borda branca dos botões (ela ligava todas as peças)
    hsv = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 0, 130), (179, 255, 255))
    m[:1990 - y0, 880 - x0:] = 0            # quadro "Temporary slots" (texto branco) no canto direito
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((21, 9), np.uint8))   # junta partes da MESMA peça (vertical)
    n, _, st, cen = cv2.connectedComponentsWithStats(m)
    out = []
    for (x, y, w, h, a), (cx, cy) in zip(st[1:], cen[1:]):
        if a >= 1800 and w >= 40 and h >= 45:
            out.append((float(cx + x0), float(cy + y0), x + x0, y + y0, x + w + x0, y + h + y0))
    return out

SLOT_PITCH = 132.0     # distância entre quadrados de uma mesma peça de expansão na bandeja (medido: 132-147)

def _slot_squares(bgr):
    """Quadrados cinza LISOS (~120 px) da bandeja. O de baixo pode vir cortado pelos botões (altura >= 70)."""
    x0, y0, x1, y1 = 0, 1700, 1080, 2100
    hsv = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    m = cv2.morphologyEx(cv2.inRange(hsv, (0, 0, 80), (179, 60, 125)), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(m)
    out = []
    for (x, y, w, h, a), (cx, cy) in zip(st[1:], cen[1:]):
        if 90 <= w <= 155 and 70 <= h <= 155 and a >= 0.75 * w * h:
            patch = bgr[y0 + y + 8:y0 + y + h - 8, x0 + x + 8:x0 + x + w - 8]
            if patch.size and float(patch.std()) < 15:
                # centro do quadrado INTEIRO (se cortado embaixo, o topo é confiável)
                out.append((float(x + w / 2 + x0), float(y + 60 + y0)))
    return out

def tray_slot_pieces(bgr):
    """Peças de EXPANSÃO da grade (pedido do usuário 2026-10-01: existem 1x1, 2x1, 1x2 e em L de 3 casas).
    Quadrados vizinhos (lado a lado ou empilhados, ~132-147 px) formam UMA peça. Devolve
    [{"cells": [(dc, dr), ...] relativo ao canto sup. esquerdo, "grab": (x, y) do quadrado agarrado,
      "grab_rel": (dc, dr) desse quadrado}]."""
    sq = _slot_squares(bgr)
    parent = list(range(len(sq)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(len(sq)):
        for j in range(i + 1, len(sq)):
            dx, dy = abs(sq[i][0] - sq[j][0]), abs(sq[i][1] - sq[j][1])
            if (100 <= dx <= 180 and dy < 40) or (100 <= dy <= 180 and dx < 40):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(sq)):
        groups.setdefault(find(i), []).append(sq[i])
    out = []
    for g in groups.values():
        xm, ym = min(p[0] for p in g), min(p[1] for p in g)
        rel = [(int(round((p[0] - xm) / SLOT_PITCH)), int(round((p[1] - ym) / SLOT_PITCH))) for p in g]
        # agarra pelo quadrado mais ALTO (o de baixo pode estar atrás dos botões)
        k = min(range(len(g)), key=lambda i: (g[i][1], g[i][0]))
        out.append({"cells": rel, "grab": g[k], "grab_rel": rel[k]})
    return out

def tray_slots(bgr):
    """(compatibilidade) centros das peças de expansão."""
    return [p["grab"] for p in tray_slot_pieces(bgr)]

AD_THR = 0.75

def tray_ad_icons(bgr):
    """Ícones de CLAQUETE na bandeja = peça premium que só sai assistindo ANÚNCIO (cápsula de vidro).
    Ao vivo (2026-10-01) o bot arrastou uma e abriu anúncios, travando a fase. Template: assets/claq_tray.png
    (1.0 na peça com anúncio x <=0.51 em todos os outros prints)."""
    from .states import _t
    t = _t("claq_tray")
    # começa em 1700: cápsula ALTA (canhão comprido) tem o ícone em y~1787 (ao vivo escapou com início em 1780)
    x0, y0, x1, y1 = 0, 1700, 1080, 2125
    reg = bgr[y0:y1, x0:x1]
    out = []
    for s in (0.7, 0.85, 1.0, 1.15):
        ts = cv2.resize(t, None, fx=s, fy=s)
        r = cv2.matchTemplate(reg, ts, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.nonzero((r >= AD_THR) & (r >= cv2.dilate(r, np.ones((21, 21), np.uint8))))
        for x, y in zip(xs, ys):
            p = (x + x0 + ts.shape[1] / 2, y + y0 + ts.shape[0] / 2)
            if all(abs(p[0] - q[0]) > 40 for q in out):
                out.append(p)
    return out

def capsule_boxes(bgr):
    """Contorno BRANCO fino da cápsula de vidro (peça premium): 3ª camada contra anúncio, independe do ícone.
    Colunas verticais brancas longas + topo arredondado. Devolve [(x0, x1)] das cápsulas."""
    x0, y0, y1 = 0, 1700, 2090
    g = cv2.cvtColor(bgr[y0:y1], cv2.COLOR_BGR2HSV)
    white = cv2.inRange(g, (0, 0, 205), (179, 45, 255))
    col = cv2.morphologyEx(white, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 90)))
    xs = np.nonzero(col.any(axis=0))[0]
    if len(xs) == 0:
        return []
    groups, cur = [], [xs[0]]
    for x in xs[1:]:
        if x - cur[-1] <= 3:
            cur.append(x)
        else:
            groups.append(int(np.mean(cur))); cur = [x]
    groups.append(int(np.mean(cur)))
    out = []
    for i in range(len(groups) - 1):
        a, b = groups[i], groups[i + 1]
        if 90 <= b - a <= 260:                    # par de laterais da cápsula
            out.append((a, b))
    return out

def is_ad_piece(pos, icons, caps=()):
    """Peça com anúncio: abaixo/à direita do ícone de claquete OU dentro de uma cápsula de vidro."""
    if any(ix - 60 <= pos[0] <= ix + 240 and pos[1] >= iy - 20 for ix, iy in icons):
        return True
    return any(a - 20 <= pos[0] <= b + 20 for a, b in caps)

def tray_pieces(bgr):
    """Peças na bandeja: [{type, level, pos:(x,y) p/ agarrar, badge, score}]. NUNCA inclui peça com anúncio."""
    icons, caps = tray_ad_icons(bgr), capsule_boxes(bgr)
    pieces = _tray_pieces_all(bgr)
    return [p for p in pieces if not is_ad_piece(p["pos"], icons, caps)]

def _tray_pieces_all(bgr):
    out = _tray_by_badge(bgr)
    for sp in tray_slot_pieces(bgr):
        sx, sy = sp["grab"]
        out.append({"type": "slot", "level": None, "pos": (sx, sy), "badge": (sx, sy), "score": 1.0,
                    "look": None, "no_badge": True, "slot_cells": sp["cells"], "grab_rel": sp["grab_rel"]})
    cands = None
    # peças SEM círculo lido: objeto na bandeja que não contém nenhuma peça já achada pelo círculo
    for cx, cy, bx0, by0, bx1, by1 in tray_blobs(bgr):
        if any(bx0 - 30 <= p["badge"][0] <= bx1 + 30 and by0 - 30 <= p["badge"][1] <= by1 + 30 for p in out):
            continue
        # tenta o TIPO pelos templates da bandeja cujo círculo previsto cai dentro do objeto
        cands = _tray_candidates(bgr) if cands is None else cands
        best = max(((s, k, lv) for s, k, lv, (px, py) in cands
                    if bx0 - 40 <= px <= bx1 + 40 and by0 - 40 <= py <= by1 + 60), default=None)
        kind, lv, sc = ("unknown", None, 0.0) if best is None or best[0] < TYPE_THR else (best[1], best[2], best[0])
        p = {"type": kind, "level": lv, "pos": (cx, cy), "badge": (cx, cy), "score": round(sc, 2),
             "look": None, "no_badge": True, "box": (bx0, by0, bx1, by1)}
        if kind == "unknown":
            # formato estimado pelo TAMANHO na bandeja (~100-110 px por casa): evita tentar pôr peça de 2 casas
            # numa casa só (ao vivo: peça comprida mirada na casa de expansão)
            p["shape"] = (max(1, round((bx1 - bx0) / 112)), max(1, round((by1 - by0) / 112)))
        out.append(p)
    return out

def _tray_by_badge(bgr):
    out = []
    cands = _tray_candidates(bgr)
    for b in find_badges(bgr, TRAY_BOX, 38):
        kind, _, score = _identify_tray(cands, b)
        x0, y0, x1, y1 = tray_box(b, PIECES.get(kind, {}).get("shape", (1, 1)))
        if kind == "unknown":
            _save_unknown("tray", bgr[max(0, int(b[1]) - 250):int(b[1]) + 30, max(0, int(b[0]) - 240):int(b[0]) + 30])
        out.append({"type": kind, "level": badge_level(bgr, b, 38), "pos": ((x0 + x1) / 2, (y0 + y1) / 2),
                    "badge": b, "score": round(score, 2), "look": appearance(bgr, b, _badge_w(bgr, b, 38) / 52.0)})
    return out

# ---------- prévia verde durante o arrasto ----------
def preview_cells(bgr):
    """Casas pintadas de verde (destino da peça sendo arrastada). Medido no canto sup. esquerdo da casa:
    prévia H~46 S~180 V~142; casa vazia H~107 S~33."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    out = set()
    for c in range(COLS):
        for r in range(ROWS):
            x, y = cell_center(c, r)
            # 4 cantos: a própria peça arrastada cobre alguns (visto ao vivo: asa da Gatling no canto sup. esq.)
            green = 0
            for dx, dy in ((-60, -60), (60, -60), (-60, 60), (60, 60)):
                h, s, v = np.median(hsv[y + dy - 10:y + dy + 10, x + dx - 10:x + dx + 10].reshape(-1, 3), axis=0)
                green += 38 <= h <= 56 and s >= 120 and v >= 90
            if green >= 2:
                out.add((c, r))
    return out

# ---------- dinheiro ----------
def read_cash(bgr):
    """Valor do dinheiro da partida, ou None se algum dígito não for reconhecido com segurança."""
    x0, y0, x1, y1 = CASH_BOX
    g = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    m = (g > 200).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    boxes = sorted((x, y, w, h) for x, y, w, h, a in st[1:] if h >= 18 and a >= 40)
    if not boxes:
        return None
    digits = _tpls("digit_")
    val = ""
    for x, y, w, h in boxes:
        crop = g[y:y + h, x:x + w]
        scores = []
        for d, t in digits.items():
            t = cv2.resize(cv2.cvtColor(t, cv2.COLOR_BGR2GRAY), (w, h))
            scores.append((float(cv2.matchTemplate(crop, t, cv2.TM_CCOEFF_NORMED)[0, 0]), d))
        scores.sort(reverse=True)
        # 2,3,4,6,8,9 são sintéticos (fonte Malgun Gothic Bold, ~0.92 de semelhança com os reais):
        # exige margem sobre o 2º colocado para não confundir; na dúvida -> None (sem Refresh)
        if scores[0][0] < DIGIT_THR or (len(scores) > 1 and scores[0][0] - scores[1][0] < 0.06):
            _save_unknown("digit", crop)
            return None
        val += scores[0][1]
    return int(val)

def _save_unknown(kind, crop):
    d = os.path.join("logs", "grid", "unknown")
    os.makedirs(d, exist_ok=True)
    cv2.imwrite(os.path.join(d, f"{kind}_{int(time.time() * 1000)}.png"), crop)

PLAY_V = 204.0      # brilho (V) normal do botão Play amarelo (medido em 5 prints da montagem)

def normalize_brightness(bgr):
    """Tela escurecida (fade ao abrir a montagem: botão Play com V=160 em vez de 204) quebrava os limites
    fixos de brilho (círculo de nível, dígitos) e a leitura vinha vazia. Clareia pela referência do Play."""
    v = float(cv2.cvtColor(bgr[2130:2210, 780:960], cv2.COLOR_BGR2HSV)[:, :, 2].mean())
    if v < 0.92 * PLAY_V and v > 60:
        return cv2.convertScaleAbs(bgr, alpha=PLAY_V / v)
    return bgr

# ---------- seta verde de merge ----------
def merge_arrows(bgr):
    """Seta verde dupla que o JOGO desenha em cima da peça da bandeja que pode ser mesclada com uma da grade
    (conferido em 2026-09-30: aparece só nas peças com par). Duas divisas ~44x30 px, uma ~22 px abaixo da
    outra; as notas verdes do cash não formam esse par. Devolve [(x, y_topo)]."""
    y0 = 1650
    hsv = cv2.cvtColor(bgr[y0:2110], cv2.COLOR_BGR2HSV)
    n, _, st, cen = cv2.connectedComponentsWithStats(cv2.inRange(hsv, (45, 150, 170), (75, 255, 255)))
    ch = [(cen[i][0], cen[i][1]) for i in range(1, n) if 36 <= st[i][2] <= 56 and 22 <= st[i][3] <= 38 and st[i][4] >= 520]
    return [(a[0], a[1] + y0 - 15) for a in ch for b in ch if abs(a[0] - b[0]) < 6 and 16 <= b[1] - a[1] <= 30]

ARROW_MATCH = 0.62   # medido: par certo 0.65-0.95; sem par <= 0.49

def locate_in_grid(bgr, box):
    """Onde a peça da bandeja (recorte `box`) aparece na grade: (score, (x, y) centro). Serve p/ peça cujo círculo
    de nível ficou atrás dos botões (ao vivo 2026-09-30: canhão comprido nv1 com par na grade e sem merge)."""
    gx0, gy0 = X0 - PITCH // 2, Y0 - PITCH // 2
    k = 0.5                                                   # meia resolução (2.2 s -> ~0.2 s)
    hay = cv2.resize(bgr[gy0:gy0 + ROWS * PITCH, gx0:gx0 + COLS * PITCH], None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    bx0, by0, bx1, by1 = (int(v) for v in box)
    t = bgr[max(0, by0):by1, max(0, bx0):bx1]
    best = (-1.0, None)
    if t.size == 0:
        return best
    for s in np.arange(0.52, 0.92, 0.04):
        ts = cv2.resize(t, None, fx=k / s, fy=k / s, interpolation=cv2.INTER_AREA)
        if ts.shape[0] > hay.shape[0] or ts.shape[1] > hay.shape[1]:
            continue
        _, v, _, loc = cv2.minMaxLoc(cv2.matchTemplate(hay, ts, cv2.TM_CCOEFF_NORMED))
        if v > best[0]:
            best = (float(v), (gx0 + (loc[0] + ts.shape[1] / 2) / k, gy0 + (loc[1] + ts.shape[0] / 2) / k))
    return best

def _arrow_merges(bgr, grid, tray):
    """Marca peças da bandeja com seta de merge; a sem nível lido herda tipo/nível/aparência da peça da grade
    onde ela casa (o planejador então faz o merge normal)."""
    arrows = merge_arrows(bgr)
    for t in tray:
        if t["type"] == "slot":
            continue
        px, py = t["pos"]
        bx0, by0, bx1, by1 = t.get("box") or (px - 110, py - 200, px + 110, py + 60)
        t["arrow"] = any(bx0 - 30 <= ax <= bx1 + 30 and by0 - 140 <= ay <= min(by1, py) for ax, ay in arrows)
        if not t["arrow"]:
            continue
        # par já lido na grade (mesmo tipo reconhecido e nível) -> o planejador resolve sem busca
        if t["type"] != "unknown" and t["level"] and any(g["type"] == t["type"] and g["level"] == t["level"] for g in grid):
            continue
        box = t.get("box") or tray_box(t["badge"], PIECES.get(t["type"], {}).get("shape", (1, 1)))
        v, c = locate_in_grid(bgr, box)
        if v < ARROW_MATCH or c is None:
            continue
        col, row = round((c[0] - X0) / PITCH), round((c[1] - Y0) / PITCH)
        t["arrow_cell"] = (col, row)       # onde o par está na grade (usado se o par não foi lido: notas voando)
        if t["level"] or "box" not in t:
            continue
        # dona da casa; peça desconhecida só registra a casa do círculo (embaixo) -> mesma coluna, logo abaixo
        own = [g for g in grid if (col, row) in g["cells"]] or \
              sorted((g for g in grid if (col, row + 1) in g["cells"] or (col, row + 2) in g["cells"]),
                     key=lambda g: max(r for _, r in g["cells"]))
        if own and own[0]["level"]:
            g = own[0]
            t.update(type=g["type"], level=g["level"], look=g["look"], arrow_match=round(v, 2))

def read(bgr):
    """Leitura completa da montagem."""
    bgr = normalize_brightness(bgr)
    grid, tray = grid_pieces(bgr), tray_pieces(bgr)
    _arrow_merges(bgr, grid, tray)
    return {"cash": read_cash(bgr), "cells": cell_states(bgr), "grid": grid, "tray": tray}
