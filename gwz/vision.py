"""Visão computacional do Galaxy Wing Zero (tela 1080x2340 retrato).
Todas as funções recebem um frame BGR (numpy) e devolvem dados simples.
"""
import cv2
import numpy as np

# ---------- balas inimigas ----------
# Sprites recortados de gravações reais (assets/bullet_*.png). Template matching no
# recorte central (60%) para reduzir influência do fundo; NMS para remover duplicatas.
import os, math, time   # math/time: detect_comets e o filtro de movimento dos cometas
_ASSETS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
_TPL = {}

FLAME_TIGHT = True
COMET_ANGLES = (-90, -60, -30, 0, 30)   # em relação ao recorte (que já cai ~30° p/ a direita)

def _low(kind):
    """Famílias buscadas em METADE da resolução de busca (sprites grandes, continuam nítidos reduzidos)."""
    return kind.startswith("needle") or kind.startswith("comet")
FLAME_ANGLES = (-30, -12, 12, 30)  # ±30 (2026-10-04): chama inclinada do tiro 3 da floresta 18:12 só aparece com
                                   # ela; 120 prints de tiro: +1 detecção, 0 perdidas, +4 ms/quadro

def _rotated(t, ang, inner=None):
    """Template t girado de `ang` graus; recorta a caixa justa do objeto girado (inner=(h,w) do objeto
    sem fundo, ou o próprio t)."""
    h, w = t.shape[:2]; side = int(np.hypot(h, w)) + 2
    canvas = cv2.copyMakeBorder(t, (side - h) // 2, (side - h + 1) // 2, (side - w) // 2, (side - w + 1) // 2,
                                cv2.BORDER_REPLICATE)
    M = cv2.getRotationMatrix2D((side / 2, side / 2), ang, 1.0)
    r = cv2.warpAffine(canvas, M, (side, side), borderMode=cv2.BORDER_REPLICATE)
    ih, iw = inner or (h, w)
    c, s = abs(np.cos(np.radians(ang))), abs(np.sin(np.radians(ang)))
    hh, ww = int(ih * c + iw * s), int(iw * c + ih * s)
    y0, x0 = max(0, (side - hh) // 2), max(0, (side - ww) // 2)
    return r[y0:y0 + hh, x0:x0 + ww]

def _load_tpl():
    if _TPL:
        return
    for kind, fn in (("normal", "bullet_normal.png"), ("boss", "bullet_boss.png")):
        t = cv2.imread(os.path.join(_ASSETS, fn))
        h, w = t.shape[:2]
        m = int(min(h, w) * 0.2)
        _TPL[kind] = t[m:h - m, m:w - m]
    # 'flame': míssil/chama alongada (naves azuis, fase da estrada) — visto em
    # Screen_Recording_20260928_225604. Sem corte de margem: é fino e alongado,
    # cortar bordas jogaria fora a maior parte do sprite.
    flame = os.path.join(_ASSETS, "bullet_flame.png")
    if os.path.exists(flame):
        _TPL["flame"] = cv2.imread(flame)
        # "flame_t": a mesma chama com POUCO fundo (corte 25% x / 20% y). Fase da FLORESTA (ao vivo 2026-10-02): rajadas
        # de chamas coladas em COLUNA -> a vizinha invadia a margem do template e a nota caía 0.93 -> 0.68 (< 0.75);
        # a nave morreu dentro de colunas que o bot não via. Com o corte: 0.88 na coluna, 0.97 isolada.
        if FLAME_TIGHT:
            t = _TPL["flame"]; h, w = t.shape[:2]
            _TPL["flame_t"] = t[int(h * 0.2):h - int(h * 0.2), int(w * 0.25):w - int(w * 0.25)]
            # colunas INCLINADAS (leque do chefe da floresta): gira a chama inteira e recorta a caixa justa
            th, tw = _TPL["flame_t"].shape[:2]
            for ang in FLAME_ANGLES:
                _TPL[f"flame_t{ang:+d}"] = _rotated(t, ang, (th, tw))
            # o "flame" original fica redundante (flame_t dá 0.97 na chama isolada): em 5 quadros de rajada a
            # combinação sem ele e só com ±12 acha 359/359 balas da referência completa, 175 -> 130 ms/quadro
            del _TPL["flame"]
    # "flame2": chama do DESERTO (miolo amarelo-branco embaixo + 2 pontas vermelhas), ao vivo 2026-10-01 acertou a
    # nave (-216) sem ser vista: nenhum template casava
    flame2 = os.path.join(_ASSETS, "bullet_flame2.png")
    if os.path.exists(flame2):
        _TPL["flame2"] = cv2.imread(flame2)
    # "needle": AGULHA vermelho-alaranjada fina e vertical, em CHUVA (fileiras). Ao vivo 2026-10-01: nos contatos com a
    # nave (dano + esquiva do jogo), 52% foram "bala não vista" e várias eram essas agulhas (recortada do vídeo)
    needle = os.path.join(_ASSETS, "bullet_needle.png")
    if os.path.exists(needle):
        _TPL["needle"] = cv2.imread(needle)
        # agulhas INCLINADAS (vêm na diagonal; ao vivo 2026-10-01 as verticais eram vistas e as tortas não)
        t = _TPL["needle"]; h, w = t.shape[:2]; side = int(np.hypot(h, w)) + 2
        canvas = cv2.copyMakeBorder(t, (side - h) // 2, (side - h + 1) // 2, (side - w) // 2, (side - w + 1) // 2,
                                    cv2.BORDER_REPLICATE)
        for ang in NEEDLE_ANGLES:
            M = cv2.getRotationMatrix2D((side / 2, side / 2), ang, 1.0)
            r = cv2.warpAffine(canvas, M, (side, side), borderMode=cv2.BORDER_REPLICATE)
            # recorta a caixa justa da agulha girada (menos fundo = menos falso)
            hh = int(abs(h * np.cos(np.radians(ang))) + abs(w * np.sin(np.radians(ang))))
            ww = int(abs(w * np.cos(np.radians(ang))) + abs(h * np.sin(np.radians(ang))))
            y0, x0 = max(0, (side - hh) // 2), max(0, (side - ww) // 2)
            _TPL[f"needle{ang:+d}"] = r[y0:y0 + hh, x0:x0 + ww]
        # a agulha VERTICAL ficou redundante com o flame_t (2026-10-02: 3377 balas de referência, 0 perdidas sem
        # ela; -5 ms/quadro). As inclinadas ficam (sem as ±20 perderia 22).
        if FLAME_TIGHT:
            del _TPL["needle"]
    # "swirl": disco ROSA com espiral branca do CHEFE do deserto (2026-10-01: no chefe 70% dos contatos foram
    # "bala não vista"; esse disco não casava com o "boss" das ilhas)
    swirl = os.path.join(_ASSETS, "bullet_swirl.png")
    if os.path.exists(swirl):
        _TPL["swirl"] = cv2.imread(swirl)
    # "comet": COMETA (cabeça branco-amarela + cauda vermelha em V) caindo na diagonal, em ângulos variados. Ao vivo
    # 2026-10-02 (floresta) acertou a nave 2x sem ser visto; nos prints de tiro do deserto/estrada há vários.
    # O recorte cai a ~30° p/ a direita; as rotações cobrem quedas de ~-60° a +60°. Família em 1/4 de resolução.
    comet = os.path.join(_ASSETS, "bullet_comet.png")
    if os.path.exists(comet):
        t = cv2.imread(comet)
        for ang in COMET_ANGLES:
            _TPL[f"comet{ang:+d}"] = t if ang == 0 else _rotated(t, ang)

_TPL_SCALED = {}
_STRIP = 256         # altura (px, já na escala reduzida) de cada faixa de busca paralela. 96 -> 256 (2026-10-04, com
                     # lotes + cinza): estrada 78 -> 66 ms, prints pesados 98 -> 81 ms, mesmas balas
_POOL = None
NEEDLE_ANGLES = (-40, -20, 20, 40)   # rotações extras do template da agulha
_BATCH = True         # lotes (1 por thread). Sem ganho em 2026-09-30/10-01 (tarefas coloridas, pesadas); com a busca em
                      # cinza (tarefas leves, o despacho pesa) 2026-10-04: estrada 98 -> 80 ms, prints pesados 114 -> 97
PARALLEL_SHIP = True  # nave e balas em paralelo no detect_threats (quando há posição anterior)

def _pool():
    global _POOL
    if _POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _POOL = ThreadPoolExecutor(max_workers=max(2, min(8, (os.cpu_count() or 4) - 2)))
    return _POOL

# CANAL do template matching (2026-10-04, velocidade): None = cor (BGR, 3 canais); "gray"/"r"/"s"/"rg" = 1 canal
# (~3.4x mais rápido: matchTemplate era ~890 ms de CPU por quadro somando as threads). Com canal, a detecção é em
# DUAS ETAPAS (candidatos no canal com CH_THR + confirmação colorida). 166 quadros / 3404 balas de referência:
# cor 148 ms; cinza 0.70 (agulhas na cor) 110 ms, perde 4 (0.1%), 0 falsas; cinza 0.65 125 ms, perde 0;
# saturação 0.70 95 ms mas perde 42 (discos, chamas, cometas). Canal sozinho sem confirmação = milhares de falsas.
CHANNEL = "gray"

def to_channel(img, ch="cur"):
    ch = CHANNEL if ch == "cur" else ch
    if ch is None:
        return img
    if ch == "gray":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if ch == "r":
        return np.ascontiguousarray(img[..., 2])
    if ch == "s":
        return np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1])
    if ch == "rg":                                  # "vermelhidão": R - G (balas vermelhas/laranjas com miolo claro)
        return cv2.subtract(img[..., 2], img[..., 1] // 2)
    raise ValueError(ch)

def _templates(scale, ch="cur"):
    _load_tpl()
    ch = CHANNEL if ch == "cur" else ch
    key = (scale, ch)
    if key not in _TPL_SCALED:
        _TPL_SCALED[key] = {k: to_channel(t if scale == 1.0 else cv2.resize(t, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), ch)
                            for k, t in _TPL.items()}
    return _TPL_SCALED[key]

CH_THR = 0.70        # limiar da 1ª etapa (1 canal); a 2ª etapa confirma com o template COLORIDO no limiar normal
COLOR_FAMILIES = ("needle",)   # buscadas direto na cor (no cinza a agulha perdia 8/22)
VERIFY_M = 3         # px (na escala de busca) de folga em volta do candidato na confirmação colorida

CORE_V = 240         # brilho mínimo do "miolo claro" das balas no pré-filtro. 215 -> 240 (2026-10-01): areia clara do
                     # deserto passava; deserto 135 -> 85 ms/quadro, perde 3 de 4559 balas (0.07%), 0 novas

STREAM_W = 90        # px (na escala de busca): coluna de balas coladas até essa largura é varrida
STREAM_FILL = 0.25   # região grande com menos que isso de pixels de bala = leque/fileira rala (não é explosão)

def _color_boxes(img, pad):
    """Caixas (x0, y0, x1, y1) em volta de pixels com a cor de alguma bala. Os sprites (normal,
    boss, flame) têm matiz 0-29 ou 170-179 (medido nos assets); folga para 0-30 e 160-179.
    Saturação >= 130: o anel das balas fica em ~140+; a calçada da fase da estrada (~84) fica de fora.
    Caixas sobrepostas são fundidas para não correlacionar a mesma área várias vezes."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 130, 100), (30, 255, 255)) | cv2.inRange(hsv, (160, 130, 100), (179, 255, 255))
    # só cor de bala PERTO de um miolo claro (as balas têm centro branco/amarelo-claro): a areia laranja da
    # fase do deserto passava no filtro de cor e fazia varrer a tela quase toda (72 ms por quadro)
    core = cv2.inRange(hsv, (0, 0, CORE_V), (179, 110, 255))
    m &= cv2.dilate(core, np.ones((9, 9), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    H, W = img.shape[:2]
    # bala na escala 0.5 tem no máx ~25x50 px (flame); regiões bem maiores são explosão, números de dano,
    # a própria nave — varrer em volta delas custava ~60 ms/quadro no deserto
    # EXCEÇÃO (2026-10-02, fase da floresta): RAJADAS de chamas/agulhas coladas umas nas outras viram UMA região
    # comprida -> eram descartadas como "explosão" e a nave morria dentro de colunas que o bot não via. Mantém
    # região ESTREITA e alta (coluna) e região RALA (leque diagonal: pouco preenchida); explosão/números são
    # blocos cheios.
    def keep(w, h, area):
        if w <= 70 and h <= 80:
            return True
        if w <= STREAM_W and h <= 1400:
            return True
        return w <= 450 and h <= 1400 and area < STREAM_FILL * w * h
    boxes = [[max(x - pad, 0), max(y - pad, 0), min(x + w + pad, W), min(y + h + pad, H)]
             for x, y, w, h, area in stats[1:] if area >= 6 and keep(w, h, area)]
    # junta só caixas que se sobrepõem MUITO (>50% da menor): juntar qualquer encosto fazia poucas balas
    # espalhadas virarem um retângulo cobrindo ~90% da tela (sem ganho nenhum do pré-filtro)
    def area(b):
        return (b[2] - b[0]) * (b[3] - b[1])
    merged = True
    while merged:
        merged, out = False, []
        for b in boxes:
            for o in out:
                ix = min(b[2], o[2]) - max(b[0], o[0]); iy = min(b[3], o[3]) - max(b[1], o[1])
                if ix > 0 and iy > 0 and ix * iy > 0.5 * min(area(b), area(o)):
                    o[:] = [min(o[0], b[0]), min(o[1], b[1]), max(o[2], b[2]), max(o[3], b[3])]
                    merged = True; break
            else:
                out.append(b)
        boxes = out
    return boxes

def detect_bullets(bgr, hud_top=350, thr=0.62, scale=1.0, prefilter=True):
    """Retorna lista de (x, y, raio, tipo). scale: use <1 para acelerar (frame reduzido).
    prefilter: só roda o template matching em volta de regiões com cor de bala (~10x mais rápido
    que varrer o quadro todo; mesmo resultado nos quadros de teste)."""
    img = bgr[int(hud_top):]
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    tpls = _templates(scale)
    k = 2 * max(1, int(12 * scale)) + 1   # janela de máximo local (< distância de duplicata de 24 px)
    kern = np.ones((k, k), np.uint8)
    # (margem por template testada 2026-10-01: 86 x 75 ms — caixas menores se juntam menos, mais tarefas; revertido)
    pad = max(max(t.shape[:2]) for t in tpls.values())
    # (testado: janelinha só em volta de cada candidata perdeu 61/537 balas coladas -> mantidas as caixas)
    boxes = _color_boxes(img, pad) if prefilter else [(0, 0, img.shape[1], img.shape[0])]
    H_, W_ = img.shape[:2]

    # família da AGULHA (vertical + inclinadas, compridas) em METADE da resolução: 4 rotações extras derrubavam o loop
    # de 16 p/ 12.5 q/s (2026-10-01); o sprite tem 104 px e continua nítido reduzido
    img_low = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    low_tpls = _templates(scale * 0.5)
    # DUAS ETAPAS (2026-10-04, velocidade): com CHANNEL, procura candidatos num canal só (~3x mais barato) com limiar
    # mais baixo (CH_THR) e CONFIRMA cada um com o template colorido numa janelinha em volta (precisão da cor).
    two = CHANNEL is not None
    img_c, low_c = img, img_low
    tpls_c, low_tpls_c = (_templates(scale, None), _templates(scale * 0.5, None)) if two else (tpls, low_tpls)
    if two:                                       # (as caixas acima já usaram a COR; a 1ª etapa usa 1 canal)
        img, img_low = to_channel(img), to_channel(img_low)
    thr1 = min(thr, CH_THR) if two else thr

    def peaks(job):
        src, tt, x0, y0, x1, y1, kind, low, two_j = job
        t1 = thr1 if two_j else thr
        res = cv2.matchTemplate(src[y0:y1, x0:x1], tt, cv2.TM_CCOEFF_NORMED)
        if res.max() < t1:
            return []          # saída rápida: a maioria das caixas não tem ESSE tipo de bala (economiza dilate/nonzero)
        # só picos locais: sem isso cada bala gera dezenas de candidatos e o laço abaixo fica O(n²)
        ys, xs = np.nonzero((res >= t1) & (res >= cv2.dilate(res, kern)))
        if not two_j:
            return list(zip(res[ys, xs], xs + x0, ys + y0))
        csrc = low_c if low else img_c
        ct = (low_tpls_c if low else tpls_c)[kind]
        th, tw = ct.shape[:2]; m = VERIFY_M
        out = []
        for px, py in zip(xs + x0, ys + y0):
            ax, ay = max(0, px - m), max(0, py - m)
            patch = csrc[ay:py + th + m, ax:px + tw + m]
            if patch.shape[0] < th or patch.shape[1] < tw:
                continue
            r2 = cv2.matchTemplate(patch, ct, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(r2)
            if mx >= thr:
                out.append((mx, ax + loc[0], ay + loc[1]))
        return out

    # tarefas (template x faixa): caixas altas viram faixas com sobreposição = altura do template,
    # rodadas em paralelo (o OpenCV solta o GIL). Picos repetidos na sobreposição caem no NMS abaixo.
    jobs = {}
    for kind, tt in tpls.items():
        low = _low(kind)
        # famílias que perdem balas no canal único (agulha: 8/22 em cinza) ficam na COR, sem 2ª etapa
        two_j = two and not any(kind.startswith(fm) for fm in COLOR_FAMILIES)
        if two_j:
            src, f = (img_low, 2) if low else (img, 1)
            if low:
                tt = low_tpls[kind]
        else:
            src, f = (low_c, 2) if low else (img_c, 1)
            tt = (low_tpls_c if low else tpls_c)[kind]
        th, tw = tt.shape[:2]
        jobs[kind] = []
        for bx0, by0, bx1, by1 in boxes:
            x0, y0, x1, y1 = bx0 // f, by0 // f, bx1 // f, by1 // f
            if x1 - x0 < tw or y1 - y0 < th:
                continue
            step = _STRIP // f
            for sy in range(y0, y1 - th + 1, step):
                jobs[kind].append((src, tt, x0, sy, x1, min(y1, sy + step + th + k), kind, low, two_j))
    flat = [(kind, j) for kind, js in jobs.items() for j in js]
    # LOTES (1 por thread): cena cheia gerava ~115 tarefas minúsculas e o despacho/espera do pool custava
    # mais que o próprio matchTemplate (perfil 2026-09-30: 113 ms/quadro, travadas de 0.3 s ao vivo)
    nw = _pool()._max_workers
    if len(flat) > 1 and _BATCH:
        chunks = [[j for _, j in flat[i::nw]] for i in range(min(nw, len(flat)))]
        parts = list(_pool().map(lambda js: [peaks(j) for j in js], chunks))
        results = [None] * len(flat)
        for i, part in enumerate(parts):
            results[i::nw] = part
    elif len(flat) > 1:
        results = list(_pool().map(peaks, [j for _, j in flat]))
    else:
        results = [peaks(j) for _, j in flat]
    per_kind = {kind: [] for kind in tpls}
    for (kind, _), r in zip(flat, results):
        per_kind[kind] += r
    out = []
    for kind, tt in tpls.items():
        cand = sorted(per_kind[kind], reverse=True)
        f = 2 if _low(kind) else 1
        if f == 2:
            tt = low_tpls[kind]
        for sc, x, y in cand:
            cx = (x + tt.shape[1] / 2) * f / scale
            cy = (y + tt.shape[0] / 2) * f / scale + hud_top
            r = 26.0 if kind == "normal" else 30.0
            if all((cx - o[0]) ** 2 + (cy - o[1]) ** 2 > 24 ** 2 for o in out):
                out.append((float(cx), float(cy), r, kind))
    # a cabeça do COMETA parece o disco vermelho: "cometa" colado (< 60 px) em outra bala é o disco visto de novo,
    # deslocado (virava bala fantasma ao lado de cada disco)
    others = [o for o in out if not o[3].startswith("comet")]
    out = [o for o in out if not o[3].startswith("comet")
           or all((o[0] - p[0]) ** 2 + (o[1] - p[1]) ** 2 > 60 ** 2 for p in others)]
    return out

# ---------- nave do jogador ----------
# A barra de HP verde segmentada (com coração) fica logo acima e à esquerda da nave.
_SHIP = {}
SHIP_THR = 0.45
_SHIP_PAD = 50

def detect_ship(bgr, near=None):
    """Posição (x, y) da nave do jogador, ou None.
    Pelo DESENHO da nave (template em tons de cinza), não pela cor: na fase do deserto o chão laranja
    virava "nave" na detecção por cor (visto ao vivo 2026-09-29: nave lida em x=60/y=948 etc.).
    Cinza aguenta o flash vermelho de dano; borda estendida acha a nave encostada na lateral.
    near = posição anterior: procura só numa janela em volta (rápido); sem near varre a parte de baixo."""
    if "t" not in _SHIP:
        t = cv2.imread(os.path.join(_ASSETS, "ship.png"))
        _SHIP["t"] = cv2.resize(cv2.cvtColor(t, cv2.COLOR_BGR2GRAY), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        # MÁSCARA só com os pixels da NAVE (2026-10-01): o template tem o asfalto cinza da estrada em volta; na
        # areia do deserto esse fundo derrubava a nota (nave visível: 0.51 sem máscara x 0.90 com) e a nave era
        # "perdida" em 8-37% dos quadros
        hsv = cv2.cvtColor(t, cv2.COLOR_BGR2HSV)
        m = ((hsv[..., 1] > 90) | (hsv[..., 2] > 215)).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        _SHIP["m"] = cv2.resize(m, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        _SHIP["hw"], _SHIP["hh"] = 85, 90          # centro da nave dentro de assets/ship.png (170x190)
    if near is not None:
        # janela perto da última posição (rápido); se não achar ali, varre a tela toda — sem isso a nave que
        # saiu da janela nunca mais era achada e o bot ficava parado levando tiro (visto ao vivo 13:32)
        p = _ship_search(bgr, near)
        if p is None and (near[0] < 180 or near[0] > bgr.shape[1] - 180):
            # nave metade fora da tela casa pior (score ~0.31 ao vivo): perto da borda aceita mais fraco,
            # desde que seja perto de onde ela estava
            p = _ship_search(bgr, near, thr=EDGE_THR)
            if p is not None and (math.hypot(p[0] - near[0], p[1] - near[1]) > 200 or not _is_orange(bgr, p)):
                p = None
        if p is None:
            # míssil-ala / barra de vida / balas por cima derrubam a nota (ao vivo 2026-09-30: 0.41 no lugar
            # certo, nave "perdida" e o bot desviou uma posição velha): aceita NEAR_THR bem perto da anterior
            p = _ship_search(bgr, near, thr=NEAR_THR)
            # + casco laranja: ao vivo 2026-10-01 o míssil verde do Missile Pod ao lado da nave passava aqui
            if p is not None and (math.hypot(p[0] - near[0], p[1] - near[1]) > 150 or not _is_orange(bgr, p)):
                p = None
        if p is not None:
            return p
    # tela toda: candidatos em 1/4 de resolução + refino em janelinha (2026-10-01: a varredura em meia resolução
    # + a "fraca" custavam ~300 ms por quadro perdido -> o loop congelava 0.3-0.6 s logo depois de perder a nave,
    # 153 pausas na partida T e 7 tiros NAVE_PERDIDA; o grosso acha a nave no top 6 em 219/219 quadros)
    cands = _coarse_cands(bgr)
    p = _ship_full(bgr, cands)
    return p if p is not None else _ship_weak(bgr, cands)

WEAK_THR = 0.28
BLINK_THR = 0.20
WEAK_EVERY = 0.25
_WEAK = {"t": 0.0}

COARSE_K = 6
REFINE_WIN = (110, 120)   # meia-largura/altura da janela de refino em volta de cada candidato grosso

def _coarse_cands(bgr):
    """Melhores picos do template da nave em 1/4 de resolução (y >= 1300, borda estendida): [(nota, x, y)]."""
    if "t4" not in _SHIP:
        _SHIP["t4"] = cv2.resize(_SHIP["t"], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        _SHIP["m4"] = cv2.resize(_SHIP["m"], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    g = cv2.resize(cv2.cvtColor(bgr[1300:], cv2.COLOR_BGR2GRAY), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    g = cv2.copyMakeBorder(g, 0, _SHIP_PAD // 2, _SHIP_PAD // 2, _SHIP_PAD // 2, cv2.BORDER_REPLICATE)
    r = np.nan_to_num(cv2.matchTemplate(g, _SHIP["t4"], cv2.TM_CCOEFF_NORMED, mask=_SHIP["m4"]),
                      nan=-1.0, posinf=-1.0, neginf=-1.0)
    pk = np.argwhere(r >= cv2.dilate(r, np.ones((9, 9), np.uint8)))
    return sorted(((float(r[y, x]), float(x * 4 - _SHIP_PAD * 2 + _SHIP["hw"]), float(y * 4 + 1300 + _SHIP["hh"]))
                   for y, x in pk), reverse=True)[:COARSE_K]

def _ship_full(bgr, cands, thr=None):
    """Tela toda = refino de cada candidato grosso; com >1 achado quase tão bom (CLONE) fica com o da barra verde."""
    found = []
    for _, x, y in cands:
        p = _ship_search(bgr, (x, y), thr=thr, win=REFINE_WIN)
        if p is not None and all(math.hypot(p[0] - q[0], p[1] - q[1]) > 40 for _, q in found):
            found.append((_SHIP["mx"], p))
    if not found:
        return None
    found.sort(key=lambda s: -s[0])
    close = [p for s, p in found if s >= found[0][0] - 0.2]
    if len(close) > 1:
        with_bar = [p for p in close if _has_hp_bar(bgr, p)]
        if len(with_bar) == 1:
            return with_bar[0]
    return found[0][1]

def _ship_weak(bgr, cands=None):
    """Último recurso (tela inteira, nota baixa): só aceita candidato COM a barra de vida verde acima/à esquerda.
    Ao vivo 2026-10-01 (fase do DESERTO): a nave sobre a areia laranja dava 0.36 (< 0.45) no lugar certo, ficou
    'perdida' 10 s e o bot parou de pilotar (5 tiros seguidos)."""
    # no máx. a cada WEAK_EVERY s (rodar em todo quadro perdido derrubou o loop p/ 8-11 q/s)
    now = time.time()
    if now - _WEAK["t"] < WEAK_EVERY:
        return None
    _WEAK["t"] = now
    cands = cands if cands is not None else _coarse_cands(bgr)
    for _, x, y in cands:
        p = _ship_search(bgr, (x, y), thr=WEAK_THR, win=REFINE_WIN)
        if p is not None and _has_hp_bar(bgr, p) and _is_orange(bgr, p):
            return p
    # nave PISCANDO (translúcida logo depois de levar dano, ao vivo 2026-10-02): casco claro -> sem laranja e nota
    # ~0.25; a barra de vida verde no lugar certo basta (só a nave do jogador tem barra verde segmentada)
    for _, x, y in cands:
        p = _ship_search(bgr, (x, y), thr=BLINK_THR, win=REFINE_WIN)
        # (1 falso em 29 numa área vazia em y=1538: a nave nunca fica tão alta -> y >= 1650)
        if p is not None and p[1] >= 1650 and _has_hp_bar(bgr, p):
            return p
    # nave METADE FORA da tela numa borda (a barra de vida também fica fora): busca nas duas laterais com a nota de
    # borda; só aceita se UM lado achar (ao vivo 2026-10-01: nave em x~53 perdida 22 s, o bot parou de pilotar)
    sides = [p for p in (_ship_search(bgr, (90, 2000), thr=EDGE_THR), _ship_search(bgr, (990, 2000), thr=EDGE_THR))
             if p is not None and (p[0] < 160 or p[0] > bgr.shape[1] - 160) and _is_orange(bgr, p)]
    return sides[0] if len(sides) == 1 else None

def _is_orange(bgr, p, frac=0.25):   # medido: nave 0.45-0.52 (deserto/borda); rocha 0.17, míssil verde <=0.16, areia 0
    """Casco laranja/vermelho saturado da nave na caixa em volta de p (calçada/carros na borda não têm)."""
    x, y = int(p[0]), int(p[1])
    box = bgr[max(0, y - 70):y + 50, max(0, x - 55):min(bgr.shape[1], x + 55)]
    if box.size == 0:
        return False
    hsv = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 150, 150), (18, 255, 255)) | cv2.inRange(hsv, (170, 150, 150), (179, 255, 255))
    return float((m > 0).mean()) >= frac

EDGE_THR = 0.25
NEAR_THR = 0.33

SHIP_MASK = True

def _ship_match(g):
    """Correlação do template da nave (meia resolução) sobre g; com máscara ignora o fundo do template."""
    if SHIP_MASK:
        r = cv2.matchTemplate(g, _SHIP["t"], cv2.TM_CCOEFF_NORMED, mask=_SHIP["m"])
        return np.nan_to_num(r, nan=-1.0, posinf=-1.0, neginf=-1.0)
    return cv2.matchTemplate(g, _SHIP["t"], cv2.TM_CCOEFF_NORMED)

def _ship_search(bgr, near, thr=None, win=(300, 260)):
    thr = SHIP_THR if thr is None else thr
    ts = _SHIP["t"]
    H, W = bgr.shape[:2]
    if near is not None:
        x0, x1 = int(near[0]) - win[0], int(near[0]) + win[0]
        y0, y1 = int(near[1]) - win[1], int(near[1]) + win[1]
    else:
        x0, x1, y0, y1 = -_SHIP_PAD * 2, W + _SHIP_PAD * 2, 1300, H + _SHIP_PAD * 2
    # recorte com borda estendida (nave encostada na lateral/embaixo ainda casa com o template)
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if cx1 - cx0 < 100 or cy1 - cy0 < 120:
        return None
    # a mesma janela é buscada até 3x seguidas (nota normal / borda / "perto"): guarda o mapa da última
    key = (bgr.__array_interface__["data"][0], x0, y0, x1, y1, int(bgr[cy0 + 5, cx0:cx0 + 64].sum()))
    if _SHIP.get("key") == key:
        r = _SHIP["r"]
    else:
        g = cv2.cvtColor(bgr[cy0:cy1, cx0:cx1], cv2.COLOR_BGR2GRAY)
        g = cv2.copyMakeBorder(g, cy0 - y0 if y0 < 0 else 0, y1 - cy1 if y1 > H else 0,
                               cx0 - x0 if x0 < 0 else 0, x1 - cx1 if x1 > W else 0, cv2.BORDER_REPLICATE)
        g = cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        if g.shape[0] < ts.shape[0] or g.shape[1] < ts.shape[1]:
            return None
        r = _ship_match(g)
        _SHIP["key"], _SHIP["r"] = key, r
    _, mx, _, loc = cv2.minMaxLoc(r)
    _SHIP["mx"] = mx
    if mx < thr:
        return None
    best = (float(loc[0] * 2 + x0 + _SHIP["hw"]), float(loc[1] * 2 + y0 + _SHIP["hh"]))
    # CLONE (skill cria cópia idêntica da nave, visto ao vivo 2026-10-01): outros picos quase tão bons?
    # A nave REAL tem a barra de vida verde a (-224, -111) dela; o clone não.
    peaks = np.argwhere((r >= max(thr, mx - 0.2)) & (r >= cv2.dilate(r, np.ones((25, 25), np.uint8))))
    if len(peaks) > 1:
        cands = [(float(px * 2 + x0 + _SHIP["hw"]), float(py * 2 + y0 + _SHIP["hh"])) for py, px in peaks]
        with_bar = [c for c in cands if _has_hp_bar(bgr, c)]
        if len(with_bar) == 1:
            return with_bar[0]
    return best

def hp_fraction(bgr, ship):
    """Fração de VIDA da nave (0-1) pela barra verde de 11 segmentos acima dela (x-228..x-75, y-112..y-104 do
    centro; medido 2026-10-03). None se a barra estiver fora da tela. Usado pelo modo defensivo do run.py."""
    if ship is None:
        return None
    x0, x1 = int(ship[0]) - 228, int(ship[0]) - 75
    y0, y1 = int(ship[1]) - 112, int(ship[1]) - 104
    if x0 < 0 or x1 > bgr.shape[1] or y0 < 0:
        return None
    hsv = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    g = cv2.inRange(hsv, (40, 120, 120), (80, 255, 255)).any(0)
    if not g.any():
        return None            # sem verde nenhum = barra coberta/fora (vida 0 = já morreu, não chega aqui)
    # vida = até onde vai o verde (da esquerda), não a contagem (a borda entre segmentos é escura).
    # (2026-10-04: exigir verde CONTÍNUO desde a esquerda deu MAIS saltos — 4.7% x 2.4% — explosão cobrindo um
    #  segmento cortava a leitura; o run.py usa a MEDIANA de 7 leituras, que zerou os saltos > 0.3)
    return float(np.nonzero(g)[0].max() + 1) / g.size

def _has_hp_bar(bgr, ship):
    """Barra de vida verde da própria nave: começa em (x-224, y-111), ~15 px de altura."""
    x, y = int(ship[0]) - 224, int(ship[1]) - 111
    # nave perto da borda esquerda: a barra fica parcialmente fora da tela -> usa o pedaço visível (>= 20 px)
    # (ao vivo 2026-10-01: nave em x~160 e a checagem sempre dava falso)
    xa, xb = max(0, x), min(bgr.shape[1], x + 60 if x >= 0 else x + 200)   # barra tem ~200 px de largura
    if y < 0 or xb - xa < 20:
        return False
    hsv = cv2.cvtColor(bgr[y:y + 15, xa:xb], cv2.COLOR_BGR2HSV)
    return float((cv2.inRange(hsv, (40, 120, 120), (80, 255, 255)) > 0).mean()) > 0.25

def detect_ship_color(bgr, search_top=900, near=None, near_r=200):
    """Retorna (x, y) do centro aproximado da nave, ou None.
    Processa em meia resolução (só abaixo de search_top): ~8x mais rápido, erro de ~1 px.
    near=(x, y) da nave no quadro anterior: prefere a mancha mais próxima dela (clone de skill,
    número de dano laranja e explosões também passam no filtro de cor/tamanho)."""
    sub = bgr[search_top::2, ::2]
    # nave: laranja-avermelhado muito saturado; excluir balas (pequenas) por área
    m = cv2.inRange(cv2.cvtColor(sub, cv2.COLOR_BGR2HSV), (5, 170, 170), (16, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    cand = []
    for i in range(1, n):
        x, y, w, hh, area = stats[i]
        if 625 <= area <= 7500 and 30 <= w <= 130 and 35 <= hh <= 150:
            cand.append((area, float(cents[i][0]) * 2, float(cents[i][1]) * 2 + search_top))
    if not cand:
        return None
    if near is not None:
        d2, c = min(((cx - near[0]) ** 2 + (cy - near[1]) ** 2, (cx, cy)) for _, cx, cy in cand)
        if d2 <= near_r ** 2:
            return c
    _, cx, cy = max(cand)
    return (cx, cy)

# ---------- dano recebido ----------
def damage_flash(bgr):
    """Vermelhidão das bordas laterais (R - média(G,B)). Ao levar dano o jogo pinta a tela de
    vermelho: ~170 no flash, ~46 normal (fase da estrada, 2026-09-28). > 100 = acerto."""
    e = np.concatenate([bgr[500:2200:4, 0:24:2], bgr[500:2200:4, -24::2]]).reshape(-1, 3).astype(np.int16)
    return float(np.mean(e[:, 2] - (e[:, 1] + e[:, 0]) / 2))

# ---------- ícone de anúncio opcional (claquete azul) ----------
def find_template(bgr, tmpl, thr=0.72, region=None):
    """Template matching multi-escala simples. Retorna (x, y, score) ou None."""
    img = bgr if region is None else bgr[region[1]:region[3], region[0]:region[2]]
    ox, oy = (0, 0) if region is None else (region[0], region[1])
    best = None
    for sc in (0.8, 0.9, 1.0, 1.1, 1.25):
        t = cv2.resize(tmpl, None, fx=sc, fy=sc)
        if t.shape[0] >= img.shape[0] or t.shape[1] >= img.shape[1]:
            continue
        r = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(r)
        if best is None or mx > best[2]:
            best = (loc[0] + t.shape[1] / 2 + ox, loc[1] + t.shape[0] / 2 + oy, mx)
    return best if best and best[2] >= thr else None

_KIND_COLOR = {"normal": (0, 255, 0), "boss": (255, 0, 255), "flame": (0, 165, 255)}
def debug_draw(bgr, ship=None, bullets=()):
    im = bgr.copy()
    for (x, y, r, k) in bullets:
        cv2.circle(im, (int(x), int(y)), int(r) + 6, _KIND_COLOR.get(k, (0, 200, 255)), 3)
    if ship:
        cv2.circle(im, (int(ship[0]), int(ship[1])), 45, (255, 255, 0), 4)
    return im


def detect_comets(bgr, hud_top=350, ship=None):
    """Cometas laranjas (cabeça amarelo-alaranjada muito brilhante + rastro vermelho) que caem em ângulos
    variados — visto na fase do deserto (2026-09-30); a análise pós-partida os apontava como 'bala não vista'.
    Por cor/forma (template não serve: cada um vem num ângulo). Devolve (x, y, r, 'comet')."""
    sm = cv2.resize(bgr[int(hud_top):], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
    head = cv2.inRange(hsv, (8, 110, 235), (32, 255, 255))
    trail_raw = cv2.inRange(hsv, (0, 150, 150), (12, 255, 255)) | cv2.inRange(hsv, (170, 150, 150), (179, 255, 255))
    n, _, st, cen = cv2.connectedComponentsWithStats(head)
    out = []
    for (x, y, w, h, a), (cx, cy) in zip(st[1:], cen[1:]):
        if not (4 <= a <= 90 and max(w, h) <= 18):
            continue
        # rastro vermelho encostado E de UM LADO só: o disco vermelho comum tem vermelho em volta por igual
        # (centro do vermelho ~ centro da cabeça); no cometa o vermelho fica atrás (deslocado)
        R = 12
        y0, y1, x0, x1 = max(0, int(cy) - R), int(cy) + R + 1, max(0, int(cx) - R), int(cx) + R + 1
        tw = trail_raw[y0:y1, x0:x1]
        ys, xs = np.nonzero(tw)
        if len(xs) < 8:
            continue
        off = math.hypot(xs.mean() + x0 - cx, ys.mean() + y0 - cy)
        if off < 3.5:
            continue
        X, Y = float(cx * 2), float(cy * 2 + hud_top)
        if ship is not None and (X - ship[0]) ** 2 + (Y - ship[1]) ** 2 < 170 ** 2:
            continue              # motor da própria nave e dos drones/mísseis aliados que a acompanham
        out.append((X, Y, 24.0, "comet"))
    return out

def detect_lasers(bgr, top=350, min_len=320, bottom=None, S=0.5):
    """Raios LASER (linhas vermelho-alaranjadas longas e finas que cruzam a tela, vistos no chefe/naves amarelas da
    fase do deserto, 2026-10-01). Devolve segmentos [(x1, y1, x2, y2)] em coordenadas da tela."""
    S = 0.5
    sm = cv2.resize(bgr[top:], None, fx=S, fy=S, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 150, 170), (14, 255, 255)) | cv2.inRange(hsv, (172, 150, 170), (179, 255, 255))
    lines = cv2.HoughLinesP(m, 1, np.pi / 180, threshold=int(120 * S), minLineLength=int(min_len * S), maxLineGap=max(4, int(24 * S)))
    out = []
    if lines is None:
        return out
    for x1, y1, x2, y2 in lines.reshape(-1, 4):   # formato varia entre versões do OpenCV
        if abs(y2 - y1) < 0.6 * abs(x2 - x1):
            continue                                   # quase horizontal: HUD/barras, não é laser
        # confere a cor AO LONGO da linha (mediana): descarta traço de areia/explosão que passou no Hough
        ts = np.linspace(0, 1, 24)
        xs = np.clip((x1 + (x2 - x1) * ts).astype(int), 0, sm.shape[1] - 1)
        ys = np.clip((y1 + (y2 - y1) * ts).astype(int), 0, sm.shape[0] - 1)
        px = hsv[ys, xs]
        if np.median(px[:, 1]) < 180 or np.median(px[:, 2]) < 200:
            continue
        seg = (x1 / S, y1 / S + top, x2 / S, y2 / S + top)
        # borda de areia/estrada (faixas laterais) e borda do flash de dano: segmento inteiro numa lateral = falso
        if max(seg[0], seg[2]) < 240 or min(seg[0], seg[2]) > 840:
            continue
        # funde segmentos quase iguais (Hough devolve vários por linha grossa)
        if all(min(abs(seg[0] - o[0]) + abs(seg[2] - o[2]), abs(seg[0] - o[2]) + abs(seg[2] - o[0])) > 40 for o in out):
            out.append(seg)
    return out

def laser_points(segs, H=2340, step=70):
    """Pontos ao longo de cada laser, ESTENDIDO até a borda de baixo (o raio continua além do que foi detectado).
    Viram 'balas paradas' p/ a esquiva (obstáculo)."""
    pts = []
    for x1, y1, x2, y2 in segs:
        if y2 < y1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        if y2 == y1:
            continue
        k = (x2 - x1) / (y2 - y1)
        y = y1
        while y <= H:
            pts.append((x1 + k * (y - y1), y))
            y += step
    return pts

def detect_threats(bgr, thr=0.75, scale=0.5, ship_exclude=60, near=None, roi_above=None):
    """Nave + balas, já removendo o falso positivo do nariz branco da própria nave.
    roi_above: só procura balas até essa distância (px) acima da nave — é o que importa para
    esquivar e corta ~60% do custo (a maioria das balas fica perto dos inimigos, lá em cima)."""
    if near is not None and roi_above is not None and PARALLEL_SHIP:
        # nave e balas EM PARALELO: a faixa de busca usa a posição anterior da nave (ela quase não anda na
        # vertical; margem de 80 px). Antes: nave (~20 ms) e depois balas, em série.
        top = max(350, int(near[1] - roi_above - 80))
        fut = _pool().submit(detect_ship, bgr, near)
        bl = detect_bullets(bgr, hud_top=top, thr=thr, scale=scale)
        ship = fut.result()
    else:
        ship = detect_ship(bgr, near=near)
        top = 350 if (roi_above is None or ship is None) else max(350, int(ship[1] - roi_above))
        bl = detect_bullets(bgr, hud_top=top, thr=thr, scale=scale)
    if ship is not None:
        bl = [b for b in bl if (b[0] - ship[0]) ** 2 + (b[1] - ship[1]) ** 2 > ship_exclude ** 2]
    if COMETS:
        # cometas (sem duplicar o que já é disco/chama detectado), só os que ANDAM rápido desde o quadro
        # anterior: chamas de motor dos inimigos e fogo pintado no chão passavam no filtro de cor
        now = time.time()
        cs = [c for c in detect_comets(bgr, hud_top=top, ship=ship)
              if all((c[0] - b[0]) ** 2 + (c[1] - b[1]) ** 2 > 30 ** 2 for b in bl)]
        prev, dt = _COMET_PREV.get("c", []), now - _COMET_PREV.get("t", 0)
        if 0 < dt < 0.3:
            lo, hi = 250 * dt, 1900 * dt
            bl += [c for c in cs if any(lo <= math.hypot(c[0] - p[0], c[1] - p[1]) <= hi and c[1] > p[1]
                                        for p in prev)]
        _COMET_PREV.update(c=cs, t=now)
    return ship, bl

_COMET_PREV = {}

# cometas laranjas (análise pós-partida 2026-09-30: 'bala não vista'). DESLIGADO por padrão: nos vídeos
# sem cometas ainda gera ~0.5-1 falso por quadro (chama de motor inimigo, fogo pintado no chão); falta um
# vídeo do deserto para calibrar. `run.py --cometas` liga.
COMETS = False


# EXPERIMENTAL — ainda gera falsos positivos (água, explosões, HUD).
# NÃO usado pelo bot ainda; ver calibrate.py e a conversa no README sobre cores por fase.
# ---------- detecção por FORMATO (independe de cor) ----------
# Complementa o template matching: acha discos com miolo claro cercados por anel
# bem saturado, não importa o matiz. Serve de (a) rede de segurança se a fase
# mudar a cor das balas antes de recalibrar, e (b) fonte "limpa" de amostras de
# cor para a calibração (gwz/calibrate.py), já que exige disco CHEIO, não só um
# pixel saturado qualquer (o que causava contaminação por água/HUD).
def detect_bullet_shapes(bgr, hud_top=350, rmin=7, rmax=16, hud_card=(0, 700, 260, 2340), ship=None):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    colored = (s >= 120) & (v >= 140)
    white = (s < 90) & (v > 200)
    m = (colored | white).astype(np.uint8) * 255
    m[:hud_top] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    peak = (dist == cv2.dilate(dist, np.ones((25, 25), np.uint8))) & (dist >= rmin) & (dist <= rmax)
    ys, xs = np.nonzero(peak)
    hx0, hy0, hx1, hy1 = hud_card
    out = []
    for x, y in zip(xs, ys):
        if hx0 <= x <= hx1 and hy0 <= y <= hy1:
            continue
        if ship is not None and (x - ship[0]) ** 2 + (y - ship[1]) ** 2 < 130 ** 2:
            continue
        q = m[max(y - 10, 0):y + 11, max(x - 10, 0):x + 11]
        if q.mean() < 0.88 * 255:              # precisa ser disco CHEIO (não borda/linha)
            continue
        hue = h[max(y - 12, 0):y + 13, max(x - 12, 0):x + 13][s[max(y - 12, 0):y + 13, max(x - 12, 0):x + 13] > 120]
        if len(hue) < 30:
            continue
        out.append((float(x), float(y), float(dist[y, x]), float(np.median(hue))))
    return _dedupe_xy(out, dist=22)

def _dedupe_xy(bs, dist=18):
    keep = []
    for b in bs:
        if all((b[0] - k[0]) ** 2 + (b[1] - k[1]) ** 2 > dist * dist for k in keep):
            keep.append(b)
    return keep
