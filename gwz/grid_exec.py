"""Executor da montagem: aplica as ações de grid_plan com o dedo virtual (scrcpy) e relê a tela a cada passo.

Mecânica (gravações de 2026-09-29): a peça segue o dedo ~150 px ACIMA dele; soltando em (centro das
casas + (40, 150)) ela cai no lugar (a tolerância de encaixe é meia casa, ±92 px). Durante o arrasto o
jogo pinta de verde as casas de destino -> conferimos antes de soltar. Depois de soltar aparece um balão
com o nome da peça que cobre casas vizinhas e confunde a leitura -> toque numa casa com X para sumir.
Testado ao vivo em 2026-09-29: encaixes (M, Beamer, Tool Kit, Boost Fuel, Cash Purge) e merge de Gatling.
"""
import time, logging, cv2
from . import grid_read, grid_plan, states
from .grid import PLAY, REFRESH, GET_SLOTS, PITCH, cell_center

log = logging.getLogger("gwz")
DROP_OFF = (40, 150)          # dedo - centro das casas de destino
# toque p/ sumir o balão de nome: casa com X (4,6) da própria grade. NÃO usar perto da borda da tela:
# (60,1700) coincidiu com o jogo abrindo o menu de pausa e "Exit battle?" (gesto de voltar do Android?)
NEUTRAL = (908, 1546)
EDGE = 130                    # px mínimos até a lateral da tela para qualquer toque de arrasto
STEP_PX, HZ = 30, 60
EMPTY_MAX = 6                 # leituras vazias seguidas até apertar Play (antes esperava os 25 passos, ~2.5 min)
MAX_STEPS = 25                # ações por visita à montagem (evita laço infinito)
FAIL_TTL = 120                # s que uma peça da bandeja com arrasto falho fica ignorada
_FAILED = {}                  # (x, y) da peça na bandeja -> quando falhou (vale entre visitas à montagem)
LAST_GRID_TYPES = set()       # tipos na grade na última leitura (a escolha de skill prioriza dano DESSAS armas)
_SHAPES = {}                  # posição da peça na bandeja -> formato REAL visto na prévia (peças desconhecidas)
_TRAY_TYPES = {}              # (x, y) na loja -> (tipo, nível) da peça que o bot acabou de mesclar ali

def _pos_key(p):
    return (round(p[0] / 60), round(p[1] / 60))

def _center(cells):
    xs, ys = zip(*(cell_center(*c) for c in cells))
    return sum(xs) / len(xs), sum(ys) / len(ys)

def _glide(dev, a, b):
    """Move o dedo (já encostado) de a até b em passos pequenos."""
    dist = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    n = max(1, int(dist / STEP_PX))
    for i in range(1, n + 1):
        dev.touch_move(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n)
        time.sleep(1 / HZ)

PROBE = [(0, -92), (0, 92), (0, -184), (0, 184), (-92, 0), (92, 0)]   # meia casa / uma casa em volta

def _probe(dev, finger, target, pv):
    """Prévia vazia (ou de outro tamanho): move o dedo em volta até aparecer uma prévia; fica na que mais se
    parece com o alvo. Devolve (dedo, prévia)."""
    best = (finger, pv, -1)          # o dedo físico está SEMPRE em best[0] ao fim de cada tentativa
    for dx, dy in PROBE:
        nf = (min(max(finger[0] + dx, EDGE), 1080 - EDGE), finger[1] + dy)
        _glide(dev, best[0], nf); time.sleep(0.2)
        p = grid_read.preview_cells(dev.screenshot(fresh=True))
        score = len(p & target) * 10 - len(p ^ target) if p else -1
        if score > best[2]:
            best = (nf, p, score)
        else:
            _glide(dev, nf, best[0])
        if p == target:
            break
    return best[0], best[1]

TRAY_MAX_Y = 2085      # dedo nunca desce até os botões de baixo (Get slots = anúncio, Refresh, Play)
TRAY_DROP = (20, 95)   # dedo - peça de destino no merge da loja

def tray_merge(dev, src, dst):
    """Solta a peça da loja em `src` em cima da peça igual em `dst` (merge dentro da loja, pedido do usuário
    2026-10-01). NÃO TESTADO AO VIVO: confere relendo a loja — deu certo se sobrou 1 peça a menos."""
    img = dev.screenshot(fresh=True)
    if img is None:
        return False
    icons, caps = grid_read.tray_ad_icons(img), grid_read.capsule_boxes(img)
    if grid_read.is_ad_piece(src, icons, caps) or grid_read.is_ad_piece(dst, icons, caps):
        log.error("montagem: merge na loja envolve peça com ANÚNCIO -> NÃO arrasto"); return False
    before = len(grid_read.tray_pieces(img))
    a = (min(max(src[0], EDGE), 1080 - EDGE), min(src[1], TRAY_MAX_Y))
    # a peça arrastada fica ~150 px ACIMA do dedo (DROP_OFF): ao vivo 2026-10-01, soltando com o dedo em cima da
    # peça de destino, 3/3 não mesclaram (a peça caía acima dela). Dedo mais abaixo (limite: os botões em y~2100+)
    b = (min(max(dst[0] + TRAY_DROP[0], EDGE), 1080 - EDGE), min(dst[1] + TRAY_DROP[1], TRAY_MAX_Y))
    dev.touch_down(*a); time.sleep(0.15)
    _glide(dev, a, b); time.sleep(0.35)
    dev.touch_up(*b); time.sleep(0.8)
    dev.tap(*NEUTRAL); time.sleep(0.5)
    after = len(grid_read.tray_pieces(_wait_still(dev)))
    ok = after < before
    log.info("montagem: merge na loja %s -> %s: peças %d -> %d (%s)", tuple(map(round, src)), tuple(map(round, dst)),
             before, after, "ok" if ok else "NÃO mesclou")
    return ok

def drag_piece(dev, src, cells, merge, free=None, expand=False, grab_cell=None):
    """Arrasta a peça de src para as casas `cells`. Retorna as casas onde ela caiu (set, com prévia
    conferida quando não é merge) ou None se desistiu (peça devolvida à bandeja).
    free: casas livres — se a prévia mostrar casas TODAS livres que incluem o alvo, aceita (peça de formato
    desconhecido: o jogo mostra o formato real; ao vivo, peça "1x1" era 1x2 e o bot desistia sem fim)."""
    # TRAVA DE ANÚNCIO (2ª camada): nunca encostar em peça da bandeja com claquete (abre anúncio e trava a fase)
    if src[1] > 1750:
        img = dev.screenshot(fresh=True)
        if img is None:
            log.error("montagem: sem imagem p/ conferir anúncio -> NÃO arrasto"); return None
        icons, caps = grid_read.tray_ad_icons(img), grid_read.capsule_boxes(img)
        if grid_read.is_ad_piece(src, icons, caps):
            log.error("montagem: peça em %s tem ANÚNCIO (claquete) -> NÃO arrasto", tuple(map(round, src)))
            return None
    tx, ty = _center(cells)
    if expand and grab_cell is not None:
        tx, ty = cell_center(*grab_cell)          # o quadrado AGARRADO vai p/ a casa dele (peça 2x1/1x2/L)
    finger = (tx + DROP_OFF[0], ty + DROP_OFF[1])
    # nunca começar/passar perto da lateral: toque junto à borda pode virar o gesto "voltar" do Android
    src = (min(max(src[0], EDGE), 1080 - EDGE), src[1])
    finger = (min(max(finger[0], EDGE), 1080 - EDGE), finger[1])
    dev.touch_down(*src); time.sleep(0.15)
    _glide(dev, src, finger); time.sleep(0.3)
    ok = True
    if expand:
        # peça de EXPANSÃO: o alvo são casas BLOQUEADAS (X); o jogo pinta prévia verde (visto ao vivo).
        # prévia = alvo -> solta; prévia deslocada (mesmo formato) -> corrige; prévia vazia -> procura em volta.
        tgt = set(cells)
        pv = grid_read.preview_cells(dev.screenshot(fresh=True))
        for _ in range(3):
            if pv == tgt:
                break
            if pv and len(pv) == len(tgt):
                (px, py), (qx, qy) = _center(pv), _center(tgt)
                nf = (min(max(finger[0] + qx - px, EDGE), 1080 - EDGE), finger[1] + qy - py)
                _glide(dev, finger, nf); finger = nf; time.sleep(0.25)
                pv = grid_read.preview_cells(dev.screenshot(fresh=True))
            else:
                finger, pv = _probe(dev, finger, tgt, pv)
        ok = pv == tgt or (not pv and len(tgt) == 1)      # 1 casa sem prévia: solta no alvo calculado
        if not ok:
            log.warning("montagem: expansão com prévia %s x alvo %s -> devolvendo", sorted(pv), sorted(tgt))
            _glide(dev, finger, src); finger = src
        else:
            log.info("montagem: expansão solta em %s (prévia %s)", sorted(tgt), sorted(pv) or "nenhuma")
        dev.touch_up(*finger); time.sleep(0.6)
        dev.tap(*NEUTRAL); time.sleep(0.4)
        return tgt if ok else None
    if not merge:
        target, reshaped, pv = set(cells), False, set()
        free_set = set(free) if free is not None else None
        for _ in range(4):
            pv = grid_read.preview_cells(dev.screenshot(fresh=True))
            if not pv:
                # prévia VAZIA: peça alta/comprida pega pelo meio fica deslocada do dedo e cai em lugar inválido
                # (ao vivo: canhão comprido, 3x "prévia [] -> devolvendo"). Procura em volta antes de desistir.
                finger, pv = _probe(dev, finger, target, pv)
                if not pv:
                    break
            if pv == target:
                break
            if free_set is not None and target <= pv <= free_set:
                break                                   # peça maior, mas tudo livre: serve
            if len(pv) != len(target) and free_set is not None and not reshaped:
                # FORMATO REAL diferente do previsto (peça desconhecida tratada como 1x1): mede pela prévia,
                # procura lugar LIVRE com esse formato e leva a peça até lá. Antes: a prévia cobria uma arma
                # vizinha (ao vivo: Tool Kit nv2), o bot desistia e repetia sem fim — e soltar ali trocaria a
                # arma melhor pela nova.
                cs = [c for c, _ in pv]; rs = [r for _, r in pv]
                shape = (max(cs) - min(cs) + 1, max(rs) - min(rs) + 1)
                _SHAPES[_pos_key(src)] = shape          # lembra o formato real desta peça da bandeja
                opts = grid_plan.placements(shape, frozenset(free_set))
                reshaped = True
                if not opts:
                    log.warning("montagem: peça é %dx%d, não cabe em casa livre -> devolvendo", *shape)
                    break
                new = max(opts, key=lambda c: grid_plan._fit_score(c, frozenset(free_set)))
                log.info("montagem: peça é %dx%d (prévia %s) -> levando para %s", *shape, sorted(pv), sorted(new))
                # mira o centro do destino novo com o mesmo deslocamento dedo->peça do início. (Peça de 2 casas
                # mirada no centro de UMA casa fica no meio de dois encaixes; medir pela prévia dali oscilava.)
                qx, qy = _center(new)
                nf = (qx + DROP_OFF[0], qy + DROP_OFF[1])
                target = set(new)
            else:
                # mesmo tamanho, lugar errado: corrige pela diferença de centros
                (px, py), (qx, qy) = _center(pv), _center(target)
                nf = (finger[0] + qx - px, finger[1] + qy - py)
            nf = (min(max(nf[0], EDGE), 1080 - EDGE), nf[1])
            _glide(dev, finger, nf); finger = nf; time.sleep(0.25)
        else:
            pv = grid_read.preview_cells(dev.screenshot(fresh=True))
        # só solta se a prévia estiver TODA em casas livres (nunca cobrir outra peça = nunca trocar arma)
        ok = bool(pv) and (pv == target or (free_set is not None and target <= pv)) and \
            (free_set is None or pv <= free_set)
        if not ok:
            log.warning("montagem: prévia %s x alvo %s -> devolvendo a peça", sorted(pv), sorted(target))
            _glide(dev, finger, src); finger = src
        cells = pv if ok else cells
    dev.touch_up(*finger); time.sleep(0.5)
    dev.tap(*NEUTRAL); time.sleep(0.4)       # some o balão de nome
    return set(cells) if ok else None

MOTION_THR = 2.0     # diferença média (0-255) na área da grade entre 2 quadros: acima disso há animação

def _wait_still(dev, max_wait=4.0):
    """Espera a grade PARAR de mexer (notas de dinheiro do Cash Purge voam por cima das peças por alguns
    segundos no começo da montagem e escondiam as peças: grade lida como 'unknown' -> merges perdidos)."""
    import numpy as np
    t0 = time.time()
    a = dev.screenshot(fresh=True)
    while time.time() - t0 < max_wait:
        time.sleep(0.3)
        b = dev.screenshot(fresh=True)
        if a is None or b is None:
            return b
        diff = float(np.abs(a[350:1700:4, ::4].astype(np.int16) - b[350:1700:4, ::4].astype(np.int16)).mean())
        if diff < MOTION_THR:
            return b
        a = b
    log.info("montagem: grade ainda animando após %.0fs -> seguindo assim mesmo", max_wait)
    return a

def assemble(dev, tap):
    """Laço da montagem. tap = função de toque segura do Bot (respeita FORBIDDEN). Retorna ao apertar Play
    ou ao sair da tela GRID."""
    refreshes, last, arrow_waits, empty_reads = 0, None, 0, 0
    tk_waits = 0                                 # releituras esperando lugar p/ um Tool Kit da loja
    _SHAPES.clear(); _TRAY_TYPES.clear()     # bandeja nova a cada visita
    placed = {}      # casas -> (tipo, nível) do que o bot pôs nesta visita (a leitura logo após soltar
                     # pode dar "unknown" por balão/animação; ao vivo isso fez colocar 2 Gatling em vez de mesclar)
    seen = {}        # casas -> (tipo, nível) lidos com tipo nesta visita
    done = {}        # ação -> quantas vezes foi feita nesta visita
    for step in range(MAX_STEPS):
        time.sleep(0.3)
        frame = _wait_still(dev)
        st, _ = states.classify(frame)
        if st != "GRID":
            log.info("montagem: saiu da tela (%s)", st); return
        s = grid_read.read(frame)
        if s["grid"]:
            LAST_GRID_TYPES.clear(); LAST_GRID_TYPES.update(p["type"] for p in s["grid"] if p["type"] != "unknown")
        if not s["grid"] and not s["tray"] and s["cash"] is None:
            # leitura vazia = tela em animação (visto ao vivo logo após devolver uma peça): decidir "Play"
            # com base nisso travou o bot 7 min. Espera e relê.
            empty_reads += 1
            if empty_reads == 1:
                cv2.imwrite(f"logs/grid/empty_{int(time.time())}.png", frame)     # p/ ver o que cobre a grade
            if empty_reads >= EMPTY_MAX:
                # ao vivo 2026-10-01: 2.5 min lendo "vazio" (algo por cima da grade); o toque em Play destravou
                tap(*PLAY, why=f"Play ({empty_reads} leituras vazias seguidas)"); time.sleep(1.5)
                return
            log.info("montagem: leitura vazia (animação?) -> esperando"); time.sleep(0.8); continue
        empty_reads = 0
        # peças cujo arrasto falhou há pouco (mesmo em visitas anteriores) ficam de fora por FAIL_TTL s
        now = time.time()
        for k in [k for k, t in _FAILED.items() if now - t > FAIL_TTL]:
            del _FAILED[k]
        s["tray"] = [t for t in s["tray"] if not any(abs(t["pos"][0] - fx) < 40 and abs(t["pos"][1] - fy) < 40
                                                     for fx, fy in _FAILED)]
        for t in s["tray"]:                 # formato real já descoberto num arrasto anterior
            if t["type"] == "unknown":
                # peça que o bot MESCLOU na loja: o visual do nível novo pode não ter template (ao vivo 2026-10-05:
                # Tool Kit nv1+nv1 -> lido "unknown nv2", não coube e foi perdido no Play)
                # (ao vivo 2026-10-06: Core Reactor nv1+nv1 -> a peça nova NÃO ficou na posição do destino; com
                #  90 px de tolerância ela continuou "unknown" e não foi mesclada com o nv2 da grade) -> a mais perto
                near = [(abs(t["pos"][0] - mx) + abs(t["pos"][1] - my), kind, lv)
                        for (mx, my), (kind, lv) in _TRAY_TYPES.items()]
                near = [n for n in near if n[0] < 450]
                if near:
                    _, kind, lv = min(near)
                    t["type"] = kind; t["level"] = t["level"] or lv; t["tray_mem"] = True
            if _pos_key(t["pos"]) in _SHAPES:
                t["shape"] = _SHAPES[_pos_key(t["pos"])]
        for p in s["grid"]:
            if p["type"] == "unknown":
                for cells, (kind, lv) in placed.items():
                    if set(p["cells"]) <= cells:
                        # aprende o visual: peça que o próprio bot encaixou (prévia conferida), balão já
                        # dispensado, círculo na casa inferior direita e nível lido == nível esperado
                        br = max(cells, key=lambda c: (c[1], c[0]))
                        if p["cells"] == [br] and p["level"] == lv and kind:
                            if grid_read.learn_piece(frame, kind, lv, cells):
                                log.info("montagem: aprendi o visual de %s nv%s na grade", kind, lv)
                        p["type"], p["cells"] = kind, sorted(cells)
                        p["level"] = p["level"] or lv
                        break
            if p["type"] == "unknown":
                # peça LIDA com tipo numa leitura anterior desta visita, nas mesmas casas (ou parte delas: com as notas
                # "$" por cima o Tool Kit 2x1 vira "unknown" de 1 casa): herda o tipo (ao vivo 2026-10-06: Core Reactor
                # nv2 e Tool Kit nv2 lidos certo, logo depois "unknown" -> merge perdido / troca em vai-e-volta)
                for cells, (kind, lv) in seen.items():
                    if set(p["cells"]) <= set(cells):
                        p["type"], p["cells"] = kind, sorted(cells)
                        p["level"] = p["level"] or lv
                        break
        for p in s["grid"]:
            if p["type"] != "unknown" and not p.get("from_memory"):
                key = tuple(sorted(p["cells"]))
                for k in [k for k in seen if set(k) & set(key)]:
                    del seen[k]          # o que estava nessas casas foi trocado/mesclado: vale a leitura nova
                seen[key] = (p["type"], p["level"])
        # peça que o bot pôs e SUMIU da leitura (casas ainda ocupadas, círculo coberto pelas notas "$" voando —
        # ao vivo 2026-09-30: cash nv1 recém-posto não foi lido e o par da bandeja foi perdido no Refresh)
        owned = {c for p in s["grid"] for c in p["cells"]}
        for cells, (kind, lv) in placed.items():
            if not cells & owned and all(s["cells"].get(c) == "occupied" for c in cells):
                s["grid"].append({"type": kind, "level": lv, "cells": sorted(cells), "badge": (0, 0),
                                  "score": 0.0, "look": None, "from_memory": True})
        act = grid_plan.plan(s, refreshes, tk_force=tk_waits >= 3)
        # seta verde de merge na bandeja sem merge planejado -> o par está na grade mas não foi lido (animação):
        # espera e relê antes de um Refresh/Play que jogaria a peça fora
        if act["op"] in ("refresh", "play") and arrow_waits < 3 and any(t.get("arrow") for t in s["tray"]):
            arrow_waits += 1
            log.info("montagem: peça com seta de merge sem par lido -> esperando a animação (%d/3)", arrow_waits)
            time.sleep(1.2); continue
        # TOOL KIT (cura) na loja e nenhum na grade, mas o plano é Refresh/Play: ao vivo 2026-10-04 o bot DESCARTOU
        # o único Tool Kit da partida num Refresh — logo depois de outro Refresh as notas "$" cobriam as peças, 2
        # viraram "unknown" (nunca descartáveis) e não sobrou lugar. Espera a animação e relê (3x); depois o Tool
        # Kit pode tirar até peça "unknown" (grid_plan tk_force)
        tk_tray = any(t["type"] in grid_plan.PRIORITY for t in s["tray"])
        tk_grid = any(p["type"] in grid_plan.PRIORITY for p in s["grid"])
        if act["op"] in ("refresh", "play") and tk_tray and not tk_grid and tk_waits < 3:
            tk_waits += 1
            log.info("montagem: Tool Kit na loja sem lugar lido -> esperando e relendo (%d/3)", tk_waits)
            time.sleep(1.2); continue
        log.info("montagem %d: dinheiro=%s grade=%s bandeja=%s -> %s %s (%s)", step, s["cash"],
                 [(p["type"], p["level"]) for p in s["grid"]], [(p["type"], p["level"]) for p in s["tray"]],
                 act["op"], sorted(act.get("cells", [])), act["why"])
        key = (act["op"], tuple(sorted(act.get("cells", []))), act.get("src") and tuple(map(round, act["src"])))
        if key == last and act["op"] == "drag":
            log.warning("montagem: mesma ação duas vezes seguidas sem efeito -> Play"); act = {"op": "play"}
        elif act["op"] == "drag" and done.get(key, 0) >= 2:
            # vai-e-volta (ao vivo 2026-10-06: troca <-> "merge" alternando por 2.5 min até o limite de passos)
            log.warning("montagem: ação repetida 3x nesta visita (vai-e-volta) -> Play"); act = {"op": "play"}
        last = key
        done[key] = done.get(key, 0) + 1
        if act["op"] == "drag":
            free = {k for k, v in s["cells"].items() if v == "empty"}
            if not act.get("merge") and not act.get("expand"):
                # TROCA planejada: as casas do alvo podem estar ocupadas pela peça que sai. Sem isso a prévia
                # certa era recusada ("prévia [(2,2),(3,2)] x alvo [(2,2),(3,2)] -> devolvendo", ao vivo
                # 2026-10-05: Tool Kit no lugar de Boost Fuel + Core Reactor nunca acontecia).
                free |= set(act["cells"])
            landed = drag_piece(dev, act["src"], act["cells"], act.get("merge", False), free=free,
                                expand=act.get("expand", False), grab_cell=act.get("grab_cell"))
            if landed is None:
                _FAILED[(round(act["src"][0]), round(act["src"][1]))] = time.time()
            else:
                # a peça mesclada na loja SAIU da loja: o rótulo não vale mais (senão a peça que a troca devolve à
                # loja herdava o tipo — ao vivo 2026-10-06 um Core Reactor virou "Tool Kit" e a montagem entrou em loop)
                if act["src"][1] > 1750:
                    for k in [k for k, (kind, _) in _TRAY_TYPES.items() if kind == act.get("type")
                              and abs(k[0] - act["src"][0]) + abs(k[1] - act["src"][1]) < 450]:
                        del _TRAY_TYPES[k]
                cells = frozenset(landed)
                gone = cells | frozenset(act.get("src_cells", ()))   # merge na grade: a origem some
                for k in [k for k in placed if k & gone]:
                    del placed[k]
                if act.get("type") not in (None, "unknown", "slot"):
                    placed[cells] = (act["type"], act.get("level"))
        elif act["op"] == "tray_merge":
            if not tray_merge(dev, act["src"], act["dst"]):
                _FAILED[(round(act["src"][0]), round(act["src"][1]))] = time.time()
            elif act.get("type") not in (None, "unknown"):
                _TRAY_TYPES[tuple(act["dst"])] = (act["type"], act.get("level"))
        elif act["op"] == "refresh":
            tap(*REFRESH, why="Refresh (dinheiro >= 30)"); refreshes += 1; _SHAPES.clear(); _TRAY_TYPES.clear()
            time.sleep(1.2)
        else:
            tap(*PLAY, why="Play (montagem pronta)"); time.sleep(1.5)
            return
    tap(*PLAY, why="Play (limite de passos da montagem)")
