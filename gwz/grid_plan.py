"""Planejador da montagem: decide UMA ação por vez a partir de grid_read.read(). O executor aplica a ação,
relê a tela e chama de novo — assim erros de leitura/arrasto se corrigem sozinhos no passo seguinte.

Regras combinadas com o usuário (2026-09-29):
- Objetivo: o MAIOR NÚMERO DE ARMAS DIFERENTES na grade; ignorar sinergias de suporte.
- A bandeja não é guardada: tudo que fica nela some no Play e no Refresh.
- Refresh sempre que o dinheiro da partida estiver >= 30 (limite do jogo: 3 por wave).
- Arma M (Gatling Gun) obrigatória.
"""
from itertools import product
from .grid import PIECES, COLS, ROWS

REFRESH_AT = 15      # custo do Refresh ("x15"). Usuário 2026-10-01: refresh SEMPRE que terminar a loja (antes só com >= 30)
MAX_REFRESH = 3
LOW_PRIORITY = {"cash_purge"}       # não é arma: só entra se sobrar casa (ajuda a juntar p/ Refresh)
PRIMARY = "gatling_gun"
PRIORITY = {"tool_kit"}   # usuário (2026-09-30): sempre priorizar o Tool Kit (cura) — entra antes e nunca é descartado

SAME_LOOK = 0.80     # semelhança p/ "mesma peça" quando o tipo não foi reconhecido (0.62 deixou passar merge
                     # errado ao vivo: peça diferente solta por cima = o jogo TROCA, e armas nv2 saíram da grade)
                     # (medido: mesma peça grade x bandeja 0.73; tipos diferentes <= 0.50)

SAME_LOOK_ARROW = 0.60   # com a seta: mesma peça medida 0.73; tipos diferentes <= 0.50

def shape_of(kind):
    return PIECES.get(kind, {}).get("shape", (1, 1))

def same_piece(a, b):
    """Mesmo tipo e mesmo nível (condição de merge). Tipo reconhecido nos dois: compara o tipo.
    Algum 'unknown' (peça nova/sem template): compara a APARÊNCIA — antes esses merges eram perdidos."""
    if not a["level"] or a["level"] != b["level"]:
        return False
    if a["type"] != "unknown" and b["type"] != "unknown":
        return a["type"] == b["type"]
    from .grid_read import similar
    # seta verde de merge do jogo na peça da bandeja = confirmação extra -> limiar menor
    thr = SAME_LOOK_ARROW if a.get("arrow") or b.get("arrow") else SAME_LOOK
    return similar(a.get("look"), b.get("look")) >= thr

def placements(shape, allowed):
    """Todas as posições (conjunto de casas) em que a forma cabe inteira dentro de `allowed`."""
    w, h = shape
    out = []
    for c, r in product(range(COLS), range(ROWS)):
        cells = frozenset((c + i, r + j) for i in range(w) for j in range(h))
        if cells <= allowed:
            out.append(cells)
    return out

def _fit_score(cells, free):
    """Quanto espaço útil sobra depois de ocupar `cells` (mais = melhor p/ caber mais armas depois)."""
    rest = free - cells
    return sum(len(placements(s, rest)) for s in ((1, 1), (1, 2), (2, 1), (1, 3))) - 0.01 * min(cells)[1]

def plan(state, refreshes_done=0, tk_force=False):
    """tk_force: depois de reler sem achar lugar, o Tool Kit também pode tirar peça "unknown" (grid_exec)."""
    global _TK_FORCE
    _TK_FORCE = tk_force
    return _plan(state, refreshes_done)

_TK_FORCE = False

def _plan(state, refreshes_done=0):
    """state = grid_read.read(frame). Retorna um dict de ação:
    {"op": "drag", "src": (x,y), "cells": set, "why": str} | {"op": "refresh"} | {"op": "play"}"""
    cells, grid, tray, cash = state["cells"], state["grid"], state["tray"], state["cash"]
    free = frozenset(k for k, v in cells.items() if v == "empty")
    # liberada = vazia OU coberta por peça detectada. "occupied" sem peça = balão de nome por cima de
    # casas com X (visto em g020) ou algo não reconhecido -> fica fora de qualquer plano.
    unlocked = free | frozenset(c for p in grid for c in p["cells"])
    known = [p for p in grid if p["type"] != "unknown"]
    grid_types = {p["type"] for p in known}

    # 1) merge bandeja -> grade (mesmo tipo e nível; tipo desconhecido compara pela aparência)
    for t in tray:
        for g in grid:
            if same_piece(t, g):
                kind = t["type"] if t["type"] != "unknown" else g["type"]
                return {"op": "drag", "src": t["pos"], "cells": set(g["cells"]), "merge": True, "type": kind, "level": t["level"] + 1,
                        "why": f"merge {kind} nv{t['level']} (bandeja) na grade"}
    # 1a) seta verde de merge na bandeja, mas o par da grade não foi lido (ao vivo 2026-09-30: cash nv1 coberto
    #     pelas notas "$" voando) -> solta em cima da casa onde a peça casou, se ela está ocupada e sem peça lida
    owned = {c for p in grid for c in p["cells"]}
    for t in tray:
        ac = t.get("arrow_cell")
        if ac and t["level"] and cells.get(ac) == "occupied" and ac not in owned:
            return {"op": "drag", "src": t["pos"], "cells": {ac}, "merge": True, "type": t["type"], "level": t["level"] + 1,
                    "why": f"merge {t['type']} nv{t['level']} (seta verde; par da grade coberto) em {ac}"}
    # 1b) merge grade -> grade
    for a, b in product(grid, grid):
        if a is not b and same_piece(a, b):
            kind = a["type"] if a["type"] != "unknown" else b["type"]
            return {"op": "drag", "src": _center(a["cells"]), "cells": set(b["cells"]), "merge": True, "type": kind, "level": a["level"] + 1,
                    "src_cells": set(a["cells"]), "why": f"merge {kind} nv{a['level']} dentro da grade"}

    # 1t) merge DENTRO DA BANDEJA (pedido do usuário 2026-10-01): 2 peças iguais (tipo e nível) na loja viram 1 de
    #     nível maior; se a grade tem essa peça no nível novo, o passo 1 mescla em seguida (nv1+nv1 -> nv2 -> +nv2 da
    #     grade = nv3). Peças sem nível lido (atrás dos botões) ficam de fora.
    for a, b in product(tray, tray):
        # só tipos RECONHECIDOS (ao vivo 2026-10-01: 2 das 3 tentativas eram "unknown" comparadas pela aparência)
        if a is b or a["type"] in ("slot", "unknown", None) or not a["level"] or a.get("no_badge") or b.get("no_badge"):
            continue
        if same_piece(a, b):
            kind = a["type"] if a["type"] != "unknown" else b["type"]
            return {"op": "tray_merge", "src": a["pos"], "dst": b["pos"], "type": kind, "level": a["level"] + 1,
                    "why": f"merge {kind} nv{a['level']} + nv{b['level']} na LOJA"}

    # 1e) TOOL KIT no lugar do SHIELD PULSAR (pedido do usuário 2026-10-01: o kit de cura é muito importante), só se
    #     ainda não há Tool Kit na grade. Mesmo formato (2x1): soltar em cima troca (o shield volta p/ a loja).
    if "tool_kit" not in grid_types:
        kits = [t for t in tray if t["type"] == "tool_kit"]
        shields = [g for g in grid if g["type"] == "shield_pulsar"]
        if kits and shields:
            g = min(shields, key=lambda g: g["level"] or 0)
            k = max(kits, key=lambda t: t["level"] or 0)
            return {"op": "drag", "src": k["pos"], "cells": set(g["cells"]), "merge": True, "type": "tool_kit",
                    "level": k["level"], "why": f"Tool Kit nv{k['level']} no lugar do Shield Pulsar nv{g['level']}"}

    # 1c) UPGRADE (sugestão do usuário): peça na bandeja do MESMO tipo e nível MAIOR que uma da grade, sem par p/
    #     merge -> solta por cima da mais fraca (o jogo troca: a melhor entra, a pior volta p/ a bandeja).
    for t in tray:
        if t["type"] in ("unknown", "slot", None) or not t["level"]:
            continue
        weaker = [g for g in grid if g["type"] == t["type"] and g["level"] and g["level"] < t["level"]]
        if weaker:
            g = min(weaker, key=lambda g: g["level"])
            return {"op": "drag", "src": t["pos"], "cells": set(g["cells"]), "merge": True, "type": t["type"],
                    "level": t["level"], "why": f"trocar {t['type']} nv{g['level']} da grade pelo nv{t['level']} da bandeja"}

    # 1d) EXPANSÃO da grade (pedido do usuário: sempre colocar, SEMPRE encostada na área liberada). A casa nova
    #     só entra nos planos depois que a leitura seguinte a mostrar liberada (o executor relê a tela a cada ação).
    slots = [t for t in tray if t["type"] == "slot"]
    for sl in sorted(slots, key=lambda t: -len(t.get("slot_cells") or [(0, 0)])):   # maiores primeiro
        rel = sl.get("slot_cells") or [(0, 0)]
        place = expansion_placement(cells, unlocked, rel)
        if place is not None:
            target, anchor = place
            g = sl.get("grab_rel") or rel[0]
            return {"op": "drag", "src": sl["pos"], "cells": target, "merge": False, "type": "slot",
                    "expand": True, "grab_cell": (anchor[0] + g[0], anchor[1] + g[1]),
                    "why": f"expandir a grade com peça {len(rel)} casa(s) em {sorted(target)} (encostada)"}
    tray = [t for t in tray if t["type"] != "slot"]

    # 2) arma principal
    if PRIMARY not in grid_types:
        m = [t for t in tray if t["type"] == PRIMARY]
        if m:
            act = _place(m[0], free, unlocked, grid, force=True)
            if act:
                act["why"] = "colocar a arma principal (M)"
                return act

    # 3) armas de tipo novo nas casas livres (menores primeiro: cabem mais tipos)
    new = [t for t in tray if t["type"] not in grid_types and t["type"] not in LOW_PRIORITY | {"unknown"}]
    # prioridade do usuário: Tool Kit primeiro (recupera vida); depois as menores (cabem mais tipos)
    new.sort(key=lambda t: (t["type"] not in PRIORITY, shape_of(t["type"])[0] * shape_of(t["type"])[1]))
    seen = set()
    for t in new:
        if t["type"] in seen:
            continue
        seen.add(t["type"])
        # Tool Kit sem casa livre já força a troca aqui (antes de pares/repetidas ocuparem o espaço)
        act = _place(t, free, unlocked, grid, force=t["type"] in PRIORITY)
        if act:
            return act
    # 3b) par igual na bandeja (mesmo tipo e nível): põe um numa casa livre; o outro mescla no passo
    #     seguinte (depois da M e dos tipos novos, para não roubar a casa deles)
    for a, b in product(tray, tray):
        if a is not b and same_piece(a, b):
            act = _place(a, free, unlocked, grid, force=False)
            if act:
                act["why"] = f"par de {a['type']} nv{a['level']} na bandeja: coloca um p/ mesclar o outro"
                return act
    # 4) sem espaço: troca repetida/cash por tipo novo. Tipo novo com PAR na bandeja vale como nível+1 (o outro
    #    mescla em seguida) — ao vivo 2026-09-30: par de Missile Pod nv1 descartado com a grade cheia de nv2
    for t in new:
        pair = any(o is not t and same_piece(t, o) for o in tray)
        act = _place(t, free, unlocked, grid, force=True, eff_level=(t["level"] or 1) + 1 if pair else None)
        if act:
            return act

    # 5) sobrou casa: armas repetidas (mais poder de fogo; numa wave futura viram "descartáveis" p/ um tipo
    #    novo), depois peças desconhecidas (podem ser armas sem template) e por último cash purge
    rest = [t for t in tray if t["type"] in grid_types and t["type"] not in LOW_PRIORITY]
    rest += [t for t in tray if t["type"] == "unknown"] + [t for t in tray if t["type"] in LOW_PRIORITY]
    for t in rest:
        act = _place(t, free, unlocked, grid, force=False)
        if act:
            return act

    # 6) refresh
    if cash is not None and cash >= REFRESH_AT and refreshes_done < MAX_REFRESH:
        return {"op": "refresh", "why": f"dinheiro {cash} >= {REFRESH_AT}"}
    return {"op": "play", "why": "nada mais a fazer"}

def expansion_cell(cells, unlocked):
    """Casa bloqueada ENCOSTADA (lado, não diagonal) na área liberada; prefere a que encosta em mais casas
    liberadas (grade compacta) e, no empate, a mais perto do centro da área."""
    if not unlocked:
        return None
    cc = sum(c for c, _ in unlocked) / len(unlocked); rc = sum(r for _, r in unlocked) / len(unlocked)
    best, key = None, None
    for (c, r), v in cells.items():
        if v != "locked":
            continue
        touch = sum((c + dc, r + dr) in unlocked for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        if touch == 0:
            continue
        k = (touch, -abs(c - cc) - abs(r - rc))
        if key is None or k > key:
            best, key = (c, r), k
    return best

def expansion_placement(cells, unlocked, rel):
    """Onde encaixar uma peça de expansão de formato `rel` (lista de (dc, dr)): TODAS as casas bloqueadas,
    pelo menos uma encostada (lado) na área liberada; prefere mais contato e mais perto do centro.
    Devolve (conjunto de casas, âncora) ou None."""
    if not unlocked:
        return None
    cc = sum(c for c, _ in unlocked) / len(unlocked); rc = sum(r for _, r in unlocked) / len(unlocked)
    best, key = None, None
    for (ac, ar) in cells:
        tgt = {(ac + dc, ar + dr) for dc, dr in rel}
        if any(cells.get(t) != "locked" for t in tgt):
            continue
        touch = sum((c + dc, r + dr) in unlocked for c, r in tgt for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        if touch == 0:
            continue
        dist = sum(abs(c - cc) + abs(r - rc) for c, r in tgt) / len(tgt)
        k = (touch, -dist)
        if key is None or k > key:
            best, key = (tgt, (ac, ar)), k
    return best

def _center(cells):
    from .grid import cell_center
    xs, ys = zip(*(cell_center(*c) for c in cells))
    return sum(xs) / len(xs), sum(ys) / len(ys)

def _place(t, free, unlocked, grid, force, eff_level=None):
    """Acha onde pôr a peça t. force=False: só casas livres. force=True: pode cobrir peças descartáveis
    (repetidas de nível menor, cash_purge, unknown) — elas voltam para a bandeja."""
    shape = t.get("shape") or shape_of(t["type"])     # formato aprendido na prévia tem prioridade
    opts = placements(shape, free)
    if opts:
        best = max(opts, key=lambda c: _fit_score(c, free))
        return {"op": "drag", "src": t["pos"], "cells": set(best), "merge": False, "type": t["type"], "level": t["level"],
                "why": f"colocar {t['type']} nv{t['level']} em casas livres"}
    if not force:
        return None
    # peças que podem ser sacrificadas
    count = {}
    for p in grid:
        count[p["type"]] = count.get(p["type"], 0) + 1
    vip = t["type"] in PRIORITY          # Tool Kit (cura): pode tirar QUALQUER peça comum, até de nível maior
    def expendable(p):
        if vip and p["type"] not in PRIORITY | {PRIMARY, "unknown"}:
            return True
        if vip and _TK_FORCE and p["type"] == "unknown":
            return True         # Tool Kit sem lugar mesmo depois de reler: pode tirar peça não reconhecida
        if p["type"] == PRIMARY:
            # arma principal: nunca a ÚNICA nem a de maior nível; uma 2ª Gatling de nível menor pode sair
            # para dar lugar a um tipo novo (visto ao vivo: Gatling 2 + Gatling 1 e Shield sem espaço)
            prim = [q for q in grid if q["type"] == PRIMARY]
            return len(prim) >= 2 and (p["level"] or 0) < max(q["level"] or 0 for q in prim)
        if p["type"] in PRIORITY and count.get(p["type"], 0) < 2:
            return False        # único Tool Kit nunca sai
        if p["type"] == "unknown":
            return False        # pode ser arma sem template (ex.: nível 3+): nunca descartar às cegas
        if p["type"] in LOW_PRIORITY:
            return True
        if count.get(p["type"], 0) >= 2:           # repetida: só a de menor nível
            lv = min(q["level"] or 0 for q in grid if q["type"] == p["type"])
            return (p["level"] or 0) == lv
        return False
    owner = {c: p for p in grid for c in p["cells"]}
    best, cost = None, None
    new_lv = eff_level or t.get("level") or 1
    for cells in placements(shape, unlocked):
        hit = {id(owner[c]): owner[c] for c in cells if c in owner}
        # NUNCA tirar peça de nível MAIOR que a que entra (ao vivo: reator nv1 tirou reator nv2 e core nv2);
        # cash_purge é a exceção (não é arma)
        if not vip and any((p["level"] or 0) > new_lv and p["type"] not in LOW_PRIORITY for p in hit.values()):
            continue
        if all(expendable(p) for p in hit.values()):
            # Tool Kit: perde o mínimo de poder (soma de níveis, cash primeiro); resto: menos peças
            if vip:
                k = (sum((p["level"] or 1) * (p["type"] not in LOW_PRIORITY) for p in hit.values()), len(hit))
            else:
                k = (len(hit), sum(p["level"] or 0 for p in hit.values()))
            if cost is None or k < cost:
                best, cost = cells, k
    if best is None:
        return None
    lost = [owner[c]["type"] for c in best if c in owner]
    return {"op": "drag", "src": t["pos"], "cells": set(best), "merge": False, "type": t["type"], "level": t["level"],
            "why": f"trocar {sorted(set(lost))} por {t['type']} (tipo novo)"}
