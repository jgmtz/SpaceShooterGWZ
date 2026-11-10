"""Bateria de testes OFFLINE (sem celular):  python tools/selftest.py
Confere, com prints/vídeos reais já gravados:
  1. estados das telas conhecidas;  2. trava de ANÚNCIO na bandeja;  3. leitura da montagem sem erro e sem
  peça 'unknown' onde já existe template;  4. planos da montagem (merges esperados);  5. nave (clone/borda);
  6. laser;  7. esquiva sintética;  8. loop real sobre gravação (sem travar).
Sai com código 1 se algo falhar."""
import sys, os, glob, time, math, traceback
import cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
from gwz import states, vision, evade, grid_read as R, grid_plan as P

FAILS, OKS = [], []

def check(name, cond, info=""):
    (OKS if cond else FAILS).append(name)
    print(("  ok   " if cond else "  FALHA ") + name + (f"  ({info})" if info else ""), flush=True)

def img(p):
    im = cv2.imread(p)
    if im is None:
        raise FileNotFoundError(p)
    return im

def t_states():
    print("1) estados")
    cases = {r"logs\unknown\1790657510.png": "EXIT_DIALOG", r"logs\grid\failed_close.png": "FAILED",
             r"logs\grid\live_merge.png": "GRID", r"logs\unknown\1790650780.png": "FAILED"}
    for p, e in cases.items():
        if os.path.exists(p):
            s = states.classify(img(p))[0]
            check(f"estado {os.path.basename(p)} = {e}", s == e, s)

def t_ads():
    print("2) trava de anúncio na bandeja")
    ad = {"1790748196", "1790748910", "1790749145", "1790749239", "1790749713", "1790749806", "1790749905", "1790736259",
          "1790781026", "1790781107", "1790781266", "1790830287",
          "1790835018", "1790835195", "1790835319", "1790835479", "1790836439"}   # sessão longa 2026-10-01: cápsulas nv3   # sessão 2026-09-30: cápsulas nv3 e L de expansão
    for n in sorted(ad):
        p = rf"logs\grid\{n}.png"
        if not os.path.exists(p):
            continue
        im = img(p)
        allp = R._tray_pieces_all(im)
        kept = R.tray_pieces(im)
        icons, caps = R.tray_ad_icons(im), R.capsule_boxes(im)
        check(f"anúncio detectado em {n}", bool(icons or caps), f"ícones={len(icons)} cápsulas={len(caps)}")
        check(f"peça com anúncio fora da leitura em {n}", len(kept) < len(allp) or not allp,
              f"{len(allp)} -> {len(kept)}")
    # nenhum falso: prints sem anúncio não perdem peças
    # só prints já REVISADOS à mão (até a sessão longa de 2026-10-01): depois disso cada sessão traz dezenas de
    # cápsulas de anúncio novas (21 conferidas, todas corretas) e a lista manual não escala
    clean = [p for p in glob.glob(r"logs\grid\1*.png") if os.path.basename(p)[:10] not in ad and int(os.path.basename(p)[:10]) < 1790838000]
    lost = [os.path.basename(p) for p in clean if len(R.tray_pieces(img(p))) < len(R._tray_pieces_all(img(p)))]
    check("nenhuma peça normal descartada como anúncio", not lost, ", ".join(lost[:5]))

def t_grid():
    print("3/4) montagem")
    fs = sorted(glob.glob(r"logs\watch\20260929_005807\g0*_before.png")) + sorted(glob.glob(r"logs\grid\*.png"))
    errs, unk_grid, n = [], 0, 0
    for p in fs:
        try:
            s = R.read(img(p)); P.plan(s); n += 1
            unk_grid += sum(1 for g in s["grid"] if g["type"] == "unknown")
        except Exception as e:
            errs.append(f"{os.path.basename(p)}: {e}")
    check(f"leitura+plano sem erro em {len(fs)} prints", not errs, "; ".join(errs[:3]))
    print(f"       (peças 'unknown' na grade somando todos os prints: {unk_grid})")
    exp = {  # print -> trecho esperado no motivo do plano
        "1790749905": "merge gatling_gun",
        "1790749340": "merge cash_purge",
        "1790749041": "expandir a grade",
    }
    for n_, want in exp.items():
        p = rf"logs\grid\{n_}.png"
        if os.path.exists(p):
            why = P.plan(R.read(img(p)))["why"]
            check(f"plano {n_}: '{want}'", want in why, why)
    # regras sintéticas
    T = lambda k, l, x=500: {"type": k, "level": l, "pos": (x, 2000), "badge": (x, 2000), "score": 1}
    G = lambda k, l, cells: {"type": k, "level": l, "cells": cells, "badge": (0, 0), "score": 1}
    cells = {(c, r): "empty" for c in (1, 2, 3) for r in (2, 3, 4)}
    def st(grid, tray):
        c = dict(cells)
        for g in grid:
            for q in g["cells"]:
                c[q] = "occupied"
        return {"cells": c, "grid": grid, "tray": tray, "cash": 15}
    a = P.plan(st([G("gatling_gun", 2, [(1, 2), (1, 3)]), G("core_reactor", 1, [(2, 2)])], [T("core_reactor", 2)]))
    check("upgrade de mesmo tipo com nível maior", "trocar core_reactor nv1" in a["why"], a["why"])
    full = [G("gatling_gun", 2, [(1, 2), (1, 3)]), G("core_reactor", 2, [(2, 2)]), G("shield_pulsar", 2, [(3, 2), (3, 3)]),
            G("boost_fuel", 2, [(2, 3), (2, 4)]), G("missile_pod", 2, [(1, 4)]), G("beamer", 2, [(3, 4)])]
    a = P.plan(st(full, [T("arc_shield", 1)]))
    check("nunca troca arma nv2 por peça nv1", a["op"] != "drag" or "trocar" not in a["why"], a["why"])
    a = P.plan(st(full, [T("tool_kit", 1)]))
    check("Tool Kit entra mesmo com a grade cheia (sem tirar a Gatling)",
          a["op"] == "drag" and a["type"] == "tool_kit" and not a["cells"] & {(1, 2), (1, 3)}, a.get("why", ""))
    # peças de expansão de vários formatos (usuário: 2x1, 1x2, L de 3 em todas as direções)
    for n, want, adv in (("1790780957.png", [(0, 0), (0, 1)], False), ("1790781026.png", [(0, 0), (1, 0), (1, 1)], True)):
        im = img(os.path.join("logs", "grid", n))
        if im is None:
            continue
        sl = R.tray_slot_pieces(im)
        check(f"forma da expansão em {n}", any(sorted(p["cells"]) == want for p in sl), str([p["cells"] for p in sl]))
        s_ = R.read(im)
        if not adv:     # só a peça de expansão (o merge da engrenagem vem antes e tem teste próprio)
            s_["tray"] = [t for t in s_["tray"] if t["type"] == "slot"]
        a = P.plan(s_)
        if adv:     # esse L está na cápsula de anúncio -> nunca pode ser planejado
            check(f"expansão com anúncio ignorada em {n}", not a.get("expand"), a.get("why", ""))
        else:
            check(f"expansão 1x2 ocupa 2 casas em {n}", a.get("expand") and len(a["cells"]) == 2, str(a.get("cells")))
    # seta verde de merge (sessão 2026-09-30): merges perdidos ao vivo
    for n, want in (("1790781026.png", "merge unknown nv1"), ("1790780957.png", "merge core_reactor nv1")):
        im = img(os.path.join("logs", "grid", n))
        if im is not None:
            a = P.plan(R.read(im))
            check(f"merge pela seta verde em {n}", want in a.get("why", ""), a.get("why", ""))
    im = img(os.path.join("logs", "grid", "1790781107.png"))
    if im is not None:
        check("seta verde detectada (2 em 1790781107)", len(R.merge_arrows(R.normalize_brightness(im))) == 2)
    ex = P.expansion_placement({(c, r): ("empty" if (c, r) in {(2, 2), (2, 3)} else "locked") for c in range(5) for r in range(7)},
                               {(2, 2), (2, 3)}, [(0, 0), (1, 0), (1, 1)])
    check("L de 3 cabe todo em casas bloqueadas encostadas", ex is not None and len(ex[0]) == 3, str(ex))

def t_ship():
    print("5) nave")
    p = r"logs\hits\1790738608_1_raw.png"
    if os.path.exists(p):
        s = vision.detect_ship(img(p))
        check("clone: segue a nave com barra de vida", s is not None and abs(s[0] - 283) < 30, str(s))
    p = r"logs\hits\1790731457_3_raw.png"
    if os.path.exists(p):
        s = vision.detect_ship(img(p), near=(1050, 1930))
        check("nave metade fora da tela (borda) encontrada", s is not None, str(s))

def t_laser():
    print("6) laser")
    v = r"logs\frames\1790749893648.mp4"
    if os.path.exists(v):
        c = cv2.VideoCapture(v); hits = 0
        for k in (8, 12):
            c.set(1, k); ok, im = c.read()
            hits += bool(ok and vision.detect_lasers(cv2.resize(im, (1080, 2340))))
        check("laser do chefe detectado", hits == 2, f"{hits}/2")
    S = glob.glob(r"logs\frames_estrada\*.png")     # quadros da estrada sem laser (opcional)
    fp = sum(len(vision.detect_lasers(img(p))) for p in S[::4])
    if S:
        check("sem laser falso na estrada", fp == 0, str(fp))

def t_evade():
    print("7) esquiva sintética")
    t = evade.choose_target((540, 2000), [(540, 1500, 0, 850)], t0=0.1)
    check("foge de bala reta", abs(t[0] - 540) >= 100, str(t))
    t = evade.choose_target((540, 2000), [], t0=0.1)
    check("sem balas fica parada", abs(t[0] - 540) < 1, str(t))
    t = evade.choose_target((150, 2000), [], t0=0.1, prev=(150, 2000))
    check("esquece destino no canto sem balas", t[0] > 150, str(t))
    wall = [(x, 1500 + 70 * i, 0, 0) for i in range(10) for x in (540,)]
    t = evade.choose_target((540, 2000), wall, t0=0.1)
    check("sai da parede de laser", abs(t[0] - 540) >= 100, str(t))
    # planejador espaço-tempo (padrão do run.py desde 2026-10-02): muro com UM vão -> vai para o vão a tempo
    from gwz import planner
    wall = [(x, 1300, 0, 900) for x in range(40, 1080, 90) if not 600 <= x <= 840]
    tgt, clear, path = planner.plan((300, 2000), planner.linear_predictor(wall), t0=0.15)
    xs_at_wall = [x for t_, x, y in path if 0.7 <= t_ <= 0.9]
    check("planejador atravessa o muro pelo vão", clear > 0 and all(560 <= x <= 880 for x in xs_at_wall),
          f"folga {clear:.0f}, x no muro {xs_at_wall[:3]}")
    tgt, clear, path = planner.plan((540, 2000), planner.linear_predictor([]), t0=0.15)
    check("planejador sem balas não sai do lugar", abs(tgt[0] - 540) <= 30 and abs(tgt[1] - 2000) <= 40, str(tgt))

def t_loop():
    print("8) loop real sobre gravação")
    v = r"C:\Users\joaog\Downloads\Screen_Recording_20260928_225604_Galaxy_Wing_Zero.mp4"
    if not os.path.exists(v):
        return
    import subprocess
    before = set(glob.glob(r"logs\frames\*"))
    t = time.time()
    r = subprocess.run([sys.executable, "tools/replay_loop.py", v, "23"], capture_output=True, text=True, timeout=300)
    for p in set(glob.glob(r"logs\frames\*")) - before:
        os.remove(p)
    if os.path.exists(r"logs\trace.csv"):
        os.remove(r"logs\trace.csv")
    line = [l for l in r.stdout.splitlines() if "loop rodou" in l]
    ok = r.returncode == 0 and bool(line)
    rate = float(line[0].split("(")[1].split("/s")[0]) if line else 0
    check("loop roda sem erro", ok, (r.stderr.strip().splitlines() or [""])[-1][:120])
    # 14 -> 13 (2026-10-01): 3 templates novos de bala custam ~10%. 13 -> 10 (2026-10-02): planejador espaço-tempo
    # (~13 ms) + ROI 1500 (~+20 ms); no simulador fiel ele ganha MESMO decidindo menos (6/s: 13 contatos x 65 antigo)
    # 10 -> 9 (2026-10-02 tarde): flame_t (chamas em rajada) — antes o bot perdia 12% das balas de referência
    # 9 -> 8: + família do COMETA (~1 q/s; matou a nave 2x na floresta). Vídeo de todo quadro: ~0.2 q/s.
    # tools/eval_all.py mede a esquiva também decidindo a 6/s (pior caso) — ainda bem melhor que a antiga.
    check("loop >= 8 quadros/s", rate >= 8, f"{rate}/s")

def t_projectiles():
    print("13) projéteis/inimigos novos (2026-10-01)")
    from gwz import enemies
    cases = (("needles.png", "needle", 6), ("swirl.png", "swirl", 2), ("flame2.png", "flame2", 1))
    alias = {"needle": ("needle", "flame_t")}     # 2026-10-02: a agulha vertical é achada pelo flame_t
    for fn, kind, need in cases:
        p = os.path.join("logs", "tests", fn)
        if os.path.exists(p):
            n = sum(1 for *_, k in vision.detect_bullets(img(p), hud_top=350, thr=0.75, scale=0.5)
                    if k == kind or k in alias.get(kind, ()))
            check(f"detecta '{kind}' ({fn})", n >= need, f"{n} (mín {need})")
    p = os.path.join("logs", "tests", "needles_tilted.png")     # agulhas em leque (partida de 2026-10-01 22:20)
    if os.path.exists(p):
        n = sum(1 for *_, k in vision.detect_bullets(img(p), hud_top=350, thr=0.75, scale=0.5)
                if k.startswith("needle") and k != "needle")
        check("detecta agulhas INCLINADAS (leque)", n >= 5, f"{n} (mín 5)")
    p = os.path.join("logs", "tests", "flame_fan.png")     # rajadas de chamas coladas (chefe da floresta, 2026-10-02)
    if os.path.exists(p):
        n = sum(1 for *_, k in vision.detect_bullets(img(p), hud_top=350, thr=0.75, scale=0.5) if k.startswith("flame"))
        check("detecta chamas em RAJADA (colunas coladas)", n >= 60, f"{n} (mín 60; antes 22)")
    # vida pela barra verde (modo defensivo, 2026-10-03): barra quase vazia (3/11) x cheia
    for fn, lo, hi in (("hp_low.png", 0.15, 0.35), ("hp_full.png", 0.8, 1.01)):
        p = os.path.join("logs", "tests", fn)
        if os.path.exists(p):
            im = img(p); vision._WEAK["t"] = 0
            h = vision.hp_fraction(im, vision.detect_ship(im))
            check(f"vida lida pela barra ({fn})", h is not None and lo <= h <= hi, str(h))
    p = os.path.join("logs", "tests", "comet.png")         # cometa que acertou a nave na floresta (2026-10-02)
    if os.path.exists(p):
        c = [(x, y) for x, y, r, k in vision.detect_bullets(img(p), hud_top=350, thr=0.75, scale=0.5) if k.startswith("comet")]
        check("detecta o COMETA perto da nave", any(abs(x - 1034) < 40 and abs(y - 2100) < 40 for x, y in c), str(c))
    p = os.path.join("logs", "tests", "jets.png")
    if os.path.exists(p):
        check("detecta os jatos roxos", len(enemies.detect_jets(img(p))) >= 5, str(len(enemies.detect_jets(img(p)))))
    # nenhum projétil novo em quadros de outras fases (falso positivo)
    fp = 0
    for q in sorted(glob.glob(r"logs\hits\1790[67]*_raw.png"))[:40]:
        fp += sum(1 for *_, k in vision.detect_bullets(img(q), hud_top=350, thr=0.75, scale=0.5) if k in ("swirl",))
    check("espiral do chefe não aparece em quadros de outras fases", fp == 0, str(fp))
    # classificação: tela de skill nunca vira GAMEPLAY pelo critério suave de 0.97
    sk = sorted(glob.glob(r"logs\skills\*.png"))[:30]
    bad = [os.path.basename(q) for q in sk if states.classify(img(q))[0] == "GAMEPLAY"]
    check("tela de skill nunca é confundida com partida", not bad, ", ".join(bad[:3]))

def t_revive():
    print("12) derrota: o bot sempre recusa o reviver com anúncio")
    from gwz import bot as B
    p = r"logs\watch\20260930_224108\g000_before.png"
    if not os.path.exists(p):
        return
    im = img(p)
    st, _ = states.classify(im)
    check("tela 'Try again?' reconhecida como DEFEATED", st == "DEFEATED", st)
    for wave in (5, 13, None):
        b = B.Bot(None, dry=True); b.wave = wave
        check(f"onda {wave}: recusa o reviver (X)", b.act("DEFEATED", im, {}) is True and b.actions[0][1] == B.XY["defeated_close"])
    p7 = sorted(glob.glob(r"logs\grid\1790835*.png"))
    if p7:
        w7 = states.read_next_wave(img(p7[0]))
        check("lê o número da onda na montagem", w7 is not None, str(w7))
    x, y = B.XY["defeated_close"]
    check("toque do X fora da área de compra", not any(a <= x <= c and bb <= y <= d for a, bb, c, d in B.FORBIDDEN))

def t_ship_desert():
    print("11) nave no deserto (nota baixa) longe da última posição")
    p = r"logs\hits\1790824213_1_raw.png"
    if os.path.exists(p):
        s = vision.detect_ship(img(p), near=(605, 1918))
        check("nave achada pela barra de vida (deserto, borda esquerda)", s is not None and abs(s[0] - 162) < 40, str(s))

def t_skill():
    print("10) carta de Life Steal (prioridade do usuário)")
    from gwz import bot as B
    p = r"logs\watch\20260930_224108\g125_before.png"
    if os.path.exists(p):
        check("Life Steal escolhido (g125, carta do meio)", B.Bot(None, dry=True).choose_skill(img(p)) == (540, 1125))
    # esquiva antes do Recommend (pedido do usuário 2026-10-01): g054 tem "Dodge Rate +10%" no meio
    p2 = r"logs\watch\20260930_224108\g054_before.png"
    if os.path.exists(p2):
        check("carta de esquiva escolhida (g054, meio)", B.Bot(None, dry=True).choose_skill(img(p2)) == (540, 1125))
    p3 = r"logs\watch\20260930_224108\g016_before.png"
    if os.path.exists(p3):
        im3 = img(p3)
        check("sem carta de esquiva em g016", max(B.dodge_score(im3, c, 150, 330) for c in B.SKILL_CARDS_3) < B.DODGE_THR)
    # dano da arma equipada antes do Recommend (usuário 2026-10-01): g067 carta 0 = "Missile Pod / ATK +20%"
    from gwz import grid_exec
    p4 = r"logs\watch\20260930_224108\g067_before.png"
    if os.path.exists(p4):
        grid_exec.LAST_GRID_TYPES.clear(); grid_exec.LAST_GRID_TYPES.update({"missile_pod", "gatling_gun"})
        a = B.Bot(None, dry=True).choose_skill(img(p4))
        grid_exec.LAST_GRID_TYPES.clear()
        im4 = img(p4)
        hm = B.header_score(im4, B.SKILL_CARDS_4[0], 270, "missile_pod"); hg = B.header_score(im4, B.SKILL_CARDS_4[0], 270, "gatling_gun")
        # 2026-10-04: sobrevivência ACIMA do dano (pedido do usuário) — g067 carta 1 = "Shield Generation +15%"
        check("Shield Generation escolhida antes do dano da arma (g067)", a == B.SKILL_CARDS_4[1], str(a))
        check("cabeçalho da arma equipada lido (Missile Pod x Gatling)", hm >= B.HEADER_THR > hg,
              f"missile {hm:.2f} x gatling {hg:.2f}")
    # HP Regen e Shield Generation (2026-10-04): telas da floresta
    for fn, want in (("1791097616.png", (540, 1125)), ("1791096814.png", (540, 1125))):
        p5 = os.path.join("logs", "skills", fn)
        if os.path.exists(p5):
            check(f"sobrevivência escolhida em {fn} (HP Regen / Shield Generation)",
                  B.Bot(None, dry=True).choose_skill(img(p5)) == want)
    # raridade pela cor do cabeçalho (desempate depois do Recommend, usuário 2026-10-01)
    p5, p6 = r"logs\watch\20260930_224108\g016_before.png", r"logs\watch\20260930_224108\g139_before.png"
    if os.path.exists(p5) and os.path.exists(p6):
        a5, a6 = img(p5), img(p6)
        got = [B.card_rarity(a5, c, 330)[0] for c in B.SKILL_CARDS_3] + [B.card_rarity(a6, B.SKILL_CARDS_4[3], 270)[0]]
        check("raridade pela cor (vermelha, roxa, laranja, multicolorida)", got == ["red", "purple", "orange", "multi"], str(got))
    # NUNCA dinheiro (nem Recommend): nenhuma tela gravada pode escolher carta "Gold" se havia outra sem anúncio
    bad = []
    for q in sorted(glob.glob(r"logs\skills\*.png")):
        im = img(q)
        c = B.Bot(None, dry=True).choose_skill(im)
        cards = B.SKILL_CARDS_4 if im[1500:1650, 250:400].mean() > 90 else B.SKILL_CARDS_3
        hh = 270 if len(cards) == 4 else 330
        if c and B.text_score(im, c, 150, hh, B.GOLD_TPLS) >= B.GOLD_THR:
            bad.append(os.path.basename(q))
    check("nunca escolhe carta de dinheiro", not bad, ", ".join(bad[:4]))
    worst = 0.0
    known = {"1790830365", "1790841629", "1790867840"}                       # telas com Life Steal DE VERDADE (escolhido ao vivo 2026-10-01)
    for q in glob.glob(r"logs\skills\*.png"):
        if os.path.basename(q)[:10] in known or int(os.path.basename(q)[:10]) > 1790868500:   # depois: não revisadas
            continue
        im = img(q)
        cards = B.SKILL_CARDS_4 if im[1500:1650, 250:400].mean() > 90 else B.SKILL_CARDS_3
        hh = 270 if len(cards) == 4 else 330
        worst = max([worst] + [B.lifesteal_score(im, c, 150, hh) for c in cards])
    check("nenhuma outra carta parece Life Steal", worst < B.LIFESTEAL_THR, f"maior {worst:.2f}")

def t_drag():
    """Arrasto com aparelho FALSO que imita a prévia verde do jogo (sem celular)."""
    print("9) arrasto (aparelho simulado)")
    from gwz import grid_exec as E
    from gwz.grid import X0, Y0, PITCH
    saved = (E.time.sleep, R.preview_cells, R.tray_ad_icons, R.capsule_boxes)
    E.time.sleep = lambda s: None
    R.tray_ad_icons = R.capsule_boxes = lambda i: []
    class Dev:
        f = None
        def touch_down(s, x, y): s.f = (x, y)
        def touch_move(s, x, y): s.f = (x, y)
        def touch_up(s, x, y): pass
        def tap(s, x, y): pass
        def screenshot(s, fresh=False): return s.f or (0, 0)
    try:
        # canhão 1x3 lido como 1x2 e agarrado pelo pé: prévia começa vazia -> procura em volta
        free = {(3, 2), (3, 3), (3, 4), (0, 2)}
        unl = free | {(1, 2), (1, 3), (2, 2), (2, 3)}
        off = (-E.DROP_OFF[0], -E.DROP_OFF[1] - 1.5 * PITCH)
        def pv_long(f):
            x, y = f[0] + off[0], f[1] + off[1]
            c0, r0 = round((x - X0) / PITCH), round((y - Y0) / PITCH - 1)
            cs = {(c0, r0 + j) for j in range(3)}
            return cs if cs <= unl else set()
        R.preview_cells = pv_long
        r = E.drag_piece(Dev(), (500, 1990), {(3, 3), (3, 4)}, False, free=free)
        check("prévia vazia: acha o encaixe real da peça comprida", r == {(3, 2), (3, 3), (3, 4)}, str(r))
        # expansão em L agarrada por um quadrado: esse quadrado vai p/ a casa dele
        REL = [(0, 0), (1, 0), (1, 1)]
        cells = {(c, r_): "locked" for c in range(5) for r_ in range(7)}
        for k in [(1, 2), (1, 3), (2, 2), (2, 3)]:
            cells[k] = "empty"
        def pv_L(f):
            c0 = round((f[0] - E.DROP_OFF[0] - X0) / PITCH); r0 = round((f[1] - E.DROP_OFF[1] - Y0) / PITCH)
            t = {(c0 + dc, r0 + dr) for dc, dr in REL}
            return t if all(cells.get(q) == "locked" for q in t) else set()
        R.preview_cells = pv_L
        tgt, anchor = P.expansion_placement(cells, {k for k, v in cells.items() if v != "locked"}, REL)
        r = E.drag_piece(Dev(), (464, 1922), tgt, False, expand=True, grab_cell=anchor)
        check("expansão em L cai inteira no lugar planejado", r == tgt, f"{r} x {tgt}")
    finally:
        E.time.sleep, R.preview_cells, R.tray_ad_icons, R.capsule_boxes = saved

if __name__ == "__main__":
    for f in (t_states, t_ads, t_grid, t_ship, t_laser, t_evade, t_drag, t_skill, t_ship_desert, t_revive, t_projectiles, t_loop):
        try:
            f()
        except Exception:
            FAILS.append(f.__name__); traceback.print_exc()
    print(f"\n{len(OKS)} ok, {len(FAILS)} falhas" + (": " + ", ".join(FAILS) if FAILS else ""))
    sys.exit(1 if FAILS else 0)
