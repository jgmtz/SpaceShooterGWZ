"""Classificação de estado por templates recortados de gravações reais do S22 (1080x2340).
Cada tela tem um marcador único + região de busca + limiar. Ordem importa (mais específico primeiro).
"""
import os, cv2
import numpy as np

_A = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
_cache = {}

def _t(name):
    if name not in _cache:
        _cache[name] = cv2.imread(os.path.join(_A, name + ".png"))
    return _cache[name]

def match(bgr, name, region, scales=(1.0,)):
    """Melhor pontuação (0..1) e posição do centro do template em `region`=(x0,y0,x1,y1)."""
    x0, y0, x1, y1 = region
    reg = bgr[y0:y1, x0:x1]
    best = (0.0, None)
    for sc in scales:
        t = _t(name)
        if sc != 1.0:
            t = cv2.resize(t, None, fx=sc, fy=sc)
        if t.shape[0] >= reg.shape[0] or t.shape[1] >= reg.shape[1]:
            continue
        r = cv2.matchTemplate(reg, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(r)
        if mx > best[0]:
            best = (float(mx), (x0 + loc[0] + t.shape[1] // 2, y0 + loc[1] + t.shape[0] // 2))
    return best

# nome do estado -> (template, região de busca, limiar)
MARKERS = [
    # diálogo "Exit battle?" (aparece após BACK no menu de pausa). Ação: X. NUNCA OK (OK fica dentro de FORBIDDEN).
    ("EXIT_DIALOG",   "exit_battle",    (300, 1000, 780, 1260), 0.85),
    ("DEFEATED",      "hdr_tryagain",   (200, 700, 880, 900),  0.80),
    # "There is an unfinished battle, go back to the battle?" (ao reabrir o jogo no meio de uma partida) -> OK
    ("RESUME_BATTLE", "tips_unfinished", (150, 900, 930, 1170), 0.80),
    ("ADFREE_POPUP",  "hdr_adfree",     (200, 700, 880, 900),  0.80),
    ("VICTORY",       "victory_banner", (260, 440, 820, 860),  0.75),
    # tela de derrota com recompensas (depois de recusar o reviver); o banner racha numa animação,
    # por isso dois templates (intacto / rachado). Visto ao vivo em 2026-09-28.
    ("FAILED",        "failed_banner",  (200, 560, 880, 1020), 0.80),
    ("FAILED",        "failed_banner2", (200, 560, 880, 1020), 0.80),
    ("AD_BREAK",      "adbreak_cup",    (330, 1050, 750, 1500), 0.70),
    ("SKILL_SELECT",  "select_skill",   (330, 340, 750, 640),  0.80),
    ("GRID",          "play_grid",      (650, 2060, 1080, 2300), 0.85),
    ("REWARDS_CLOSE", "tap_to_close",   (360, 1980, 720, 2160), 0.80),
    ("LEVEL_MENU",    "play_level",     (250, 1740, 830, 1990), 0.85),
]

def gameplay_score(bgr):
    """Ícone verde de dinheiro no HUD (canto superior direito)."""
    sc, pos = match(bgr, "hud_cash", (820, 90, 1080, 260))
    if pos is None:                  # quadro de outro tamanho (anúncio na horizontal): ao vivo 2026-10-06 isso
        return 0.0                   # derrubou o watch_ad no meio do reviver
    x, y = pos
    # A correlação ignora o brilho: na tela FAILED o HUD fica ESCURECIDO atrás do banner e ainda dava 0.985 ->
    # o loop de esquiva pilotou 9+ min na tela de derrota (ao vivo 2026-10-05). Ícone em partida: V médio ~200
    # (59 prints de tiro, mín 201); escurecido: ~20.
    if sc >= GAMEPLAY_SOFT:
        patch = bgr[max(0, y - 25):y + 25, max(0, x - 30):x + 30]
        if patch.size and cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[..., 2].mean() < HUD_MIN_V:
            return 0.0
    return sc

HUD_MIN_V = 100

# Checagem RÁPIDA (sem marcadores) continua 0.98. No CHEFE os números de dano por cima do HUD davam 0.975-0.979 -> a
# esquiva saía 37x e a nave ficou 129 s sem piloto (2026-10-01): a classificação COMPLETA aceita 0.97 desde que
# nenhum marcador de outra tela (skill, derrota...) apareça — a tela de skill de uma gravação passava de 0.97.
GAMEPLAY_THR = 0.98
GAMEPLAY_SOFT = 0.97

def is_gameplay(bgr, thr=GAMEPLAY_THR):
    return gameplay_score(bgr) >= thr

def classify(bgr):
    """Retorna (estado, detalhes). Estados desconhecidos viram 'UNKNOWN' (nunca clicar às cegas)."""
    gs = gameplay_score(bgr)
    if gs >= GAMEPLAY_THR:
        return "GAMEPLAY", {}
    for state, tpl, reg, thr in MARKERS:
        sc, pos = match(bgr, tpl, reg)
        if sc >= thr:
            return state, {"score": round(sc, 2), "pos": pos}
    if gs >= GAMEPLAY_SOFT:
        return "GAMEPLAY", {"soft": round(gs, 3)}       # HUD parcialmente coberto (chefe), sem outra tela
    return "UNKNOWN", {}


# ---- onda atual: "Next wave: N/15" no topo da tela de montagem (usuário 2026-10-01: anúncio de reviver só
#      depois da onda 12). 48 exemplos rotulados de logs/grid (1..15) em assets/waves.npz; vizinho mais próximo.
_WAVES = {}

def read_next_wave(bgr, min_corr=0.75):
    """Número N de "Next wave: N/15" (a onda que vai começar), ou None. Correlação em CINZA com os exemplos
    (tolera brilho apagado no meio da transição: com limiar fixo um "7" apagado foi lido como 2, ao vivo 2026-10-01)."""
    if "g" not in _WAVES:
        p = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "waves.npz")
        if not os.path.exists(p):
            return None
        d = np.load(p)
        if "g" not in d:
            return None
        _WAVES["g"], _WAVES["y"] = d["g"], d["y"]
    band = cv2.cvtColor(bgr[204:271, 372:768], cv2.COLOR_BGR2GRAY)     # 6-8 px de folga p/ deslocamento
    if band.std() < 8:
        return None
    best, lab = -1.0, None
    for s, y in zip(_WAVES["g"], _WAVES["y"]):
        r = float(cv2.matchTemplate(band, s, cv2.TM_CCOEFF_NORMED).max())
        if r > best:
            best, lab = r, int(y)
    return lab if best >= min_corr else None