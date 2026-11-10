"""Máquina de estados do bot. Orientada a ESTADO VISUAL, nunca a cliques às cegas.
Modo replay (sem celular): Bot(dry=True) só decide e registra; útil para validar em gravações.
"""
import time, logging, os
import cv2
import numpy as np
from . import states, vision

log = logging.getLogger("gwz")

# ---- coordenadas (tela 1080x2340; lidas das gravações do S22) ----
XY = {
    "defeated_close": (1005, 805),    # X do popup "Try again?" (recusa o anúncio de reviver)
    "adfree_close":   (995, 800),     # X do popup "Ad-free play"
    "victory_get":    (785, 1918),    # botão Get (Double = anúncio, NÃO usar)
    "failed_get":     (796, 1930),    # Get da tela FAILED (Double à esquerda = anúncio, NÃO usar)
    "exit_dialog_x":  (1010, 810),    # X do diálogo "Exit battle?" (fica na batalha)
    "pause_continue": (540, 1644),    # Continue do menu de pausa (Home/Continue/Settings)
    "grid_play":      (870, 2175),
    "resume_ok":      (850, 1387),    # OK do "unfinished battle" (botão x~600-945; FORA do FORBIDDEN, que vai até x=760)
    "rewards_close":  (540, 2070),    # "Tap To Close"
    "level_play":     (538, 1862),
    "level_prev":     (212, 1012),     # seta "<" do menu de níveis (print 2026-10-02, Level 16)
}
# retângulos onde JAMAIS tocar (botões de compra em dinheiro real): (x0,y0,x1,y1)
FORBIDDEN = [(380, 1380, 760, 1540)]  # botão "R$ 5,49" do popup Ad-free
SKILL_CARDS_3 = [(195, 1125), (540, 1125), (878, 1125)]
SKILL_CARDS_4 = [(325, 950), (740, 950), (325, 1575), (740, 1575)]
# templates da carta de lifesteal (assets/skill_lifesteal*.png) — ainda não existem: criar quando a carta
# aparecer em logs/skills/. Enquanto isso a regra cai para "Recommend".
LIFESTEAL_TPLS = [os.path.basename(p)[:-4] for p in
                  __import__("glob").glob(os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "skill_lifesteal*.png"))]

LIFESTEAL_THR = 0.82   # 0.75 -> 0.82 (2026-10-04): a carta do Tool Kit "Life recovery Effect +35%" dava 0.77 (falso);
                       # as de Life Steal reais dão 0.89-1.00
_LS = {}
# esquiva (pedido do usuário 2026-10-01: depois do Life Steal, antes do Recommend): "Dodge Rate +x%" e
# "Base dodge rate +x%" (textos recortados das gravações do usuário)
DODGE_TPLS = [os.path.basename(p)[:-4] for p in
              __import__("glob").glob(os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "skill_dodge*.png"))]
DODGE_THR = 0.75

# 1.2/1.3 (2026-10-04): o print da tela de skill às vezes pega a carta do meio AINDA AUMENTADA pela animação
# ("Shield Generation +5%" dava 0.67 e a carta foi ignorada)
TEXT_SCALES = (0.75, 0.85, 0.95, 1.0, 1.1, 1.2, 1.3)

def text_score(frame, card, half_w, half_h, names):
    """Melhor casamento (cinza, multi-escala) de algum template de TEXTO `names` na carta."""
    cache = {}
    for n in names:
        if n not in _LS:
            t = cv2.imread(os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", n + ".png"))
            if t is not None:
                _LS[n] = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
        if n in _LS:
            cache[n] = _LS[n]
    cx, cy = card
    reg = cv2.cvtColor(frame[max(0, cy - half_h):min(2340, cy + half_h), max(0, cx - half_w):min(1080, cx + half_w)],
                       cv2.COLOR_BGR2GRAY)
    best = 0.0
    for t in cache.values():
        for s in TEXT_SCALES:
            ts = cv2.resize(t, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            if ts.shape[0] < reg.shape[0] and ts.shape[1] < reg.shape[1]:
                best = max(best, float(cv2.matchTemplate(reg, ts, cv2.TM_CCOEFF_NORMED).max()))
    return best

HPREGEN_THR = 0.80
SHIELDGEN_THR = 0.88            # carta geral "Shield Generation +x%"
SHIELDGEN_THR_PULSAR = 0.72     # inclui as da arma Shield Pulsar (~0.77) quando ela está na grade
GOLD_TPLS = ["skill_gold"]       # "Gold +10" (carta de dinheiro)
GOLD_THR = 0.80
ATTACK_WEAPONS = {"gatling_gun", "missile_pod", "beamer", "phaser_cannon", "plasma_orb"}
HEADER_THR = 0.80

def header_score(frame, card, half_h, kind):
    """Nome da ARMA no cabeçalho da carta (assets/skill_hdr_<tipo>.png, texto branco): faixa do topo do cartão."""
    cx, cy = card
    top = cy - half_h
    return text_score(frame, (cx, top + 25), 165, 55, [f"skill_hdr_{kind}"])

# RARIDADE pela cor do cabeçalho da carta (escada no topo da tela de skill: Blue < Purple < Orange < Red <
# Multicolored). Usuário 2026-10-01: depois do Recommend, escolher a de MAIOR raridade em vez da 1ª.
RARITY = {"blue": 1, "purple": 2, "orange": 3, "red": 4, "multi": 5}

def card_rarity(frame, card, half_h):
    """Cor do cabeçalho (sem o texto branco): devolve ('blue'|'purple'|'orange'|'red'|'multi'|None, matizes)."""
    cx, cy = card
    top = cy - half_h
    band = frame[max(0, top - 10):top + 60, max(0, cx - 150):min(1080, cx + 150)]
    if band.size == 0:
        return None, None
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    col = hsv[(hsv[:, 1] > 110) & (hsv[:, 2] > 110)]
    if len(col) < 200:
        return None, None
    h = col[:, 0].astype(int)
    fr = {"blue": np.mean((h >= 95) & (h <= 120)), "purple": np.mean((h > 120) & (h <= 155)),
          "orange": np.mean((h >= 8) & (h <= 25)), "red": np.mean((h > 155) | (h < 8)), "cyan": np.mean((h >= 80) & (h < 95))}
    # multicolorido: gradiente ciano + rosa/roxo no mesmo cabeçalho
    if fr["cyan"] + fr["blue"] > 0.2 and fr["purple"] + fr["red"] > 0.2:
        return "multi", fr
    k = max(("blue", "purple", "orange", "red"), key=lambda k: fr[k])
    return (k if fr[k] > 0.4 else None), fr

def dodge_score(frame, card, half_w, half_h):
    return text_score(frame, card, half_w, half_h, DODGE_TPLS)

def lifesteal_score(frame, card, half_w, half_h):
    """Texto "Life Steal" (assets/skill_lifesteal*.png, recortado da gravação do usuário 2026-09-30) na carta,
    em CINZA: o fundo da carta muda de cor (roxa/laranja/vermelha) e o ícone de coração parece o do HP Regen.
    (Só os templates de Life Steal: o cache _LS também guarda os de esquiva — bug achado no selftest 2026-10-01.)"""
    return text_score(frame, card, half_w, half_h, LIFESTEAL_TPLS)

class Bot:
    def __init__(self, dev=None, dry=False):
        self.dev, self.dry = dev, dry
        self.wave = None                     # onda em jogo (lida no "Next wave: N/15" da montagem; contagem se não ler)
        self.same_since, self.last_state, self.attempts = time.time(), None, 0
        self.stats = {"deaths": 0, "victories": 0, "ad_breaks": 0, "recoveries": 0, "skills": 0}
        self.actions = []           # (estado, ação) — usado no modo replay

    # ---------- ações seguras ----------
    def tap(self, x, y, why=""):
        for (a, b, c, d) in FORBIDDEN:
            if a <= x <= c and b <= y <= d:
                log.error("TOQUE BLOQUEADO (área de compra) em %s,%s", x, y)
                return False
        log.info("TAP %s,%s  (%s)", x, y, why)
        self.actions.append(("TAP", (x, y), why))
        if not self.dry:
            self.dev.tap(x, y)
        return True

    # ---------- decisões por estado ----------
    def choose_skill(self, frame):
        """Escolhe carta SEM ícone de claquete (anúncio opcional). Layout 4 cartas se há carta clara embaixo."""
        v = frame[1500:1650, 250:400].mean()
        cards = SKILL_CARDS_4 if v > 90 else SKILL_CARDS_3
        half_w, half_h = 150, (270 if len(cards) == 4 else 330)
        ok = []
        for i, (cx, cy) in enumerate(cards):
            region = (cx - half_w, cy - half_h, cx + half_w, min(cy + half_h, 1600))
            sc, _ = states.match(frame, "claq_card", region, scales=(0.9, 1.0, 1.1))
            has_ad = sc >= 0.62
            log.info("carta %d (%d,%d): claquete=%.2f -> %s", i, cx, cy, sc, "ANÚNCIO" if has_ad else "ok")
            if not has_ad:
                ok.append(i)
        if not ok:
            log.warning("todas as cartas parecem anúncio; esperando")
            return None
        # acervo: toda tela de skill vai p/ logs/skills/ (ainda não vimos a carta de LIFESTEAL p/ fazer o template)
        if not self.dry:              # (em teste não grava: a cópia virava "carta conhecida" no selftest)
            os.makedirs("logs/skills", exist_ok=True)
            cv2.imwrite(f"logs/skills/{int(time.time())}.png", frame)
        # prioridade do usuário (2026-09-30): 1) lifesteal  2) "Recommend"  3) qualquer uma — sempre sem anúncio
        best = max(((lifesteal_score(frame, cards[i], half_w, half_h), i) for i in ok), default=(0.0, None))
        if best[0] >= LIFESTEAL_THR:
            log.info("carta %d: LIFESTEAL (%.2f) -> escolhida", best[1], best[0])
            return cards[best[1]]
        best = max(((dodge_score(frame, cards[i], half_w, half_h), i) for i in ok), default=(0.0, None))
        if best[0] >= DODGE_THR:
            log.info("carta %d: ESQUIVA (%.2f) -> escolhida", best[1], best[0])
            return cards[best[1]]
        # SOBREVIVÊNCIA (usuário 2026-10-04: a vida virou o gargalo — na floresta ficou 25-50% da onda 2 em diante):
        # HP Regen e depois Shield Generation, ACIMA do dano de arma. Textos em assets/skill_hpregen.png /
        # skill_shieldgen.png (569 cartas: HP Regen 26 >= 0.9, resto <= 0.66 fora uma 0.87; Shield Generation geral
        # >= 0.9; as cartas da ARMA Shield Pulsar "Shield generation +20%/interval -20%" dão ~0.77 e só contam com
        # Shield Pulsar na grade)
        from . import grid_exec as _ge
        hp = max(((text_score(frame, cards[i], half_w, half_h, ["skill_hpregen"]), i) for i in ok), default=(0.0, None))
        if hp[0] >= HPREGEN_THR:
            log.info("carta %d: HP REGEN (%.2f) -> escolhida", hp[1], hp[0])
            return cards[hp[1]]
        sg_thr = SHIELDGEN_THR_PULSAR if "shield_pulsar" in _ge.LAST_GRID_TYPES else SHIELDGEN_THR
        sg = max(((text_score(frame, cards[i], half_w, half_h, ["skill_shieldgen"]), i) for i in ok), default=(0.0, None))
        if sg[0] >= sg_thr:
            log.info("carta %d: SHIELD GENERATION (%.2f) -> escolhida", sg[1], sg[0])
            return cards[sg[1]]
        # DINHEIRO ("Gold +x") nunca, nem se for Recommend (usuário 2026-10-01)
        gold = {i for i in ok if text_score(frame, cards[i], half_w, half_h, GOLD_TPLS) >= GOLD_THR}
        for i in sorted(gold):
            log.info("carta %d: DINHEIRO -> ignorada", i)
        pick = [i for i in ok if i not in gold]
        if not pick:
            log.warning("só sobraram cartas de dinheiro sem anúncio -> pega mesmo assim (não há outra)")
            pick = ok
        # dano das ARMAS EQUIPADAS (usuário 2026-10-01: antes do Recommend; nome da arma no cabeçalho da carta)
        from . import grid_exec
        weapons = [w for w in grid_exec.LAST_GRID_TYPES if w in ATTACK_WEAPONS]
        best = (0.0, None, None)
        for i in pick:
            for w in weapons:
                s = header_score(frame, cards[i], half_h, w)
                if s > best[0]:
                    best = (s, i, w)
        if best[0] >= HEADER_THR:
            log.info("carta %d: DANO da arma equipada %s (%.2f) -> escolhida", best[1], best[2], best[0])
            return cards[best[1]]
        sc, pos = states.match(frame, "recommend", (0, 300, 1080, 1900), scales=(0.8, 0.9, 1.0, 1.1))
        if sc >= 0.65:
            # o selo fica logo ACIMA da carta: a carta dele é a de centro mais próximo abaixo do selo
            below = [i for i, (cx, cy) in enumerate(cards) if cy > pos[1]]
            rec = min(below, key=lambda i: abs(cards[i][0] - pos[0]) + 0.3 * (cards[i][1] - pos[1])) if below else None
            if rec is not None and rec in pick:
                log.info("carta %d: Recommend (%.2f) sem anúncio -> escolhida", rec, sc)
                return cards[rec]
            log.info("Recommend na carta %s tem anúncio/dinheiro/indefinido -> outra", rec)
        # MAIOR RARIDADE pela cor do cabeçalho (usuário 2026-10-01; escada Blue < Purple < Orange < Red < Multicolored)
        rar = [(RARITY.get(card_rarity(frame, cards[i], half_h)[0], 0), -i, i) for i in pick]
        r, _, i = max(rar)
        log.info("carta %d: maior raridade (%s) -> escolhida", i, next((k for k, v in RARITY.items() if v == r), "?"))
        return cards[i]

    def act(self, state, frame, info):
        if state == "GAMEPLAY":
            return "steer"
        if state == "DEFEATED":
            # o bot NUNCA assiste anúncio para reviver: recusa sempre (X do "Try again?")
            return self.tap(*XY["defeated_close"], why=f"recusar reviver (onda {self.wave})")
        if state == "RESUME_BATTLE":
            return self.tap(*XY["resume_ok"], why="voltar à partida inacabada (OK)")
        if state == "ADFREE_POPUP":
            return self.tap(*XY["adfree_close"], why="fechar oferta paga")
        if state == "VICTORY":
            return self.tap(*XY["victory_get"], why="Get (sem Double)")
        if state == "EXIT_DIALOG":
            self.tap(*XY["exit_dialog_x"], why="NÃO sair da batalha (X do 'Exit battle?')")
            time.sleep(1.0)
            # atrás dele fica o menu de pausa: Continue (posição lida da gravação de 2026-09-29)
            return self.tap(*XY["pause_continue"], why="Continue (menu de pausa)")
        if state == "FAILED":
            # 2 versões: com recompensas (Double=anúncio + Get) ou só "Close" (visto ao vivo 2026-09-29)
            sc, pos = states.match(frame, "btn_close", (250, 1600, 830, 1950))
            if sc >= 0.8:
                return self.tap(*pos, why="Close (derrota sem recompensas)")
            return self.tap(*XY["failed_get"], why="Get (sem Double) após derrota")
        if state == "AD_BREAK":
            return "wait_ad_break"       # nada a clicar; o anúncio automático roda sozinho
        if state == "SKILL_SELECT":
            # espera as cartas PARAREM de animar (2026-10-04: o print pegava a carta do meio ainda aumentada e o texto
            # "Shield Generation" não casava): decide quando 2 quadros seguidos são iguais, ou depois de 2 s
            if not self.dry:
                thumb = cv2.resize(cv2.cvtColor(frame[600:1800], cv2.COLOR_BGR2GRAY), (108, 120)).astype(np.int16)
                prev, t0 = getattr(self, "_skill_thumb", None), getattr(self, "_skill_t0", None)
                now = time.time()
                if t0 is None or now - t0 > 10:
                    self._skill_t0, t0 = now, now
                self._skill_thumb = thumb
                if (prev is None or float(np.abs(thumb - prev).mean()) > 2.0) and now - t0 < 2.0:
                    return "wait"
                self._skill_t0 = None
            c = self.choose_skill(frame)
            if c:
                self.stats["skills"] += 1
                return self.tap(*c, why="skill sem anúncio")
            return "wait"
        if state == "GRID":
            if getattr(self, "assemble_grid", False):
                return "assemble"          # run.py chama grid_exec.assemble (merges/encaixe/refresh/Play)
            return self.tap(*XY["grid_play"], why="Play (montagem)")
        if state == "REWARDS_CLOSE":
            return self.tap(*XY["rewards_close"], why="fechar recompensas")
        if state == "LEVEL_MENU":
            self.wave = None
            # `run.py --voltar N` (pedido do usuário 2026-10-02: jogar a NEVE, nível 15, em vez da floresta, 16):
            # só na 1ª visita ao menu da sessão, toca a seta da esquerda N vezes (o menu lembra o nível escolhido)
            if getattr(self, "go_back", 0) > 0:
                self.go_back -= 1
                return self.tap(*XY["level_prev"], why=f"nível anterior (faltam {self.go_back})")
            return self.tap(*XY["level_play"], why="jogar fase")       # TODO: checar energia
        return "wait_unknown"

    # ---------- passo do loop ----------
    def step(self, frame):
        state, info = states.classify(frame)
        now = time.time()
        if state != self.last_state:
            log.info("ESTADO %s -> %s %s", self.last_state, state, info)
            if state == "GRID":
                n = states.read_next_wave(frame)
                prev = self.wave
                # dentro da partida a onda só cresce: leitura menor que a anterior = erro de leitura -> anterior + 1
                if n and (prev is None or n >= prev):
                    self.wave = n
                else:
                    self.wave = (prev + 1) if prev else n
                log.info("onda: próxima = %s (lida %s, antes %s)", self.wave, n, prev)
            self.last_state, self.same_since, self.attempts = state, now, 0
            # conta na ENTRADA da tela (o bot toca 2-3x na mesma tela; antes cada toque contava)
            key = {"DEFEATED": "deaths", "VICTORY": "victories", "AD_BREAK": "ad_breaks"}.get(state)
            if key:
                self.stats[key] += 1
        result = self.act(state, frame, info)
        self.attempts += 1
        return state, result
