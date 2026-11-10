"""Planejador de esquiva do GWZ = `shooter_core.SpaceTimePlanner` com os números medidos NESTE jogo.

O algoritmo (caminho que maximiza a menor folga até as balas, com termo de ofensiva) mora no core
(AutoSpaceShooterCore). Aqui ficam só os parâmetros do Galaxy Wing Zero (tela 1080x2340 do S22) e o histórico
de como cada um foi escolhido. A API de módulo continua a mesma de antes (plan, linear_predictor, setup e as
constantes), então run.py e tools/ (tune.py altera as constantes com setattr) não mudam.

predict(t) -> array (n,2) das balas no instante t (s a partir de agora): previsão linear (`linear_predictor`)
ou o futuro verdadeiro (oráculo, só no simulador).
"""
from shooter_core.config import PlannerConfig
from shooter_core.planner import SpaceTimePlanner, linear_predictor, young_xsv as _young_xsv

W = 1080
GX_STEP = 30.0
GY_STEP = 40.0
DT = 0.04
HORIZON = 1.2
Y_RANGE = (1650.0, 2150.0)  # tools/tune.py (3 gravações): mín. 1500/1650/1800/1880 = balas 68/68/71/75, colisões c/
                            # naves 49/40/30/31 (ao vivo vários tiros com a nave em y 1536-1758) -> 1650
X_RANGE = (60.0, 1020.0)    # 2026-10-06: (40,1040) testado — simulador 73 -> 57 balas, mas AO VIVO (nível 19, 20:39) a
                            # nave ficou encostada em x=1003-1080 e levou 2 de 7 tiros sem conseguir sair -> voltou a 60
SPEED = 1000.0              # px/s efetivos (margem sobre os ~1100 medidos)
REAL_SPEED = 1100.0         # px/s medidos (onde a nave estará quando o novo comando valer)
LEAD = 0.12                 # s depois de t0: casa do caminho usada como destino imediato (simulador fiel:
                            # 0.04/0.08/0.12/0.16 -> 0.12 melhor; chegar ANTES do plano é tão ruim quanto depois)
SIGMA0 = 6.0                # px de incerteza no instante 0
SIGMA_V = 60.0              # px de incerteza por s de previsão (balas curvas, velocidade mal medida)
CLEAR_CAP = 160.0           # folga acima disso não importa (já está seguro)
QUANT = 8.0                 # px: folgas dentro disso contam como iguais (desempate pelo custo)
PREFER_Y = 2000.0           # altura preferida da nave (embaixo = mais tempo para ver as balas)
VECTOR = False

# ---- OFENSIVA (NEXT_STEPS §1, 2026-10-02): o bot levava 2x o tempo do usuário p/ limpar a fase (25.6 x 12.1 min na
# neve) com MENOS contato por minuto -> faltava matar rápido. As armas principais (Gatling/Beamer) atiram RETO p/
# cima: ficar na coluna de um inimigo = dano. Custo por passo fora da coluna dos inimigos acima da nave, e folga
# acima de SAFE_CAP conta como "igualmente seguro" (só foge da coluna quando o risco passa desse limiar).
# tools/tune.py (neve+ilhas, 24 min): W_DPS 0 -> 20 balas, 35% na mira; 10/AIM 90 -> 24 balas, 56% (= usuário:
# 54-56%); 20 -> 28, 62%; 40+ ou SAFE_CAP 60 -> muito mais balas. Partida do próprio bot: 38 -> 42 balas, 37 -> 56%.
W_DPS = 10.0         # peso do termo de ofensiva (0 = desligado)
SAFE_CAP = CLEAR_CAP  # folga (px) a partir da qual o caminho já é "seguro o bastante" (60 piorou)
AIM_W = 90.0         # meia-largura (px) da coluna de tiro de um inimigo

YOUNG_AGE = 2        # bala vista até este nº de vezes = velocidade incerta (tools/pred_eval: 40% dos erros > 100 px)
YOUNG_SV = 0.0       # px/s de incerteza extra p/ essas balas (0 = desligado; ajustado no simulador)

_core = None
_geom = None


def _geometry():
    return (W, tuple(X_RANGE), tuple(Y_RANGE), GX_STEP, GY_STEP, DT, SPEED, PREFER_Y)


def config():
    """PlannerConfig do core com as constantes atuais deste módulo."""
    return PlannerConfig(width=float(W), x_range=tuple(X_RANGE), y_range=tuple(Y_RANGE), gx_step=GX_STEP,
                         gy_step=GY_STEP, dt=DT, horizon=HORIZON, speed=SPEED, real_speed=REAL_SPEED, lead=LEAD,
                         sigma0=SIGMA0, sigma_v=SIGMA_V, clear_cap=CLEAR_CAP, quant=QUANT, prefer_x=W / 2,
                         prefer_y=PREFER_Y, w_dps=W_DPS, safe_cap=SAFE_CAP, aim_w=AIM_W, vectorized=VECTOR)


def setup():
    """(Re)monta o planejador do core a partir das constantes (chamar após mudar GX_STEP/DT/...)."""
    global _core, _geom, GX, GY
    _core = SpaceTimePlanner(config())
    _geom = _geometry()
    GX, GY = _core.GX, _core.GY


def _sync():
    """Constantes que não mexem na grade são copiadas a cada plano (tools/tune.py muda com setattr)."""
    if _geometry() != _geom:
        setup()
    c = _core.cfg
    c.horizon, c.lead, c.sigma0, c.sigma_v = HORIZON, LEAD, SIGMA0, SIGMA_V
    c.clear_cap, c.quant, c.real_speed, c.vectorized = CLEAR_CAP, QUANT, REAL_SPEED, VECTOR
    c.w_dps, c.safe_cap, c.aim_w = W_DPS, SAFE_CAP, AIM_W
    return _core


def start_after(ship, prev_target, t0, speed=None):
    p = _sync()
    if speed is None:
        return p.start_after(ship, prev_target, t0)
    old = p.cfg.real_speed
    p.cfg.real_speed = speed
    try:
        return p.start_after(ship, prev_target, t0)
    finally:
        p.cfg.real_speed = old


def plan(ship, predict, t0=0.10, prev_end=None, prev_target=None, enemies=None, w_dps=None, sigma0=None, y_min=None):
    """Devolve (destino (x,y), folga do melhor caminho em px, caminho [(t, x, y)]). Ver SpaceTimePlanner.plan."""
    return _sync().plan(ship, predict, t0=t0, prev_end=prev_end, prev_target=prev_target, enemies=enemies,
                        w_dps=w_dps, sigma0=sigma0, y_min=y_min)


def young_xsv(n_total, ages):
    """Vetor xsv: YOUNG_SV para as balas vistas até YOUNG_AGE vezes; 0 para o resto (memória/inimigos)."""
    return _young_xsv(n_total, ages, YOUNG_AGE, YOUNG_SV)


__all__ = ["plan", "linear_predictor", "setup", "young_xsv", "start_after", "config"]

setup()
