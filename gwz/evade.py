"""Esquiva reativa (padrões do jogo são aleatórios -> nada de decorar).

v2 (2026-09-29), a partir dos quadros salvos em logs/hits/: o dano vinha de "muros" de balas (duas
fileiras de discos na largura toda, vãos de ~110 px). O modelo antigo (raio de risco 126 px, grade de
60 px) via a tela toda igualmente perigosa e ficava parado. Agora:
 1. velocidade de cada bala pela posição PREVISTA no quadro anterior;
 2. para cada destino candidato (a cada 20 px na horizontal), a MENOR distância de cada bala ao
    destino dentro da janela de tempo em que a nave estará lá (cálculo exato, sem "pular" bala rápida),
    mais o risco no caminho até lá;
 3. custo = risco + deslocamento + leve preferência por ficar embaixo e longe das paredes.
Vetorizado em numpy (dezenas de candidatos x dezenas de balas em ~1 ms).
"""
import math
import numpy as np

from shooter_core.config import TrackerConfig
from shooter_core.tracking import Tracker

W, H = 1080, 2340
Y_RANGE = (1880, 2150)         # faixa de altura da nave (embaixo = mais tempo p/ reagir; 1750 deixava subir até os mini aviões/drones — ao vivo 2026-09-30 5 tiros em y~1750)
HIT_R = 120.0                  # colisão real medida ~80 px (logs/hits); 120 = folga p/ latência e ruído.
                               # Escolhido no simulador offline (tools/sim_evade.py, balas reais gravadas):
                               # 80/margem 8 -> ~10-13 acertos/min e ~43 trocas de lado/min;
                               # 120/margem 90 -> ~8.3 acertos/min e ~18.5 trocas/min (nave parada: 19.4)
RISK_R = 1.4 * HIT_R
SWITCH_MARGIN = 90.0           # histerese: só troca de destino se o novo for MUITO melhor (tremedeira)
PREV_KEEP = 320.0              # destino anterior mais longe que isso (px) é esquecido pela histerese
SWITCH_MARGIN_CALM = 15.0      # sem balas à vista (só importa com a mira ligada)
# bordas: o usuário mostrou que a nave pode ir até ~metade para fora da tela, então a lateral é rota de fuga
# Simulador: (80, 250) 4.55 balas/min; (30, 120) 4.43; sem custo 4.55-4.62; borda 10 px 5.26 (pior).
# AO VIVO (2026-09-30 22:24) com (30, 120): nave passou de x~1060 / ficou em x~31-62, metade fora da tela ->
# detecção perdia a nave (score 0.31), o bot parava o dedo e ela apanhava encurralada: 2 mortes em 2.5 min.
# O simulador não "perde" a nave, por isso não viu. Voltou a afastar das bordas.
X_EDGE = 140                   # destino mais perto da borda permitido (centro da nave, px). 100 ainda deixava a
                               # nave presa/perdida no canto esquerdo ao vivo (partida I, ondas 10+)
WALL_ZONE, W_WALL = 260, 0.0004   # simulador (deserto 8.5 min): 100/220 9.96, 170/300 11.4 acertos/min; ao vivo a
                                  # borda causou 30% dos tiros (nave perdida) -> meio-termo + empurrão de recuperação
SHIP_SPEED = 1000.0            # px/s efetivos da nave (dedo 18 px/evento a 60 Hz x ganho ~1.1)
DEFAULT_V = (0.0, 850.0)       # px/s: medido ao vivo na fase da estrada (vy ~700-870); usado sem histórico
MATCH_GATE = 60.0              # px: distância máxima entre a posição prevista e a observada

V_MAX = 1800.0                 # px/s: balas rápidas medidas na fase do deserto chegam a ~1600

# GIRO das balas (curvas): tools/curve_eval.py (2026-10-01) — 25% das balas giram > 30 graus/s e nelas a previsão
# em linha reta erra 2-3.5x mais (mediana 76-120 px a 0.4 s, x 34 px nas retas); prever o giro direto ficou PIOR
# (ruído) -> margem extra no risco proporcional ao giro medido e ao tempo até a bala chegar.
_TRACKER = Tracker()       # rastreador do core (shooter_core.tracking), um só para balas, inimigos e drones
_TURN = _TRACKER.turn     # (x//4, y//4) da bala -> giro suavizado (graus/s), atualizado pelo rastreador das balas
TURN_MIN = 45.0           # abaixo disso é ruído de medida
TURN_K = 1.0              # px de margem por (grau/s x s). Simulador c/ 120 trechos RECENTES do bot (21 min, com
                          # balas curvas): 0 -> 3.03, 0.3 -> 2.91, 0.6 -> 2.83, 1.0 -> 2.78 (-8%), 1.5 -> 2.83 acertos/min
TURN_CAP = 150.0          # margem máxima (px)

def _tkey(x, y):
    return (int(x) // 4, int(y) // 4)

NEIGHBOR_R = 0.0     # px (0 = desligado); simulador: 80/150 px = 72 x 68 balas (pior) -> desligado
LSQ_N = 0            # reta nas últimas N observações (0 = suavização antiga). Simulador: N=4 -> 68 x 68 balas (igual),
                     # N=6 -> 73: a previsão melhora na mediana mas os erros GRANDES (que acertam) não -> desligado

def tracker_config():
    """TrackerConfig do core com as constantes atuais deste módulo (tools/tune.py muda com setattr)."""
    return TrackerConfig(default_v=tuple(DEFAULT_V), v_max=V_MAX, lsq_n=LSQ_N, neighbor_r=NEIGHBOR_R,
                         coast_s=COAST_S, coast_near=COAST_NEAR, coast_vmin=COAST_VMIN,
                         enemy_obst=ENEMY_OBST, enemy_vy_min=ENEMY_VY_MIN, enemy_near=ENEMY_NEAR,
                         enemy_half=ENEMY_HALF, enemy_self=ENEMY_SELF)

def track_velocities(prev, cur, dt, smooth=0.5, store_turn=False):
    """Rastreio 1-para-1 usando a velocidade que CADA bala já tinha (prev = saída anterior, (x,y,vx,vy)).
    Implementado no core (shooter_core.Tracker.update). Por que assim (2026-09-30): balas de 1200-1600 px/s andam
    ~75 px por quadro; o método antigo (previsão fixa 850 px/s para baixo + portão de 60 px) perdia a associação
    e chutava a velocidade -> 'bala tardia' / previsão errada. tools/track_eval.py: erro a 0.5 s 30.6 -> 19.9 px."""
    _TRACKER.cfg = tracker_config()
    return _TRACKER.update(prev, cur, dt, smooth=smooth, store_turn=store_turn)

def ages_of(lst):
    """Quantas vezes cada bala de uma lista devolvida por track_velocities já foi vista (None se desconhecido)."""
    return _TRACKER.ages_of(lst)


def estimate_velocities(prev, cur, dt):
    """prev/cur: listas de (x,y,r,k). Devolve lista de (x,y,vx,vy).
    Associação pela posição PREVISTA (anterior + DEFAULT_V*dt), não pela mais próxima: em rajadas
    densas a vizinha mais próxima costuma ser a bala de trás, o que dava velocidades "subindo"."""
    out = []
    ex, ey = DEFAULT_V[0] * dt, DEFAULT_V[1] * dt
    for (x, y, *_ ) in cur:
        best, bd = None, MATCH_GATE ** 2
        for (px, py, *_ ) in prev:
            d = (x - px - ex) ** 2 + (y - py - ey) ** 2
            if d < bd:
                best, bd = (px, py), d
        if best and dt > 0:
            vx, vy = (x - best[0]) / dt, (y - best[1]) / dt
            if abs(vx) > 1500 or abs(vy) > 1800 or vy < -200:
                vx, vy = DEFAULT_V
        else:
            vx, vy = DEFAULT_V
        out.append((x, y, vx, vy))
    return out

# (testado e descartado: memória curta de balas "piscando" — simulador ANTIGO piorou de 4.4 p/ 4.8-5.6 acertos/min;
#  re-testado no simulador fiel em 2026-10-02 com `coast`, ver NEXT_STEPS §3)

COAST_S = 0.3        # s que uma bala rastreada que sumiu continua prevista
COAST_NEAR = 260.0   # só perto da nave (px); None = qualquer distância
COAST_VMIN = 300.0   # só balas rápidas (px/s) — paradas/lentas costumam ser falso positivo

def coast(prev_vel, vel, coast_prev, dt, ship, max_age=None, near=None, vmin=None):
    """Balas que SUMIRAM do detector continuam previstas em linha reta por até max_age s (oclusão por explosão,
    sobreposição com a nave...). Implementado no core (shooter_core.Tracker.coast)."""
    _TRACKER.cfg = tracker_config()
    return _TRACKER.coast(prev_vel, vel, coast_prev, dt, ship, max_age=max_age, near=near, vmin=vmin)

def _min_dist(P, V, Q, t_lo, t_hi):
    """Menor distância entre as balas (P + V t) e os pontos Q no intervalo [t_lo, t_hi].
    P,V: (n,2); Q: (m,2); t_lo,t_hi: (m,) -> dist (m,n) e instante (m,n)."""
    D = P[None, :, :] - Q[:, None, :]                      # (m,n,2)
    vv = (V ** 2).sum(1) + 1e-9                            # (n,)
    t = -(D * V[None]).sum(2) / vv[None]                   # instante de maior aproximação sem limites
    t = np.clip(t, t_lo[:, None], t_hi[:, None])
    R = D + V[None] * t[..., None]
    return np.sqrt((R ** 2).sum(2)), t

def _risk(dist, t):
    """Risco por (candidato, bala): cresce perto e mais cedo (menos tempo p/ corrigir)."""
    w = np.clip(1.0 - dist / RISK_R, 0.0, None) ** 2
    return (w / (0.25 + t)).sum(1)

W_OBS = 0.0      # peso do inimigo como obstáculo (0 = desligado; ajustar no simulador)
W_AIM = 0.0      # bônus por ficar embaixo de inimigo = mirar (0 = desligado; ajustar no simulador)

W_SHOOTER = 80.0    # custo de ficar na "coluna de tiro" logo abaixo de um inimigo próximo. Simulador (7 min, com
                    # inimigos): colisões com inimigo 5.7 -> 4.0/min, balas ~iguais (5.5 x 5.6). O benefício
                    # principal (sair da mira ANTES do disparo) o simulador não mede: avaliar ao vivo.

def _enemy_terms(Q, S, enemies, w_obs, w_aim, w_shooter=0.0):
    """Custo extra por inimigos (x, y, r): obstáculo no destino/caminho, bônus de mira (negativo) e coluna de
    tiro (as balas 'tardias' nascem no meio da tela, disparadas por inimigos próximos: antecipar quem atira)."""
    E = np.asarray(enemies, float).reshape(-1, 3)
    if not len(E):
        return 0.0
    out = np.zeros(len(Q))
    if w_shooter:
        above = E[:, 1] < S[1] - 150
        for ex, ey, er in E[above]:
            near = np.clip(1.0 - (S[1] - ey) / 1400.0, 0.0, 1.0)      # inimigo mais perto = coluna mais perigosa
            out += w_shooter * near * np.exp(-((Q[:, 0] - ex) / max(er, 90.0)) ** 2)
    if w_obs:
        for P in (Q, (Q + S) / 2):                         # destino e ponto médio do caminho
            d = np.hypot(P[:, None, 0] - E[None, :, 0], P[:, None, 1] - E[None, :, 1]) - E[None, :, 2]
            out += w_obs * (np.clip(1.0 - d / HIT_R, 0.0, None) ** 2).sum(1)
    if w_aim:
        above = E[:, 1] < S[1] - 100                       # só inimigo ACIMA (as armas atiram p/ cima)
        if above.any():
            A = E[above]
            width = np.maximum(A[:, 2], 60.0)
            out -= w_aim * np.exp(-((Q[:, None, 0] - A[None, :, 0]) / width[None, :]) ** 2).max(1)
    return out

def choose_target(ship, bullets, t0=0.2, horizon=0.7, max_dx=320, step=20, prev=None,
                  enemies=None, w_obs=None, w_aim=None, w_shooter=None, shooters=None):
    """ship=(x,y); bullets=[(x,y,vx,vy)]; t0 = atraso até a nave começar a obedecer;
    prev = destino escolhido no quadro anterior (histerese); enemies = [(x, y, r)] (gwz.enemies).
    Devolve (tx,ty)."""
    w_obs = W_OBS if w_obs is None else w_obs
    w_aim = W_AIM if w_aim is None else w_aim
    sx, sy = ship
    xs = np.clip(np.arange(sx - max_dx, sx + max_dx + 1, step), X_EDGE, W - X_EDGE)
    ys = np.clip(sy + np.array([-80.0, -40.0, 0.0, 40.0, 80.0]), *Y_RANGE)
    pts = [(x, y) for x in np.unique(xs) for y in np.unique(ys)]
    # histerese só vale para destino PRÓXIMO: com margem 90 um destino velho do outro lado da tela
    # "ainda servia" e a nave cruzava a tela inteira pelas balas (visto ao vivo: nave x=1026, alvo x=80)
    if prev is not None and abs(prev[0] - sx) > PREV_KEEP:
        prev = None
    # destino antigo PERTO DA BORDA sem bala a menos de 500 px: esquece (a nave ficava parada no canto)
    if prev is not None and min(prev[0], W - prev[0]) < WALL_ZONE and \
            not any((b[0] - sx) ** 2 + (b[1] - sy) ** 2 < 500 ** 2 for b in bullets):
        prev = None
    # (testado e REVERTIDO: esquecer o destino quando não há balas piorou o simulador de 4.4 p/ 5.0
    #  acertos/min — a detecção falha 1-2 quadros e a nave parava no meio da esquiva)
    if prev is not None:
        pts.append((float(prev[0]), float(prev[1])))
    Q = np.array(pts, float)
    dist_move = np.hypot(Q[:, 0] - sx, Q[:, 1] - sy)
    t_arr = t0 + dist_move / SHIP_SPEED
    # perto da parede sobra só um lado para fugir (ao vivo a nave ficou presa em x~130 e apanhou ali):
    # custo cresce rápido abaixo de 250 px da borda
    wall = np.clip(WALL_ZONE - np.minimum(Q[:, 0], W - Q[:, 0]), 0, None)
    cost = 0.02 * np.abs(Q[:, 0] - sx) + 0.06 * np.abs(Q[:, 1] - sy) \
        + 0.004 * np.abs(Q[:, 0] - W / 2) + 0.03 * np.clip(Y_RANGE[1] - 150 - Q[:, 1], 0, None) \
        + W_WALL * wall ** 2
    if bullets:
        B = np.asarray(bullets, float)
        P, V = B[:, :2], B[:, 2:4]
        # bala que CURVA: margem extra = TURN_K x giro x tempo (até TURN_CAP)
        if TURN_K > 0 and _TURN:
            wexc = np.array([max(0.0, _TURN.get(_tkey(x, y), 0.0) - TURN_MIN) for x, y in P]) * TURN_K
        else:
            wexc = None
        def margin(d, t):
            return d if wexc is None else d - np.minimum(wexc[None, :] * t, TURN_CAP)
        # no destino: da chegada até o fim do horizonte
        d1, t1 = _min_dist(P, V, Q, t_arr, t_arr + horizon)
        d1 = margin(d1, t1)
        # no caminho: 3 pontos (1/4, 1/2, 3/4), cada um só no instante em que a nave passa por ali.
        # (só o ponto médio deixava passar a travessia de colunas de balas: a nave "mirava nos tiros")
        S = np.array([sx, sy])
        travel = t_arr - t0
        path = 0.0
        for f in (0.25, 0.5, 0.75):
            M = S + (Q - S) * f
            lo = t0 + travel * (f - 0.25)
            d2, t2 = _min_dist(P, V, M, lo, lo + travel * 0.5 + 0.05)
            path = path + _risk(margin(d2, t2), t2)
        cost = cost + 100.0 * (_risk(d1, t1) + 0.7 * path)
    w_shooter = W_SHOOTER if w_shooter is None else w_shooter
    S2 = np.array([sx, sy], float)
    if enemies and (w_obs or w_aim or (w_shooter and shooters is None)):
        cost = cost + _enemy_terms(Q, S2, enemies, w_obs, w_aim, w_shooter if shooters is None else 0.0)
    if shooters and w_shooter:                 # inimigos que ATIRAM (barras de vida) separados dos drones-alvo
        cost = cost + _enemy_terms(Q, S2, shooters, 0.0, 0.0, w_shooter)
    i = int(np.argmin(cost))
    # sem bala à vista a margem é pequena: deixa a MIRA trocar o destino (bônus de mira < 90), mas ainda
    # mantém o destino de fuga em andamento (andar até ele custa < 15)
    margin = SWITCH_MARGIN_CALM if (not bullets and enemies and w_aim) else SWITCH_MARGIN
    if prev is not None and cost[-1] <= cost[i] + margin:
        i = len(Q) - 1                        # o destino anterior ainda serve: mantém
    return float(Q[i, 0]), float(Q[i, 1])

# ---------- planejador de fuga (casos difíceis) ----------
# Pedido do usuário (2026-09-30): nada de regra fixa ("cortina -> vai p/ a direita"); o bot deve PENSAR a fuga
# sozinho. Teste offline na cortina da partida K: nenhuma rota em LINHA RETA escapava (nem parada, nem
# direita/cima), mas existia um caminho em ZIGUE-ZAGUE pelos vãos. Aqui: simula ~1.2 s à frente com as balas
# previstas, expande no tempo todas as casas que a nave consegue alcançar na velocidade dela sem bala a menos de
# PLAN_R, e segue o 1º trecho de um caminho que sobrevive (refeito a cada quadro = horizonte deslizante).
PLAN_STEP = 40            # px da grade de busca
PLAN_DT = 0.04            # s por passo (40 px / 0.04 s = 1000 px/s ~ velocidade da nave)
PLAN_HORIZON = 1.2        # s simulados à frente
PLAN_R = 85.0             # px: colisão real ~80
PLAN_Y = (1450, 2150)     # pode SUBIR além da faixa normal só na fuga
PLAN_LEAD = 0.3           # s de caminho à frente usados como destino imediato
_PLAN = {"end": None}

def _grow(m):
    r = m.copy()
    r[1:] |= m[:-1]; r[:-1] |= m[1:]; r[:, 1:] |= m[:, :-1]; r[:, :-1] |= m[:, 1:]
    r[1:, 1:] |= m[:-1, :-1]; r[1:, :-1] |= m[:-1, 1:]; r[:-1, 1:] |= m[1:, :-1]; r[:-1, :-1] |= m[1:, 1:]
    return r

def danger(target, ship, bullets, t0=0.1, horizon=0.7, r=HIT_R * 0.75):
    """O destino escolhido pela esquiva normal tem bala passando a menos de r? (aciona o planejador)"""
    if not bullets:
        return False
    B = np.asarray(bullets, float)
    tq = t0 + math.hypot(target[0] - ship[0], target[1] - ship[1]) / SHIP_SPEED
    Q = np.array([target, ((target[0] + ship[0]) / 2, (target[1] + ship[1]) / 2)], float)
    d, _ = _min_dist(B[:, :2], B[:, 2:4], Q, np.array([tq, t0]), np.array([tq + horizon, tq]))
    return bool(d.min() < r)

def plan_escape(ship, bullets, t0=0.1, need_full=False):
    """Destino (x, y) do 1º trecho de um caminho de fuga, ou None se nenhum lugar é melhor que ficar."""
    if not bullets:
        return None
    sx, sy = ship
    xs = np.arange(X_EDGE, W - X_EDGE + 1, PLAN_STEP, dtype=float)
    ys = np.arange(PLAN_Y[0], PLAN_Y[1] + 1, PLAN_STEP, dtype=float)
    GX, GY = np.meshgrid(xs, ys)
    B = np.asarray(bullets, float)
    B = B[(B[:, 1] > PLAN_Y[0] - 1500) & (B[:, 1] < PLAN_Y[1] + 200)]
    reach = np.zeros(GX.shape, bool)
    reach[np.argmin(np.abs(ys - sy)), np.argmin(np.abs(xs - sx))] = True
    hist, t, n = [reach], 0.0, int(PLAN_HORIZON / PLAN_DT)
    for k in range(n):
        t += PLAN_DT
        if t > t0:
            reach = _grow(reach)
        if len(B):
            P = B[:, :2] + B[:, 2:4] * t
            near = (P[:, 1] > PLAN_Y[0] - PLAN_R) & (P[:, 1] < PLAN_Y[1] + PLAN_R)
            if near.any():
                P = P[near]
                d2 = ((GX[..., None] - P[None, None, :, 0]) ** 2 + (GY[..., None] - P[None, None, :, 1]) ** 2).min(-1)
                reach = reach & (d2 > PLAN_R ** 2)
        hist.append(reach)
        if not reach.any():
            break
    # último passo com alguma casa viva (sobreviver o mais longe possível); entre elas: embaixo, perto do centro,
    # perto do plano anterior (evita trocar de rota a cada quadro)
    K = max(i for i, h in enumerate(hist) if h.any())
    if K == 0 or (need_full and K < n):
        return None             # need_full: só usa caminho que sobrevive o horizonte inteiro
    end_ok = hist[K]
    cost = 0.5 * np.abs(GY - 2000) + 0.15 * np.abs(GX - W / 2)
    if _PLAN["end"] is not None:
        cost += 0.6 * np.hypot(GX - _PLAN["end"][0], GY - _PLAN["end"][1])
    cost = np.where(end_ok, cost, np.inf)
    iy, ix = np.unravel_index(np.argmin(cost), cost.shape)
    _PLAN["end"] = (xs[ix], ys[iy])
    # volta no tempo até o passo do destino imediato (prefere ficar parado quando dá)
    lead = min(K, max(1, int(round((t0 + PLAN_LEAD) / PLAN_DT))))
    for k in range(K - 1, lead - 1, -1):
        for dy, dx in ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            jy, jx = iy + dy, ix + dx
            if 0 <= jy < len(ys) and 0 <= jx < len(xs) and hist[k][jy, jx]:
                iy, ix = jy, jx
                break
    return float(xs[ix]), float(ys[iy])

# NAVES INIMIGAS como obstáculo MÓVEL (usuário 2026-10-01: "colide com as naves que vêm para baixo"). Nos prints
# de tiro: 52% sem bala perto; desses, 27% com nave inimiga/drone a < 260 px (colunas de drones e formações
# descendo em cima da nave). Antes os inimigos só entravam como "coluna de tiro" (W_SHOOTER), nunca como
# obstáculo. Agora: velocidade de cada inimigo pelo rastreador e, os que DESCEM (ou estão perto), viram 3 pontos
# (corpo) com a velocidade deles, tratados como balas pela esquiva.
ENEMY_OBST = True
ENEMY_VY_MIN = 60.0       # px/s p/ baixo. Simulador c/ inimigos (4 vídeos): atual 5.00 balas + 3.85 colisões/min;
ENEMY_NEAR = 300.0        # vy>60/perto<300 -> 4.17 + 2.63 (-23% no total); 120/380 4.49+3.66; 30/300 4.04+2.89; 200/0 3.91+3.85
ENEMY_HALF = 55.0         # px: meia largura do corpo (pontos em -H, 0, +H)
ENEMY_SELF = 100.0        # px: "inimigo" colado no centro da nave = falso positivo da própria nave

def enemy_points(tracked, ship):
    """tracked = [(x, y, vx, vy)] dos inimigos (track_velocities); devolve pontos-obstáculo (x, y, vx, vy).
    Implementado no core (shooter_core.Tracker.enemy_points)."""
    _TRACKER.cfg = tracker_config()
    return _TRACKER.enemy_points(tracked, ship)

def risk_at(px, py, bullets, horizon=0.7, t0=0.0):
    """Risco num ponto fixo (diagnóstico/log)."""
    if not bullets:
        return 0.0
    B = np.asarray(bullets, float)
    d, t = _min_dist(B[:, :2], B[:, 2:4], np.array([[px, py]], float), np.array([t0]), np.array([t0 + horizon]))
    return float(_risk(d, t)[0])

