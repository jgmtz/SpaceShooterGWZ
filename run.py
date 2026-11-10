"""Loop principal.  python run.py [--dry]   (Ctrl+C para parar)
Ordem: captura -> classifica -> age -> verifica -> se preso, escala recuperação.
NUNCA clica às cegas: estado UNKNOWN só gera espera, screenshot e (após timeouts) BACK/relaunch.
"""
import sys, time, os, math, json, logging, collections, threading, cv2
import numpy as np
from gwz.device import Device, APP
from gwz.bot import Bot
from gwz.pilot import Pilot
from gwz import vision, evade, states, grid_exec, enemies, planner
from gwz import pilot as _P
# núcleo reutilizável (AutoSpaceShooterCore): planejador assíncrono e estimativa robusta da nave
from shooter_core.loop import AsyncPlanner
from shooter_core.ship import ShipEstimator
from shooter_core.config import ShipConfig

os.makedirs("logs/unknown", exist_ok=True)
os.makedirs("logs/grid", exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.FileHandler("logs/run.log", encoding="utf-8"), logging.StreamHandler()])
log = logging.getLogger("gwz")

T_UNKNOWN_BACK = 25      # s em UNKNOWN antes de tentar BACK (checando o app em primeiro plano)
T_UNKNOWN_RELAUNCH = 90  # s em UNKNOWN antes de reabrir o jogo
T_FROZEN = 120           # s com a tela IDÊNTICA (e o bot tocando) até considerar o jogo congelado
T_STUCK = 120            # s no mesmo estado (não-gameplay) sem mudar => recuperação
ENERGY_WAIT = 300        # s de espera se Play do menu não avança (energia insuficiente?)
GP_GRACE = 1.5           # s sem o HUD antes de sair da esquiva (o flash vermelho de dano esconde o HUD)
PLANNER = "--esquiva-antiga" not in sys.argv   # gwz/planner.py (padrão desde 2026-10-02)
PLAN_ASYNC = "--plano-sincrono" not in sys.argv  # planejador numa thread (NEXT_STEPS §5)
if "--sem-ofensiva" in sys.argv:                 # teste A/B (2026-10-03): só desliga o termo de ofensiva
    planner.W_DPS = 0.0

# MODO DEFENSIVO (pedido do usuário 2026-10-03): a nave começa com só 1000 de vida e cada tiro tira ~500-600 ->
# 2 erros na onda 1 matam (partida das 00:10 morreu em 7 s). Enquanto estiver frágil: sem ofensiva, margem maior e
# nave embaixo. tools/tune.py (neve+ilhas+partida do bot): W_DPS 0 + SIGMA0 20 + y >= 1800 = 65 balas / 19 colisões
# x modo normal (ofensiva) 70 / 45; SIGMA0 35 piorou.
DEF_WAVES = 2          # ondas iniciais sempre defensivas
DEF_HP = 0.5           # vida (barra verde) abaixo disso -> defensivo
HP_MEDIAN_N = 7        # leituras da barra (1 a cada 3 quadros, ~1.5 s) na mediana
SHIP_JUMP_PX = 90      # px: tolerância fixa entre a nave detectada e a prevista pelo dedo
SHIP_JUMP_V = 900      # px/s a mais de tolerância pelo tempo desde a última detecção aceita
SHIP_TRUST_S = 0.4     # conflito por mais que isso -> volta a confiar na detecção
DEF_ENTRY_S = 1.5      # s no começo de cada trecho de partida (a nave entra lá em cima)
DEF_PARAMS = {"w_dps": 0.0, "sigma0": 20.0, "y_min": 1800.0}

def _ship_estimator(pilot):
    """Nave detectada x prevista pelo dedo (shooter_core.ShipEstimator) com os limites medidos neste jogo."""
    def predict(s, t_then, t_now):   # nave vista em t_then + movimento do dedo enviado até t_now (x GAIN)
        a0 = pilot.virt_at(t_then - _P.LATENCY); a1 = pilot.virt_at(t_now - _P.LATENCY)
        return s[0] + (a1[0] - a0[0]) * _P.GAIN, s[1] + (a1[1] - a0[1]) * _P.GAIN
    cfg = ShipConfig(teleport_px=400, teleport_dt=0.3, jump_px=SHIP_JUMP_PX, jump_v=SHIP_JUMP_V, jump_window=0.6,
                     trust_s=SHIP_TRUST_S, hold_s=0.5, bounds=(0.0, 1080.0, 1300.0, 2340.0))
    return ShipEstimator(cfg, predict)
PLAN_T0 = 0.15           # s até o comando valer (simulador fiel: 0.10 -> 0.15 reduziu contatos ~3x)
ROI_ABOVE = 1500 if PLANNER else 750   # px acima da nave onde procurar balas (planejador olha 1.2 s à frente:
                         # simulador fiel 750 -> 1500 = 15 -> 5 contatos na neve, 30 -> 21 na outra gravação)
REACT = 0.10             # s: antecipação na previsão de risco (0.1 foi o melhor no simulador offline)

HIT_RED = 100            # vision.damage_flash acima disso = levou dano
# estados dentro da batalha: aqui BACK abre pausa -> "Exit battle?" (visto ao vivo em 2026-09-29)
BATTLE = ("GAMEPLAY", "GRID", "SKILL_SELECT", "EXIT_DIALOG")
SHIP_FP_R = 30           # px: só o falso positivo do miolo da nave (visto em dx~0,dy~+6); 60 escondia balas reais
# --mira: inimigos (gwz.enemies, pelas barras de vida) como obstáculo + ficar embaixo deles (mirar).
# Simulador (5.2 min): balas/min ~igual (4.23 x 4.62), contatos com inimigo -43%, tempo mirando 17% -> 30%,
# MAS trocas de lado 17 -> 45/min (a nave acompanha o inimigo). Desligado por padrão: o usuário prioriza
# esquiva e pouca tremedeira; ligar para avaliar no jogo.
AIM = "--mira" in sys.argv
ONE_MATCH = "--continuo" not in sys.argv  # PADRÃO (pedido do usuário): para após cada derrota, depois da análise
vision.COMETS = "--cometas" in sys.argv   # experimental (ver vision.COMETS)
DRONES = "--sem-drones" not in sys.argv   # drones kamikaze: desviar + ficar embaixo p/ derrubar (2026-09-30)
# "laser" DESLIGADO por padrão (2026-10-02): os "lasers" do chefe do deserto e da floresta eram RAJADAS DE CHAMAS
# coladas (conferido nos quadros de referência). Antes o bot não via essas chamas e o laser virava parede parada
# até o fundo da tela; com o vision.flame_t elas são balas com velocidade real, e a parede falsa só atrapalha
# (ao vivo: 20 "lasers"/367 pontos num quadro, loop a ~3 q/s e nave parada no meio do leque). `--laser` liga.
LASERS = "--laser" in sys.argv
# memória de balas sumidas: evade.COAST_S / COAST_NEAR / COAST_VMIN (ajustados no simulador fiel)
DRONE_W_AIM = 0.0        # DESLIGADO (usuário 2026-09-30: "avançando em direção aos mini aviões" -> colidia); drones seguem como obstáculo
VIDEO_EVERY = int(os.environ.get("GWZ_VIDEO_EVERY", "1"))   # grava TODO quadro da partida em logs/frames/<trecho>.mp4 (meia resolução). Era 3; com 1 o
                         # vídeo serve de "gravação" p/ o simulador fiel (tools/sim_play.py) em cada fase nova.
                         # O nº do quadro no vídeo vai no registro ("vf") — a fila pode pular quadros.
IDLE_AFTER = 1.0         # s sem balas à vista -> modo "vai e volta" (o jogo fica lento com a nave parada)
WIGGLE_X = (380, 700)    # pontos do vai e volta (longe das bordas)
AIM_W_OBS, AIM_W_AIM = 200.0, 50.0
SLOW_PROBE = "--medir-lento" in sys.argv      # medição antiga (fator já medido: ver slow_rate)
SLOW_MODE = "--lento" in sys.argv or "--lento-sempre" in sys.argv   # câmera lenta (dedo solto na tempestade). DESLIGADA
                         # por padrão (2026-10-01): simulador +-3% (ruído), ao vivo sem ganho e o usuário achou pior
SLOW_STORM_N = 0 if "--lento-sempre" in sys.argv else 6   # balas à vista p/ "tempestade" (tools/sim_slow.py, 19.7 min:
                         # nunca 10.96 acertos/min de jogo; >=6 e parada<80 10.57; sempre 10.04 mas partida 2x mais longa)
SLOW_STAY_R = 80         # px: destino a menos disso = "ficar parada" (solta o dedo)

def slow_rate(since):
    """Ritmo do jogo com o dedo solto há `since` s (gravação do usuário 2026-09-30: <0.4 s ~0.94, 0.4-1 s 0.62,
    >1 s 0.29). Usado p/ converter as velocidades medidas em câmera lenta p/ o ritmo normal."""
    return 1.0 - 0.15 * since / 0.4 if since < 0.4 else (0.62 if since < 1.0 else 0.29)
SLOW_SAFE = 550          # px: só mede se nenhuma bala estiver mais perto que isso
SLOW_PROBE_S = 0.7       # s com o dedo levantado na medição (0.5 dava só ~2 quadros úteis no replay)
DEFAULT_VY = evade.DEFAULT_V[1]
AUX_EVERY = 0.16         # s entre rodadas de inimigos/laser/drones na thread auxiliar
JET_UP = 650             # px acima da nave onde procurar jatos sem barra de vida (antes 1100)
LOST_PUSH_AFTER = 1.0    # s com a nave perdida (longe da borda conhecida) até começar a empurrar p/ o centro

class _VideoThread:
    """Grava o vídeo da partida numa thread (reduzir+codificar no loop custava ~4 quadros/s da esquiva)."""
    def __init__(self, path, fps):
        import queue
        self.q = queue.Queue(maxsize=8)
        self.w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (540, 1170))
        self.th = threading.Thread(target=self._run, daemon=True); self.th.start()
    def _run(self):
        while True:
            img = self.q.get()
            if img is None:
                break
            self.w.write(cv2.resize(img, (540, 1170), interpolation=cv2.INTER_AREA))
        self.w.release()
    def write(self, img):
        """Enfileira o quadro; devolve o nº dele no vídeo (None se a fila estava cheia e ele foi pulado)."""
        if self.q.full():                # fila cheia: pula o quadro (nunca atrasa a esquiva)
            return None
        self.q.put_nowait(img)
        self.n = getattr(self, "n", -1) + 1
        return self.n
    def release(self):
        self.q.put(None); self.th.join(timeout=5)

class _AuxThread:
    """Inimigos (barras de vida), laser e drones fora do loop da esquiva: processa sempre o quadro MAIS RECENTE
    (descarta os intermediários) e publica os resultados com o instante do quadro."""
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.foes, self.laser_pts, self.laser_seen = [], [], -9.0
        self.drones, self.t_drones = [], -9.0
        self.foes_tr, self.t_foes = [], -9.0
        self._laser_prev = []
        self._last = 0.0
        self._run = True
        self.th = threading.Thread(target=self._loop, daemon=True); self.th.start()
    def feed(self, ft, img, ship):
        with self.lock:
            self.frame = (ft, img, ship)
    def stop(self):
        self._run = False
    def _loop(self):
        while self._run:
            with self.lock:
                job, self.frame = self.frame, None
            if job is None:
                time.sleep(0.005); continue
            ft, img, ship = job
            # ~6 x/s (como antes: 1 a cada 3 quadros). Mais que isso disputa CPU com a detecção de balas.
            wait = AUX_EVERY - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
                with self.lock:
                    if self.frame is not None:
                        ft, img, ship = self.frame; self.frame = None
            self._last = time.time()
            try:
                if AIM or evade.W_SHOOTER or evade.ENEMY_OBST:
                    self.foes = [(e["x"], e["y"], e["r"]) for e in enemies.detect_enemies(img)]
                    # jatos roxos sem barra de vida (colunas que passam na altura da nave; 2026-10-01)
                    if ship is not None:
                        # só perto da altura da nave (o risco é COLISÃO): 1100 -> 650 px acima = -28% do custo
                        # (2026-10-06: o 4º template, jato sobre asfalto, levou detect_jets de 44 para 54 ms)
                        self.foes += [(x, y, 60.0) for x, y in enemies.detect_jets(img, top=max(350, int(ship[1]) - JET_UP))]
                    # velocidade de cada inimigo (os que DESCEM viram obstáculo móvel na esquiva)
                    dtf = ft - self.t_foes
                    self.foes_tr = evade.track_velocities(self.foes_tr, [(x, y) for x, y, _ in self.foes],
                                                          dtf if 0 < dtf < 0.6 else 0)
                    self.t_foes = ft
                if LASERS:
                    # linhas longas confirmadas em 2 leituras seguidas viram parede de pontos parados
                    cur = vision.detect_lasers(img)
                    ok = [s for s in cur if any(abs(s[0] - p[0]) + abs(s[2] - p[2]) < 120 or
                                                abs(s[0] - p[2]) + abs(s[2] - p[0]) < 120 for p in self._laser_prev)]
                    self._laser_prev = cur
                    if ok:
                        self.laser_pts, self.laser_seen = vision.laser_points(ok), ft
                if DRONES and ship is not None:
                    dr = enemies.detect_drones(img, top=max(350, int(ship[1]) - 1100))
                    dt = ft - self.t_drones
                    self.drones = evade.track_velocities(self.drones, dr, dt if 0 < dt < 0.4 else 0)
                    self.t_drones = ft
            except Exception:
                log.exception("aux (inimigos/laser/drones)")
                time.sleep(0.1)

def _save_hit(history, t_hit, n):
    """Salva em logs/hits/ os quadros ~0.45, ~0.25 e ~0.05 s antes do dano, com o que o bot via
    (nave ciano, balas coloridas, alvo escolhido em verde) — para entender por que a esquiva falhou."""
    if not history:
        return
    picks = [min(history, key=lambda h: abs((t_hit - h[0]) - back)) for back in (0.45, 0.25, 0.05)]
    def work():   # em thread: gravar PNG no meio da esquiva travava o loop
        os.makedirs("logs/hits", exist_ok=True)
        tiles = []
        for ft, img, ship, bl, target in picks:
            im = vision.debug_draw(img, ship, bl)
            cv2.circle(im, (int(target[0]), int(target[1])), 30, (0, 255, 0), 5)
            cv2.putText(im, f"-{t_hit - ft:.2f}s", (20, 90), 0, 2.5, (0, 255, 255), 6)
            y0 = min(max(0, int(ship[1]) - 1100), 2340 - 1300)     # recorte de tamanho FIXO
            tiles.append(cv2.resize(im[y0:y0 + 1300], (432, 520)))
        stamp = int(time.time())
        cv2.imwrite(f"logs/hits/{stamp}_{n}.png", np.hstack(tiles))
        # quadro cru em resolução cheia (~0.25 s antes): material p/ templates de inimigos/projéteis novos
        cv2.imwrite(f"logs/hits/{stamp}_{n}_raw.png", picks[1][1])
    threading.Thread(target=work, daemon=True).start()

def gameplay_loop(dev, pilot, max_seconds=None, evade_on=True, stats=None, wave=None):
    """Esquiva em tempo real enquanto a partida roda. Retorna o primeiro quadro fora do gameplay
    (ou None se max_seconds estourar — usado nos testes). evade_on=False: nave parada (controle
    para comparar). stats (dict): acumula 'hits' (flashes de dano) e 'secs'."""
    idx, ship, prev_bl, prev_t = 0, None, [], None
    prev_vel, coast_prev = [], []
    off_since, n, t_start = None, 0, time.time()
    stats = {} if stats is None else stats
    stats.setdefault("hits", 0); stats.setdefault("secs", 0.0)
    flashing, t_last, last_dbg = False, time.time(), ""
    history = collections.deque(maxlen=12)     # últimos quadros processados, p/ salvar o que levou ao dano
    last_cls, ship_seen, k_frame = 0.0, 0.0, 0
    foes = []
    t_bullets, wiggle_x = 0.0, WIGGLE_X[0]     # t_bullets=0: começa em "sem balas" -> já sai mexendo
    prev_drones, t_drones = [], 0.0
    t_nudge = 0.0
    laser_prev, laser_pts, laser_seen = [], [], 0.0
    ship_seen_prev = None
    ship_est = _ship_estimator(pilot)            # filtro da nave pelo dedo + estimativa quando some (core)
    hp_last, was_defensive, defensive = None, None, False   # vida (barra verde) e modo defensivo
    hp_hist = []                                 # últimas leituras da barra (mediana em hp_last)
    slow = {"state": "wait", "before": []}     # medição do "câmera lenta" (dedo levantado), 1x por trecho
    t_start_ft = None
    slow_up = None                              # instante em que o bot soltou o dedo (câmera lenta) ou None
    # rastro por quadro p/ diagnosticar a esquiva: nave lida x destino x dedo (logs/trace.csv)
    new_file = not os.path.exists("logs/trace.csv")
    trace = open("logs/trace.csv", "a", buffering=1)
    if new_file:
        trace.write("t,nova_leitura,nave_x,nave_y,alvo_x,alvo_y,dedo_virt_x,dedo_virt_y,dedo_x,dedo_y,balas,dano\n")
    # registro COMPLETO por quadro (nave, destino, balas com velocidade, dano) p/ a análise pós-partida
    # (tools/hit_analysis.py): responde "por que levou cada tiro e o que deveria ter feito"
    os.makedirs("logs/frames", exist_ok=True)
    seg_path = f"logs/frames/{int(time.time() * 1000)}.jsonl"
    flog = open(seg_path, "w", buffering=1 << 16)
    # vídeo da partida (material p/ ajustar a esquiva/visão offline, sem gastar vidas)
    vout = _VideoThread(seg_path[:-6] + ".mp4", 20 / VIDEO_EVERY)
    stats["segment"] = seg_path
    target = None
    plan_end = None                             # fim do caminho do planejador no quadro anterior (evita trocar de rota)
    aplan = AsyncPlanner(planner.plan, pilot, t0=PLAN_T0, threaded=PLAN_ASYNC)   # planejador assíncrono (core)
    aux = _AuxThread()                          # inimigos/laser/drones em paralelo
    def rec(ft, ship, seen, target, vel, red):
        flog.write(json.dumps({"t": round(ft, 3), "s": ship and [round(ship[0]), round(ship[1])], "v": int(seen),
                               "g": target and [round(target[0]), round(target[1])], "r": int(red), "u": int(slow_up is not None),
                               "vf": vf_now,
                               "b": [[round(x), round(y), round(vx), round(vy)] for x, y, vx, vy in vel]}) + "\n")
    while max_seconds is None or time.time() - t_start < max_seconds:
        idx, img = dev.next_frame(idx, timeout=1.0)
        if img is None:
            continue
        ft = dev.sc.frame_time
        t_start_ft = t_start_ft or ft
        stats["secs"] += ft - t_last if ft > t_last else 0; t_last = ft
        vf_now = vout.write(img) if k_frame % VIDEO_EVERY == 0 else None   # nº do quadro no vídeo (ou None)
        red = vision.damage_flash(img) > HIT_RED
        if red and not flashing:
            stats["hits"] += 1
            log.info("DANO (acerto #%d) %s", stats["hits"], last_dbg)
            _save_hit(history, ft, stats["hits"])
        flashing = red
        # HUD de partida conferido a cada 3 quadros (só serve p/ perceber que a partida acabou; ~5 ms).
        # Contador PRÓPRIO (n só anda com nave achada; na derrota a nave some e a checagem ficaria pulada)
        k_frame += 1
        if red or (k_frame % 3 and off_since is None) or states.is_gameplay(img):
            # flash de dano esconde o HUD: continua esquivando (antes classificava a tela inteira a
            # cada quadro nesse momento e o loop "travava" logo depois de cada tiro)
            off_since = None
        else:
            off_since = off_since or ft
            if ft - last_cls > 0.3:           # classificação completa é cara: no máx. ~3x/s
                last_cls = ft
                st, _ = states.classify(img)
                if st == "GAMEPLAY":
                    off_since = None          # HUD parcialmente coberto (nota 0.97-0.98) mas é partida
                elif st == "UNKNOWN" and ft - ship_seen < 0.5:
                    # HUD coberto (números de dano/efeitos do chefe) mas a NAVE está sendo vista = ainda é partida:
                    # continua pilotando (2026-10-01: 37 saídas e 129 s sem piloto no chefe)
                    off_since = ft
                elif st not in ("GAMEPLAY", "UNKNOWN") or ft - off_since > GP_GRACE:
                    pilot.release()
                    log.info("esquiva: fim (%s), %d quadros em %.0fs", st, n, time.time() - t_start)
                    flog.close(); vout.release(); aux.stop(); aplan.close(); stats["end_state"] = st
                    return img
        ship_now, bl = vision.detect_threats(img, near=ship, roi_above=ROI_ABOVE, ship_exclude=SHIP_FP_R)
        # NAVE (shooter_core.ShipEstimator):
        #  - salto impossível (> 400 px em < 0.3 s; nave anda ~1100 px/s) = leitura errada no flash de dano (visto
        #    ao vivo 673 -> 125);
        #  - FILTRO PELO DEDO (2026-10-04, floresta): na rajada a nave "vista" pulava 1705 -> 1541 -> 1645 em 0.1 s
        #    (pegava alas/mísseis/raio). Leitura longe da posição prevista (última detectada + dedo x GAIN) é
        #    descartada; conflito por mais que SHIP_TRUST_S -> volta a confiar na detecção;
        #  - sumiu há até 0.5 s (tom vermelho do dano / nave piscando): posição estimada pelo dedo (NEXT_STEPS §5).
        st_ship = ship_est.update(ship_now, ft)
        if ship_est.rejects:
            stats["ship_rejects"] = ship_est.rejects
        ship_seen = ship_est.seen_t
        if st_ship.lost:
            rec(ft, ship, False, None, evade.track_velocities(prev_vel, bl, ft - prev_t if prev_t else 0), red)
            lost_long = ft - ship_seen > LOST_PUSH_AFTER
            if ship is not None and lost_long and not (ship[0] < 220 or ship[0] > 860):
                # perdida há muito tempo LONGE da borda conhecida (ao vivo 2026-10-01: o piloto terminou o
                # movimento e a nave parou colada na borda esquerda, 22 s parada levando tiro): empurra p/ o
                # centro, no sentido contrário ao último destino (é p/ lá que ela foi)
                if ft - t_nudge > 0.7:
                    side = (target[0] if target else ship[0]) - 540
                    pilot.nudge(-250 if side > 0 else 250); t_nudge = ft; slow_up = None   # dedo encosta
                    log.info("nave perdida há %.1fs -> empurrão p/ o centro", ft - ship_seen)
            elif ship is not None and (ship[0] < 220 or ship[0] > 860):
                # perdida PERTO DA BORDA (metade fora da tela): empurrão fixo p/ dentro, repetido a cada 0.6 s
                # até reencontrar (steer com posição velha falhava: nave ficou presa no canto ao vivo 03:2x)
                if ft - t_nudge > 0.6:
                    pilot.nudge(300 if ship[0] < 540 else -300); t_nudge = ft
            elif slow_up is None:
                pilot.hold()                # (em câmera lenta não encosta o dedo só porque a nave sumiu 1 quadro)
            continue
        # sumiu há pouco: posição estimada pelo dedo; detectada: posição lida (ship_est guarda as duas)
        ship_now = st_ship.pos
        ship = ship_now
        # rastreio pela velocidade de cada bala (tools/track_eval.py: erro a 0.5 s 30.6 -> 19.9 px)
        dt_ = ft - prev_t if prev_t and ft - prev_t < 0.3 else 0
        vel = evade.track_velocities(prev_vel, bl, dt_, store_turn=True)
        tracked = vel                  # só as DETECTADAS alimentam o rastreador (lembradas/drones não)
        # bala que "some" colada na nave (sobreposição derruba o template; visto na análise da partida G):
        # continua prevista por até COAST_S, só na zona perto da nave
        # (lógica em evade.coast, compartilhada com o simulador fiel p/ testar a memória sem jogar)
        coast_prev = evade.coast(prev_vel, vel, coast_prev, dt_, ship)
        vel = vel + [(x, y, vx, vy) for x, y, vx, vy, _ in coast_prev]
        prev_vel, prev_bl, prev_t = tracked, bl, ft
        # inimigos / laser / drones: numa THREAD à parte sobre o quadro mais recente (2026-10-01: rodando no loop
        # custavam ~11 ms/quadro em média e picos de ~30 ms; o loop da esquiva fica só com nave + balas)
        aux.feed(ft, img, ship)
        foes = aux.foes
        # naves inimigas que descem/estão perto = obstáculo MÓVEL (usuário: "colide com as naves que vêm p/ baixo")
        if 0 <= ft - aux.t_foes < 0.6:
            dtf = ft - aux.t_foes
            vel = vel + evade.enemy_points([(x + vx * dtf, y + vy * dtf, vx, vy) for x, y, vx, vy in aux.foes_tr], ship)
        if LASERS and aux.laser_pts and ft - aux.laser_seen < 0.5:
            vel = vel + [(x, y, 0.0, 0.0) for x, y in aux.laser_pts if y > ship[1] - 700]
        # drones kamikaze (pedido do usuário): obstáculo largo (corpo + 2 rotores)
        drone_aim = []
        prev_drones, t_drones = aux.drones, aux.t_drones
        if DRONES and 0 <= ft - t_drones < 0.4:
            for x, y, vx, vy in prev_drones:
                dt_ = ft - t_drones
                x, y = x + vx * dt_, y + vy * dt_
                vel = vel + [(x + ox, y, vx, vy) for ox in (-45, 0, 45)]     # corpo + 2 rotores
                if y < ship[1] - 250:
                    drone_aim.append((x, y, 60.0))
        # MEDIÇÃO DO "CÂMERA LENTA" (dica do usuário 2026-09-30: levantar o dedo deixa o jogo mais lento -> mais
        # tempo p/ pensar). 1x por trecho, só em momento CALMO (balas à vista, nenhuma a < SLOW_SAFE px): solta o
        # dedo SLOW_PROBE_S s e compara a velocidade das balas antes x durante. Aborta se alguma bala chegar perto.
        real_vy = [vy for x, y, vx, vy in tracked if not (vx == 0 and vy == DEFAULT_VY) and 200 < vy < 1800]
        nearest = min((math.hypot(x - ship[0], y - ship[1]) for x, y, *_ in vel), default=9999)
        if slow["state"] == "wait":
            slow["before"] = [(t_, v_) for t_, v_ in slow["before"] if ft - t_ < 1.5] + [(ft, v) for v in real_vy]
            if SLOW_PROBE and ft - t_start_ft > 4.0 and len(slow["before"]) >= 15 and nearest > SLOW_SAFE and not red:
                slow.update(state="on", t_on=ft, during=[]); pilot.release()
                log.info("LENTO: soltando o dedo p/ medir (bala mais perto %.0f px)", nearest)
        elif slow["state"] == "on":
            slow["n"] = slow.get("n", 0) + 1
            if ft - slow["t_on"] > 0.12:            # descarta a transição
                slow["during"] += real_vy
            if ft - slow["t_on"] > SLOW_PROBE_S or nearest < 400:
                slow["state"] = "done"
                b = np.median([v for _, v in slow["before"]]) if slow["before"] else float("nan")
                d = np.median(slow["during"]) if slow["during"] else float("nan")
                log.info("LENTO: balas %.0f px/s com dedo -> %.0f px/s sem dedo (fator %.2f, %d medidas em %d quadros)%s", b, d,
                         d / b if b else float("nan"), len(slow["during"]), slow.get("n", 0), " [abortado: bala perto]" if nearest < 400 else "")
                stats.setdefault("slow", []).append((b, d, len(slow["during"])))
            else:
                history.append((ft, img, ship, bl, ship)); rec(ft, ship, ship_now is not None, None, vel, red)
                continue                              # dedo levantado: não pilota durante a medição
        # CÂMERA LENTA: com o dedo solto as balas são medidas mais lentas; a decisão de ANDAR exige recolocar o
        # dedo, e aí o jogo volta ao normal em <0.4 s -> decide com as velocidades convertidas p/ o ritmo normal
        if slow_up is not None:
            rt = slow_rate(ft - slow_up)
            vel = [(x, y, vx / rt, vy / rt) for x, y, vx, vy in vel]
        if vel:
            t_bullets = ft
        async_steer = False
        # MODO DEFENSIVO: ondas iniciais, entrada do trecho (nave lá em cima) ou vida baixa (barra verde)
        if n % 3 == 0 and ship_now is not None and ft == ship_seen:
            h = vision.hp_fraction(img, ship)
            # MEDIANA das últimas leituras (2026-10-04: a leitura pulava 0.27 -> 1.0 -> 0.5 -> 1.0 em 1 s — coisas
            # verdes passando sobre a barra — e o modo defensivo ligava/desligava sem parar)
            if h is not None:
                hp_hist.append(h)
                del hp_hist[:-HP_MEDIAN_N]
                hp_last = float(np.median(hp_hist))
        defensive = ((wave is not None and wave <= DEF_WAVES) or ft - t_start_ft < DEF_ENTRY_S
                     or (hp_last is not None and hp_last < DEF_HP))
        if defensive != was_defensive:
            log.info("modo %s (onda %s, vida %s)", "DEFENSIVO" if defensive else "normal", wave,
                     None if hp_last is None else round(hp_last, 2))
            stats["def_s"] = stats.get("def_s", 0.0)
            was_defensive = defensive
        if ft - t_bullets > IDLE_AFTER:
            # sem balas (início da partida / entre ondas): o jogo só roda em velocidade normal com a nave se
            # mexendo (relato do usuário) -> vai e volta entre dois pontos até aparecerem projéteis
            if abs(ship[0] - wiggle_x) < 40:
                wiggle_x = WIGGLE_X[1] if wiggle_x == WIGGLE_X[0] else WIGGLE_X[0]
            target = (wiggle_x, ship[1])
        elif PLANNER and not AIM and PLAN_ASYNC:
            # NEXT_STEPS §5: planejador numa THREAD — o loop já captura/detecta o próximo quadro enquanto ele
            # pensa; a própria thread manda o destino ao piloto assim que fica pronto (sem latência extra).
            # Fila de no máx. 1: plano antigo ainda não iniciado é descartado (só vale o mais novo).
            # (shooter_core.AsyncPlanner, com o planejador do GWZ = gwz.planner.plan)
            tgt = aplan.submit(ship, vel, ft, enemies=foes if 0 <= ft - aux.t_foes < 0.6 else None,
                               steer_ok=evade_on and slow_up is None, over=DEF_PARAMS if defensive else None)
            target = tgt or target or ship
            async_steer = True
        elif PLANNER and not AIM:
            # planejador no espaço-tempo (2026-10-02, tools/sim_play.py com a partida do usuário na fase da neve:
            # 65 -> 5 contatos em 12 min; outra gravação 43 -> 21): caminho que maximiza a menor folga às balas
            tgt, clear, path = planner.plan(ship, planner.linear_predictor(vel), t0=PLAN_T0,
                                            prev_end=plan_end, prev_target=target if n else None,
                                            enemies=foes if 0 <= ft - aux.t_foes < 0.6 else None,   # ofensiva
                                            **(DEF_PARAMS if defensive else {}))
            plan_end = path[-1][1:]
            target = tgt
        else:
            if drone_aim and not AIM:
                target = evade.choose_target(ship, vel, t0=REACT, prev=target if n else None,
                                             enemies=drone_aim, w_obs=0.0, w_aim=DRONE_W_AIM, shooters=foes)
            else:
                target = evade.choose_target(ship, vel, t0=REACT, prev=target if n else None,
                                             enemies=foes if AIM else None, w_obs=AIM_W_OBS, w_aim=AIM_W_AIM,
                                             shooters=None if AIM else foes)
        near3 = sorted(vel, key=lambda b: (b[0] - ship[0]) ** 2 + (b[1] - ship[1]) ** 2)[:3]
        last_dbg = "nave=(%d,%d) alvo=(%d,%d) balas(dx,dy,vx,vy)=%s" % (
            *ship, *target, [(round(x - ship[0]), round(y - ship[1]), round(vx), round(vy)) for x, y, vx, vy in near3])
        # dica do usuário (2026-09-30): na TEMPESTADE de balas, se o melhor é a nave FICAR onde está, solta o dedo ->
        # o jogo fica até ~3.5x mais lento (medido na gravação dele) e o bot ganha tempo de reação; quando precisar
        # andar, recoloca o dedo (o piloto encosta em HOME e segue). Simulador (tools/sim_slow.py): ver SLOW_STORM_N.
        stay = math.hypot(target[0] - ship[0], target[1] - ship[1]) < SLOW_STAY_R
        storm = len(bl) >= SLOW_STORM_N
        if evade_on and SLOW_MODE and stay and storm and not red:
            if slow_up is None:
                pilot.release(); slow_up = ft; stats["slow_n"] = stats.get("slow_n", 0) + 1
        elif evade_on:
            if slow_up is not None:
                stats["slow_s"] = stats.get("slow_s", 0.0) + ft - slow_up
                slow_up = None
            if not async_steer:                  # (planejador assíncrono: a thread dele já pilota)
                pilot.steer(ship, target, ft)
        else:
            pilot.hold()
        # sem .copy(): cada quadro já é um array novo (Device -> to_ndarray) e ninguém altera a imagem (~4 ms)
        history.append((ft, img, ship, bl, target))
        rec(ft, ship, ship_now is not None and ft == ship_seen, target, vel, red)
        if trace:
            trace.write("%.3f,%s,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%s,%s,%d,%d\n" % (
                ft, int(ship_now is not ship_seen_prev), ship[0], ship[1], target[0], target[1],
                pilot.virt[0], pilot.virt[1], pilot.finger and round(pilot.finger[0]), pilot.finger and round(pilot.finger[1]),
                len(bl), int(red)))
        ship_seen_prev = ship_now
        n += 1
        if n % 100 == 0:
            log.info("esquiva: %.1f quadros/s, %d balas perto, nave %s -> alvo %s",
                     n / (time.time() - t_start), len(bl), tuple(map(round, ship)), target)
    pilot.release()
    flog.close(); vout.release(); aux.stop(); aplan.close()
    return None

def _analyze_match(segments):
    """Após cada derrota: por que levou cada tiro e o que deveria ter feito (logs/analysis/). Em thread."""
    def work():
        try:
            from tools import hit_analysis
            txt, _ = hit_analysis.report(segments)
            os.makedirs("logs/analysis", exist_ok=True)
            path = f"logs/analysis/partida_{time.strftime('%m%d_%H%M%S')}.txt"
            open(path, "w", encoding="utf-8").write(txt + "\n\ntrechos: " + ", ".join(segments) + "\n")
            log.info("ANÁLISE DA PARTIDA (%s):\n%s", path, txt)
        except Exception:
            log.exception("análise pós-partida falhou")
    th = threading.Thread(target=work, daemon=True); th.start()
    return th

def main():
    dry = "--dry" in sys.argv
    # --minutes N: encerra sozinho (e limpo: solta o dedo, fecha o scrcpy) depois de N minutos
    deadline = float(sys.argv[sys.argv.index("--minutes") + 1]) * 60 + time.time() if "--minutes" in sys.argv else None
    dev = None
    for attempt in range(0 if dry else 60):
        # cabo USB oscilando (visto 2026-09-30: aparelho some do adb segundos depois de aparecer): espera voltar
        try:
            dev = Device(); break
        except Exception as e:
            if attempt == 0 or attempt % 6 == 0:
                log.warning("celular indisponível (%s) -> esperando reconectar (%d/60)", str(e).splitlines()[-1][:80], attempt + 1)
            time.sleep(5)
    if dev is None and not dry:
        log.error("celular não voltou em 5 min -> saindo"); return
    bot = Bot(dev, dry=dry)
    bot.assemble_grid = dev is not None and "--no-grid" not in sys.argv   # montagem automática (grid_exec)
    # `--voltar N`: no 1º menu de níveis, N toques na seta "<" antes do Play (ex.: --voltar 1 = neve, nível 15)
    bot.go_back = int(sys.argv[sys.argv.index("--voltar") + 1]) if "--voltar" in sys.argv else 0
    pilot = Pilot(dev) if dev else None
    unknown_since = None; backs = 0; level_taps = 0; last_shot = 0; state_since = time.time(); prev = None
    last_known = None; reconnects = 0
    frozen_sig, frozen_since, frozen_acts = None, time.time(), 0
    unknown_since_check, locked_logged = 0.0, False
    match_segments = []          # trechos de gameplay da partida atual (logs/frames/*.jsonl)
    match_started = False        # já houve gameplay nesta partida (mesmo que o trecho não tenha sido gravado)
    while deadline is None or time.time() < deadline:
        try:
            frame = dev.screenshot() if dev else None
            if frame is None: break
            if ONE_MATCH and (match_segments or match_started) and states.classify(frame)[0] == "LEVEL_MENU":
                # partida acabou SEM passar pelo FAILED (ao vivo 2026-10-06: o anúncio do reviver voltou direto ao
                # menu de níveis e o bot tocou Play, começando outra partida sozinho) -> analisa e para ANTES do Play
                th = _analyze_match(list(match_segments)) if match_segments else None
                if th:
                    th.join(timeout=120)
                log.info("--uma-partida: partida terminou no menu de níveis -> fim após a análise"); break
            state, result = bot.step(frame)
            reconnects = 0        # ciclo normal: quedas espaçadas não se acumulam até o limite
            now = time.time()
            # ---- jogo CONGELADO: imagem idêntica por T_FROZEN s mesmo com o bot tocando (2026-10-01: tela escura
            #      parada na montagem depois de um anúncio; Play tocado várias vezes sem efeito) -> reinicia o jogo
            sig = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (36, 78), interpolation=cv2.INTER_AREA).astype(np.int16)
            if frozen_sig is None or np.abs(sig - frozen_sig).mean() > 0.6:
                frozen_sig, frozen_since, frozen_acts = sig, now, len(bot.actions)
            elif now - frozen_since > T_FROZEN and len(bot.actions) > frozen_acts and state != "LEVEL_MENU":   # menu sem energia é parado mesmo
                log.error("JOGO CONGELADO há %.0fs (tela idêntica apesar de %d toques) -> reiniciando o jogo",
                          now - frozen_since, len(bot.actions) - frozen_acts)
                cv2.imwrite(f"logs/unknown/frozen_{int(now)}.png", frame)
                dev.restart_game(); bot.stats["recoveries"] += 1; match_segments.clear()
                frozen_sig = None; time.sleep(12); continue
            if state != prev:
                state_since, prev = now, state
                if state == "GRID":   # acervo p/ catalogar armas e ler a grade (próxima etapa)
                    cv2.imwrite(f"logs/grid/{int(now)}.png", frame)
            # ---- UNKNOWN: espera, guarda evidência, escala recuperação ----
            if state == "UNKNOWN":
                # tela APAGADA/BLOQUEADA (com senha o bot não desbloqueia): espera o usuário em vez de reiniciar
                # o jogo sem parar (2026-10-01: 73 reinícios em 2.5 h com o celular bloqueado)
                if now - unknown_since_check > 20 if unknown_since else True:
                    unknown_since_check = now
                    if dev.screen_locked():
                        dev.wake()
                        if not locked_logged:
                            log.error("TELA BLOQUEADA/APAGADA -> esperando o usuário desbloquear (o bot não usa senha)")
                            locked_logged = True
                        time.sleep(30); unknown_since = None; continue
                    locked_logged = False
                unknown_since = unknown_since or now
                if now - last_shot > 10:
                    cv2.imwrite(f"logs/unknown/{int(now)}.png", frame); last_shot = now
                dt = now - unknown_since
                if dt > T_UNKNOWN_RELAUNCH:
                    # fecha À FORÇA e reabre: "monkey" só trazia de volta um anúncio jogável travado (2026-10-01);
                    # ao reabrir o jogo oferece voltar à partida inacabada (RESUME_BATTLE -> OK)
                    log.warning("UNKNOWN por %.0fs -> fechando e reabrindo o jogo", dt); bot.stats["recoveries"] += 1
                    dev.restart_game(); unknown_since = None; time.sleep(12)
                elif dt > T_UNKNOWN_BACK and backs < 3:
                    pkg = dev.foreground_package(); log.warning("UNKNOWN por %.0fs, app=%s", dt, pkg)
                    if pkg != APP:
                        # jogo fora da tela (visto ao vivo: launcher em 1º plano após a derrota) -> traz de volta
                        log.warning("jogo não está em primeiro plano -> reabrindo"); dev.launch_game(); backs += 1; time.sleep(8)
                    elif dt > 45 and last_known not in BATTLE:
                        log.warning("-> BACK (fora da batalha)"); dev.back(); backs += 1
                    # dentro da batalha BACK abre pausa/"Exit battle?" -> não aperta; só espera/relança
                time.sleep(1.0); continue
            unknown_since = None; backs = 0; last_known = state
            # ---- montagem: merges, encaixe de armas novas, refresh, Play ----
            if result == "assemble":
                grid_exec.assemble(dev, bot.tap)
                continue
            # ---- gameplay: esquiva em tempo real (scrcpy) ----
            if state == "GAMEPLAY":
                if pilot:
                    st = {}
                    try:
                        gameplay_loop(dev, pilot, max_seconds=None if deadline is None else max(0, deadline - time.time()),
                                      stats=st, wave=bot.wave)
                    finally:
                        # (2026-10-04: o vídeo do scrcpy caiu no meio da partida -> exceção -> o trecho não entrava na
                        #  partida, o FAILED achava a partida "vazia" e o modo 1-partida NÃO parava)
                        if st.get("segment"):
                            match_segments.append(st["segment"])
                        match_started = True
                continue
            # ---- fim REAL da partida (derrota sem reviver): análise dos tiros de TODOS os trechos ----
            if state == "FAILED" and (match_segments or match_started):
                th = _analyze_match(list(match_segments)) if match_segments else None
                match_segments.clear(); match_started = False
                if ONE_MATCH:
                    # pedido do usuário: analisar ANTES de puxar outra partida (aprender a cada uma). Para MESMO sem
                    # trechos gravados (queda de conexão no meio) — antes começava outra partida sozinho
                    if th:
                        th.join(timeout=120)
                    log.info("--uma-partida: fim após a análise"); break
            # ---- menu de nível: Play repetido sem avançar => provável falta de energia ----
            if state == "LEVEL_MENU":
                level_taps += 1
                if level_taps >= 4:
                    wait = ENERGY_WAIT if deadline is None else max(0, min(ENERGY_WAIT, deadline - time.time()))
                    log.warning("Play do menu não avança (%dx). Assumindo energia insuficiente: aguardando %ds", level_taps, wait)
                    time.sleep(wait); level_taps = 0; continue
            else:
                level_taps = 0
            # ---- preso no mesmo estado por muito tempo ----
            if now - state_since > T_STUCK and state not in ("AD_BREAK",) and state not in BATTLE:
                log.warning("preso em %s por %.0fs -> BACK", state, now - state_since)
                cv2.imwrite(f"logs/unknown/stuck_{state}_{int(now)}.png", frame)
                dev.back(); state_since = now; bot.stats["recoveries"] += 1
            time.sleep(1.2)
        except KeyboardInterrupt:
            break
        except Exception:
            # rodando sozinho: um erro (ex.: scrcpy caiu, cabo mexido) não pode derrubar o bot com o dedo
            # virtual encostado. Solta o dedo, e se a conexão caiu, reconecta (até 5 vezes).
            log.exception("erro no loop principal")
            if pilot:
                try: pilot.release()
                except Exception: pass
            if dev and dev.sc and not dev.sc.alive:
                reconnects += 1
                if reconnects > 5:
                    log.error("scrcpy caiu mais de 5 vezes: encerrando"); break
                log.warning("reconectando ao celular (%d/5)...", reconnects)
                try:
                    if pilot: pilot.close()
                    dev.close()
                except Exception: pass
                time.sleep(5)
                try:
                    dev = Device(); bot.dev = dev; pilot = Pilot(dev)
                except Exception:
                    log.exception("reconexão falhou"); time.sleep(10)
            else:
                time.sleep(2)
    try:
        if pilot: pilot.close()
        if dev: dev.close()
    except Exception:
        log.exception("erro ao encerrar")
    log.info("FIM. stats=%s", bot.stats)

if __name__ == "__main__":
    main()
