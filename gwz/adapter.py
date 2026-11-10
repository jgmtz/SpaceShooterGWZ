"""GWZ como PRODUTO do AutoSpaceShooterCore: o adaptador mínimo para o loop genérico (shooter_core.DodgeLoop).

Tudo que é específico do Galaxy Wing Zero entra por aqui:
- visão: nave e balas por template (gwz.vision), inimigos pelas barras de vida (gwz.enemies);
- fim da partida: HUD de gameplay (gwz.states);
- regra própria: modo defensivo com a vida baixa (sem ofensiva, margem maior, nave embaixo).

O run.py continua com o loop completo do bot (menus, montagem, anúncios, gravação, câmera lenta...), que usa
os mesmos componentes do core. Este adaptador mostra que só a parte de JOGAR já roda no core genérico:

    python tools/core_loop_replay.py <video.mp4> [segundos_a_pular]
"""
from shooter_core.config import CoreConfig, LoopConfig, ShipConfig
from shooter_core.interfaces import GameAdapter, Observation

from . import vision, enemies, states, planner, evade, pilot

ROI_ABOVE = 1500          # px acima da nave onde procurar balas (o planejador olha 1.2 s à frente)
SHIP_FP_R = 30            # falso positivo do miolo da nave
DEF_HP = 0.5              # vida abaixo disso = modo defensivo
DEF_PARAMS = {"w_dps": 0.0, "sigma0": 20.0, "y_min": 1800.0}
END_GRACE = 1.5           # s sem o HUD de partida até considerar que a partida acabou


def core_config():
    """CoreConfig com os números medidos no GWZ (os mesmos módulos que o run.py usa)."""
    return CoreConfig(planner=planner.config(), tracker=evade.tracker_config(), pilot=pilot.config(),
                      ship=ShipConfig(), loop=LoopConfig(plan_t0=0.15, idle_after=1.0, wiggle_x=(380.0, 700.0)))


class GWZAdapter(GameAdapter):
    def __init__(self, clock=None):
        self.k = 0
        self.off_since = None
        self.hp = []
        self.clock = clock

    def detect(self, img, ship_hint=None):
        self.k += 1
        ship, bullets = vision.detect_threats(img, near=ship_hint, roi_above=ROI_ABOVE, ship_exclude=SHIP_FP_R)
        foes = [(e["x"], e["y"], e["r"]) for e in enemies.detect_enemies(img)] if self.k % 2 == 0 else getattr(self, "_foes", [])
        self._foes = foes
        playing = True
        if self.k % 3 == 0:                       # HUD conferido a cada 3 quadros (~5 ms)
            if states.is_gameplay(img) or vision.damage_flash(img) > 100:
                self.off_since = None
            else:
                self.off_since = self.off_since or self.k
                playing = (self.k - self.off_since) < END_GRACE * 8     # ~8 quadros/s
        return Observation(ship=ship, bullets=bullets, enemies=foes, playing=playing)

    def plan_overrides(self, ctx):
        if ctx.ship_detected and self.k % 3 == 0:
            h = vision.hp_fraction(ctx.img, ctx.ship)
            if h is not None:
                self.hp = (self.hp + [h])[-7:]
        if self.hp and sorted(self.hp)[len(self.hp) // 2] < DEF_HP:
            return DEF_PARAMS
        return None
