"""Piloto do GWZ = `shooter_core.RelativePilot` com a mecânica medida no aparelho.

Mecânica medida no aparelho (2026-09-28, fase da estrada):
- Controle RELATIVO: a nave anda o deslocamento do dedo x ~1.1 (dedo parado longe da nave = nada).
- O jogo ignora saltos grandes: até ~20 px por evento a 60 Hz é aceito; >= 30 px começa a perder.
- Levantar o dedo não pausa o jogo, então dá para reposicionar o dedo quando ele chega na borda.
- Latência toque -> vídeo ~150-190 ms: a posição vista da nave está atrasada; compensamos somando
  o movimento de dedo já enviado nesse intervalo (ainda não visível no vídeo).
A lógica (passos, borda, compensação de latência) mora no core; aqui ficam só os números deste jogo.
"""
from shooter_core.config import PilotConfig
from shooter_core.pilot import RelativePilot

GAIN = 1.12          # deslocamento da nave / deslocamento do dedo
STEP = 18            # px máximos de dedo por evento
HZ = 60
LATENCY = 0.10       # s entre enviar o toque e o efeito aparecer no quadro (medido pelo frame_time;
                     # 0.17 oscilava; 0.06-0.15 estável nos testes)
DEADBAND = 30        # px: erro menor que isso não mexe o dedo (6 deixava o ruído da detecção, 2-6 px,
                     # virar correções constantes; 30 empatou/ganhou no simulador offline)
# área onde o dedo pode ficar (longe do HUD de cima e das bordas com gestos do sistema)
FX = (120, 960)
FY = (700, 2150)
HOME = (540, 1600)


def config():
    return PilotConfig(gain=GAIN, step=STEP, hz=HZ, latency=LATENCY, deadband=DEADBAND, fx=FX, fy=FY, home=HOME)


class Pilot(RelativePilot):
    """dev: gwz.device.Device (touch_down / touch_move / touch_up pelo scrcpy)."""

    def __init__(self, dev):
        super().__init__(dev, config())
