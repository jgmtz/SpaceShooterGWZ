"""Roda o loop GENÉRICO do core (shooter_core.DodgeLoop + gwz.adapter.GWZAdapter) sobre uma gravação do jogo.
    python tools/core_loop_replay.py <video.mp4> [segundos_iniciais_a_pular] [segundos_max]

Prova de reuso: o mesmo loop que joga a arena de exemplo do core joga o GWZ trocando só o adaptador.
Um "celular falso" entrega os quadros do vídeo no ritmo em que o loop consegue processar; o piloto falso só
conta os comandos (a gravação não obedece).
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from replay_loop import FakeDevice, FakePilot          # noqa: E402
from gwz.adapter import GWZAdapter, core_config        # noqa: E402
from shooter_core import DodgeLoop                     # noqa: E402


class Source:
    def __init__(self, dev):
        self.dev = dev

    def next_frame(self, idx, timeout=1.0):
        if not self.dev.sc.alive:
            raise StopIteration
        return self.dev.next_frame(idx, timeout)

    @property
    def frame_time(self):
        return self.dev.sc.frame_time


def main():
    path = sys.argv[1]
    skip = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    secs = float(sys.argv[3]) if len(sys.argv) > 3 else 120.0
    dev, pilot = FakeDevice(path, skip), FakePilot()
    loop = DodgeLoop(GWZAdapter(), pilot, core_config())
    t = time.time()
    try:
        end = loop.run(Source(dev), max_seconds=secs)
    except StopIteration:
        end = None
    dt = time.time() - t
    loop.close()
    print(f"loop do core rodou {dt:.0f}s: {loop.frames} quadros ({loop.frames / max(dt, 1e-9):.1f}/s); "
          f"comandos ao piloto: {pilot.cmds}; saiu por: {'fim do vídeo/tempo' if end is None else 'partida acabou'}")


if __name__ == "__main__":
    main()
