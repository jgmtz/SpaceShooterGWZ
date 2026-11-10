"""Roda o loop REAL de esquiva (run.gameplay_loop) sobre uma gravação, sem celular e sem gastar vidas.
    python tools/replay_loop.py <video.mp4> [segundos_iniciais_a_pular]

Um "celular falso" entrega os quadros do vídeo no ritmo em que o loop consegue processar (como o scrcpy:
sempre o quadro mais recente) e um piloto falso só anota os comandos. Mede quadros/s reais do loop,
verifica que ele não quebra e que percebe o fim da partida.
"""
import sys, os, time, logging
import cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run

class FakeSc:
    def __init__(self): self.frame_time, self.alive, self.error = 0.0, True, None

class FakeDevice:
    """Entrega o quadro do vídeo correspondente ao tempo real decorrido (descarta os que o loop não
    conseguiu processar, como o scrcpy faz)."""
    def __init__(self, path, skip_s=0.0):
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.cap.set(cv2.CAP_PROP_POS_MSEC, skip_s * 1000)
        self.t0, self.pos, self.sc, self.size = time.time(), -1, FakeSc(), (1080, 2340)
        self.img = None
    def next_frame(self, idx, timeout=1.0):
        want = int((time.time() - self.t0) * self.fps)
        while self.pos < want:
            if not self.cap.grab():
                self.sc.alive = False
                return idx, None
            self.pos += 1
        ok, f = self.cap.retrieve()
        if not ok:
            return idx, None
        self.img = cv2.resize(f, self.size) if (f.shape[1], f.shape[0]) != self.size else f
        self.sc.frame_time = time.time()
        return idx + 1, self.img
    def screenshot(self, fresh=False):
        return self.img

class FakePilot:
    def __init__(self): self.virt, self.finger, self.cmds = [0.0, 0.0], None, 0
    def steer(self, ship, target, t): self.cmds += 1
    def hold(self): pass
    def nudge(self, dx, dy=0.0): self.cmds += 1
    def virt_at(self, t): return (0.0, 0.0)          # (gravação: o dedo não anda de verdade)
    def release(self): pass

def main():
    path = sys.argv[1]
    skip = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    dev, pilot = FakeDevice(path, skip), FakePilot()
    stats = {}
    t = time.time()
    end = run.gameplay_loop(dev, pilot, max_seconds=120, stats=stats)
    dt = time.time() - t
    print(f"loop rodou {dt:.0f}s; saiu por: {'fim do vídeo/tempo' if end is None else 'tela fora da partida'}; "
          f"comandos ao piloto: {pilot.cmds} ({pilot.cmds / max(dt, 1e-9):.1f}/s); acertos vistos no vídeo: {stats['hits']}; "
          f"câmera lenta: {stats.get('slow_n', 0)} vezes, {stats.get('slow_s', 0.0):.1f}s com o dedo solto; registro {stats.get('segment')}")

if __name__ == "__main__":
    main()
