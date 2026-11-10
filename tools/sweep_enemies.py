"""Varredura no simulador: inimigos como OBSTÁCULO e MIRA (ficar embaixo do inimigo), sem gastar vidas.
    python tools/sweep_enemies.py
Usa as gravações de tela em C:\\Users\\joaog\\Downloads\\Screen_Recording_*.mp4 + a do observador.
Na 1ª vez calcula os inimigos de cada quadro (cache <video>.en.npy, alguns minutos).
Métricas: balas/min (acertos), contatos_inimigo/min, mirando (% do tempo com inimigo acima e alinhado)."""
import sys, os, glob
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tools import sim_evade as S
from gwz import evade as E, pilot
import run

VIDS = sorted(glob.glob(r"C:\Users\joaog\Downloads\Screen_Recording_*.mp4")) + \
       sorted(glob.glob(os.path.join("logs", "watch", "*", "video.mp4")))

def main():
    frames = []
    for v in VIDS:
        fr = S.load_cached(v)
        off = frames[-1][0] + 5.0 if frames else 0.0
        frames += [(t + off,) + tuple(r) for t, *r in fr]
    frames = S.attach_enemies(frames)
    print("inimigos detectados (soma por quadro):", sum(len(f[4]) for f in frames if f[4]), flush=True)
    starts = [(360, 2000), (540, 2000), (720, 2000)]

    def go(label, w_obs, w_aim):
        fn = lambda s, v, p, en: E.choose_target(s, v, t0=run.REACT, prev=p, enemies=en, w_obs=w_obs, w_aim=w_aim)
        rs = [S.simulate(frames, fn, start=st, deadband=pilot.DEADBAND, roi=run.ROI_ABOVE) for st in starts]
        m = {k: np.mean([r[k] for r in rs]) for k in rs[0]}
        print(f"{label:28s} balas/min={m['acertos/min']:4.2f}  contatos_inimigo/min={m['contatos_inimigo/min']:4.2f}  "
              f"mirando={m['mirando_%']:4.1f}%  trocas/min={m['trocas_de_lado/min']:5.1f}", flush=True)

    if "--suave" in sys.argv:          # 2ª rodada: suavizar a tremedeira da mira
        go("sem inimigos (atual)", 0, 0)
        for calm in (15, 30, 45):
            E.SWITCH_MARGIN_CALM = calm
            for w_aim in (30, 50):
                go(f"obst 200 + mira {w_aim}, calma {calm}", 200, w_aim)
        return
    go("sem inimigos (atual)", 0, 0)
    for w_obs in (50, 200):
        go(f"obstáculo {w_obs}", w_obs, 0)
    for w_aim in (20, 50, 80):
        go(f"mira {w_aim}", 0, w_aim)
    for w_obs, w_aim in ((50, 50), (200, 50)):
        go(f"obstáculo {w_obs} + mira {w_aim}", w_obs, w_aim)

if __name__ == "__main__":
    main()
