"""Laboratório de toque: descobre COMO a nave reage ao input.
Uso (PC, celular no USB, jogo em partida com a nave visível):
    python tools/touch_lab.py latency  (tempo de captura; não toca na tela)
    python tools/touch_lab.py step   (latência real toque->nave se mexer, via scrcpy)
    python tools/touch_lab.py T1     (dedo parado 10 s)
    python tools/touch_lab.py T2     (dedo parado + micro-movimento)
    python tools/touch_lab.py T3     (arrasto lento contínuo)
    python tools/touch_lab.py T4     (motionevent DOWN/MOVE/UP em passos)
Cada teste grava logs/<teste>.csv com t, x_nave, y_nave (medidos por visão).
Padrão: scrcpy. Acrescente --adb para repetir os testes antigos com adb shell.
Os testes de toque só rodam se o jogo estiver em primeiro plano.
"""
import sys, time, csv, threading, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gwz.device import Device, APP
from gwz.vision import detect_ship

def sample_ship(dev, seconds, tag):
    rows, t0 = [], time.time()
    while time.time() - t0 < seconds:
        s = detect_ship(dev.screenshot())
        rows.append((round(time.time() - t0, 2), None if s is None else round(s[0]), None if s is None else round(s[1])))
    os.makedirs("logs", exist_ok=True)
    with open(f"logs/{tag}.csv", "w", newline="") as f:
        csv.writer(f).writerows([("t", "ship_x", "ship_y")] + rows)
    xs = [r[1] for r in rows if r[1] is not None]
    if len(xs) >= 2:
        print(f"[{tag}] amostras={len(xs)}  x: {xs[0]} -> {xs[-1]}  (deslocamento {xs[-1]-xs[0]:+d} px)")
    else:
        print(f"[{tag}] nave não detectada; confira se a partida está rodando")
    return rows

def run_async(fn, *a):
    th = threading.Thread(target=fn, args=a, daemon=True); th.start(); return th

def step_latency(dev, trials=5, dist=200, step=10):
    """Latência ponta a ponta via scrcpy: arrasta o dedo `dist` px em passos de `step` px a 60 Hz
    (o jogo IGNORA saltos grandes num evento só) e mede do 1º MOVE até a nave andar >= 12 px."""
    lat = []
    for i in range(trials):
        idx, img = dev.next_frame(0)
        s = detect_ship(img)
        if s is None:
            print("nave não detectada; confira se a partida está rodando"); break
        x0, y0 = s
        d = 1 if x0 < dev.size[0] / 2 else -1
        fx, fy, moved = x0, y0 + 150, 0
        dev.touch_down(fx, fy); time.sleep(0.15)       # dedo um pouco abaixo da nave
        t = time.time(); last = 0; got = False
        while time.time() - t < 1.0:
            if moved < dist and time.time() - last > 1 / 60:
                moved += step; dev.touch_move(fx + d * moved, fy); last = time.time()
            idx, img = dev.next_frame(idx, 0.05)
            s = detect_ship(img, near=(x0, y0)) if img is not None else None
            if not got and s is not None and abs(s[0] - x0) >= 12:
                lat.append((time.time() - t) * 1000); got = True
            if got and moved >= dist:
                break
        dev.touch_up(fx + d * moved, fy); time.sleep(0.5)
    if lat:
        print(f"latência toque->vídeo: {' '.join(f'{v:.0f}' for v in lat)} ms  (mediana {sorted(lat)[len(lat)//2]:.0f} ms)")
    else:
        print("não medi nenhum deslocamento")

def main():
    test = sys.argv[1] if len(sys.argv) > 1 else "latency"
    dev = Device(scrcpy="--adb" not in sys.argv)
    try:
        run_test(dev, test)
    finally:
        dev.close()

def run_test(dev, test):
    w, h = dev.size
    print(f"Tela {w}x{h} ({'scrcpy' if dev.sc else 'adb'}). Deixe uma partida rodando com a nave viva. Começando em 5 s...")
    time.sleep(5)
    if test != "latency" and dev.foreground_package() != APP:
        print(f"O jogo ({APP}) não está em primeiro plano. Não vou tocar na tela."); return
    cx, cy = int(w * 0.5), int(h * 0.80)
    if test == "step":
        step_latency(dev)
    elif test == "latency":
        t = time.time(); dev.screenshot(fresh=True); print(f"screenshot: {(time.time()-t)*1000:.0f} ms")
        t = time.time(); dev.sh("echo ok"); print(f"shell vazio: {(time.time()-t)*1000:.0f} ms")
    elif test == "T1":
        th = run_async(dev.hold, cx, cy, 10); sample_ship(dev, 10, "T1_hold_parado"); th.join()
    elif test == "T2":
        def micro():
            dev.motion("DOWN", cx, cy)
            for i in range(20):
                dev.motion("MOVE", cx + (1 if i % 2 else 0), cy); time.sleep(0.4)
            dev.motion("UP", cx, cy)
        th = run_async(micro); sample_ship(dev, 10, "T2_micro_movimento"); th.join()
    elif test == "T3":
        th = run_async(dev.drag, int(w*0.25), cy, int(w*0.75), cy, 8); sample_ship(dev, 8, "T3_arrasto_lento"); th.join()
    elif test == "T4":
        def steps():
            dev.motion("DOWN", int(w*.3), cy)
            for i in range(15):
                dev.motion("MOVE", int(w*.3 + i*(w*.4/15)), cy); time.sleep(0.15)
            dev.motion("UP", int(w*.7), cy)
        th = run_async(steps); sample_ship(dev, 8, "T4_motionevent_passos"); th.join()
    else:
        print(__doc__)

if __name__ == "__main__":
    main()
