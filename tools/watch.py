"""Modo OBSERVADOR: grava a tela e os toques que VOCÊ faz no celular. O bot não toca em nada
(scrcpy abre só o vídeo, control=false).   python tools/watch.py [segundos]   (Ctrl+C para parar)

Saída em logs/watch/<início>/:
  video.mp4        tela inteira em meia resolução, ~10 quadros/s (contexto)
  gestures.json    cada gesto: início/fim (tempo e x,y em pixels 1080x2340) e o caminho do dedo
  gNNN_before.png  tela em resolução cheia quando o dedo encostou
  gNNN_mid.png     durante o arrasto (0.4 s depois de encostar, se o dedo ainda estava na tela)
  gNNN_after.png   0.7 s depois de soltar (resultado do gesto)
Toques lidos do sensor via `getevent` (sec_touchscreen, 0-4095 em cada eixo).
"""
import sys, os, time, json, re, threading, subprocess
import cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gwz.device import Device

TOUCH_DEV = "/dev/input/event6"      # sec_touchscreen no S25 FE
RAW_MAX = 4095
LINE = re.compile(r"\[\s*([\d.]+)\]\s+(\S+)\s+(\S+)\s+(\S+)")

def main():
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 600
    out = os.path.join("logs", "watch", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    with Device(control=False) as dev:
        W, H = dev.size
        vw = cv2.VideoWriter(os.path.join(out, "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (W // 2, H // 2))
        gestures, lock = [], threading.Lock()
        pending = []                                   # (quando_salvar, arquivo)
        state = {"down": None, "x": None, "y": None, "path": [], "offset": None}

        def frame_now():
            img = dev.screenshot()
            return None if img is None else img.copy()

        def reader(proc):
            for line in proc.stdout:
                m = LINE.match(line)
                if not m:
                    continue
                kt, typ, code, val = float(m.group(1)), m.group(2), m.group(3), m.group(4)
                now = time.time()
                # relógio do kernel -> relógio do PC: menor atraso observado (linhas chegam quase na hora)
                off = now - kt
                state["offset"] = off if state["offset"] is None else min(state["offset"], off)
                t = kt + state["offset"]
                if code == "ABS_MT_POSITION_X":
                    state["x"] = int(val, 16) * W / RAW_MAX
                elif code == "ABS_MT_POSITION_Y":
                    state["y"] = int(val, 16) * H / RAW_MAX
                elif code == "ABS_MT_TRACKING_ID" and val != "ffffffff" and state["down"] is None:
                    state["down"], state["path"] = t, []
                    state["pending_down"] = True
                elif code == "ABS_MT_TRACKING_ID" and val == "ffffffff" and state["down"] is not None:
                    with lock:
                        g = {"i": len(gestures), "t_down": round(state["down"], 3), "t_up": round(t, 3),
                             "start": state["path"][0][1:] if state["path"] else None,
                             "end": state["path"][-1][1:] if state["path"] else None,
                             "path": state["path"][::3]}
                        gestures.append(g)
                        with open(os.path.join(out, "gestures.json"), "w") as f:   # salva a cada gesto
                            json.dump(gestures, f, indent=1)
                        pending.append((time.time() + 0.7, os.path.join(out, f"g{g['i']:03d}_after.png")))
                        d = "" if not g["start"] else f"({g['start'][0]},{g['start'][1]}) -> ({g['end'][0]},{g['end'][1]})"
                        print(f"gesto {g['i']}: {g['t_up']-g['t_down']:.2f}s {d}", flush=True)
                    state["down"] = None
                elif code == "SYN_REPORT" and state["down"] is not None and state["x"] is not None:
                    state["path"].append((round(t - state["down"], 3), round(state["x"]), round(state["y"])))
                    if state.pop("pending_down", False):
                        n = len(gestures)
                        img = frame_now()
                        if img is not None:
                            cv2.imwrite(os.path.join(out, f"g{n:03d}_before.png"), img)
                        pending.append((time.time() + 0.4, os.path.join(out, f"g{n:03d}_mid.png")))

        base = [dev.adb_bin] + (["-s", dev.serial] if dev.serial else [])
        proc = subprocess.Popen(base + ["exec-out", "getevent", "-lt", TOUCH_DEV],
                                stdout=subprocess.PIPE, text=True, errors="replace")
        threading.Thread(target=reader, args=(proc,), daemon=True).start()
        print(f"Observando por {secs:.0f}s -> {out}  (o bot NÃO toca na tela)", flush=True)
        t_end, next_frame = time.time() + secs, 0
        try:
            while time.time() < t_end:
                now = time.time()
                if now >= next_frame:
                    img = dev.screenshot()
                    if img is not None:
                        vw.write(cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA))
                    next_frame = now + 0.1
                with lock:
                    due = [p for p in pending if p[0] <= now]
                    pending[:] = [p for p in pending if p[0] > now]
                for _, path in due:
                    if path.endswith("_mid.png"):
                        n = int(os.path.basename(path)[1:4])
                        if n < len(gestures):
                            continue           # o dedo já tinha saído: não há "durante"
                    img = frame_now()
                    if img is not None:
                        cv2.imwrite(path, img)
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass
        finally:
            proc.kill(); vw.release()
            with open(os.path.join(out, "gestures.json"), "w") as f:
                json.dump(gestures, f, indent=1)
            print(f"fim: {len(gestures)} gestos salvos em {out}", flush=True)

if __name__ == "__main__":
    main()
