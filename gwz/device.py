"""Acesso ao celular.

- Tempo real (captura + toque): scrcpy-server (o mesmo que vem com o scrcpy.exe instalado),
  falando o protocolo binário direto por socket. Vídeo H.264 decodificado com PyAV.
- Tarefas lentas (app em primeiro plano, abrir o jogo, BACK): adb shell comum.

Protocolo conferido no aparelho com scrcpy 4.1 (ver `_video_loop`). Se o scrcpy for atualizado,
rode `python tools/touch_lab.py scrcpy` para conferir antes de rodar o bot.
"""
import subprocess, time, re, os, shutil, socket, struct, threading, random, logging
import numpy as np, cv2

APP = "com.space.shooter.galaxy.attack.wing.fighter"
log = logging.getLogger("gwz.device")

def find_adb():
    return shutil.which("adb") or "adb"


class ScrcpyError(RuntimeError):
    pass


def _recvn(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError("socket do scrcpy fechou")
        buf += chunk
    return bytes(buf)


class Scrcpy:
    """Cliente mínimo do scrcpy-server: vídeo (só leitura) + socket de controle (toque)."""
    REMOTE_JAR = "/data/local/tmp/scrcpy-server.jar"
    # flags do cabeçalho de 12 bytes de cada pacote de vídeo (scrcpy 4.x)
    F_SESSION, F_CONFIG, F_KEY = 1 << 63, 1 << 62, 1 << 61
    # AMOTION_EVENT_ACTION_*
    DOWN, UP, MOVE = 0, 1, 2
    MSG_INJECT_TOUCH = 2

    def __init__(self, adb="adb", serial=None, max_fps=60, bit_rate=8_000_000, max_size=0, port=27183,
                 extra_opts=(), control=True):
        """extra_opts: opções a mais para o servidor, ex. ("show_touches=true",).
        control=False: só vídeo (modo observador — impossível tocar na tela)."""
        self.adb, self.serial, self.extra_opts, self.control = adb, serial, list(extra_opts), control
        self.max_fps, self.bit_rate, self.max_size, self.port = max_fps, bit_rate, max_size, port
        self.video_size = None          # (w, h) do stream; o toque precisa ser enviado nesse sistema
        self.frame_idx = 0              # quantos quadros já foram decodificados
        self.frame_time = 0.0           # time.time() do último quadro decodificado
        self._frame = None              # último av.VideoFrame (convertido para BGR só sob demanda)
        self._bgr = None; self._bgr_idx = -1
        self._cond = threading.Condition()
        self._send_lock = threading.Lock()
        self._conv_lock = threading.Lock()
        self._proc = self._video = self._ctrl = None
        self._running = False
        self.error = None

    # ---------- ciclo de vida ----------
    def _adb(self, *args, **kw):
        base = [self.adb] + (["-s", self.serial] if self.serial else [])
        return subprocess.run(base + list(args), capture_output=True, text=True, timeout=30, **kw)

    def _server_version_and_jar(self):
        local = os.path.join(os.path.dirname(shutil.which(self.adb) or self.adb), "scrcpy-server")
        if not os.path.isfile(local):
            raise ScrcpyError(f"scrcpy-server não encontrado ao lado do adb ({local})")
        out = subprocess.run([os.path.join(os.path.dirname(local), "scrcpy"), "--version"],
                             capture_output=True, text=True, timeout=15).stdout
        m = re.search(r"scrcpy (\S+)", out)
        if not m:
            raise ScrcpyError("não consegui ler a versão do scrcpy")
        return m.group(1), local

    def start(self, timeout=10):
        import av  # só exigido quando o scrcpy é usado
        version, jar = self._server_version_and_jar()
        r = self._adb("push", jar, self.REMOTE_JAR)
        if r.returncode:
            raise ScrcpyError("adb push falhou: " + r.stderr)
        scid = "%08x" % random.randint(0, 0x7FFFFFFF)
        if self._adb("forward", f"tcp:{self.port}", f"localabstract:scrcpy_{scid}").returncode:
            raise ScrcpyError(f"adb forward falhou (porta {self.port} ocupada?)")
        opts = [f"scid={scid}", "log_level=info", "tunnel_forward=true", "audio=false",
                f"control={'true' if self.control else 'false'}",
                "cleanup=true", "clipboard_autosync=false", "stay_awake=true",
                f"max_fps={self.max_fps}", f"video_bit_rate={self.bit_rate}", f"max_size={self.max_size}"] + self.extra_opts
        base = [self.adb] + (["-s", self.serial] if self.serial else [])
        self._proc = subprocess.Popen(base + ["shell", f"CLASSPATH={self.REMOTE_JAR}", "app_process", "/",
                                              "com.genymobile.scrcpy.Server", version] + opts,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                      encoding="utf-8", errors="replace")
        threading.Thread(target=self._log_server, daemon=True).start()

        # modo "forward": o adb aceita a conexão antes do servidor existir; o servidor confirma
        # com 1 byte (dummy) no socket de vídeo. Depois conecta-se o de controle.
        deadline = time.time() + timeout
        while True:
            if self._proc.poll() is not None:
                raise ScrcpyError("scrcpy-server encerrou ao iniciar (veja o log)")
            try:
                s = socket.create_connection(("127.0.0.1", self.port), timeout=2)
                if s.recv(1):
                    self._video = s; break
                s.close()
            except OSError:
                pass
            if time.time() > deadline:
                self.close(); raise ScrcpyError("timeout conectando ao scrcpy-server")
            time.sleep(0.1)
        if self.control:
            self._ctrl = socket.create_connection(("127.0.0.1", self.port), timeout=5)
            self._ctrl.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._ctrl.settimeout(None)
        self._video.settimeout(None)
        self.device_name = _recvn(self._video, 64).rstrip(b"\0").decode(errors="replace")
        codec = _recvn(self._video, 4)
        if codec != b"h264":
            self.close(); raise ScrcpyError(f"codec inesperado: {codec!r}")
        self._decoder = av.CodecContext.create("h264", "r")
        self._decoder.flags |= 1 << 19          # AV_CODEC_FLAG_LOW_DELAY
        self._decoder.thread_type = "SLICE"     # threads por quadro atrasariam N quadros
        self._running = True
        threading.Thread(target=self._video_loop, daemon=True).start()
        if self.control:
            threading.Thread(target=self._drain_ctrl, daemon=True).start()
        if not self.wait_frame(0, timeout=timeout):
            self.close(); raise ScrcpyError("nenhum quadro de vídeo recebido")
        log.info("scrcpy %s conectado: %s, vídeo %dx%d", version, self.device_name, *self.video_size)
        return self

    def close(self):
        self._running = False
        for s in (self._video, self._ctrl):
            try:
                s and s.close()
            except OSError:
                pass
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.wait(timeout=2)   # com cleanup=true o servidor sai sozinho ao perder o socket
            except subprocess.TimeoutExpired:
                self._proc.kill()
        try:   # com o cabo desconectado o adb pode travar até o timeout: fechar não pode lançar erro
            self._adb("forward", "--remove", f"tcp:{self.port}")
        except (subprocess.SubprocessError, OSError):
            pass

    def _log_server(self):
        for line in self._proc.stdout:
            line = line.rstrip()
            if line:
                log.info("scrcpy-server: %s", line)

    def _drain_ctrl(self):
        # o servidor pode mandar mensagens (clipboard, acks); só descartamos para não travar o socket
        try:
            while self._running and self._ctrl.recv(4096):
                pass
        except OSError:
            pass

    # ---------- vídeo ----------
    def _video_loop(self):
        """Cada pacote: cabeçalho de 12 bytes = u64 (flags + pts) + u32 (tamanho), big-endian.
        Pacote de SESSÃO (bit 63): os 8 bytes seguintes às flags são u32 largura + u32 altura, sem payload.
        CONFIG (bit 62) = SPS/PPS, que é colado na frente do próximo pacote (como o scrcpy faz)."""
        import av
        config = b""
        try:
            while self._running:
                hdr = _recvn(self._video, 12)
                head = struct.unpack(">Q", hdr[:8])[0]
                if head & self.F_SESSION:
                    w, h = struct.unpack(">II", hdr[4:])
                    with self._cond:
                        self.video_size = (w, h)
                    continue
                size = struct.unpack(">I", hdr[8:])[0]
                data = _recvn(self._video, size)
                if head & self.F_CONFIG:
                    config = data; continue
                if config:
                    data, config = config + data, b""
                for frame in self._decoder.decode(av.Packet(data)):
                    with self._cond:
                        self._frame = frame
                        self.frame_idx += 1
                        self.frame_time = time.time()
                        self._cond.notify_all()
        except (EOFError, OSError) as e:
            if self._running:
                self.error = e; log.error("stream de vídeo do scrcpy caiu: %s", e)
        except Exception as e:  # erro de decodificação etc. — não pode morrer em silêncio
            self.error = e; log.exception("erro no stream do scrcpy")
        finally:
            self._running = False
            with self._cond:
                self._cond.notify_all()

    @property
    def alive(self):
        return self._running

    def wait_frame(self, after_idx, timeout=1.0):
        """Espera um quadro com índice > after_idx. Retorna o índice novo ou None (timeout/queda)."""
        with self._cond:
            self._cond.wait_for(lambda: self.frame_idx > after_idx or not self._running, timeout)
            return self.frame_idx if self.frame_idx > after_idx else None

    def frame(self):
        """(índice, imagem BGR) do quadro mais recente. A conversão YUV->BGR é feita só aqui."""
        with self._cond:
            fr, idx = self._frame, self.frame_idx
        if fr is None:
            return 0, None
        # trava: to_ndarray (swscale) em duas threads ao mesmo tempo derrubou o Python (0xc0000005)
        with self._conv_lock:
            if idx != self._bgr_idx:
                self._bgr, self._bgr_idx = fr.to_ndarray(format="bgr24"), idx
            return idx, self._bgr

    # ---------- toque ----------
    def touch(self, action, x, y, pointer_id=0):
        """x, y em pixels do VÍDEO (video_size). O servidor ignora eventos cujo tamanho de tela
        declarado não bate com o do stream atual — por isso mandamos video_size junto."""
        if not self.control:
            raise ScrcpyError("scrcpy aberto em modo observador (control=False): toque bloqueado")
        w, h = self.video_size
        pressure = 0 if action == self.UP else 0xFFFF
        msg = struct.pack(">BBqiiHHHII", self.MSG_INJECT_TOUCH, action, pointer_id,
                          int(x), int(y), w, h, pressure, 0, 0)
        with self._send_lock:
            self._ctrl.sendall(msg)


class Device:
    def __init__(self, serial=None, adb=None, scrcpy=True, **scrcpy_opts):
        """scrcpy=True: captura/toque em tempo real via scrcpy. False: só adb (lento, ~4 s por captura)."""
        self.adb_bin, self.serial = adb or find_adb(), serial
        self.size = self._size()
        self.sc = Scrcpy(self.adb_bin, serial, **scrcpy_opts).start() if scrcpy else None

    def close(self):
        if self.sc:
            self.sc.close(); self.sc = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _cmd(self, *args):
        base = [self.adb_bin] + (["-s", self.serial] if self.serial else [])
        return base + list(args)

    def sh(self, cmd, timeout=15):
        return subprocess.run(self._cmd("shell", cmd), capture_output=True, text=True, timeout=timeout).stdout

    def _size(self):
        m = re.search(r"(\d+)x(\d+)", self.sh("wm size"))
        return (int(m.group(1)), int(m.group(2))) if m else (1080, 2340)

    def _check(self):
        if self.sc and not self.sc.alive:
            raise ScrcpyError(f"conexão scrcpy perdida: {self.sc.error}")

    # --- captura ---
    def screenshot(self, fresh=False, timeout=1.0):
        """Imagem BGR em coordenadas do aparelho (self.size). Com scrcpy devolve o quadro mais recente
        (não copiar/alterar in-place: é reaproveitado). fresh=True espera um quadro novo."""
        if not self.sc:
            raw = subprocess.run(self._cmd("exec-out", "screencap", "-p"), capture_output=True, timeout=15).stdout
            return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        self._check()
        if fresh:
            self.sc.wait_frame(self.sc.frame_idx, timeout)
        _, img = self.sc.frame()
        if img is not None and (img.shape[1], img.shape[0]) != self.size:
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
        return img

    def next_frame(self, after_idx, timeout=1.0):
        """Para o loop de gameplay: (índice, imagem) do primeiro quadro depois de after_idx, ou (after_idx, None)."""
        self._check()
        if self.sc.wait_frame(after_idx, timeout) is None:
            return after_idx, None
        idx, _ = self.sc.frame()
        return idx, self.screenshot()

    # --- entrada (coordenadas do aparelho, 1080x2340) ---
    def _to_video(self, x, y):
        vw, vh = self.sc.video_size
        return (min(max(round(x * vw / self.size[0]), 0), vw - 1),
                min(max(round(y * vh / self.size[1]), 0), vh - 1))

    def touch_down(self, x, y):
        self._check(); self.sc.touch(Scrcpy.DOWN, *self._to_video(x, y))

    def touch_move(self, x, y):
        self._check(); self.sc.touch(Scrcpy.MOVE, *self._to_video(x, y))

    def touch_up(self, x, y):
        self._check(); self.sc.touch(Scrcpy.UP, *self._to_video(x, y))

    def tap(self, x, y):
        if not self.sc:
            return self.sh(f"input tap {int(x)} {int(y)}")
        self.touch_down(x, y); time.sleep(0.05); self.touch_up(x, y)

    def drag(self, x1, y1, x2, y2, seconds, hz=60):
        """Gesto contínuo único (DOWN, MOVEs interpolados, UP)."""
        if not self.sc:
            return self.sh(f"input swipe {int(x1)} {int(y1)} {int(x2)} {int(y2)} {int(seconds*1000)}", timeout=seconds + 10)
        self.touch_down(x1, y1)
        t0 = time.time()
        while (t := time.time() - t0) < seconds:
            k = t / seconds
            self.touch_move(x1 + (x2 - x1) * k, y1 + (y2 - y1) * k)
            time.sleep(1 / hz)
        self.touch_move(x2, y2); self.touch_up(x2, y2)

    def hold(self, x, y, seconds):
        """Dedo parado em (x,y)."""
        self.drag(x, y, x, y, seconds)

    def motion(self, action, x, y):
        """action: DOWN | MOVE | UP."""
        if not self.sc:
            return self.sh(f"input motionevent {action} {int(x)} {int(y)}")
        {"DOWN": self.touch_down, "MOVE": self.touch_move, "UP": self.touch_up}[action](x, y)

    def back(self):
        self.sh("input keyevent 4")

    # --- estado do app ---
    def foreground_package(self):
        out = self.sh("dumpsys window | grep -E 'mCurrentFocus|mFocusedApp'")
        m = re.search(r"([a-zA-Z0-9_.]+)/[a-zA-Z0-9_.$]+", out)
        return m.group(1) if m else None

    def launch_game(self):
        self.sh(f"monkey -p {APP} -c android.intent.category.LAUNCHER 1")

    def screen_locked(self):
        """Tela de bloqueio aparecendo (ou tela apagada)? Ao vivo 2026-10-01 a tela apagou às ~07:05 e travou com
        senha; o bot passou 2.5 h reiniciando o jogo à toa. Com senha o bot NÃO tenta desbloquear."""
        out = self.sh("dumpsys window policy | grep -E 'showing=|mIsShowing'") or ""
        pw = self.sh("dumpsys power | grep -E 'mWakefulness='") or ""
        return "showing=true" in out or "mIsShowing=true" in out or "Asleep" in pw or "Dozing" in pw

    def wake(self):
        self.sh("input keyevent KEYCODE_WAKEUP")

    def restart_game(self):
        """Jogo CONGELADO (imagem idêntica por minutos mesmo com toques; visto 2026-10-01 depois de um anúncio de
        reviver): fecha à força e abre de novo. Perde a partida em andamento (ela já estava perdida)."""
        self.sh(f"am force-stop {APP}")
        time.sleep(2.0)
        self.launch_game()
