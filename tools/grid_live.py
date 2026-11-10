"""Teste ao vivo SÓ da montagem:  python tools/grid_live.py [--no-play]
Precisa estar na tela de montagem (GRID). Faz merges/encaixes/refresh e aperta Play
(com --no-play para antes do Play, para você conferir a grade)."""
import sys, os, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gwz.device import Device
from gwz.bot import Bot
from gwz.grid import PLAY
from gwz import grid_exec, states

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
no_play = "--no-play" in sys.argv
with Device() as dev:
    if states.classify(dev.screenshot(fresh=True))[0] != "GRID":
        sys.exit("Não está na tela de montagem. Nada foi tocado.")
    bot = Bot(dev)
    def tap(x, y, why=""):
        if no_play and (x, y) == PLAY:
            logging.info("(--no-play) parei antes do Play: %s", why); raise SystemExit
        return bot.tap(x, y, why=why)
    grid_exec.assemble(dev, tap)
