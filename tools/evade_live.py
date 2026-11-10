"""Teste ao vivo SÓ da esquiva (não toca em menus): python tools/evade_live.py [segundos] [--off]
Precisa de uma partida rodando. Para sozinho ao sair do gameplay ou ao fim do tempo.
--off: não mexe a nave (controle), para comparar acertos por minuto com a esquiva ligada."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run
from gwz.device import Device, APP
from gwz.pilot import Pilot
from gwz import states

args = [a for a in sys.argv[1:] if not a.startswith("--")]
secs = float(args[0]) if args else 30
on = "--off" not in sys.argv
with Device() as dev:
    if dev.foreground_package() != APP or not states.is_gameplay(dev.screenshot(fresh=True)):
        sys.exit("A partida não está rodando na tela. Nada foi tocado.")
    pilot = Pilot(dev)
    try:
        st = {}
        end = run.gameplay_loop(dev, pilot, max_seconds=secs, evade_on=on, stats=st)
        mins = max(st["secs"], 1) / 60
        print(f"esquiva {'LIGADA' if on else 'DESLIGADA'}: {st['hits']} acertos em {st['secs']:.0f}s "
              f"= {st['hits']/mins:.1f}/min;", "tempo esgotado" if end is None else f"saiu do gameplay -> {states.classify(end)[0]}")
    finally:
        pilot.close()
