"""Produto GWZ: adaptador do AutoSpaceShooterCore para o Galaxy Wing Zero (Android, via scrcpy)."""
import os, sys

try:
    import shooter_core  # noqa: F401  (instalado com: pip install -e ../AutoSpaceShooterCore)
except ImportError:      # sem instalar: usa a pasta irmã ../AutoSpaceShooterCore, se existir
    _base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..")
    for _nome in ("AutoSpaceShooterCore", "space-shooter-core"):
        _core = os.path.join(_base, _nome)
        if os.path.isdir(os.path.join(_core, "shooter_core")):
            sys.path.insert(0, os.path.abspath(_core))
            break
    import shooter_core  # noqa: F401
