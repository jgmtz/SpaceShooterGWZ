"""Regressão contra a versão ANTERIOR ao core: o GWZ montado sobre o AutoSpaceShooterCore tem que decidir
exatamente o mesmo que o bot decidia antes da extração.

    python -m unittest discover -s tests -v

tests/golden_pre_core.json foi gerado rodando tests/run_cases.py com o gwz/planner.py e o gwz/evade.py
originais (antes de o algoritmo ir para o core).
"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from gwz import planner, evade   # noqa: E402
from run_cases import run         # noqa: E402


class CoreRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(HERE, "golden_pre_core.json"), encoding="utf-8") as f:
            cls.golden = json.load(f)
        cls.now = json.loads(json.dumps(run(planner, evade)))   # mesmo formato (tuplas -> listas)

    def test_planejador_igual_ao_anterior(self):
        g, n = self.golden["plans"], self.now["plans"]
        self.assertEqual(len(g), len(n))
        for i, (a, b) in enumerate(zip(g, n)):
            self.assertEqual(a, b, f"cenário {i}")

    def test_rastreio_memoria_e_inimigos_iguais_ao_anterior(self):
        g, n = self.golden["track"], self.now["track"]
        self.assertEqual(len(g), len(n))
        for i, (a, b) in enumerate(zip(g, n)):
            self.assertEqual(a, b, f"quadro {i}")


if __name__ == "__main__":
    unittest.main()
