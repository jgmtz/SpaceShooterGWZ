"""Montagem (GRID): geometria, catálogo de peças e regras observadas.

Tudo aqui foi medido/observado ao vivo em 2026-09-29 (tools/watch.py, sessões
logs/watch/20260929_005631 = cards de equipamento, logs/watch/20260929_005807 = 43 gestos na grade).
Ainda NÃO há leitura automática da grade nem arrasto automático — este módulo é a base para isso.
"""

# ---- geometria (tela 1080x2340) ----
COLS, ROWS = 5, 7
PITCH = 184                 # px entre centros de casas
CELL = 158                  # lado visível de uma casa
X0, Y0 = 172, 442           # centro da casa (col 0, fila 0)
# casas liberadas nesta conta/fase: bloco 3x3 central (o resto tem X). "Get slots" (anúncio) abriria mais.
UNLOCKED = {(c, r) for c in (1, 2, 3) for r in (2, 3, 4)}

def cell_center(col, row):
    return X0 + col * PITCH, Y0 + row * PITCH

# botões da montagem
PLAY = (870, 2175)
REFRESH = (540, 2175)       # custa dinheiro da partida (x15) — só com regra explícita
GET_SLOTS = (210, 2175)     # ícone de claquete = ANÚNCIO -> NUNCA tocar
TRAY_Y = 2000               # faixa da bandeja de peças (y ~1940-2060)

# ---- catálogo (formato = casas ocupadas (largura, altura); lido dos quadradinhos verdes dos cards) ----
# tipo: ataque / defesa / suporte / blindagem. "vizinhas" = bônus para as peças ADJACENTES na grade.
PIECES = {
    "gatling_gun":   {"shape": (1, 2), "kind": "ataque",    "primary": True,
                      "note": "arma principal 'M': obrigatória para dar Play (senão 'Primary Weapon Required')"},
    "missile_pod":   {"shape": (1, 2), "kind": "ataque",    "note": "o míssil verde-camuflado da partida"},
    "beamer":        {"shape": (2, 1), "kind": "ataque",    "note": "feixe; nível 1 cinza/listras amarelas, nível 2 verde/laranja"},
    "phaser_cannon": {"shape": (1, 3), "kind": "ataque"},
    "plasma_orb":    {"shape": (1, 1), "kind": "ataque",    "note": "usuário trocou o Phaser Cannon por ele (2026-10-01); sem template ainda"},
    "shield_pulsar": {"shape": (2, 1), "kind": "defesa",    "note": "nível 1 azul-aço, nível 2 verde"},
    "core_reactor":  {"shape": (1, 1), "kind": "suporte",   "note": "+20% ataque das vizinhas, esquiva"},
    "tool_kit":      {"shape": (2, 1), "kind": "suporte",   "note": "cura; +20% ataque das vizinhas"},
    "boost_fuel":    {"shape": (1, 2), "kind": "blindagem", "note": "vida; +20% crítico das vizinhas"},
    "cash_purge":    {"shape": (1, 1), "kind": "economia",  "note": "item de dinheiro da partida"},
    "arc_shield":    {"shape": (2, 1), "kind": "defesa",    "note": "peça azul em arco (nv3 visto em 2026-10-01; tipo exato incerto)"},
}

# ---- regras observadas nos gestos ----
RULES = """
- Arrastar da bandeja para casas livres encaixa a peça. A peça fica ACIMA do dedo (~150-200 px) e
  mantém o ponto onde foi agarrada; durante o arrasto o jogo pinta de VERDE as casas de destino
  -> o bot deve conferir o verde antes de soltar (malha fechada), em vez de confiar num offset fixo.
- Merge: soltar uma peça sobre outra IGUAL e do MESMO nível -> uma peça do nível seguinte
  (visto: Shield Pulsar 1 + 1 -> 2). A aparência muda com o nível; o número fica no círculo amarelo.
- Soltar sobre uma peça DIFERENTE: a nova entra, a antiga volta para a bandeja.
- Soltar em casa com X / fora da área: a peça volta para a bandeja.
- Peças da grade podem voltar para a bandeja (arrastar para baixo) e ser movidas dentro da grade.
- A arma M pode ser movida e até tirada; sem ela o Play mostra 'Primary Weapon Required' e uma mão
  apontando para a M na bandeja.
- Nenhuma rotação observada (Beamer sempre deitado, Missile Pod sempre em pé).
- A bandeja muda entre waves (peças novas aparecem após cada wave).
"""
