# Próximos passos — tornar o bot ofensivo, não só defensivo

## STATUS (2026-10-02, tudo offline; números de `tools/tune.py` nas gravações neve+ilhas+chefe+partida do bot, ~51 min)

| seção | status | resultado medido |
|---|---|---|
| 1. ofensiva no planejador | **FEITO** (`planner.W_DPS`=10, `AIM_W`=90, `SAFE_CAP`) | tempo NA MIRA 34.6% -> **53.6%** (usuário: 54-56%; bot ao vivo: 39.6%); balas 68 -> 70 (ruído) |
| 2. busca automática | **FEITO** (`tools/tune.py`, processos em paralelo) | grades: W_DPS x SAFE_CAP, W_DPS x AIM_W, memória, altura mínima, latência |
| 3. memória de balas | **TESTADO, mantido como estava** (`evade.coast`, agora compartilhado com o simulador) | sem memória 24 / atual 23 / memória em qualquer lugar 26-32 balas -> "bala não vista" NÃO é bala piscando; o simulador antes nem tinha a memória do ao vivo (corrigido) |
| 4. alarme leve por quadro | NÃO FEITO | ver abaixo: o que importa é decisões/s, e a detecção é o custo dominante |
| extra: detecção mais rápida (2026-10-04) | **FEITO** (`vision.CHANNEL`="gray" em 2 etapas, `_BATCH`, `_STRIP` 256) | 149 -> 82 ms/quadro (perde 4 de 3404 balas, 0 falsas); loop 9.2 -> 11.5 decisões/s |
| 5. pipeline assíncrono + nave prevista | **FEITO** (`run.PLAN_ASYNC`, `--plano-sincrono` desliga; nave sumida = última vista + movimento do dedo) | replay: 8.1-8.4 -> **8.9-9.3** decisões/s (+10%) |
| extra: altura mínima | **FEITO** (`planner.Y_RANGE` 1500 -> 1650) | colisões c/ naves 49 -> 40 (métrica imprecisa); balas iguais |
| 6. skill/grade por tempo de clear | NÃO FEITO | precisa de dados de DPS por peça/skill |
| extra: "manual por tipo de nave" (2026-10-03) | **MEDIDO, não vale agora** (`tools/attack_origin.py`) | só 3% dos contatos vêm de bala que nasce tarde perto de atirador visível; 89% a bala era visível >= 0.5 s antes |
| extra: previsão de trajetória (2026-10-03) | **TESTADO, desligado** (`tools/pred_eval.py`, `evade.LSQ_N`, `NEIGHBOR_R`, `planner.YOUNG_SV`) | reta em 4 obs. melhora a mediana, mas contatos no simulador iguais (68) ou piores |
| extra: modo defensivo no início (2026-10-03) | **FEITO** (`run.DEF_*`, `vision.hp_fraction`) | ondas 1-2 / entrada / vida < 50%: sem ofensiva, margem 20, y >= 1800; simulador sem diferença medível no início (6 x 5) |
| extra: 5 tiros "bala não vista" da floresta 18:12 (2026-10-04) | **FEITO, não testado ao vivo** | chamas ±30° (`vision.FLAME_ANGLES`), filtro de salto da nave pelo dedo (`run.SHIP_JUMP_*`), caça vermelho `enemy_jet3.png`; selftest 97/97 |
| 7. imitação do usuário | PARCIAL | a ofensiva foi calibrada para o MESMO % na mira do usuário (55%) |

Medidas que orientam o resto (simulador, 3 gravações): latência 0.13 -> 0.08 -> 0.05 s = 68 -> 63 -> 52 balas
(com o t0 do planejador acompanhando); **metade das decisões por segundo = 145 balas (2x)**. Ou seja: quadros/s
pesa muito mais que latência; o próximo ganho grande está em detectar mais rápido (seção 4 / detecção).
Limitação: o simulador não mostra o ganho principal da ofensiva (inimigo morrendo antes = menos balas), porque
os inimigos vêm da gravação; e ficar embaixo deles aumenta as "colisões" no simulador (31 -> 45, métrica
imprecisa). Conferir ao vivo: tempo por onda e colisões.

Contexto (ver README_WINDOWS.md §6.5): o bot leva ~2x mais tempo que o
usuário pra limpar a mesma fase, mesmo levando MENOS contato por minuto
(1.57 vs 2.88). Ou seja, a esquiva já é boa; o problema é que o planejador
de `gwz/planner.py` é puramente defensivo (minimiza risco, sem termo de
ofensiva), então ele é livre pra fugir pra qualquer posição seguro, inclusive
pra fora do alcance eficaz da arma — o que alonga a wave e gera mais tiros
no total. Resolver isso ataca a causa raiz; mexer só na esquiva ataca sintoma.

As seções abaixo estão em ordem de prioridade sugerida. Cada uma pode virar
uma etapa de trabalho isolada.

---

## 1. Termo de ofensiva no custo do planejador (maior alavanca, menor esforço)

Arquivo: `gwz/planner.py`.

Hoje o custo de cada posição candidata é só risco + custo de deslocamento.
Adicionar um termo que penaliza se afastar da faixa onde a arma atual tem
cobertura (alcance, spread, mira automática — checar como a arma mira hoje
antes de assumir):

    custo = risco * W_risco + afastamento_da_faixa_otima * W_dps

Regra prática: só aceitar fugir da faixa de dano quando o risco calculado
ultrapassar um limiar — não sempre que existir uma casa marginalmente mais
segura no grid. Começar com `W_dps` baixo e subir gradualmente (ver seção 2
para medir o efeito sem precisar jogar ao vivo).

Critério de sucesso: no `tools/eval_all.py`, tempo de clear simulado cai
sem os contatos totais voltarem a subir para o patamar do "bot ANTIGO".

---

## 2. Busca automática de parâmetros no simulador (evita tuning manual)

Ferramentas já existentes: `tools/eval_all.py`, `tools/sim_play.py`
(validados contra partida real: 38×37 contatos bate).

Em vez de ajustar `W_risco`, `W_dps`, horizonte de previsão (1.2 s),
resolução da grade etc. na mão, escrever um script novo (`tools/tune.py`)
que:
1. define um espaço de busca pra cada parâmetro (grid search simples ou
   random search — não precisa ser bayesiano);
2. roda `sim_play` nas gravações já salvas pra cada combinação;
3. pontua por uma métrica combinada, por exemplo
   `score = contatos_totais + peso * tempo_de_clear_simulado`;
4. imprime a melhor combinação encontrada.

Isso transforma tuning manual em algo reprodutível e sem precisar jogar ao
vivo — resolve diretamente o "não quero ficar interferindo nas partidas".

---

## 3. Tracking de balas entre quadros (ataca "bala não vista", 45% das causas)

Arquivo: provavelmente `gwz/vision.py` ou onde a detecção por quadro é
chamada no loop principal.

Hoje a detecção parte do zero a cada frame. Adicionar um filtro simples
(nem precisa ser Kalman completo — uma média móvel de posição+velocidade já
ajuda) que deixa uma bala detectada no quadro N "existir" por 1-2 quadros
mesmo se o detector falhar nesse instante específico (oclusão por explosão,
sprite sobreposto, etc.).

Validar com `tools/hit_analysis.py` nos logs já salvos: a fração de contatos
rotulados `BALA_NAO_VISTA` deve cair.

---

## 4. Desacoplar captura de detecção pesada

Com o loop a ~8.5 quadros/s em cena pesada, uma bala pode nascer e acertar
entre duas detecções completas. Se o scrcpy já entrega vídeo contínuo:
- rodar uma checagem leve (diff de movimento só na área de jogo, sem HUD)
  a cada quadro de vídeo que chega;
- só disparar a detecção completa por template quando esse diff indicar
  mudança relevante na região.

A detecção pesada continua no mesmo ritmo médio, mas o "alarme" de que algo
novo apareceu fica mais rápido — ajuda tanto "bala não vista" quanto
"bala tardia" (10%).

---

## 5. Pipeline assíncrono (ataca "bala tardia" e parte de "nave perdida")

Hoje o ciclo parece ser serializado: captura → detecção → decisão → toque.
Trocar para: começar a calcular a decisão do quadro N+1 enquanto o toque do
quadro N ainda está sendo enviado. Reduz a latência percebida sem mudar a
qualidade da detecção em si.

Mesma lógica de tracking da seção 3 aplicada à própria nave: usar posição
prevista por 1-2 quadros antes de declarar "nave perdida" em vez de exigir
detecção perfeita a cada frame (hoje é 8.5% das causas; já caiu de 28% com a
correção anterior da busca da nave — mostra que vale a pena continuar nessa
linha).

---

## 6. Repriorizar skill e montagem por tempo-de-clear (não por ranking fixo)

A ordem atual de skill é `Life Steal > esquiva > dano de arma equipada >
Recommend > raridade`. Dado que a causa dominante do problema é "demora
demais pra limpar a wave", faz sentido inverter a lógica: estimar o ganho
de DPS de cada skill/upgrade disponível (ATK%, crit, cadência) contra o
ganho de sobrevivência, e escolher a opção que reduz mais o tempo estimado
de clear da wave atual — não uma lista de prioridade fixa.

Mesmo raciocínio pra montagem (hoje: merge sempre que possível + Refresh
com dinheiro ≥ 15, que é uma regra genérica): calcular um "score de poder"
explícito por configuração possível da grade, baseado nos stats reais da
arma, e escolher a que maximiza poder por recurso gasto.

Esta seção é mais trabalhosa que as anteriores (precisa modelar o que cada
skill/peça realmente vale em DPS) — fazer depois que 1-3 estiverem
estáveis, pra não misturar causas ao medir o efeito.

---

## 7. Clonar o padrão de jogo do usuário (maior potencial, maior esforço)

Já existe `tools/watch.py` (grava tela + toques do usuário) e o simulador
fiel (`tools/sim_play.py`). Isso é a base pronta para um modelo de imitação:

1. Gravar algumas sessões do usuário jogando de verdade com `watch.py`,
   em fases variadas (ilhas, neve, estrada, floresta).
2. Treinar um modelo pequeno (regressão ou árvore de decisão — não precisa
   ser rede profunda) que prevê, a partir das mesmas features que o
   planejador já calcula (densidade de balas à esquerda/direita, distância
   dos inimigos, wave atual), a posição lateral que o usuário escolheria
   naquele instante.
3. Usar essa previsão como viés de posição dentro do planejador físico:
   ele continua decidindo a esquiva fina (item 1), mas a "vontade de ficar
   perto do inimigo pra matar rápido" passa a vir do padrão real do
   usuário, não de uma constante escrita à mão.

Esse item ataca direto a lacuna "o usuário leva metade do tempo do bot",
porque captura o julgamento humano sobre quando vale a pena arriscar
posição em troca de dano — exatamente o tipo de trade-off difícil de
codificar à mão em pesos fixos. Fazer por último, quando os itens 1-5 já
estiverem estáveis (senão fica difícil saber qual mudança está causando
qual efeito nos resultados).

---

## Ordem sugerida de execução

1. Seção 1 — termo de DPS no custo do planejador.
2. Seção 2 — busca automática de parâmetros (valida a seção 1 sem jogar ao vivo).
3. Seção 3 — tracking de balas (barato, ataca a maior causa isolada).
4. Seção 5 — pipeline assíncrono + tracking da nave.
5. Seção 4 — desacoplar captura/detecção.
6. Seção 6 — repriorizar skill/grid por tempo-de-clear.
7. Seção 7 — imitação do padrão do usuário.

Cada seção concluída deve ser validada com `tools/eval_all.py` antes de
passar pra próxima, pra manter a comparação limpa (mesma lógica das tabelas
do README_WINDOWS.md §6.2-6.4).
