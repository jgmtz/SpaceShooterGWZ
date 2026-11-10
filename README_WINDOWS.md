# Galaxy Wing Zero Bot — uso no Windows (S25 FE, cabo USB)

## 1. Instalação (uma vez)
    winget install Genymobile.scrcpy        # traz o adb.exe e o scrcpy-server (o bot usa os dois)
    winget install Python.Python.3.12
    cd <pasta do projeto>
    python -m venv .venv
    .venv\Scripts\python.exe -m pip install -r requirements.txt   # opencv, numpy, av (+ pillow p/ tools)

Celular: depuração USB autorizada, cabo conectado, `adb devices` deve mostrar `device`. Rodar o PC na tomada.

## 2. Rodar o bot
    .venv\Scripts\python.exe run.py                 # UMA partida (para depois da derrota e da análise) — padrão
    .venv\Scripts\python.exe run.py --continuo      # várias partidas seguidas
    .venv\Scripts\python.exe run.py --voltar 1      # no 1º menu de níveis volta N fases antes do Play
    .venv\Scripts\python.exe run.py --minutes 30    # para sozinho
    .venv\Scripts\python.exe run.py --no-grid       # não mexe na montagem (só aperta Play)
    .venv\Scripts\python.exe run.py --esquiva-antiga  # esquiva antiga (evade.choose_target) em vez do planejador
    .venv\Scripts\python.exe run.py --laser         # liga o detector de "laser" (desligado: eram rajadas de chamas)
    .venv\Scripts\python.exe run.py --plano-sincrono  # planejador no loop principal (padrão: numa thread, +10% q/s)

Ciclo: menu -> montagem (merges, expansões, Tool Kit — nunca descartado num Refresh: espera a animação e relê —,
Refresh com dinheiro >= 15, Play) -> partida com esquiva ->
skill (Life Steal > esquiva > HP Regen > Shield Generation > dano de arma equipada > Recommend > raridade;
nunca dinheiro, nunca anúncio) ->
derrota: recusa o reviver (nunca assiste anúncio) -> análise automática dos tiros ->
para (modo 1 partida). Tela bloqueada: espera (nunca usa senha). Jogo congelado 120 s: reinicia o jogo.

## 3. Ferramentas
| comando | o que faz | celular? |
|---|---|---|
| `tools\selftest.py` | **bateria offline** (91 checagens: estados, anúncios, montagem, nave, projéteis, esquiva, loop real) — rodar antes de cada sessão ao vivo | não |
| `tools\eval_all.py [pastas]` | **teste da esquiva nas fases gravadas** (simulador fiel): você x bot antigo x bot novo, contatos por onda, pior rajada | não |
| `tools\tune.py --grid "W_DPS=0,10;Y_MIN=1500,1650" --recs neve,ilhas` | **busca automática de parâmetros** no simulador (em paralelo): balas, colisões, % na mira | não |
| `tools\sim_play.py <pasta> [--calibrar]` | simulador fiel (verdade = trajetórias reconstruídas; bot vê só o que veria ao vivo); `--calibrar` mede o hitbox | não |
| `tools\extract_play.py <logs\watch\pasta>` | prepara uma gravação do watch.py p/ o simulador (play.npy) | não |
| `tools\extract_rec.py <video.mp4> [nome]` | converte uma GRAVAÇÃO DE TELA do celular p/ o simulador | não |
| `tools\extract_botrun.py <nome> <trechos.jsonl>` | converte uma PARTIDA DO PRÓPRIO BOT p/ o simulador (vídeos com todos os quadros, desde 2026-10-02) | não |
| `tools\dodge_eval.py --dia DD/MM --desde HH:MM --ate HH:MM` | tiros com dano + anulados pelo "闪避" do jogo + causa de cada contato | não |
| `tools\hit_analysis.py <logs\frames\*.jsonl>` | por que cada tiro acertou (roda sozinho após cada derrota) | não |
| `tools\progress_report.py` | evolução de todas as sessões | não |
| `tools\watch.py 1800` | grava a tela + os SEUS toques (modo observador; o bot não toca) | sim |
| `tools\grid_live.py --no-play` | só a montagem, para antes do Play | sim |

## 4. Onde olhar quando algo der errado
- `logs/analysis/partida_*.txt` — causa de cada tiro da última partida (NAVE_PERDIDA, BALA_NAO_VISTA, DECISAO_RUIM...).
- `logs/hits/` — a cada dano: 3 quadros com o que o bot via + o quadro cru (`*_raw.png`).
- `logs/frames/<ms>.mp4 + .jsonl` — vídeo de TODOS os quadros + registro por quadro (material do simulador).
- `logs/unknown/`, `logs/ads/`, `logs/skills/`, `logs/grid/` — telas não reconhecidas, anúncios, skills, montagem.

## 5. Regras de segurança (no código)
- Nunca toca em anúncio; cartas de skill e
  peças da bandeja com claquete são recusadas; área de compra (R$) bloqueada (`bot.FORBIDDEN`).
- "Exit battle?" -> X e Continue. Dentro da batalha nunca aperta BACK. Toques a >= 130 px das laterais.

## 6. Estatísticas e comparações (atualizado 2026-10-02)

### 6.1 Esquiva NOVA (planejador espaço-tempo, `gwz/planner.py`)
A cada quadro simula 1.2 s à frente numa grade de posições x tempo e escolhe o caminho em que a bala mais
perto fica o mais LONGE possível (passa pelos vãos no instante certo). Medidas que guiaram o projeto:
- Hitbox real ~40 px (contatos do usuário: bala a 23 px mediana); a esquiva antiga usava 120 px e achava que não
  havia lugar seguro nos muros de discos com vão de ~100 px.
- "Oráculo" (planejador que conhece o futuro verdadeiro): 6 contatos em 12 min na neve -> ~100% é possível
  com a latência e a velocidade da nave.
- Latência ao vivo ~0.13 s (o simulador com 0.13 s reproduz a partida real do bot: 38 x 37 contatos).

### 6.2 Teste offline nas fases gravadas (`tools/eval_all.py`, contatos com bala)
Todos julgados pela mesma "verdade" (trajetórias reconstruídas com o detector atual).

| gravação | duração | você (caminho real) | bot ANTIGO | bot NOVO | NOVO decidindo 6/s | pior rajada do NOVO |
|---|---|---|---|---|---|---|
| Neve (fase inteira, até o chefe) | 12 min | 35 | 59 | **10** | 18 | 2 em 30 s |
| Ilhas | 12 min | 13 | 19 | **10** | 15 | 2 em 30 s |
| Chefe 108K (ilhas) | 2.5 min | 7 | — | **0** | 4 | 0 |
| Estrada | 0.8 min | 2 | — | **1** | 4 | 1 |
| Ilhas (curta) | 0.7 min | 1 | — | **0** | 0 | 0 |

Validação do simulador: caminho real do usuário = estrada 2 (real 2 tiros), ilhas 13 (real 15), neve 35
(real 11 + esquivas: pessimista). Ondas 1-9 de todas as gravações: 0 contatos do bot NOVO.
"Naves" (colisão com inimigos) NÃO é medido de forma confiável no simulador (explosões viram "inimigo").

### 6.3 Detecção de projéteis
| mudança | antes | depois |
|---|---|---|
| balas de referência perdidas (3377 balas, quadros de teste + 160 prints) | 403 (12%) | **0** |
| quadro do leque de chamas da floresta (balas vistas) | 22 | **92** |
| "lasers" falsos (eram rajadas de chamas) num quadro | 20 lasers / 367 pontos | desligado |
| cometa diagonal | não detectado | detectado (23 reais em 60 prints de outras fases) |
| nave piscando (translúcida após dano) | perdida | achada pela barra de vida (0 mudanças nas 118 já certas) |
| custo da detecção em cena pesada | ~104 ms | ~130 ms (loop ~8.5 q/s) |

### 6.4 Partidas AO VIVO (dano = flash vermelho; contatos = dano + "闪避" do jogo)
| partida | fase | duração | tiros c/ dano | contatos totais | resultado |
|---|---|---|---|---|---|
| S (01/10 19:34, esquiva antiga) | neve | 31 min | 59 | 172 | chegou ao chefe da onda 15 |
| T (01/10 22:20, esquiva antiga) | neve | 15 min | 19 | 32 | onda 11 |
| U (02/10 00:35, esquiva antiga) | **fase inteira vencida** + nova | 27.7 min | 27 (0.97/min) | 86 | venceu 15 ondas + chefe |
| 02/10 14:20 (planejador) | floresta | 3.5 min | 7 | — | morreu na onda 2 (rajadas não vistas) |
| 02/10 17:01 (planejador + chamas/cometa) | floresta | 2 min | 3 | — | morreu na onda 2 (3 tiros = morte) |
| 02/10 19:09 (planejador + tudo) | neve | 25.7 min | 33 (1.29/min) | 143 (77% salvos pelo jogo) | chefe da onda 15, 3 mortes (2 revives) |
| 03/10 00:02 (com ofensiva) | neve | 4.5 min | 10 | — | morreu na onda 3 |
| 03/10 00:10 (sem ofensiva) | neve | 7 s | 2 | — | morreu na onda 1 |
| 03/10 02:11 (modo defensivo) | floresta | 2 min | 3 | — | morreu na onda 2 (onda 1 sem tiros) |
| **04/10 03:52 (detecção rápida, ~12.4 decisões/s)** | floresta | 9 min | 12 | — | **onda 6** (antes: onda 2) |
| 04/10 04:05 | floresta | ~8 min | 7 (~0.9/min) | — | onda 5 (morreu pela vida baixa) |
| 04/10 14:31 | floresta | 1.5 min | 2 | — | onda 1: o scrcpy caiu ~40 s (sem piloto) |
| 04/10 18:12 (Tool Kit + Life Steal + esquiva) | floresta | ~6 min | 9 (0/0/1/5/3 por onda) | — | onda 4 |
| 05/10 23:40 | floresta | ~3 min | 4 | — | onda 2 (começou a onda 2 com 16% de vida) |
| **06/10 01:00 (trocas na grade + Tool Kit nv2)** | floresta | ~39 min | 34 | — | **chefe da onda 15** (2 revives; o 2º perdido por bug) |
| **06/10 02:35 (memória de tipos na montagem)** | floresta | ~35 min | **16** | — | **FLORESTA VENCIDA** (3 revives, todos no chefe) |
| **06/10 04:11** | **nível 17** | ~34 min | 19 (11 nas ondas 1-3) | — | **VENCIDO SEM MORRER**, 0 anúncios |
| **06/10 12:33** | **nível 18 (neve)** | ~53 min (chefe 15 min) | 36 (13 no chefe) | — | **VENCIDO SEM MORRER**, 0 anúncios |
| 06/10 13:51 | nível 19 (estrada) | 2 min | 3 | — | onda 2 (cada tiro ~1/3 da vida) |
| 06/10 14:02 | nível 19 (estrada) | 4 min | 7 | — | onda 3 |

Nível 19, análise dos 10 tiros: paredes de discos de lado a lado com o lugar livre na borda (o planejador não ia
até lá: limite 60-1020 -> agora 40-1040; simulador: 73 -> 57 balas em 4 gravações) e coluna de jatos subindo
pelo fundo da tela que o modelo da neve não reconhecia no asfalto (modelo novo `enemy_jet4.png`).

Bugs achados ao vivo em 05/10 e corrigidos: trocas planejadas na grade nunca aconteciam (o bot só soltava em
casa livre); Tool Kit mesclado na loja virava "desconhecido" e era perdido; a tela FAILED era lida como partida
(o bot "pilotou" ~10 min na tela de derrota).

Os 5 tiros "bala não vista" da partida 18:12, olhando os prints (`logs/hits/`):
| tiro | o que era | correção |
|---|---|---|
| 3 | chama inclinada ~30° (o bot só tinha ±12°) | chamas giradas ±30° (+1 detecção em 120 prints, 0 perdidas, +4 ms) |
| 5, 6 | posição da nave lida errada (salto < 400 px passava) | leitura longe de onde o dedo levou a nave é recusada por até 0.4 s |
| 7 | chamas por cima da nave | — (sobreposição, sem solução) |
| 8 | bateu num caça vermelho de estrelas brancas sem template | `assets/enemy_jet3.png` |

Causas na última partida (todos os contatos): bala não vista 45%, decisão ruim 31%, bala tardia 10%,
nave perdida 8.5% (era 28% antes da correção da busca da nave), não chegou 3.5%.

### 6.5 Ofensiva e ajuste automático (NEXT_STEPS.md, `tools/tune.py`)
Tempo NA MIRA = % do tempo com um inimigo acima e alinhado (as armas principais atiram reto p/ cima).
Usuário: 54-56%. Bot ao vivo (neve, 19:09): 39.6%.

| config (4 gravações, ~51 min) | balas | colisões c/ naves* | na mira |
|---|---|---|---|
| sem ofensiva, altura até 1500 (antes) | 68 | 31 | 34.6% |
| **ofensiva W_DPS 10 + altura >= 1650 (agora)** | **70** | 45 | **53.6%** |

| teste no simulador (3 gravações) | balas |
|---|---|
| latência 0.13 s (atual) / 0.08 / 0.05 | 68 / 63 / 52 |
| metade das decisões por segundo | 145 |
| memória de balas: nenhuma / atual / em qualquer lugar | 24 / 23 / 26-32 (neve+ilhas) |
| planejador em thread (replay) | 8.1-8.4 -> 8.9-9.3 decisões/s |

\* métrica imprecisa: no simulador os inimigos não morrem mais rápido quando o bot fica embaixo deles.

### 6.6 Teste A/B da ofensiva ao vivo e modo defensivo (2026-10-03)
| partida | config | resultado |
|---|---|---|
| 00:02 (A) | com ofensiva | morreu na onda 3, 10 tiros (7 até a onda 3) |
| 00:10 (B) | `--sem-ofensiva` | morreu em 7 s na onda 1, 2 tiros |
| 19:09 (referência) | sem ofensiva (antes) | chefe da onda 15, 8 tiros até a onda 3 |

A nave começa toda partida com **1000 de vida** e cada tiro na neve tira ~500-600: o começo decide a partida
e uma partida de cada lado não basta. **Modo defensivo** (padrão): nas ondas 1-2, nos 1.5 s de entrada de cada
onda e com vida < 50% (barra verde) -> sem ofensiva, margem 20 px, nave abaixo de y=1800. No simulador:
65 balas/19 colisões (defensivo o tempo todo) x 70/45 (normal); no início: 6 x 5 contatos (sem diferença
medível — o simulador não tem vida).

### 6.7 De onde vêm os contatos (2026-10-03, `tools/attack_origin.py`, `tools/pred_eval.py`)
| contatos do bot na neve (37) | % |
|---|---|
| bala visível >= 0.5 s antes (havia tempo) | 89% |
| apareceu tarde, com atirador visível (um "manual por nave" anteciparia) | 3% |
| apareceu tarde, sem atirador visto | 8% |

Nos contatos com bala visível, a previsão linear errava > 40 px em ~52%. Alternativas testadas no simulador
(base 68 balas): reta em 4/6 observações 68/73, velocidade da vizinha 72, margem extra p/ bala nova 74-156 —
nenhuma ajudou. O que comprovadamente pesa: decisões por segundo (metade = 2x contatos) e latência.

### 6.8 Velocidade da detecção (2026-10-04)
| detecção (166 quadros, 3404 balas de referência) | ms/quadro | balas perdidas | falsas novas |
|---|---|---|---|
| original (cor) | 149 | — | — |
| saturação + confirmação colorida | 95 | 42 | 0 |
| **cinza + confirmação colorida, agulhas na cor, lotes, faixas 256 (atual)** | **82** | **4** | **0** |

Loop real (replay): 9.1-9.4 -> **11.5-11.6 decisões/s**. No simulador, metade das decisões dobrava os contatos.

### 6.9 Conclusões atuais
- O bot leva **2x mais tempo que o usuário** na mesma fase (neve: 25.6 x 12.1 min): armas/grade mais fracas ->
  inimigos vivos por mais tempo -> mais balas no total e chega fraco ao chefe. Por minuto ele leva MENOS contato
  que o caminho do usuário no simulador (1.57 x 2.88/min).
- Na floresta (nível 16) cada tiro tira ~600 de vida: 3 tiros matam. Está acima da força atual da nave.
- Próximos passos: força da grade (matar mais rápido), simulador com as partidas do próprio bot em cada fase
  nova (vídeo de todos os quadros já é gravado), e ver por que "bala não vista" ainda é a 1ª causa ao vivo.
