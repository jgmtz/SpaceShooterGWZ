# SpaceShooterGWZ: produto de exemplo do AutoSpaceShooterCore

Bot que joga **Galaxy Wing Zero** (shooter vertical de Android) a partir do PC. É um **produto derivado** do [AutoSpaceShooterCore](https://github.com/jgmtz/AutoSpaceShooterCore): a esquiva, a ofensiva, o rastreio e o piloto vêm do core, e este repositório contém só o que é específico deste jogo.

Projeto de estudo de automação e visão computacional. Não é afiliado ao jogo nem aos desenvolvedores dele.

## O que vem do core e o que é deste jogo

| Do core (`shooter_core`) | Específico do GWZ (`gwz/`) |
|---|---|
| Planejador espaço-tempo com ofensiva (`SpaceTimePlanner`) | Números medidos no jogo: faixa da nave, velocidades, pesos (`gwz/planner.py`) |
| Rastreio, memória de balas e inimigos-obstáculo (`Tracker`) | Constantes do rastreio e a esquiva reativa antiga (`gwz/evade.py`) |
| Piloto relativo com compensação de latência (`RelativePilot`) | Ganho, passo e área do dedo medidos no aparelho (`gwz/pilot.py`) |
| Estimativa robusta da nave (`ShipEstimator`) | Limites de salto e de confiança (`run.py`) |
| Planejador assíncrono (`AsyncPlanner`) e loop genérico (`DodgeLoop`) | Visão por templates: nave, balas, chamas, cometas, drones, barras de vida (`gwz/vision.py`, `gwz/enemies.py`) |
| | Telas do jogo e navegação pelos menus (`gwz/states.py`, `gwz/bot.py`) |
| | Montagem automática da grade de armas (`gwz/grid*.py`) |
| | Acesso ao celular por scrcpy e adb (`gwz/device.py`) |
| | Adaptador para o loop genérico do core (`gwz/adapter.py`) |

## Prova de que o core funciona
- **Regressão contra a versão anterior ao core** (`tests/test_core_regression.py`): as saídas do bot original, gravadas antes da extração (`tests/golden_pre_core.json`), são reproduzidas **exatamente** pela versão montada sobre o core (80 planos e 150 quadros de rastreio).
- Na extração, a mesma comparação passou em 2.400 quadros reais de partidas gravadas: planejador, rastreio, memória, inimigos-obstáculo, piloto e estimativa da nave idênticos.
- **Loop completo sobre uma gravação** (`tools/replay_loop.py`): o `run.py` refatorado roda a 8,3 decisões/s e percebe o fim da partida.
- **Loop genérico do core** (`tools/core_loop_replay.py`): o mesmo `DodgeLoop` que joga a arena sintética do core joga o GWZ trocando só o adaptador (9,8 quadros/s na gravação).

## Tecnologias
- Python 3.12, NumPy, OpenCV (template matching, segmentação por cor)
- PyAV (decodificação H.264 do vídeo do scrcpy)
- scrcpy e adb (captura de tela e toque no Android, protocolo binário por socket)

## Como rodar
1. Clonar este repositório e o core lado a lado:
   ```bash
   git clone https://github.com/jgmtz/AutoSpaceShooterCore.git
   git clone https://github.com/jgmtz/SpaceShooterGWZ.git
   cd SpaceShooterGWZ
   ```
2. Instalar as dependências (o core entra como pacote editável da pasta irmã):
   ```bash
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
3. Celular com depuração USB e `scrcpy` instalado. Detalhes, opções e estatísticas em [README_WINDOWS.md](README_WINDOWS.md).
4. Testes offline (sem celular):
   ```bash
   .venv\Scripts\python.exe -m unittest discover -s tests -v
   ```

## Contexto
Projeto pessoal. O bot foi desenvolvido primeiro para este jogo; depois a parte comum foi extraída para o core, aplicando conceitos de **linha de produto de software** da disciplina de Reuso de Software (Engenharia de Software, PUCPR), 2026.

## Uso responsável
Feito para estudo de automação e visão computacional. Respeite os termos de uso do jogo; o projeto não incentiva burlar regras de jogos online.

## Licença
Código sob [MIT](LICENSE). A arte, os nomes e as marcas de Galaxy Wing Zero pertencem aos seus donos e não fazem parte deste repositório.
