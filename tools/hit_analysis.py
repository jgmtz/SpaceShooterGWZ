"""Análise pós-partida: POR QUE a nave levou cada tiro e O QUE deveria ter feito.
    python tools/hit_analysis.py <logs/frames/xxx.jsonl> [...]      (o run.py roda sozinho após cada derrota)

Para cada acerto (início do flash vermelho), olha o segundo anterior e classifica a causa:
  NAVE_PERDIDA     a visão não achava a nave logo antes -> o bot decidiu às cegas
  BALA_NAO_VISTA   nenhuma bala detectada perto da nave nos quadros antes/depois (projétil de outro formato,
                   explosão, colisão com inimigo, ou bala escondida por efeito)
  BALA_TARDIA      a bala culpada só foi vista < 0.35 s antes (não dava tempo de reagir: latência ~0.2 s)
  PREVISAO_ERRADA  o destino escolhido parecia seguro pela previsão, mas as balas REAIS passaram por lá
                   (bala mudou de direção/velocidade, ou a velocidade estimada estava errada)
  NAO_CHEGOU       o destino era seguro de verdade, mas a nave ainda estava longe dele (lenta/decidiu tarde)
  DECISAO_RUIM     havia um lugar seguro alcançável, mas o bot escolheu outro que não era
  SEM_SAIDA        nenhum lugar alcançável ficava livre (encurralado) — tinha que ter saído antes
E calcula, "em retrospecto" (com as posições REAIS das balas), o melhor lugar alcançável.
"""
import sys, os, json, math, collections
import numpy as np

HIT_R = 80.0          # colisão real (px), medida nos quadros de dano
SPEED = 1100.0        # px/s da nave
LAT = 0.15            # s até a nave obedecer
W = 1080

def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]

def hits_of(fr):
    return [i for i in range(1, len(fr)) if fr[i]["r"] and not fr[i - 1]["r"]]

def _dmin(pos, frames_after):
    """Menor distância de pos a qualquer bala REAL nos quadros dados."""
    best = 1e9
    for f in frames_after:
        for x, y, *_ in f["b"]:
            best = min(best, math.hypot(x - pos[0], y - pos[1]))
    return best

def analyze_hit(fr, i):
    th = fr[i]["t"]
    win = [f for f in fr[:i] if th - f["t"] <= 1.0]
    near_hit = [f for f in fr if abs(f["t"] - th) <= 0.12]
    ship = next((f["s"] for f in reversed(fr[:i + 1]) if f["s"]), None)
    res = {"t": th, "ship": ship}
    if ship is None:
        return dict(res, cause="NAVE_PERDIDA", why="nave nunca localizada neste trecho")
    # 1) nave perdida logo antes?
    recent = [f for f in win if th - f["t"] <= 0.4]
    if recent and sum(f["v"] for f in recent) / len(recent) < 0.5:
        res.update(cause="NAVE_PERDIDA", why=f"nave vista em só {sum(f['v'] for f in recent)}/{len(recent)} quadros nos 0.4 s antes")
        return res
    # 2) bala culpada: a mais próxima da nave no momento do acerto
    # culpada: balas dos quadros ANTES do dano (0-0.35 s), levadas até o instante do acerto pela velocidade.
    # (no flash vermelho do dano a detecção falha — procurar só ali dava 'bala não vista' falso)
    before = [f for f in fr[:i] if th - f["t"] <= 0.35 and not f["r"]]
    cands = [(math.hypot(x + vx * (th - f["t"]) - ship[0], y + vy * (th - f["t"]) - ship[1]),
              x, y, vx, vy, f["t"])                      # guarda onde/quando foi VISTA (p/ voltar no tempo dali)
             for f in before for x, y, vx, vy in f["b"]]
    cands += [(math.hypot(x - ship[0], y - ship[1]), x, y, vx, vy, f["t"]) for f in near_hit for x, y, vx, vy in f["b"]]
    culprit = min(cands) if cands else None
    if culprit is None or culprit[0] > HIT_R * 1.6:
        res.update(cause="BALA_NAO_VISTA",
                   why="nenhuma bala detectada perto da nave no acerto (outro projétil, inimigo ou efeito)")
        return res
    # 3) desde quando essa bala era vista? (volta no tempo pela posição prevista)
    d, cx, cy, cvx, cvy, ct = culprit
    first = ct
    for f in reversed([f for f in win if f["t"] < ct]):
        px, py = cx - cvx * (ct - f["t"]), cy - cvy * (ct - f["t"])
        if any(math.hypot(x - px, y - py) < 60 for x, y, *_ in f["b"]):
            first = f["t"]
        else:
            break
    res["seen_for"] = round(th - first, 2)
    if th - first < 0.35:
        res.update(cause="BALA_TARDIA", why=f"bala culpada vista só {th - first:.2f} s antes (reação precisa ~0.3 s)")
        return res
    # 4) decisão em t0 = ~0.45 s antes: o destino escolhido era seguro DE VERDADE?
    dec = min(win, key=lambda f: abs((th - f["t"]) - 0.45))
    goal = dec["g"] or ship
    after = [f for f in fr if dec["t"] + LAT <= f["t"] <= th + 0.25]
    goal_safe = _dmin(goal, [f for f in after if f["t"] >= th - 0.15]) > HIT_R and \
        math.hypot(goal[0] - ship[0], goal[1] - ship[1]) > 40      # destino = onde apanhou -> não era seguro
    # 5) retrospecto: melhor x alcançável (mesma altura), usando as balas REAIS
    s0 = dec["s"] or ship
    reach = SPEED * max(0.0, (th - dec["t"]) - LAT)
    best, bestd = None, -1
    late = [f for f in after if f["t"] >= th - 0.15]
    for x in range(40, W - 39, 20):
        if abs(x - s0[0]) > reach:
            continue
        # avalia na altura do ACERTO; perto de onde a nave apanhou não conta como livre (a bala que encosta
        # na nave some da detecção — é confundida com o miolo da própria nave)
        if abs(x - ship[0]) < HIT_R:
            continue
        dd = _dmin((x, ship[1]), late)
        if dd > bestd:
            best, bestd = x, dd
    res.update(goal=goal, best_x=best, best_clear=round(bestd), decided_at=round(th - dec["t"], 2))
    dist_goal = math.hypot(ship[0] - goal[0], ship[1] - goal[1])
    if bestd <= HIT_R:
        res.update(cause="SEM_SAIDA", why=f"nenhuma posição a até {reach:.0f} px ficava livre; tinha que ter saído antes")
    elif goal_safe and dist_goal > 60:
        res.update(cause="NAO_CHEGOU", why=f"destino x={goal[0]} era seguro, mas a nave estava a {dist_goal:.0f} px dele")
    elif goal_safe:
        res.update(cause="PREVISAO_ERRADA", why="nave no destino previsto como seguro, mas a bala real passou por ali "
                                                f"(vel. estimada {cvx:.0f},{cvy:.0f} px/s)")
    else:
        res.update(cause="DECISAO_RUIM", why=f"escolheu x={goal[0]} (não era seguro); x={best} estava livre "
                                             f"(folga {bestd:.0f} px)")
    return res

def report(paths):
    all_hits, lines = [], []
    for p in paths:
        fr = load(p)
        for i in hits_of(fr):
            h = analyze_hit(fr, i); h["file"] = os.path.basename(p); all_hits.append(h)
    t0 = all_hits[0]["t"] if all_hits else 0
    for k, h in enumerate(all_hits, 1):
        extra = f" | deveria: ir para x={h['best_x']}" if h.get("best_x") is not None and h["cause"] in ("DECISAO_RUIM", "NAO_CHEGOU", "PREVISAO_ERRADA") else ""
        lines.append(f"tiro {k} (t+{h['t'] - t0:5.1f}s, nave {h['ship']}): {h['cause']} — {h['why']}{extra}")
    c = collections.Counter(h["cause"] for h in all_hits)
    lines.append("")
    lines.append("RESUMO: " + ", ".join(f"{k}={v}" for k, v in c.most_common()) if c else "RESUMO: nenhum tiro")
    tips = {
        "NAVE_PERDIDA": "melhorar a detecção da nave (bordas / efeitos por cima)",
        "BALA_NAO_VISTA": "projétil de outro formato ou colisão: novo template de bala / desvio de inimigos",
        "BALA_TARDIA": "ver balas mais cedo (região acima maior) ou ficar mais baixo na tela",
        "PREVISAO_ERRADA": "prever melhor a trajetória (balas que curvam/aceleram) ou margem maior",
        "NAO_CHEGOU": "decidir mais cedo / nave mais rápida (passo do dedo) / destino mais perto",
        "DECISAO_RUIM": "ajustar custo/risco da esquiva (o lugar certo existia)",
        "SEM_SAIDA": "antecipar: evitar entrar em regiões que fecham (bordas, entre colunas)",
    }
    if c:
        lines.append("O QUE MELHORAR: " + "; ".join(f"{k}: {tips[k]}" for k, _ in c.most_common()))
    # CÂMERA LENTA (dedo solto na tempestade): deu certo? tiros por minuto de partida com o dedo solto x na tela
    up_s = dn_s = 0.0; up_h = dn_h = 0
    for p in paths:
        fr = load(p)
        for a, b in zip(fr, fr[1:]):
            dt = b["t"] - a["t"]
            if 0 < dt < 0.5:
                if a.get("u"):
                    up_s += dt
                else:
                    dn_s += dt
        for i in hits_of(fr):
            was_up = any(f.get("u") for f in fr[max(0, i - 6):i])     # dedo solto em algum quadro ~0.35 s antes
            up_h += was_up; dn_h += not was_up
    if up_s > 0:
        lines.append(f"CÂMERA LENTA: dedo solto {up_s:.0f}s ({up_h} tiros, {60 * up_h / up_s:.2f}/min) x dedo na tela "
                     f"{dn_s:.0f}s ({dn_h} tiros, {60 * dn_h / max(dn_s, 1e-9):.2f}/min) — tempo real")
    return "\n".join(lines), all_hits

if __name__ == "__main__":
    txt, _ = report(sys.argv[1:])
    print(txt)
