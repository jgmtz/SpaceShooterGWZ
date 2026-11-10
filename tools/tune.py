"""Busca automática de parâmetros da esquiva no simulador fiel (NEXT_STEPS §2). Sem celular.

    python tools/tune.py                         # grade padrão (W_DPS x SAFE_CAP), gravações padrão
    python tools/tune.py --grid "W_DPS=0,20,40;SAFE_CAP=160,60" --recs neve,ilhas --par 6

Cada combinação roda num PROCESSO separado (em paralelo) sobre as gravações e devolve: contatos com bala,
colisões "naves" (pouco confiável), % do tempo NA MIRA (inimigo acima e alinhado — armas principais atiram reto
p/ cima; substituto do "tempo de clear", que o simulador não mede porque os inimigos vêm da gravação) e o
score = contatos - PESO_MIRA x mira%. Parâmetros aceitos: qualquer constante de gwz/planner.py (W_DPS, SAFE_CAP,
AIM_W, HORIZON, SIGMA_V, LEAD, ...) e T0 (t0 do planejador).
"""
import sys, os, json, itertools, subprocess, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)

RECS = {"neve": r"logs\watch\20261002_022540", "ilhas": r"logs\watch\20260930_224108",
        "chefe": r"logs\watch\rec_ilhas_chefe", "bot_neve": r"logs\watch\bot_neve_v",
        "estrada19": r"logs\watch\bot_estrada19"}       # 2 derrotas do bot no nível 19 (2026-10-06, ~6 min)
PESO_MIRA = 0.3      # 1 ponto de % na mira vale 0.3 contato (ajuste conforme a prioridade)

def worker(params, recs):
    import sim_play as SP, eval_all as EA
    from gwz import evade, planner as PL
    t0 = params.pop("T0", 0.15)
    # memória de balas (NEXT_STEPS §3): COAST_AGE (s; 0 = sem memória), COAST_NEAR (px; 0 = qualquer), COAST_VMIN
    co = (params.pop("COAST_AGE", None), params.pop("COAST_NEAR", None), params.pop("COAST_VMIN", None))
    co = False if co[0] == 0 else co
    for k in [k for k in params if k.startswith("EV_")]:   # EV_<nome> = constante de gwz/evade.py
        setattr(evade, k[3:], params.pop(k))
    y_min = params.pop("Y_MIN", None)             # altura mínima da nave no planejador (Y_RANGE[0])
    x_edge = params.pop("X_EDGE", None)           # margem lateral do planejador: X_RANGE = (x_edge, 1080 - x_edge)
    if x_edge is not None:
        PL.X_RANGE = (x_edge, 1080.0 - x_edge)
    def_waves = int(params.pop("DEF", 0))         # modo defensivo nos N primeiros trechos (0 = desligado)
    def_on = def_waves > 0 or params.pop("DEF_ENTRY", 0)
    def_over = {"w_dps": 0.0, "sigma0": params.pop("DEF_SIGMA", 20.0), "y_min": params.pop("DEF_YMIN", 1800.0)}
    lat = params.pop("LAT", 0.13)                 # latência comando -> nave (s) no simulador
    skip = int(params.pop("SKIP", 0)) or None     # 1 de cada N quadros sem decisão (loop mais lento)
    for k, v in params.items():
        setattr(PL, k, v)
    if y_min is not None:
        PL.Y_RANGE = (y_min, PL.Y_RANGE[1])
    PL.setup()
    def choose(ship, vel, prev, foes, foes_tr, state):
        v = vel + evade.enemy_points(foes_tr, ship)
        if not v and not foes:
            return prev
        # modo defensivo como no run.py: 1as DEF ondas (trechos) + começo de cada trecho (vida não existe aqui)
        over = {}
        if def_on and (state.get("seg_no", 99) < def_waves or state.get("t_in_seg", 99) < 1.5):
            over = dict(def_over)
        xsv = PL.young_xsv(len(v), state.get("ages"))     # incerteza extra das balas recém-vistas (PL.YOUNG_SV)
        tgt, best, path = PL.plan(ship, PL.linear_predictor(v, xsv=xsv), t0=t0, prev_end=state.get("end"),
                                  prev_target=prev, enemies=foes, **over)
        state["end"] = path[-1][1:]
        return tgt
    out = {}
    for name in recs:
        d = RECS[name]
        play = list(np.load(os.path.join(d, "play.npy"), allow_pickle=True))
        sf, tf = os.path.join(d, "step.txt"), os.path.join(d, "times.npy")
        SP.STEP = float(open(sf).read()) if os.path.exists(sf) else 0.1112
        SP.TIMES = list(np.load(tf)) if os.path.exists(tf) else None
        if SP.TIMES is not None:
            dts = np.diff(SP.TIMES); SP.STEP = float(np.median(dts[(dts > 0) & (dts < 0.5)]))
        aux = SP.load_aux(d, play)
        r = SP.run_sim(play, aux, {"choose": choose, "hit_r": 40.0, "roi": 1500, "coast": co, "lat": lat,
                                   "skip_every": skip})
        # contatos no INÍCIO (2 primeiros trechos + 1.5 s de entrada de cada trecho): é onde a nave tem só 1000 de vida
        segs = SP.segments(play); starts = {s[0]: (k, SP._t(s[0])) for k, s in enumerate(segs)}
        early = sum(1 for ts, _, s0 in r["_contacts"] if starts[s0][0] < 2 or ts - starts[s0][1] < 1.5)
        out[name] = {"balas": r["balas"], "naves": r["naves"], "mira": round(r["mira_%"], 1), "min": round(r["min"], 1),
                     "inicio": early}
    return out

def main():
    a = sys.argv
    if "--worker" in a:
        params = json.loads(a[a.index("--worker") + 1]); recs = a[a.index("--recs") + 1].split(",")
        print(json.dumps(worker(params, recs))); return
    grid = a[a.index("--grid") + 1] if "--grid" in a else "W_DPS=0,20,40,80;SAFE_CAP=160,60"
    recs = a[a.index("--recs") + 1] if "--recs" in a else "neve,ilhas"
    par = int(a[a.index("--par") + 1]) if "--par" in a else 6
    keys, vals = [], []
    for part in grid.split(";"):
        k, v = part.split("="); keys.append(k); vals.append([float(x) for x in v.split(",")])
    combos = [dict(zip(keys, c)) for c in itertools.product(*vals)]
    print(f"{len(combos)} combinações x gravações [{recs}], {par} em paralelo", flush=True)
    py = sys.executable; pending = list(combos); running = []; results = []
    t0 = time.time()
    while pending or running:
        while pending and len(running) < par:
            c = pending.pop(0)
            p = subprocess.Popen([py, os.path.abspath(__file__), "--worker", json.dumps(c), "--recs", recs],
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            running.append((c, p))
        time.sleep(2)
        for c, p in list(running):
            if p.poll() is not None:
                running.remove((c, p))
                try:
                    res = json.loads(p.stdout.read().strip().splitlines()[-1])
                except Exception:
                    print("falhou:", c, flush=True); continue
                balas = sum(v["balas"] for v in res.values()); naves = sum(v["naves"] for v in res.values())
                mira = np.mean([v["mira"] for v in res.values()])
                score = balas - PESO_MIRA * mira
                results.append((score, c, balas, naves, mira, res))
                ini = sum(v.get("inicio", 0) for v in res.values())
                print(f"[{time.time() - t0:5.0f}s] {c}: balas={balas} (no início {ini}) naves={naves} mira={mira:.1f}% "
                      f"score={score:.1f}  " + " | ".join(f"{k}: {v['balas']}b/{v.get('inicio', 0)}ini {v['mira']}%"
                                                         for k, v in res.items()), flush=True)
    results.sort(key=lambda r: r[0])
    print("\nMELHORES (menor score = menos contato e mais tempo na mira):")
    for s, c, b, n, m, _ in results[:5]:
        print(f"  score={s:6.1f}  balas={b:3d} naves={n:3d} mira={m:5.1f}%  {c}")

if __name__ == "__main__":
    main()
