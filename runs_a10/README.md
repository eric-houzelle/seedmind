# Données brutes de la campagne de calibration A10 (4-7 août 2026)

Archivées ici parce que la machine A10 a été rendue après la campagne. Analyse
et verdicts : `docs/CALIBRATION_DREAMERV3_REF.md` §7-9.

| Fichier | Contenu |
|---|---|
| `ref_metrics.jsonl` | Référence NM512/dreamerv3-torch sur le micro-fouloïde, 514k steps (§7) |
| `run_500k.log` | Exp 2 — notre agent, config dreamerfix, buffer 100k, seed 0, 500k steps (§8) |
| `run_buf500k_s0.log` | Exp 3 — buffer 500k, seed 0 ; tué par l'OOM-killer à 350k (§9) |
| `run_buf500k_s1.log` | Exp 3 — buffer 500k, seed 1, 500k steps complets (§9) |
| `metrics_seed{0,1}.json` | Métriques structurées des runs exp 3 (fenêtres de 500 pas) |

Baseline aléatoire de comparaison (reproductible sans GPU) :
`python scripts/calibration/random_baseline.py` → 6,25 fourrages/1000 pas.

Régénérer les tableaux par fenêtres de 50k :

```bash
python3 - runs_a10/run_500k.log <<'EOF'
import re, sys, collections
rows = [tuple(map(float, m.groups())) for m in re.finditer(
    r"step\s+(\d+) \| wellbeing ([\d.]+) \| .*? eau (\d+) \| bouffe (\d+) \| morts (\d+)",
    open(sys.argv[1]).read())]
buckets = collections.defaultdict(list)
for s, w, e, b, m in rows:
    buckets[int(s) // 50000].append((w, e, b, m))
prev = 0
for k in sorted(buckets):
    forage = sum(e + b for _, e, b, _ in buckets[k]); n = len(buckets[k]) * 500
    d = buckets[k][-1][3] - prev; prev = buckets[k][-1][3]
    print(f"{k*50:>3}k-{(k+1)*50}k : fourrage {1000*forage/n:5.1f}/1000 | morts {d:.0f}")
EOF
```
