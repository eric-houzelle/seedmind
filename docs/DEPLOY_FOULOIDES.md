# Déployer la démo Fouloïdes

La démo est séparée en deux parties :

- **Backend Docker** : simulation Python + WebSocket.
- **Frontend Vercel** : viewer statique HTML/JS.

## 1. Backend Docker

Construire l'image :

```bash
docker build -t seedmind-fouloides .
```

Lancer le backend en mode stub :

```bash
docker run --rm \
  -p 8787:8787 \
  -p 8788:8788 \
  -e HOST=0.0.0.0 \
  -e PORT=8787 \
  -e WS_PORT=8788 \
  -e SOURCE=stub \
  seedmind-fouloides
```

Ou avec Compose :

```bash
docker compose -f docker-compose.fouloides.yml up --build
```

Vérifier :

```bash
curl http://localhost:8787/healthz
```

Le WebSocket local est :

```text
ws://localhost:8788
```

En production derrière HTTPS, expose-le en `wss://`, par exemple :

```text
wss://fouloides-backend.example.com
```

Variables utiles :

```text
SOURCE=stub | micro | live
HOST=0.0.0.0
PORT=8787
WS_PORT=8788
TICK_MS=150
SIZE=96
FOULOIDES=14
DEVICE=cpu
LIVE_CHECKPOINT=runs/fouloide_live_homeostatic/checkpoint_live.pt
LIVE_CHECKPOINT_EVERY=5000
```

Pour le mode `live`, garde un volume persistant sur `/app/runs` afin de conserver le checkpoint.

## 2. Frontend Vercel

Dans Vercel, importe le repo et configure :

```text
Install Command: npm install --ignore-scripts
Framework Preset: Other
```

Ajoute une variable d'environnement Vercel :

```text
SEEDMIND_WS_URL=wss://fouloides-backend.example.com
```

`SEEDMIND_WS_URL` est l'URL WebSocket publique de ton backend. Elle doit
commencer par `wss://` si le site Vercel est en HTTPS. Exemple :

```text
SEEDMIND_WS_URL=wss://api.example.com/fouloides
```

Le build génère `public/index.html` depuis `seedmind/visualization/fouloides_viewer.html` et injecte cette URL.

Sans variable Vercel, tu peux tester avec un paramètre d'URL :

```text
https://ton-projet.vercel.app/?ws=wss://fouloides-backend.example.com
```

## 3. Test local complet

Terminal 1 :

```bash
docker compose -f docker-compose.fouloides.yml up --build
```

Terminal 2 :

```bash
SEEDMIND_WS_URL=ws://localhost:8788 npm run build:vercel
npx serve public
```

Ouvre l'URL donnée par `serve`.
