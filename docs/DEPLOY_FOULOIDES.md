# Déployer la démo Fouloïdes

La démo est séparée en deux parties :

- **Backend Docker** : simulation Python + WebSocket.
- **Frontend Vercel** : viewer statique HTML/JS.

## 1. Backend Docker local

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

## 2. Production sur ta machine

### Option recommandée : tout gérer avec Docker

Cette option lance deux conteneurs :

- `fouloides-backend` : simulation Python + WebSocket interne.
- `fouloides-proxy` : Nginx Docker qui expose HTTPS/WSS avec tes certificats.

Elle utilise les certificats déjà présents sur l'hôte :

```text
/etc/letsencrypt/live/www.releaskills.com-0001/fullchain.pem
/etc/letsencrypt/live/www.releaskills.com-0001/privkey.pem
```

Pré-requis :

```text
www.releaskills.com pointe vers l'IP publique de ta machine
le port choisi, par défaut 8443, est ouvert vers cette machine
aucun autre service n'écoute sur ce port
Docker et Docker Compose sont installés
```

Sur la machine :

```bash
git clone https://github.com/eric-houzelle/seedmind.git
cd seedmind
git checkout sandbox-world
cp .env.fouloides.example .env.fouloides
```

Édite `.env.fouloides` si besoin :

```text
FOULOIDES_DOMAIN=www.releaskills.com
ACME_EMAIL=ton-email@example.com
FOULOIDES_TLS_PORT=8443
SOURCE=live
LIVE_CONFIG=configs/micro_fouloide_online_properties.yaml
LIVE_CHECKPOINT=runs/fouloide_live/checkpoint_live.pt
TICK_MS=60
```

Lance le backend et le proxy TLS :

```bash
docker compose --env-file .env.fouloides -f docker-compose.fouloides.tls.yml up -d --build
```

Le backend reste privé dans le réseau Docker. Seul le conteneur Nginx expose :

```text
https://www.releaskills.com:8443
wss://www.releaskills.com:8443/fouloides
```

Vérifie :

```bash
curl https://www.releaskills.com:8443/fouloides-healthz
```

Logs :

```bash
docker compose --env-file .env.fouloides -f docker-compose.fouloides.tls.yml logs -f
```

Si ça ne démarre pas, les causes les plus fréquentes sont :

```text
un autre service utilise déjà le port 8443
les certificats n'existent pas aux chemins attendus
le DNS www.releaskills.com ne pointe pas vers cette machine
```

L'URL WebSocket publique est :

```text
wss://www.releaskills.com:8443/fouloides
```

Commandes utiles :

```bash
docker compose --env-file .env.fouloides -f docker-compose.fouloides.tls.yml ps
docker compose --env-file .env.fouloides -f docker-compose.fouloides.tls.yml restart
docker compose --env-file .env.fouloides -f docker-compose.fouloides.tls.yml down
docker compose --env-file .env.fouloides -f docker-compose.fouloides.tls.yml up -d --build
```

### Option alternative : backend seul derrière un proxy hôte

Si tu préfères utiliser un Nginx installé sur l'hôte, lance seulement :

```bash
docker compose --env-file .env.fouloides -f docker-compose.fouloides.backend.yml up -d --build
```

Puis utilise `deploy/fouloides/nginx.releaskills.full.conf` comme base de
configuration Nginx hôte.

### Option Caddy si tu veux des certificats automatiques

La config `docker-compose.fouloides.prod.yml` utilise Caddy devant le backend
Python. Utilise-la seulement si tu veux que Docker/Caddy obtienne et renouvelle
les certificats lui-même.

```bash
docker compose --env-file .env.fouloides -f docker-compose.fouloides.prod.yml up -d --build
```

Vérifie :

```bash
curl https://www.releaskills.com/healthz
```

Si cette commande échoue au premier lancement, regarde les logs Caddy :

```bash
docker compose --env-file .env.fouloides -f docker-compose.fouloides.prod.yml logs -f caddy
```

Les causes les plus fréquentes sont un DNS qui ne pointe pas encore vers la
machine, un firewall qui bloque 80/443, ou un autre service déjà branché sur
80/443.

L'URL WebSocket publique est :

```text
wss://www.releaskills.com/fouloides
```

Commandes utiles :

```bash
docker compose --env-file .env.fouloides -f docker-compose.fouloides.prod.yml logs -f
docker compose --env-file .env.fouloides -f docker-compose.fouloides.prod.yml restart
docker compose --env-file .env.fouloides -f docker-compose.fouloides.prod.yml pull
docker compose --env-file .env.fouloides -f docker-compose.fouloides.prod.yml up -d --build
```

## 3. Frontend Vercel

Dans Vercel, importe le repo et configure :

```text
Install Command: npm install --ignore-scripts
Framework Preset: Other
```

Ajoute une variable d'environnement Vercel :

```text
SEEDMIND_WS_URL=wss://www.releaskills.com:8443/fouloides
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

## 4. Test local complet

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
