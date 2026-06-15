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

### Option recommandée avec ton Nginx/Apache existant

Si `www.releaskills.com` est déjà servi par ta machine avec les certificats :

```text
/etc/letsencrypt/live/www.releaskills.com-0001/fullchain.pem
/etc/letsencrypt/live/www.releaskills.com-0001/privkey.pem
```

ne lance pas Caddy sur 80/443. Lance seulement le backend Docker en localhost,
puis ajoute une route WebSocket dans ton reverse proxy existant.

Pré-requis :

```text
www.releaskills.com pointe vers l'IP publique de ta machine
ton reverse proxy HTTPS existant sert déjà www.releaskills.com
Docker et Docker Compose sont installés
```

Sur la machine :

```bash
git clone https://github.com/eric-houzelle/seedmind.git
cd seedmind
git checkout main
cp .env.fouloides.example .env.fouloides
```

Édite `.env.fouloides` si besoin :

```text
FOULOIDES_DOMAIN=www.releaskills.com
ACME_EMAIL=ton-email@example.com
SOURCE=stub
```

Lance uniquement le backend Python :

```bash
docker compose --env-file .env.fouloides -f docker-compose.fouloides.backend.yml up -d --build
```

Le backend écoute alors seulement en local :

```text
http://127.0.0.1:8787/healthz
ws://127.0.0.1:8788
```

Si ton reverse proxy est Nginx, ajoute le contenu de
`deploy/fouloides/nginx.releaskills.conf` dans le bloc HTTPS existant :

```nginx
server {
    server_name www.releaskills.com;

    ssl_certificate /etc/letsencrypt/live/www.releaskills.com-0001/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/www.releaskills.com-0001/privkey.pem;

    # colle ici deploy/fouloides/nginx.releaskills.conf
}
```

Puis recharge Nginx :

```bash
sudo nginx -t
sudo systemctl reload nginx
```

Vérifie :

```bash
curl https://www.releaskills.com/fouloides-healthz
```

L'URL WebSocket publique est :

```text
wss://www.releaskills.com/fouloides
```

Commandes utiles :

```bash
docker compose --env-file .env.fouloides -f docker-compose.fouloides.backend.yml logs -f
docker compose --env-file .env.fouloides -f docker-compose.fouloides.backend.yml restart
docker compose --env-file .env.fouloides -f docker-compose.fouloides.backend.yml up -d --build
```

### Option Caddy si le domaine n'est pas déjà servi

La config `docker-compose.fouloides.prod.yml` utilise Caddy devant le backend
Python. Caddy obtient le certificat SSL automatiquement et expose le WebSocket
en `wss://`. Utilise-la seulement si Docker peut prendre les ports 80 et 443.

Lance :

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
SEEDMIND_WS_URL=wss://www.releaskills.com/fouloides
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
