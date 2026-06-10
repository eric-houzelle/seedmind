# SeedMind — Schémas d'Architecture et de Décision

Ce document regroupe les diagrammes décrivant le fonctionnement des différents modes de décision (dont le nouveau mode `latent-q`) et le pipeline d'entraînement parallèle sur le Replay Buffer.

---

## 1. Flux de Décision (Inférence de l'Agent)

Ce schéma décrit comment l'observation brute de l'environnement est traitée selon le mode de décision choisi (`q_only`, `latent_q` ou `planner`).

```mermaid
graph TD
    %% Entrée
    Obs["Observation Brute (Grille, Scalars)"] --> Enc["Encoder (MLP/CNN)"]
    Enc --> Lat["Vecteur Latent (z)"]

    %% Aiguillage du mode de décision
    Obs --> DecMode{Decision Mode?}
    Lat --> DecMode

    %% Branche Q-Only (Observationnel)
    DecMode -- "q_only (Default)" --> QObs["Q-Network (CNN + MLP)"]
    QObs --> ActionQ["Action Q-only"]

    %% Branche Latent-Q
    DecMode -- "latent_q" --> QLat["Latent Q-Network (MLP)"]
    QLat --> ActionLatQ["Action Latent-Q"]

    %% Branche Planner (Rollouts dans le futur)
    DecMode -- "planner" --> Planner["Planner (Random Shooting)"]
    Planner --> WM["World Model (z, a) -> z', r"]
    WM --> VM["Value Model (z') -> V"]
    
    %% Loop du planner
    subgraph "Imagination (Horizon N)"
        WM -.->|Simule N pas| WM
    end
    
    WM --> Score["Score = r_imaginés + V_terminale"]
    Score --> ActionPlan["Action Planner"]

    %% Fin de boucle
    ActionQ --> EnvStep["Env.step(action)"]
    ActionLatQ --> EnvStep
    ActionPlan --> EnvStep
    
    EnvStep --> NextObs["Nouvelle Observation"]

    style DecMode fill:#f9f,stroke:#333,stroke-width:2px
    style Planner fill:#bbf,stroke:#333,stroke-width:1px
    style QLat fill:#bfb,stroke:#333,stroke-width:2px
    style QObs fill:#fbb,stroke:#333,stroke-width:1px
```

---

## 2. Flux d'Entraînement (Parallèle sur le Replay Buffer)

Ce schéma décrit comment l'agent met à jour ses différents réseaux en parallèle à partir des transitions réelles du Replay Buffer.

```mermaid
graph LR
    subgraph "Replay Buffer"
        Buffer[("Transition réelle: s, a, r, s', done, events")]
    end

    %% Entraînement de l'Encoder
    Buffer --> |"Projection"| Enc["Encoder"]
    
    %% Entraînement Q-only
    Buffer --> |"Loss TD (Observationnelle)"| QObs["Q-Network (s)"]
    
    %% Entraînement Latent-Q
    Buffer --> |"Loss TD (sur latents z, z')"| QLat["Latent Q-Network (z)"]
    Enc -.->|Fournit z, z'| QLat

    %% Entraînement World Model
    Buffer --> |"Loss Causal & Transitions"| WM["World Model (z, a) -> z', r, events"]
    Enc -.->|Fournit z, z'| WM

    %% Entraînement Value Model
    Buffer --> |"Loss Value (TD ou Monte Carlo)"| VM["Value Model (z)"]
    Enc -.->|Fournit z| VM
    
    %% Option Dyna (Futur)
    WM -.->|Rêves / Dyna (z_synthétique)| QLat

    style Buffer fill:#ddd,stroke:#333,stroke-width:2px
    style QLat fill:#bfb,stroke:#333,stroke-width:2px
    style WM fill:#bbf,stroke:#333,stroke-width:2px
    style VM fill:#bbf,stroke:#333,stroke-width:2px
```
