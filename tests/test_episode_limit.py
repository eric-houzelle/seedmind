"""Tests du reset périodique (online.episode_limit) — expérience 4.

Garanties contractuelles :
- une troncature reset le monde SANS compter comme une mort (le continue
  predictor ne doit pas apprendre que l'agent meurt à N pas) ;
- une troncature ouvre un nouvel episode_id (les séquences RSSM ne doivent
  jamais mélanger deux mondes) ;
- episode_limit=0 (défaut) laisse le comportement historique inchangé.
"""
from __future__ import annotations

import torch

from scripts.run_fouloide_online import OnlineFouloideSession
from scripts.run_micro_fouloide import load_config


CONFIG = "configs/micro_fouloide_online_homeostatic_rssm_v3_dreamerfix.yaml"


def _session(episode_limit: int, size: int = 12) -> OnlineFouloideSession:
    config = load_config(CONFIG)
    config["env"] = {**config.get("env", {}), "size": size, "max_steps": 0}
    config["online"] = {**config.get("online", {}), "episode_limit": episode_limit,
                        "warmup_steps": 10_000_000}  # pas d'entraînement : test rapide
    return OnlineFouloideSession(config, seed=0, device=torch.device("cpu"))


def test_truncation_resets_without_counting_a_death():
    session = _session(episode_limit=20)
    for _ in range(60):
        session.step()
    assert session.truncations >= 2, "le monde doit avoir été tronqué"
    # lives ne compte QUE les morts ; episode_counter compte les deux causes
    # de reset. L'invariant lie les trois compteurs.
    assert session.episode_counter == session.lives + session.truncations


def test_truncation_opens_a_new_episode_id():
    session = _session(episode_limit=20)
    for _ in range(50):
        session.step()
    ids = {str(e.get("episode_id")) for e in session.learner.buffer._data}
    assert len(ids) >= 2, f"un seul episode_id malgré les troncatures : {ids}"


def test_transitions_are_not_marked_terminal_on_truncation():
    """La transition à l'instant de la troncature ne doit pas avoir done=True."""
    session = _session(episode_limit=20)
    for _ in range(50):
        session.step()
    data = session.learner.buffer._data
    # Sans mort, aucune transition ne doit être terminale.
    if session.lives == 1:
        assert not any(bool(e.get("done")) for e in data), (
            "une troncature a été marquée terminale (le cont_head apprendrait "
            "à prédire une mort fantôme)"
        )


def test_disabled_by_default_keeps_historic_behaviour():
    session = _session(episode_limit=0)
    for _ in range(50):
        session.step()
    assert session.truncations == 0
    assert session.episode_counter == session.lives
