from seedmind.agent.curiosity import CausalCuriosityModule


def test_causal_curiosity_rewards_new_and_repeated_events():
    curiosity = CausalCuriosityModule(
        enabled=True, weight=0.2, max_reward=1.0,
        novelty_bonus=1.0, repeat_bonus=0.2,
    )
    first = curiosity.compute("craft_tool", event_amount=1)
    repeat = curiosity.compute("craft_tool", event_amount=1)
    assert first > repeat > 0.0


def test_causal_curiosity_ignores_noops_and_disabled():
    curiosity = CausalCuriosityModule(enabled=True, weight=0.2)
    assert curiosity.compute("craft_noop") == 0.0
    assert curiosity.compute("MOVE_UP") == 0.0
    disabled = CausalCuriosityModule(enabled=False, weight=0.2)
    assert disabled.compute("craft_tool") == 0.0
