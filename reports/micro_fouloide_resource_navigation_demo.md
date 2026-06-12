# Micro-Fouloide Resource Navigation Demo

```text
Base checkpoint family:
  runs/micro_fouloide_v0_rough_valueplanner_resource_navigation_seed{seed}/checkpoint_final.pt

Config:
  configs/micro_fouloide_v0_rough_valueplanner_resource_navigation.yaml

Planner preset:
  wm-calibrated

Deployment mode:
  runtime guards enabled
  survival objective disabled
  resource memory enabled
  hydration threshold = 0.55
  energy threshold = 0.25
```

```text
seed | rollout_seed | lifespan | dead | capped | drive | planner_used | resmem_used | water | food | damage | health_loss
-----|--------------|----------|------|--------|-------|--------------|-------------|-------|------|--------|------------
   1 |        10047 |      120 | no   | yes    | 0.276 |        65.8% |         n/a |     1 |    0 |      0 |         32
   2 |        10039 |      120 | no   | yes    | 0.668 |        92.5% |       15.0% |     3 |    1 |      1 |         25
   3 |        10027 |      120 | no   | yes    | 0.612 |        86.7% |       13.3% |     3 |    1 |      8 |          0
```

```text
Decision:
  GO for short visual demo candidate with resource memory.

Caveat:
  This is not a final trained policy. Resource memory is a deployment-only
  spatial memory patch. The next architecture step is to integrate this memory
  cleanly into the agent and planner.
```
