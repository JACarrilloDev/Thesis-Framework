# Advanced Guide & Technical Reference

## 1. Potential-Based Shaping Details
We add:
Φ(s) = -α * d(s)
Reward term: r_shaping = γ * Φ(s') - Φ(s)
Keeps optimal policy invariant. Suggested defaults: α=1.0, γ=0.99.

## 2. Distance Progress & Clipping
Let Δ = d_prev - d_cur. Clip to [-0.4, 0.5] to avoid reward explosions from teleport / reset jitter. Tune upward if robot moves faster.

## 3. Idle / Stuck Logic
- Idle if |Δ| < stuck_delta (default 0.01)
- Idle counter resets on sufficient progress
- Adaptive threshold: idle_base_threshold + dist / idle_dist_coeff
- Stuck penalty triggers after stuck_threshold consecutive idle steps

## 4. Velocity Alignment
Forward velocity component:
r_align = max(0, vx) * cos( heading_error ) * vel_align_scale  
Encourages both rotation toward target and propulsion forward.

## 5. Spin Penalty
If idle for >10 steps and |wz|>0.6: add spin_penalty_scale * |wz| (negative). Prevents oscillatory spinning without translation.

## 6. Near-Target Ramp
When d < near_target_radius_mult * success_dist:
r_ramp = ((span - d)/span) * near_target_bonus  
Smooths approach, reduces hovering just outside success radius.

## 7. Multi-Agent Coordination
Finisher advantage: first robot finishing Phase 1 chooses the closer remaining target. Encourages race + strategic pathing. Could add small assist reward to other robot upon assignment (not active by default).

## 8. Custom Curriculum (Future Hook)
Add to YAML:
```
curriculum:
  stages:
    - { success_dist: 0.6, max_episode_steps: 400, min_iters: 50 }
    - { success_dist: 0.4, max_episode_steps: 500, min_iters: 100 }
```
Runner can watch moving average success and promote stage.

## 9. Packaging (Transition Plan)
If publishing to PyPI:
- Move `run_rl_task.py` → `src/cli/runner.py`
- Add `entry_points={"console_scripts":["run_rl_task=cli.runner:main"]}`
- Exclude large `.ttt` scenes; keep examples downloadable via separate artifact.

## 10. Performance Notes
| Lever | Effect |
|-------|--------|
| Headless mode | Reduces renderer overhead (pr.launch(headless=True)) |
| Frame stack size | Higher stacks raise memory & per-step copy cost |
| Camera resolution | Quadratic cost; drop from 128→84 saves significant GPU memory |
| Progress clip | Too high → unstable gradients; too low → slow learning |

## 11. Debug Tips
- Log raw distances every N steps
- Temporarily raise `w_progress` to test motion responsiveness
- Visual check: enable CoppeliaSim GUI for first episodes then switch to headless

## 12. Common Extensibility Patterns
- Add additional sensors: extend observation vector (update space bounds)
- Shared team reward: average or sum agent progress deltas; keep individual shaping for credit assignment
- Domain randomization: randomize lights, friction, wheel slip in `reset()`

## 13. Troubleshooting Matrix
| Symptom | Likely Cause | Action |
|---------|--------------|--------|
| Reward flat ~0 | Robot velocities near zero | Check action scaling / controller mapping |
| Large negative spikes | Repeated collision penalty | Increase spacing or add collision avoidance shaping |
| Oscillation near target | Overweight heading vs forward motion | Reduce heading bonus or add ramp bonus (already implemented) |
| Idle penalties too early | Threshold too low far from goal | Increase `idle_dist_coeff` or `idle_warmup_steps` |

## 14. File-Based vs Dotted Env Import
File-based (`env_file`, `env_class_name`) avoids needing `__init__.py`. Dotted path cleaner for distributable packages. Both supported with fallback order: dotted → file.

## 15. Recommended Version Control for Experiments
Store:
```
experiments/
  exp_001/
    config.yaml
    notes.md
    checkpoints/
    metrics.csv
```
Embed git commit hash into log header for reproducibility.

---

End of advanced reference.