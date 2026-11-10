# Runtime configuration

Pass a profile with `--config <path>` to `scripts/collect.py`, `scripts/eval.py`,
`scripts/eval_vla.py`, or `scripts/eval_expert_set.py`. Values in the profile
override `default.yml`.

| Files or directory | Purpose |
| --- | --- |
| `default.yml` | General task defaults, merged under every profile |
| `franka.yml`, `arx_x5.yml`, `ur5_wsg.yml`, `xarm7.yml` | Embodiment selection |
| `avatars/` | Explicit avatar model profiles |
| `latest_genesis/`, `pour_water_no_avatar.yml` | Pinned-runtime smoke and rendering profiles |
| `neutral_motion_*.yml` | Force one co-presence human motion for `*_copresence` tasks |
| `object_aug_*.yml` | Model-specific object augmentation examples |
| `with_irrelevant_objects.yml` | Opt-in random background tabletop objects |
| `vla.yml` | VLA client defaults for `scripts/eval_vla.py` (`--vla-config`) |
| `collect_vla.yml` | Default profile for `scripts/collect_vla.sh` |

Keep named profiles stable: similar filenames can represent different render
settings or layouts. Machine-local experiments belong outside the repository or
in ignored `config/debug_*.yml` files.
