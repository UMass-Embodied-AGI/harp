"""Constants shared between BaseTask and its split-out mixins."""

TABLE_HEIGHT = 0.74

# Default renderer when config["renderer"] is absent (e.g. a task constructed
# from a bare dict, bypassing config/default.yml). Matches config/default.yml so
# the two can't drift. nyx is a SMART default: _resolve_renderer() transparently
# falls back to the rasterizer when nyx can't run (CPU backend, no CUDA GPU, or
# an incompatible gs_nyx_plugin), so this never breaks CPU/debug runs.
DEFAULT_RENDERER = "nyx"
