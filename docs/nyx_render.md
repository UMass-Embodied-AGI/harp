# NYX Rendering

This compatibility page is retained for existing links. The maintained setup,
runtime, camera, fallback, and troubleshooting guide is
[`general/nyx_render_howto.md`](general/nyx_render_howto.md).

Before submitting a NYX job, also read
[`general/gpu_resources.md`](general/gpu_resources.md): NYX needs a
graphics-capable GPU, so compute-only A100/H100/H200 devices do not work. Run
one NYX episode per process (for example `scripts/collect.py --episodes 1`
under the release environment contract), writing artifacts under `data/` and
scheduler logs under `job_logs/`.
