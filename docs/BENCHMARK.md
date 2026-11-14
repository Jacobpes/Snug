# Benchmark notes

Measured 14 November 2026 (Europe/Helsinki) on the build box.
qemu-user aarch64 is **slower** than a real Cortex-A53 @ 2.0 GHz (no native NEON, instruction emulation).
onnxruntime aarch64 wheels SIGSEGV under qemu-user; the NumPy detector was used for the arm64 numbers (`SNUG_NO_ONNX=1`). On the real UNO Q, ONNX Runtime should be tried first.

| Environment | FPS | decode ms | detect ms | analyze ms | total ms | peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| Native x86_64 (unrestricted) | 341.2 | 0.09 | 0.40 | 1.52 | 2.93 | 187 |
| Native x86_64, taskset 4 cores | 308.7 | 0.07 | 0.42 | 1.82 | 3.24 | 187 |
| aarch64 qemu-user, taskset 4 cores | 10.6 | 1.05 | 61.55 | 21.66 | 94.35 | 178 |
| aarch64 qemu-user, 4 cores + 2 GiB ulimit | 11.1 | 0.93 | 58.93 | 20.57 | 90.06 | 153 |
