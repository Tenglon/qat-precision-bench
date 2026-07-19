"""GPU-utilization sampler for benchmark validity.

Samples nvidia-smi at 200 ms during ONLY the timed window and attaches
util/power/clock statistics to every record, so each reported number carries
proof of whether the GPU was busy while it was measured.

Caveat (documented in the report): utilization.gpu means "a kernel was
resident this sample period" — memory-bound kernels also show ~100%. Compute
saturation is judged separately via achieved-TFLOPS / MFU in the analysis.
"""

from __future__ import annotations

import subprocess
import threading


class GpuSampler:
    def __init__(self, interval_ms: int = 200):
        self.interval_ms = interval_ms
        self.samples = []
        self.proc = None
        self.thread = None

    def __enter__(self):
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,power.draw,clocks.sm",
                 "--format=csv,noheader,nounits",
                 "-lms", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            self.thread = threading.Thread(target=self._reader, daemon=True)
            self.thread.start()
        except Exception:  # noqa: BLE001 — sampling must never break the bench
            self.proc = None
        return self

    def _reader(self):
        for line in self.proc.stdout:
            try:
                u, p, c = (float(x) for x in line.strip().split(","))
                self.samples.append((u, p, c))
            except (ValueError, AttributeError):
                pass

    def __exit__(self, *exc):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.thread is not None:
            self.thread.join(timeout=2)
        return False

    def stats(self) -> dict:
        if not self.samples:
            return {"gpu_util_avg": None}
        us = [s[0] for s in self.samples]
        ps = [s[1] for s in self.samples]
        cs = [s[2] for s in self.samples]
        return {
            "gpu_util_avg": round(sum(us) / len(us), 1),
            "gpu_util_min": min(us),
            "power_w_avg": round(sum(ps) / len(ps), 1),
            "sm_clock_mhz_avg": round(sum(cs) / len(cs)),
            "util_samples": len(us),
        }
