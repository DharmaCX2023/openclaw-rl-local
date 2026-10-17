"""Run-local import overlay for the audited seeded-sampler endpoint defect.

The installed SGLang file is never edited. A private sitecustomize imports this
hook in the engine and every spawned Python worker via an inherited PYTHONPATH.
"""
from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import time

MODULE = "sglang.srt.layers.sampler"
SOURCE_SHA256 = "33262b0cdc984d9a5d9481a3ff71c736f626449c19265d1da5e898f169669468"
OLD = "x = hashed.to(torch.float64) / torch.iinfo(torch.uint32).max"
NEW = "x = (hashed.to(torch.float64) + 0.5) / float(2**32)"
VERSION = "uint32-midpoint-open-interval-v1"


def patched_source(data: bytes) -> str:
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("SGLang sampler changed; re-audit before applying overlay")
    text = data.decode("utf-8")
    if text.count(OLD) != 1:
        raise RuntimeError("seeded sampler patch point is not unique")
    return text.replace(OLD, NEW).replace(
        "# x is a uniform sample in [0, 1]. get gumbel noise from it.",
        "# Midpoint mapping keeps x strictly in (0, 1), including hash endpoints.",
    )


class _SamplerLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        # Do not reuse the shared installation's unpatched bytecode cache.
        return compile(patched_source(self.get_data(self.path)), self.path, "exec")

    def exec_module(self, module):
        super().exec_module(module)
        module.verified_repair_sampler_patch = VERSION
        destination = os.environ.get("VERIFIED_REPAIR_SAMPLER_AUDIT")
        if destination:
            path = Path(destination).resolve()
            relative = path.relative_to("/root/openclaw-rl-local/outputs")
            if not relative.parts[0].startswith("verified-repair-"):
                raise RuntimeError("sampler audit path must be run-owned")
            path.parent.mkdir(parents=True, exist_ok=True)
            event = json.dumps({
                "pid": os.getpid(), "time": time.time(), "version": VERSION,
                "source_file": self.path, "source_sha256": SOURCE_SHA256,
                "shared_installation_modified": False,
            }) + "\n"
            fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            try:
                os.write(fd, event.encode())
            finally:
                os.close(fd)


class _SamplerFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin or not spec.origin.endswith(".py"):
            raise RuntimeError("cannot locate audited SGLang sampler source")
        spec.loader = _SamplerLoader(fullname, spec.origin)
        return spec


def install():
    if MODULE in sys.modules:
        if getattr(sys.modules[MODULE], "verified_repair_sampler_patch", None) != VERSION:
            raise RuntimeError("sampler imported before the experiment overlay")
        return
    if not any(isinstance(finder, _SamplerFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _SamplerFinder())


SITE_CUSTOMIZE = """import os
if os.environ.get("VERIFIED_REPAIR_SAMPLER_PATCH") == "1":
    try:
        from verified_repair_sampler_patch import install
        install()
    except BaseException:
        import sys
        sys.stderr.write("Verified-repair sampler overlay initialization failed\\n")
        os._exit(78)
"""


def prepare_environment(output: Path, scripts: Path) -> dict[str, str]:
    """Generate only this run's startup hook, leaving the shared venv intact."""
    directory = Path(output) / "sampler_overlay"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "sitecustomize.py"
    if path.exists() and path.read_text() != SITE_CUSTOMIZE:
        raise ValueError("existing run overlay differs; refusing overwrite")
    path.write_text(SITE_CUSTOMIZE)
    env = os.environ.copy()
    env["VERIFIED_REPAIR_SAMPLER_PATCH"] = "1"
    env["VERIFIED_REPAIR_SAMPLER_AUDIT"] = str(Path(output) / "sampler_overlay_imports.jsonl")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(directory), str(scripts), env.get("PYTHONPATH", "")]
    )
    return env
