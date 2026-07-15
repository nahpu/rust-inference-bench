#!/usr/bin/env python3
"""Canonical per-machine results-folder key: `<os>-<arch>-<cpu-slug>`.

Single source of truth so run.sh (current machine), the migration of legacy
files (per result JSON), and the folder each new result lands in all agree —
avoids bash/Rust/Python computing three slightly different slugs.

Usage:
  machine-key.py            # key for THIS machine (detect os/arch/cpu)
  machine-key.py FILE.json  # key for the machine that produced a result JSON
                            # (reads its `environment` block)
"""
import json
import platform
import re
import subprocess
import sys


def cmd(*a):
    try:
        return subprocess.run(a, capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def cpu_brand():
    s = cmd("sysctl", "-n", "machdep.cpu.brand_string")
    if s:
        return s
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def slug(s):
    s = s.lower()
    # drop trademark noise ((R)/(TM)/"CPU") and the "@ 3.20GHz" clock suffix
    s = re.sub(r"\(r\)|\(tm\)|\bcpu\b|@.*", " ", s)
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "unknown"


def norm_os(o):
    o = o.lower()
    if "darwin" in o or "mac" in o:
        return "macos"
    if "linux" in o:
        return "linux"
    if any(w in o for w in ("win", "mingw", "msys", "cygwin")):
        return "windows"
    return slug(o)


def norm_arch(a):
    a = a.lower()
    if a in ("arm64", "aarch64"):
        return "arm64"
    if a in ("x86_64", "amd64"):
        return "x86_64"
    return slug(a)


if len(sys.argv) > 1:
    # bench.rs writes env.os as "macos aarch64"; secondary.sh as "Darwin arm64".
    # norm_os/norm_arch collapse both forms to the same key.
    env = json.load(open(sys.argv[1])).get("environment", {})
    parts = str(env.get("os", "")).split()
    os_name = parts[0] if parts else platform.system()
    arch = parts[1] if len(parts) > 1 else platform.machine()
    cpu = env.get("cpu", "") or platform.machine()
else:
    os_name = platform.system()
    arch = platform.machine()
    cpu = cpu_brand()

print(f"{norm_os(os_name)}-{norm_arch(arch)}-{slug(cpu)}")
