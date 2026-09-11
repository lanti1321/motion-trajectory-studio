from __future__ import annotations

import importlib
import os
import platform
import sys


def environment_report() -> dict[str, object]:
    dependencies: dict[str, str] = {}
    problems: list[str] = []
    for module_name in ("numpy", "scipy", "mujoco", "PySide6"):
        try:
            module = importlib.import_module(module_name)
            dependencies[module_name] = str(getattr(module, "__version__", "available"))
        except Exception as error:
            dependencies[module_name] = f"unavailable: {error}"
            problems.append(f"cannot import {module_name}")
    if sys.version_info < (3, 10):
        problems.append("Python 3.10 or newer is required")
    display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    if not display and os.environ.get("QT_QPA_PLATFORM") != "offscreen":
        problems.append("no desktop display detected; GUI requires X11 or Wayland")
    return {
        "ok": not problems,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "display": display or os.environ.get("QT_QPA_PLATFORM", "none"),
        "dependencies": dependencies,
        "problems": problems,
    }
