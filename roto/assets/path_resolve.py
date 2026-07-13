"""Resolve an absolute path against known roto checkout locations.

Some paths (the hand's USD asset, real-data recording directories) are
hardcoded absolute paths in config/asset files. This repo is checked out at
different absolute locations on different machines -- this lets those
hardcoded paths keep working on any of them without hand-editing the config
every time you switch machines.
"""

import os

# Every known absolute root this repo has been checked out under. Add a new
# entry here (not at each call site) the next time you start developing from
# a new machine/location -- resolve_path automatically retries every path
# below against whichever of these the configured path was written against.
KNOWN_ROTO_ROOTS = [
    "/home/ayush/icra/roto",
    "/home/ayush/Desktop/real_to_sim/roto",
]


def resolve_path(path: str) -> str:
    """Return `path` unchanged if it exists; otherwise, if it starts with one
    of KNOWN_ROTO_ROOTS, try substituting each of the other known roots for
    that prefix and return the first rewritten path that exists on this
    machine. Raises FileNotFoundError (listing every path tried) if none do.
    """
    if os.path.exists(path):
        return path
    matched_root = next((r for r in KNOWN_ROTO_ROOTS if path.startswith(r)), None)
    tried = [path]
    if matched_root is not None:
        suffix = path[len(matched_root) :]
        for root in KNOWN_ROTO_ROOTS:
            if root == matched_root:
                continue
            candidate = root + suffix
            tried.append(candidate)
            if os.path.exists(candidate):
                return candidate
    raise FileNotFoundError(f"resolve_path: none of these exist: {tried}")
