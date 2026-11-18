"""
Write paper numbers and figures into ``paper_exports/<paper>/manifest.yaml``.

The manifest is the contract with the ASGARD writing repository: every
simulation number and every figure a paper uses is listed here with its
provenance, and the writing side pulls the manifest and the listed figures at a
tag ``<paper>-results-vN``. Format: ``docs/manifest_spec.md`` of the writing
repository. Entries are appended or replaced by ``key``.

Usage, in an analysis script::

    from paper_exports.export import export_number, export_figure

    export_number("paper1", "MdotAfiveKpc", 7.10, unit="Msun/yr",
                  status="converged", convergence="converged",
                  description="Mean net outflow through both |z| = 5 kpc planes, "
                              "15-40 Myr, A-2048",
                  runs=["A-2048"], data=cache_path)
    export_figure("paper1", "slices_A2048", png_path,
                  caption_hint="...", runs=["A-2048"])

Provenance kept per entry (not only once per file, so an entry exported at an
older commit keeps its own record):

* ``commit``: HEAD of the repository the calling script lives in, with
  ``+dirty`` if tracked files outside ``paper_exports/`` have uncommitted
  changes or the calling script itself is not committed. Untracked files
  elsewhere do not count.
* ``script``: the calling script, relative to the repository root.
* ``data``: ``host:/absolute/path`` and the sha256 of the file. Measurement
  caches live on scratch or JARVIS, not in git, so this is a record, not
  something the writing side can fetch.

Several scripts may export at the same time: the read-modify-write of the
manifest holds an exclusive file lock.
"""

# general
import datetime
import fcntl
import hashlib
import inspect
import math
import os
import socket
import subprocess
from contextlib import contextmanager
from pathlib import Path

# config
import yaml

BASE = Path(__file__).absolute().parent
STATUSES = ("converged", "not_converged", "single_snapshot", "provisional")
CONVERGENCE = ("converged", "not_converged", "single_snapshot", "unknown")
UNCERTAINTY_KINDS = ("noise", "richardson", "range")
FIGURE_STATUSES = ("draft", "final")
#: Files above this size get no sha256 (hashing a 365 GB checkpoint is pointless).
SHA256_MAX_BYTES = 4 * 1024**3


def _git(*args, cwd):
    """Run git in ``cwd``; stdout stripped, '' on failure."""
    result = subprocess.run(["git", *args], check=False, cwd=cwd, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def _repo_root(path):
    """The git top level containing ``path``."""
    root = _git("rev-parse", "--show-toplevel", cwd=Path(path).parent)
    if not root:
        raise RuntimeError(f"{path} is not inside a git repository")
    return Path(root)


def _caller_file():
    """The first source file on the stack outside this module."""
    here = Path(__file__).absolute()
    for frame in inspect.stack()[1:]:
        candidate = Path(frame.filename).absolute()
        if candidate != here:
            return candidate
    raise RuntimeError("no caller outside paper_exports/export.py")


def _commit(repo, script):
    """``<sha>`` or ``<sha>+dirty`` for the repository the script lives in."""
    sha = _git("rev-parse", "HEAD", cwd=repo)
    exclude = f":!{BASE.relative_to(repo)}" if BASE.is_relative_to(repo) else None
    status_args = ["status", "--porcelain", "--untracked-files=no", "--", "."]
    if exclude:
        status_args.append(exclude)
    dirty = bool(_git(*status_args, cwd=repo))
    script_rel = str(script.relative_to(repo))
    script_committed = bool(_git("ls-files", "--error-unmatch", script_rel, cwd=repo))
    script_clean = not _git("status", "--porcelain", "--", script_rel, cwd=repo)
    if not (script_committed and script_clean):
        dirty = True
    return sha + ("+dirty" if dirty else "")


def _data_record(data):
    """``{"path": "host:/abs/path", "sha256": ...}`` for a data file, or None."""
    if data is None:
        return None
    path = Path(data).resolve()  # the real location, not a symlinked view
    record = {"path": f"{socket.gethostname()}:{path}", "sha256": None}
    if path.is_file() and path.stat().st_size <= SHA256_MAX_BYTES:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 22), b""):
                digest.update(block)
        record["sha256"] = digest.hexdigest()
    return record


def _check_uncertainty(uncertainty):
    """Validate ``{kind: noise|richardson|range, lo, hi}`` (or None)."""
    if uncertainty is None:
        return None
    if set(uncertainty) != {"kind", "lo", "hi"}:
        raise ValueError(f"uncertainty needs exactly kind, lo, hi; got {sorted(uncertainty)}")
    if uncertainty["kind"] not in UNCERTAINTY_KINDS:
        raise ValueError(f"uncertainty kind must be one of {UNCERTAINTY_KINDS}")
    return {"kind": uncertainty["kind"], "lo": float(uncertainty["lo"]), "hi": float(uncertainty["hi"])}


@contextmanager
def _locked_manifest(paper):
    """Yield (manifest dict, write function) under an exclusive lock."""
    path = BASE / paper / "manifest.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            manifest = yaml.safe_load(path.read_text()) if path.exists() else None
            manifest = manifest or {"paper": paper, "numbers": [], "figures": []}

            def write(updated):
                tmp = path.with_suffix(".yaml.tmp")
                tmp.write_text(yaml.safe_dump(updated, sort_keys=False, allow_unicode=True, width=100))
                os.replace(tmp, path)

            yield manifest, write
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _update(paper, section, entry, commit):
    """Append or replace ``entry`` (by key) in ``section`` of the paper's manifest."""
    with _locked_manifest(paper) as (manifest, write):
        items = [item for item in manifest.get(section) or [] if item["key"] != entry["key"]]
        manifest[section] = sorted(items + [entry], key=lambda item: item["key"])
        manifest["generated"] = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
        manifest["astronomix_commit"] = commit  # of the latest export; each entry has its own
        manifest.setdefault("numbers", [])
        manifest.setdefault("figures", [])
        ordered = {key: manifest[key] for key in ("paper", "generated", "astronomix_commit", "numbers", "figures")}
        ordered.update({key: value for key, value in manifest.items() if key not in ordered})
        write(ordered)


def export_number(paper, key, value, *, unit, status, description, runs, data=None,
                  uncertainty=None, convergence=None, digits=3):
    """
    Add or replace a number in the paper's manifest.

    Args:
        paper: e.g. "paper1".
        key: letters only, it becomes the LaTeX macro ``\\<key>``. A key keeps its
            meaning once a paper uses it; a new definition needs a new key.
        value: the number (None for a placeholder).
        unit: e.g. "Msun/yr", "erg/s", "" for dimensionless.
        status: converged | not_converged | single_snapshot | provisional.
        description: what exactly was measured (quantity, window, cadence, run).
        runs: run tags, e.g. ["A-2048"].
        data: the cache the number was computed from (recorded with host and sha256).
        uncertainty: None or {"kind": noise|richardson|range, "lo": .., "hi": ..}.
        convergence: None or converged | not_converged | single_snapshot | unknown.
        digits: significant digits to print.
    """
    if not (key.isascii() and key.isalpha()):
        raise ValueError(f"number key {key!r} must be ASCII letters only (it becomes a LaTeX macro)")
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    if convergence is not None and convergence not in CONVERGENCE:
        raise ValueError(f"convergence must be one of {CONVERGENCE}")
    if value is not None and not math.isfinite(float(value)):
        raise ValueError(f"{key}: value {value} is not finite")
    script = _caller_file()
    repo = _repo_root(script)
    commit = _commit(repo, script)
    entry = {
        "key": key,
        "value": None if value is None else float(value),
        "unit": unit,
        "uncertainty": _check_uncertainty(uncertainty),
        "status": status,
        "convergence": convergence,
        "description": description,
        "runs": list(runs),
        "script": str(script.relative_to(repo)),
        "commit": commit,
        "data": _data_record(data),
        "digits": int(digits),
    }
    _update(paper, "numbers", entry, commit)


def export_figure(paper, key, path, *, caption_hint, runs, status="draft"):
    """
    Add or replace a figure in the paper's manifest.

    The figure stays where it is; the manifest stores its path relative to the
    repository root. Commit the figure before tagging, or the writing side
    cannot fetch it.
    """
    if status not in FIGURE_STATUSES:
        raise ValueError(f"figure status must be one of {FIGURE_STATUSES}")
    script = _caller_file()
    repo = _repo_root(script)
    figure = Path(path).absolute()
    if not figure.is_file():
        raise FileNotFoundError(figure)
    try:
        figure_rel = figure.relative_to(repo)
    except ValueError as error:
        raise ValueError(f"figure {figure} is outside the repository {repo}") from error
    commit = _commit(repo, script)
    entry = {
        "key": key,
        "path": str(figure_rel),
        "caption_hint": caption_hint,
        "runs": list(runs),
        "script": str(script.relative_to(repo)),
        "commit": commit,
        "status": status,
    }
    _update(paper, "figures", entry, commit)
