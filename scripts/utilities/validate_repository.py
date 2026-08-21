#!/usr/bin/env python3
"""Static consistency checks for the ``gedi-alphaearth-biomass`` repository.

This is a stdlib-only sanity checker. It does not run any model training or read
any remote-sensing data; it only inspects files already in the repository and
fails (non-zero exit) on internal inconsistencies such as:

  * a non-MIT or appended LICENSE (GitHub would not detect it as MIT);
  * ``config.yaml`` re-enabling wall-to-wall mapping or a random test split;
  * a leftover ``scripts/archive/`` directory or per-sample label-draw manifest;
  * stale figure file names referenced from the docs;
  * a source-model manifest whose structure or safeguards are wrong;
  * missing key dependencies in ``requirements.txt``.

Run from anywhere; the repository root is derived from this file's location
(``scripts/utilities/validate_repository.py`` -> parents[2]).
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8", errors="ignore")


def scan_scripts_for_stale_paths() -> None:
    """Forbid path patterns that contradict the repository layout.

    The repository uses ``outputs/tables/{main_results,diagnostics,audits}/`` and
    ``figures/``. Any script still referencing ``root / "reports"``,
    ``root / "outputs/figures"``, a bare ``outputs/tables/<file>.csv`` (instead of a
    subdir), or one of the three historically-misplaced tables is inconsistent and
    must be fixed.
    """
    forbidden_substrings = [
        'root / "reports"', "root / 'reports'",
        'root / "outputs/figures"', "root / 'outputs/figures'",
        '"outputs/tables/representation_transfer_summary.csv"',
        '"outputs/tables/kaihua_fewshot_summary.csv"',
        '"outputs/tables/kaihua_label_thresholds.csv"',
    ]
    # The validator itself mentions these strings (in this docstring and below), so
    # it must be excluded from its own scan.
    self_path = ROOT / "scripts/utilities/validate_repository.py"
    hits = []
    for py in (ROOT / "scripts").rglob("*.py"):
        if py.resolve() == self_path.resolve():
            continue
        try:
            text = py.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for sub in forbidden_substrings:
            if sub in text:
                hits.append(f"{py.relative_to(ROOT)}: {sub}")
    check("no stale path patterns in scripts", len(hits) == 0, str(hits))

    # A bare outputs/tables/<non-subdir>/ path is forbidden; the only allowed
    # continuations are main_results/, diagnostics/, audits/.
    bare = []
    pattern = re.compile(r'outputs/tables/(?!(main_results|diagnostics|audits)/)')
    for py in (ROOT / "scripts").rglob("*.py"):
        if py.resolve() == self_path.resolve():
            continue
        try:
            text = py.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for m in pattern.finditer(text):
            snippet = text[max(0, m.start() - 25):m.end() + 25]
            bare.append(f"{py.relative_to(ROOT)}: ...{snippet}...")
            break
    check("all outputs/tables paths use main_results/diagnostics/audits/",
          len(bare) == 0, str(bare))


def check_source_manifest_summary_consistency() -> None:
    """The source nested-CV metrics recorded in the manifest must match the
    committed ``source_model_comparison_summary.csv``. Both are derived from the same
    out-of-fold predictions and are the single authoritative source estimate; if they
    disagree the release is internally inconsistent.
    """
    mpath = ROOT / "outputs/manifests/frozen_source_model_manifest.json"
    spath = ROOT / "outputs/tables/main_results/source_model_comparison_summary.csv"
    if not mpath.exists() or not spath.exists():
        check("source manifest & summary agree (source_cv)", False,
              "missing manifest or summary CSV")
        return
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    sm = manifest.get("selected_models", {})
    tol = 1e-6
    mism = []
    with spath.open(encoding="utf-8", newline="") as fh:
        rows = {r["representation"]: r for r in csv.DictReader(fh)}
    for rep in ("alphaearth", "conventional"):
        if rep not in sm:
            mism.append(f"{rep}: missing from manifest")
            continue
        if rep not in rows:
            mism.append(f"{rep}: missing from summary")
            continue
        cv = sm[rep].get("source_cv", {})
        srow = rows[rep]
        for key in ("R2", "RMSE", "MAE", "Bias"):
            mv = cv.get(key)
            try:
                sv = float(srow[key])
            except (KeyError, ValueError):
                mism.append(f"{rep}.{key}: absent from summary")
                continue
            if mv is None or abs(float(mv) - sv) > tol:
                mism.append(f"{rep}.{key}: manifest={mv} summary={sv}")
    check("source manifest & summary agree (source_cv)", len(mism) == 0, str(mism))


def check_manifest_paths_posix() -> None:
    """Manifest path strings (e.g. ``model_file``) must use POSIX ``/`` separators so
    the JSON is portable across Windows/macOS/Linux and does not embed backslashes.
    """
    mpath = ROOT / "outputs/manifests/frozen_source_model_manifest.json"
    if not mpath.exists():
        check("manifest paths use / not \\", True)
        return
    manifest = json.loads(mpath.read_text(encoding="utf-8"))

    PATH_TOKENS = ("outputs", "data", "figures", "models",
                   "manifests", "tables")
    bad = []

    def _walk(o: object) -> None:
        if isinstance(o, str):
            if "\\" in o and any(tok in o for tok in PATH_TOKENS):
                bad.append(o)
        elif isinstance(o, dict):
            for v in o.values():
                _walk(v)
        elif isinstance(o, list):
            for v in o:
                _walk(v)
    _walk(manifest)
    check("manifest paths use / not \\", len(bad) == 0, str(bad))


def check_source_n_train_consistency() -> None:
    """The source_spatial_cv N_train in final_project_summary.csv must equal the
    actual outer-training size used by the nested evaluation, derived from the
    recorded per-fold provenance and the manifest train_cap (NOT the uncapped
    total). It must also never exceed train_cap.
    """
    mpath = ROOT / "outputs/manifests/frozen_source_model_manifest.json"
    spath = ROOT / "outputs/tables/main_results/final_project_summary.csv"
    if not mpath.exists() or not spath.exists():
        check("source N_train matches recorded training provenance", False,
              "missing manifest or final_project_summary.csv")
        return
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    train_cap = int(manifest.get("train_cap", 60000))
    n_train_recorded = manifest.get("nested_cv_n_train", {})
    tol = 1.0
    mism = []
    with spath.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["stage"] != "source_spatial_cv":
                continue
            rep = r["representation"].lower()
            try:
                csv_n = float(r["N_train"])
            except (KeyError, ValueError):
                mism.append(f"{rep}: N_train absent")
                continue
            if csv_n > train_cap + tol:
                mism.append(f"{rep}: N_train={csv_n} exceeds train_cap={train_cap}")
            expected = n_train_recorded.get(rep)
            if expected is None:
                mism.append(f"{rep}: manifest missing nested_cv_n_train")
            elif abs(csv_n - float(expected)) > tol:
                mism.append(f"{rep}: csv N_train={csv_n} != manifest {expected}")
    check("source N_train matches recorded training provenance", len(mism) == 0, str(mism))


def check_source_metrics_pooled_not_five_fold_mean() -> None:
    """The headline source metrics are a pooled estimate over the aggregated
    nested out-of-fold prediction vector, NOT the arithmetic mean of the five
    fold-level R2/RMSE values. The reporting must not label them a 'five-fold
    mean', and must say 'pooled'/'out-of-fold'.
    """
    spath = ROOT / "outputs/tables/main_results/final_project_summary.csv"
    bad = []
    if spath.exists():
        with spath.open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                if r["stage"] != "source_spatial_cv":
                    continue
                notes = (r.get("notes") or "").lower()
                if "five-fold mean" in notes or "5-fold mean" in notes:
                    bad.append(f"{r['representation']}: notes say 'five-fold mean'")
                if "pooled" not in notes and "out-of-fold" not in notes:
                    bad.append(f"{r['representation']}: notes missing pooled/out-of-fold wording")
    # Also forbid the exact misleading phrase anywhere in the docs.
    for d in (ROOT / "docs", ROOT):
        for md in d.glob("*.md"):
            if "five-fold mean" in md.read_text(encoding="utf-8", errors="ignore").lower():
                bad.append(f"{md.name}: contains 'five-fold mean'")
    check("source metrics labeled pooled OOF, not five-fold mean", len(bad) == 0, str(bad))


def check_nested_evaluation_distinct_from_deployment() -> None:
    """The nested source-evaluation estimate (source_cv) must be explicitly
    distinguishable from the final deployment model in the manifest.
    """
    mpath = ROOT / "outputs/manifests/frozen_source_model_manifest.json"
    if not mpath.exists():
        check("nested evaluation distinct from deployment model", False, "manifest missing")
        return
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    sm = manifest.get("selected_models", {})
    mism = []
    for rep in ("alphaearth", "conventional"):
        info = sm.get(rep)
        if not info:
            mism.append(f"{rep}: missing")
            continue
        se = info.get("source_evaluation")
        fd = info.get("final_deployment_model")
        if not isinstance(se, dict) or se.get("model") != "nested_selected":
            mism.append(f"{rep}: source_evaluation missing or wrong")
        if not isinstance(fd, dict) or "model" not in fd or "config" not in fd:
            mism.append(f"{rep}: final_deployment_model missing")
    check("nested evaluation distinct from deployment model", len(mism) == 0, str(mism))


def main() -> int:
    # --- 1. Required root files --------------------------------------------
    for f in ["README.md", "LICENSE", "requirements.txt",
              "config.yaml", "DATA_LICENSE.md", "environment.yml"]:
        check(f"root file exists: {f}", (ROOT / f).exists())

    # --- 2. LICENSE is pure MIT -------------------------------------------
    lic = read("LICENSE")
    check("LICENSE starts with 'MIT License'",
          lic.lstrip().startswith("MIT License"))
    check("LICENSE has no appended data notice",
          "Data licensing notice" not in lic)

    # --- 3. config.yaml settings ------------------------------------------
    cfg = read("config.yaml")
    check("config wall_to_wall.target_map is false", "target_map: false" in cfg)
    check("config has no 'target_map: true'", "target_map: true" not in cfg)
    check("config has no random_test_fraction", "random_test_fraction" not in cfg)

    # --- 4. scripts/archive removed ---------------------------------------
    check("scripts/archive absent", not (ROOT / "scripts/archive").exists())

    # --- 5. per-sample label-draw manifest removed -----------------------
    manifests_dir = ROOT / "outputs/manifests"
    bad = list(manifests_dir.glob("*label_draws*")) if manifests_dir.exists() else []
    check("no label_draws manifest present", len(bad) == 0,
          str([p.name for p in bad]))

    # --- 6. figures: canonical name present, old name absent -------------
    check("figures/kaihua_label_efficiency.png exists",
          (ROOT / "figures/kaihua_label_efficiency.png").exists())
    check("figures/kaihua_label_efficiency_final.png absent",
          not (ROOT / "figures/kaihua_label_efficiency_final.png").exists())

    # --- 7. no stale figure name in docs ---------------------------------
    stale = []
    for d in (ROOT / "docs", ROOT):
        for md in d.glob("*.md"):
            if "kaihua_label_efficiency_final" in md.read_text(encoding="utf-8",
                                                              errors="ignore"):
                stale.append(md.name)
    check("no stale figure name in docs", len(stale) == 0, str(stale))

    # --- 8. source-model manifest structure & safeguards -----------------
    mpath = ROOT / "outputs/manifests/frozen_source_model_manifest.json"
    if mpath.exists():
        m = json.loads(mpath.read_text())
        sm = m.get("selected_models", {})
        ok_struct, detail = True, ""
        for rep in ("alphaearth", "conventional"):
            if rep not in sm:
                ok_struct, detail = False, f"{detail} missing {rep}"
                continue
            info = sm[rep]
            for k in ("model", "config", "source_cv"):
                if k not in info:
                    ok_struct, detail = False, f"{detail} {rep}.{k} missing"
            for k in ("R2", "RMSE", "MAE", "Bias"):
                if k not in info.get("source_cv", {}):
                    ok_struct, detail = False, f"{detail} {rep}.source_cv.{k} missing"
        check("source manifest selected_models structure", ok_struct, detail)
        check("target_label_locked is true", m.get("target_label_locked") is True)
        check("zhejiang_labels_used is false", m.get("zhejiang_labels_used") is False)
        check("zhejiang_performance_used is false",
              m.get("zhejiang_performance_used") is False)
    else:
        check("source manifest exists", False, str(mpath))

    # --- 9. requirements.txt key dependencies ----------------------------
    req = read("requirements.txt").lower()
    for dep in ("pyarrow", "scipy", "pyproj", "shapely", "requests"):
        check(f"requirements.txt has {dep}", dep in req)

    # --- 10. no stale path patterns in scripts ----------------------------
    scan_scripts_for_stale_paths()

    # --- 12. source manifest vs summary consistency -----------------------
    check_source_manifest_summary_consistency()

    # --- 13. manifest paths are POSIX (no backslashes) --------------------
    check_manifest_paths_posix()

    # --- 14. source N_train matches recorded training provenance ----------
    check_source_n_train_consistency()

    # --- 15. source metrics are pooled OOF, not a five-fold mean ----------
    check_source_metrics_pooled_not_five_fold_mean()

    # --- 16. nested evaluation distinct from final deployment model -------
    check_nested_evaluation_distinct_from_deployment()

    # --- report -----------------------------------------------------------
    failed = [c for c in CHECKS if not c[1]]
    for name, ok, detail in CHECKS:
        line = f"[{'PASS' if ok else 'FAIL'}] {name}"
        if detail:
            line += f"  -> {detail}"
        print(line)
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
