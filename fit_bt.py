#!/usr/bin/env python3
"""Fit method scores from JSONL pairwise judgments, with optional A/B correction.

Each record: input_id, rule_id, language, input_form, method_a, method_b, winner.
winner is A/B/TIE in the actual presentation order. Requires numpy and scipy.
This module also supplies the numerical routines used by transfer_analysis.py.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit
from scipy.stats import norm


OUTCOMES = {"A": 1.0, "B": 0.0, "TIE": 0.5}


def normalize_record(row: dict) -> dict:
    """Accept the public format and the original experiment's field names."""
    if not isinstance(row, dict):
        raise ValueError("Each comparison record must be a JSON object")
    scenario = str(row.get("scenario") or "").split("_", 1)
    language = row.get("language") or (scenario[0] if len(scenario) == 2 else None)
    form = row.get("input_form") or row.get("input_type") or (scenario[1] if len(scenario) == 2 else None)
    input_id = row.get("input_id", row.get("sample_id"))
    result = {
        "input_id": input_id, "rule_id": row.get("rule_id", row.get("sample_id", input_id)),
        "language": language, "input_form": form,
        "method_a": row.get("method_a") or row.get("candidate_a"),
        "method_b": row.get("method_b") or row.get("candidate_b"),
        "winner": row.get("winner"),
    }
    for field, value in result.items():
        if value is None or not isinstance(value, (str, int)) or not str(value).strip():
            raise ValueError(f"Missing {field} in comparison record")
        result[field] = str(value).strip()
    if result["winner"] not in OUTCOMES:
        raise ValueError("winner must be A, B or TIE; invalid judgments are not ties")
    if result["method_a"] == result["method_b"]:
        raise ValueError("The two candidates must have different method identifiers")
    if "fold" in row:
        fold = row["fold"]
        if isinstance(fold, bool) or not isinstance(fold, int) or fold < 0:
            raise ValueError("fold must be a nonnegative integer")
        result["fold"] = fold
    return result


def load_comparisons(path: str | Path) -> list[dict]:
    text = sys.stdin.read() if str(path) == "-" else Path(path).read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        raw = json.loads(text)
    else:
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not raw:
        raise ValueError("No comparisons supplied")
    rows = [normalize_record(row) for row in raw]
    inputs = {}
    for row in rows:
        key = (row["language"], row["input_form"], row["input_id"])
        group = (row["language"], row["rule_id"])
        if inputs.setdefault(key, group) != group:
            raise ValueError(f"An input is assigned to multiple original rules: {key}")
    return rows


def select_methods(rows, methods=None):
    available = {r[k] for r in rows for k in ("method_a", "method_b")}
    methods = sorted(available) if methods is None else list(methods)
    if len(methods) < 2 or len(set(methods)) != len(methods):
        raise ValueError("Select at least two distinct methods")
    missing = set(methods) - available
    if missing:
        raise ValueError("Selected methods have no comparison records: " + ", ".join(sorted(missing)))
    rows = [r for r in rows if r["method_a"] in methods and r["method_b"] in methods]
    retained = {r[k] for r in rows for k in ("method_a", "method_b")}
    missing = set(methods) - retained
    if missing:
        raise ValueError("Selected methods have no comparisons against another selected method after filtering: "
                         + ", ".join(sorted(missing)))
    return rows, methods


def bt_design(methods, reference=None, position=True):
    reference = reference or ("direct" if "direct" in methods else methods[0])
    if reference not in methods:
        raise ValueError(f"Reference method is absent: {reference}")
    nonreference = [m for m in methods if m != reference]
    pairs = list(itertools.permutations(methods, 2))
    x = np.zeros((len(pairs), len(nonreference) + int(position)))
    for i, (a, b) in enumerate(pairs):
        for method, sign in ((a, 1), (b, -1)):
            if method != reference:
                x[i, nonreference.index(method)] = sign
        if position:
            x[i, -1] = 1
    return pairs, x, nonreference, reference


def fit_logistic(n, y, x):
    """Batched unpenalized binomial/fractional logistic maximum likelihood.

    n,y: (replicates, ordered_pairs); x: (ordered_pairs, parameters) or one
    design per replicate. The tiny solve jitter is not a coefficient penalty.
    """
    n, y, x = map(lambda a: np.asarray(a, dtype=float), (n, y, x))
    if n.ndim == 1:
        n, y = n[None], y[None]
    if x.ndim == 2:
        x = np.broadcast_to(x, (len(n), *x.shape))
    if n.shape != y.shape or x.shape[:2] != n.shape:
        raise ValueError("Inconsistent logistic data dimensions")
    if np.any(n < 0) or np.any(y < 0) or np.any(y > n) or not np.isfinite(n+y).all():
        raise ValueError("Invalid comparison counts")
    gram = np.einsum("bo,bop,boq->bpq", n, x, x)
    if np.any(np.linalg.matrix_rank(gram) < x.shape[-1]):
        raise ValueError("Scores are not identifiable. Check method coverage and A/B order variation; "
                         "for a fit without position correction, use --no-position.")
    theta = np.zeros((len(n), x.shape[-1]))
    identity = np.eye(x.shape[-1]) * 1e-10
    for _ in range(100):
        eta = np.einsum("bp,bop->bo", theta, x)
        p = expit(eta)
        gradient = np.einsum("bo,bop->bp", n*p-y, x)
        hessian = np.einsum("bo,bop,boq->bpq", n*p*(1-p), x, x)
        step = np.linalg.solve(hessian + identity, gradient[..., None])[..., 0]
        old_loss = (n*np.logaddexp(0, eta)-y*eta).sum(axis=1)
        scale = np.ones(len(n))
        for _ in range(30):
            candidate = theta-scale[:, None]*step
            linear = np.einsum("bp,bop->bo", candidate, x)
            new_loss = (n*np.logaddexp(0, linear)-y*linear).sum(axis=1)
            bad = new_loss > old_loss+1e-9
            if not bad.any():
                break
            scale[bad] *= 0.5
        else:
            raise ValueError("Logistic optimization failed to reduce the loss")
        theta = candidate
        if np.max(np.abs(scale[:, None]*step)) < 1e-9:
            break
    else:
        raise ValueError("No finite converged fit; comparisons may be completely separated")
    eta = np.einsum("bp,bop->bo", theta, x)
    p = expit(eta)
    information = np.einsum("bo,bop,boq->bpq", n*p*(1-p), x, x)
    if not np.isfinite(theta).all() or np.any(np.linalg.eigvalsh(information)[:, 0] < 1e-9):
        raise ValueError("Insufficient information for a finite logistic fit")
    return theta, p


def fit_bt(rows, *, methods=None, reference=None, position=True, contrasts=()):
    rows, methods = select_methods(rows, methods)
    pairs, x, nonreference, reference = bt_design(methods, reference, position)
    pair_index = {pair: i for i, pair in enumerate(pairs)}
    idx = np.array([pair_index[(r["method_a"], r["method_b"])] for r in rows])
    targets = np.array([OUTCOMES[r["winner"]] for r in rows])
    n = np.bincount(idx, minlength=len(pairs)).astype(float)
    y = np.bincount(idx, weights=targets, minlength=len(pairs))
    theta, probabilities = fit_logistic(n, y, x)
    theta, probabilities = theta[0], probabilities[0]
    vectors = {m: np.zeros(len(theta)) for m in methods}
    for i, m in enumerate(nonreference):
        vectors[m][i] = 1
    scores = {m: float(vectors[m] @ theta) for m in methods}
    groups = sorted({(r["language"], r["rule_id"]) for r in rows})
    group_index = {group: i for i, group in enumerate(groups)}
    group_scores = np.zeros((len(groups), len(theta)))
    for row, j, target in zip(rows, idx, targets):
        group_scores[group_index[(row["language"], row["rule_id"])]] += x[j]*(target-probabilities[j])
    covariance = None
    if len(groups) > 1 and len(rows) > len(theta):
        info = x.T @ ((n*probabilities*(1-probabilities))[:, None]*x)
        inverse = np.linalg.inv(info)
        cr1 = len(groups)/(len(groups)-1)*(len(rows)-1)/(len(rows)-len(theta))
        covariance = inverse @ (group_scores.T @ group_scores) @ inverse * cr1

    def estimate(vector):
        value = float(vector @ theta)
        if covariance is None:
            return {"estimate": value, "cluster_se": None, "ci95": None}
        se = float(np.sqrt(max(0, vector @ covariance @ vector)))
        return {"estimate": value, "cluster_se": se, "ci95": [value-1.95996398454*se, value+1.95996398454*se]}

    comparison_results = []
    for a, b in contrasts:
        if a not in methods or b not in methods or a == b:
            raise ValueError(f"Invalid contrast: {a}, {b}")
        detail = estimate(vectors[a]-vectors[b])
        se, delta = detail["cluster_se"], detail["estimate"]
        pvalue = None if se is None else (float(2*norm.sf(abs(delta/se))) if se > 0 else (1.0 if delta == 0 else 0.0))
        comparison_results.append({"method_a": a, "method_b": b, **detail,
                                   "win_rate": float(expit(delta)),
                                   "win_rate_ci95": (expit(detail["ci95"]).tolist()
                                                     if detail["ci95"] is not None else None),
                                   "p_value": pvalue})
    if comparison_results and all(c["p_value"] is not None for c in comparison_results):
        order = sorted(range(len(comparison_results)), key=lambda i: comparison_results[i]["p_value"])
        adjusted = 0.0
        for rank, i in enumerate(order):
            adjusted = max(adjusted, min(1, (len(order)-rank)*comparison_results[i]["p_value"]))
            comparison_results[i]["holm_p_value"] = adjusted
    position_vector = np.zeros(len(theta))
    if position:
        position_vector[-1] = 1
    return {
        "comparisons": len(rows), "original_rule_clusters": len(groups),
        "reference_method": reference, "position_adjusted": position,
        "tie_value": 0.5, "scores": {m: estimate(vectors[m]) for m in methods},
        "ranking": sorted(methods, key=lambda m: -scores[m]),
        "position_effect": estimate(position_vector) if position else None,
        "win_rates_without_position_effect": {a: {b: float(expit(scores[a]-scores[b])) for b in methods} for a in methods},
        "contrasts": comparison_results,
        "uncertainty": "CR1 standard errors clustered by (language, rule_id); Wald 95% intervals on the score scale, sigmoid-transformed for win rates",
    }


def write_result(value, output=None):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    if output:
        Path(output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparisons", help="JSONL or JSON array; - reads stdin")
    parser.add_argument("--methods", nargs="+", help="Restrict to these method identifiers")
    parser.add_argument("--reference", help="Method fixed at zero; defaults to direct if present")
    parser.add_argument("--no-position", action="store_true")
    parser.add_argument("--by-setting", action="store_true", help="Also fit each language/input-form setting")
    parser.add_argument("--contrast", nargs=2, action="append", default=[], metavar=("METHOD_A", "METHOD_B"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        rows, methods = select_methods(load_comparisons(args.comparisons), args.methods)
        options = dict(methods=methods, reference=args.reference, position=not args.no_position, contrasts=args.contrast)
        report = {"overall": fit_bt(rows, **options)}
        if args.by_setting:
            settings = defaultdict(list)
            for row in rows:
                settings[(row["language"], row["input_form"])].append(row)
            report["by_setting"] = []
            for (lang, form), group in sorted(settings.items()):
                try:
                    result = fit_bt(group, **options)
                except ValueError as exc:
                    raise ValueError(f"Setting {lang}/{form}: {exc}") from exc
                report["by_setting"].append({"language": lang, "input_form": form, **result})
        write_result(report, args.output)
    except (ValueError, OSError, np.linalg.LinAlgError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
