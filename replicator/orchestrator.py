"""Deterministic orchestrator (DESIGN.md §0.3, §3.7). Stages: spec -> setup -> build+run -> verify -> report.
Owns budgets, the freeze, the stop, the run log, and the always-produced report.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from . import __version__, report as report_mod
from .build.sandbox import Sandbox
from .data.adapters import ADAPTERS
from .data import checkpoint as ckpt
from .kb import family_kb
from .llm import LLM
from .schema import (ClaimResult, GridPoint, LeakageResult, Outcome, Report, RunRecord, Spec, stable_hash)
from .spec import extract as spec_extract, intake
from .templates import cs_eval
from .triage import triage
from .verify import compare, grid, leakage
from .verify.tolerance import derived_tolerance

TEMPLATE_DIR = Path(__file__).parent / "templates"


def _claim_value(metrics: dict, c) -> float | None:
    """Value a claim is compared against: metrics[claim_id] (the builder contract), else
    metrics[metric], else a disambiguated key "<metric>_<token>" where the token comes from the
    claim id (e.g. claim c_dim_tetrahedron, metric feature_dimensionality ->
    feature_dimensionality_tetrahedron). Several claims sharing one metric name is the common case
    the plain lookup gets wrong."""
    if c.id in metrics:
        return metrics[c.id]
    tokens = [t for t in c.id.lower().replace("-", "_").split("_") if len(t) > 3 and t not in ("claim", "table", "figure")]
    for k in metrics:
        kl = k.lower()
        if kl.startswith(c.metric.lower() + "_") and any(t in kl for t in tokens):
            return metrics[k]
    if c.metric in metrics:
        return metrics[c.metric]
    return None


class Orchestrator:
    def __init__(self, run_root: Path, batch: bool = False, human_approve: Optional[Callable[[Spec, list[str]], bool]] = None):
        self.root = run_root
        self.root.mkdir(parents=True, exist_ok=True)
        self.runs_dir = self.root / "runs"
        self.work = self.root / "work"
        self.batch = batch
        self.human_approve = human_approve
        self.human_minutes = 0.0
        self.t0 = time.time()
        self.stages_done: list[str] = []
        self.llm: Optional[LLM] = None
        self._sandbox: Optional[Sandbox] = None
        self.log_path = self.root / "orchestrator.log"

    # ------------------------------------------------------------------ utils
    def log(self, msg: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
        with self.log_path.open("a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    def _llm(self, max_cost: float) -> LLM:
        if self.llm is None:
            self.llm = LLM(self.root / "llm", max_cost_usd=max_cost)
        return self.llm

    def elapsed_min(self) -> float:
        return (time.time() - self.t0) / 60

    # ------------------------------------------------------------------ stage 1: spec
    def stage_spec(self, pdf: Path, track: str, family: str, hint: str = "", osap_acronym: str | None = None,
                   user_data: dict[str, str] | None = None, reported_gpu_hours: float | None = None) -> Spec:
        pages = self.root / "pages"
        meta = intake.render(pdf, pages)
        self.log(f"intake: {meta['n_pages']} pages, companions={meta['companions']}, arxiv={meta['arxiv_version']}")
        llm = self._llm(150.0)
        spec = spec_extract.extract_spec(llm, pdf, track, family, hint)
        spec = spec_extract.referee_spec(llm, pdf, spec)
        spec = spec_extract.apply_kb_defaults(spec, osap_acronym)
        problems = spec_extract.reread_claims(llm, spec, pages)
        if problems:
            self.log(f"claim re-read disagreements: {problems}")
            (self.root / "claim_reread_problems.json").write_text(json.dumps(problems, indent=1))
            if not self.batch:
                raise RuntimeError(f"claim values disagree with page images: {problems}")
        return self.stage_triage(spec, user_data=user_data, reported_gpu_hours=reported_gpu_hours)

    def stage_triage(self, spec: Spec, user_data: dict[str, str] | None = None, reported_gpu_hours: float | None = None) -> Spec:
        note = triage(spec, reported_gpu_hours=reported_gpu_hours, user_data=user_data)
        for l in note.lines:
            self.log(f"triage: {l}")
        spec.save(self.root / "spec.yaml")
        # the one human checkpoint (auto-approve in batch)
        if not self.batch and self.human_approve is not None:
            t = time.time()
            ok = self.human_approve(spec, note.lines)
            self.human_minutes += (time.time() - t) / 60
            if not ok:
                raise RuntimeError("stopped at checkpoint (dry run)")
        self.stages_done.append("spec")
        return spec

    # ------------------------------------------------------------------ stage 2: setup
    def stage_setup(self, spec: Spec) -> dict[str, Any]:
        spec.assert_frozen()
        self.work.mkdir(parents=True, exist_ok=True)
        self.log(self.sandbox.banner)  # which execution boundary this run gets, in the run log, every run
        fam = family_kb(spec.paper.family.value)
        tmpl = fam["template"].split(".")[-1] + ".py"
        shutil.copy(TEMPLATE_DIR / tmpl, self.work / f"template_{tmpl}")
        (self.work / "README.md").write_text(
            f"Family template: template_{tmpl}. Write pipeline.py (finance) or reproduce.sh (CS).\n"
            f"Config keys available: {[a.config_key for a in spec.ambiguities]}\n")
        lock = subprocess.run(["uv", "lock", "--check"], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
        lock_hash = None
        lp = Path(__file__).resolve().parents[1] / "uv.lock"
        if lp.exists():
            lock_hash = hashlib.sha256(lp.read_bytes()).hexdigest()[:16]
        info = {"env_lock_hash": lock_hash, "python": sys.version.split()[0], "uv_lock_ok": lock.returncode == 0}
        # data probes already ran in triage; descriptive-stats checkpoint runs here when the pipeline exposes it
        (self.work / "setup.json").write_text(json.dumps(info, indent=1))
        self.stages_done.append("setup")
        return info

    # ------------------------------------------------------------------ pipeline adapters
    @property
    def sandbox(self) -> Sandbox:
        """The one execution boundary for model-authored code. `pipeline.py` is never imported into
        this process: every call crosses as JSON/parquet through `Sandbox.call`."""
        if self._sandbox is None:
            self._sandbox = Sandbox(self.work, log_path=self.root / "sandbox_log.jsonl")
        return self._sandbox

    def _require_pipeline(self) -> None:
        if not (self.work / "pipeline.py").exists():
            raise RuntimeError("pipeline.py not found in work dir")

    def run_fn_for(self, spec: Spec) -> Callable[[dict, int], dict]:
        if spec.paper.track.value == "finance":
            self._require_pipeline()
            timeout = spec.plan.budget.run_timeout_s
            return lambda cfg, seed: self.sandbox.call("pipeline", "run", {}, args=[cfg, seed], timeout_s=timeout)
        def _cs(cfg, seed):
            r = cs_eval.run_reproduce(self.work, cfg, seed, timeout_s=spec.plan.budget.run_timeout_s, sandbox=self.sandbox)
            return {**r["metrics"], "_shuffled": bool(r.get("shuffled")), "_shuffle_na": bool(r.get("shuffle_not_applicable")),
                    "_intermediates": {"n_examples": r.get("n_examples"), "split": r.get("split"), "seed": r.get("seed"),
                                       "matched_scale": bool(r.get("matched_scale")), "scale": r.get("scale")}}
        return _cs

    def smoke(self, spec: Spec, probe_scale: bool = True) -> dict[str, Any]:
        """Contract + invariants + a tiny run. Passing this is what ends the build stage.
        probe_scale=False skips the timed 10%-scale probe (re-verify of an already-run work dir)."""
        fam = family_kb(spec.paper.family.value)
        cfg = {**spec.default_config(), **fam.get("smoke", {}), "_smoke": True}
        out: dict[str, Any] = {"passed": False, "problems": []}
        try:
            if spec.paper.track.value == "finance":
                self._require_pipeline()
                timeout = min(600, spec.plan.budget.run_timeout_s)
                present = self.sandbox.call("_runner", "has_fns", {"module": "pipeline", "fns": ["run", "raw_data", "build_features"]},
                                            timeout_s=timeout)
                for fn in ("run", "raw_data", "build_features"):
                    if not present.get(fn):
                        out["problems"].append(f"pipeline.py missing {fn}()")
                if out["problems"]:
                    return out
                if "years" in fam.get("smoke", {}) and spec.claims and spec.claims[0].sample_start:
                    y0 = int(str(spec.claims[0].sample_start)[:4])
                    cfg["sample_start"], cfg["sample_end"] = f"{y0}-01", f"{y0 + fam['smoke']['years']}-12"
                metrics = self.sandbox.call("pipeline", "run", {}, args=[cfg, 0], timeout_s=timeout)
                out["metrics"] = {k: v for k, v in metrics.items() if not k.startswith("_")}
                need = {c.metric for c in spec.claims if c.id in spec.plan.target_claims}
                missing = [m for m in need if m not in metrics]
                if missing:
                    out["problems"].append(f"metrics missing for claims: {missing}")
                for k, v in metrics.items():
                    if isinstance(v, float) and np.isnan(v):
                        out["problems"].append(f"metric {k} is NaN")
                if metrics.get("n_periods", 1) < 12:
                    out["problems"].append("fewer than 12 periods in smoke run")
            else:
                if not (self.work / "reproduce.sh").exists():
                    out["problems"].append("reproduce.sh missing")
                    return out
                r = cs_eval.run_reproduce(self.work, cfg, 0, timeout_s=min(600, spec.plan.budget.run_timeout_s), sandbox=self.sandbox)
                out["metrics"] = r["metrics"]
                if probe_scale:
                    self._scale_probe(spec, out)
                    self._verifier_preconditions(spec, cfg, r, out)
                else:
                    out["scale_probe"] = "skipped (re-verify)"
                targets = [c for c in spec.claims if c.id in spec.plan.target_claims]
                missing = [c.id for c in targets if _claim_value(r["metrics"], c) is None]
                if missing:
                    out["problems"].append(f"metrics.json must contain a value for each target claim, keyed by claim id: missing {missing}")
                for k in ("seed", "split", "n_examples"):
                    if k not in r:
                        out["problems"].append(f"metrics.json missing field {k}")
        except Exception as e:  # noqa: BLE001
            out["problems"].append(f"{type(e).__name__}: {e}")
            out["traceback"] = traceback.format_exc()[-4000:]
        out["passed"] = not out["problems"]
        return out


    def _scale_probe(self, spec: Spec, out: dict[str, Any]) -> None:
        """Progressive scaling (design stage 7): time a SCALE=0.1 run and extrapolate the full run.
        Rejects a build whose full run would not fit the per-run timeout."""
        t = time.time()
        try:
            cs_eval.run_reproduce(self.work, {**spec.default_config(), "_scale": 0.1}, 0, sandbox=self.sandbox,
                                  timeout_s=min(600, spec.plan.budget.run_timeout_s))
            t10 = time.time() - t
            est = t10 * 10
            out["scale_probe"] = {"scale_0.1_seconds": round(t10), "estimated_full_seconds": round(est),
                                  "run_timeout_s": spec.plan.budget.run_timeout_s}
            if est > 0.9 * spec.plan.budget.run_timeout_s:
                out["problems"].append(f"a SCALE=0.1 run took {t10:.0f}s, so the full run would take ~{est/60:.0f} min > "
                                       f"timeout {spec.plan.budget.run_timeout_s/60:.0f} min: reduce the default scale "
                                       f"(steps/epochs/data/model) until SCALE=0.1 finishes in under {0.09*spec.plan.budget.run_timeout_s:.0f}s")
        except Exception as e:  # noqa: BLE001
            out["problems"].append(f"SCALE=0.1 probe failed or exceeded 10 min: {str(e)[:300]}. reproduce.sh must honor SCALE.")

    def _verifier_preconditions(self, spec: Spec, cfg: dict, r: dict, out: dict[str, Any]) -> None:
        """Everything the verify stage will assume about the generated code is checked here, at
        build time, where a failure costs one builder turn instead of a wrong verdict:
        (1) SCALE is honored (metrics.json echoes it), (2) the shuffle flag is honored and moves
        the headline, or the script declares shuffling not applicable, (3) a null reference for
        the headline metric is reported so the shuffle test can be judged."""
        head = next((c for c in spec.claims if c.priority == "headline"), spec.claims[0] if spec.claims else None)
        if head is None:
            return
        hm = head.metric
        if str(r.get("scale", "")) not in ("1", "1.0", "0.1"):
            out["problems"].append('metrics.json must echo "scale": <SCALE> so the orchestrator can confirm SCALE is honored')
        if r.get("shuffle_not_applicable"):
            return
        if hm not in r["metrics"]:
            return  # already reported as a missing metric
        try:
            rs = cs_eval.run_reproduce(self.work, {**cfg, "_shuffle_labels": True}, 0, sandbox=self.sandbox,
                                       timeout_s=min(600, spec.plan.budget.run_timeout_s))
        except Exception as e:  # noqa: BLE001
            out["problems"].append(f"shuffled SMOKE run failed: {str(e)[:200]}")
            return
        if not rs.get("shuffled"):
            out["problems"].append('with REPLICATOR_CONFIG._shuffle_labels=true the script must shuffle labels/targets and write "shuffled": true '
                                   '(or "shuffle_not_applicable": true with a reason if the method has no labels or rewards)')
        from .verify.leakage import null_reference, metric_known
        if null_reference(rs["metrics"], hm) is None and null_reference(r["metrics"], hm) is None and "t_stat" not in r["metrics"]:
            out["problems"].append(f'report a no-skill reference for the headline metric "{hm}": "chance_level", "{hm}_random_policy", '
                                   f'or "{hm}_baseline", in the same units, so the shuffle test can be judged')
        if not metric_known(hm) and null_reference(r["metrics"], hm) is None:
            out["problems"].append(f'headline metric "{hm}" has no known better-direction; either report a null reference for it or '
                                   f'name it with a standard suffix (accuracy, error, loss, return, bleu, f1, ...)')
        if rs.get("shuffled") and hm in rs["metrics"] and r["metrics"].get(hm) is not None:
            if abs(float(rs["metrics"][hm]) - float(r["metrics"][hm])) < 1e-9:
                out["problems"].append(f'the shuffled run reproduced the exact unshuffled "{hm}"; the shuffle is not reaching the training labels')
        out["shuffle_precheck"] = {"unshuffled": r["metrics"].get(hm), "shuffled": rs["metrics"].get(hm),
                                   "null": null_reference(rs["metrics"], hm)}

    # ------------------------------------------------------------------ stage 3: build
    def stage_build(self, spec: Spec, prebuilt: Path | None = None) -> dict[str, Any]:
        spec.assert_frozen()
        if prebuilt is not None:
            if prebuilt.resolve() != self.work.resolve():
                for f in prebuilt.iterdir():
                    if f.is_file():
                        shutil.copy(f, self.work / f.name)
            sm = self.smoke(spec, probe_scale=prebuilt.resolve() != self.work.resolve())
            res = {"smoke_passed": sm.get("passed"), "turns": 0, "prebuilt": str(prebuilt), "blacklist_hits": [], "last_smoke": sm}
        else:
            from .build.builder import Builder
            llm = self._llm(spec.plan.budget.max_cost_usd)
            b = Builder(llm, spec, self.work, lambda: self.smoke(spec), spec.plan.budget.build, sandbox=self.sandbox)
            res = b.run()
        if res.get("blacklist_hits") and spec.plan.kind_of_test == "re_implementation":
            spec.plan.kind_of_test = "reproduction"
            self.log(f"author-code blacklist hit -> kind_of_test downgraded to reproduction: {res['blacklist_hits'][:3]}")
        (self.work / "build_result.json").write_text(json.dumps(res, indent=1, default=str))
        self.stages_done.append("build")
        if not res.get("smoke_passed"):
            raise RuntimeError(f"build did not pass smoke: {res.get('last_smoke', res)}")
        return res

    # ------------------------------------------------------------------ stage 3b: run
    def stage_run(self, spec: Spec, seeds: list[int] | None = None) -> list[RunRecord]:
        spec.assert_frozen()
        run_fn = self.run_fn_for(spec)
        recs: list[RunRecord] = []
        seeds = seeds or (spec.plan.seeds if spec.paper.track.value == "cs" else [0])
        # Headline variants first; at most three distinct procedures per paper. At reduced scale
        # extra variants are usually the same run relabeled, and each costs a full timeout.
        head_v = [c.method_variant for c in spec.claims if c.priority == "headline" and c.id in spec.plan.target_claims]
        rest_v = [c.method_variant for c in spec.claims if c.id in spec.plan.target_claims]
        variants = list(dict.fromkeys(head_v + rest_v))[:3] or ["default"]
        for variant in variants:
            cfg = spec.config_for_variant(variant)
            cfg["_variant"] = variant
            for seed in seeds:
                rid = f"full_{variant}_s{seed}_{stable_hash(cfg)}"
                t = time.time()
                rec = RunRecord(run_id=rid, purpose="full", config=cfg, seed=seed, started_at=datetime.now(timezone.utc).isoformat())
                try:
                    try:
                        m = run_fn(cfg, seed)
                    except RuntimeError as e:
                        if "timed out" not in str(e) or spec.paper.track.value != "cs":
                            raise
                        self.log(f"run {rid} timed out; retrying once at SCALE=0.3")
                        cfg = {**cfg, "_scale": 0.3}
                        m = run_fn(cfg, seed)
                        rec.config = cfg
                    rec.metrics = {k: float(v) for k, v in m.items() if isinstance(v, (int, float)) and not k.startswith("_")}
                    rec.intermediates = m.get("_intermediates", {}) if isinstance(m.get("_intermediates"), dict) else {}
                    if m.get("_shuffle_na"):
                        rec.intermediates["shuffle_not_applicable"] = True
                    rec.data_hashes = m.get("_data_hashes", {}) if isinstance(m.get("_data_hashes"), dict) else {}
                except Exception as e:  # noqa: BLE001
                    rec.error = f"{type(e).__name__}: {e}"
                    self.log(f"run {rid} failed: {rec.error}")
                rec.wall_seconds = time.time() - t
                rec.save(self.runs_dir)
                recs.append(rec)
        self.stages_done.append("run")
        return recs

    # ------------------------------------------------------------------ stage 4: verify
    def stage_verify(self, spec: Spec, recs: list[RunRecord], report: Report, do_grid: bool = True) -> Report:
        spec.assert_frozen()
        run_fn = self.run_fn_for(spec)
        is_cs = spec.paper.track.value == "cs"
        by_variant: dict[str, list[RunRecord]] = {}
        for r in recs:
            if not r.error:
                by_variant.setdefault(r.config.get("_variant", "default"), []).append(r)
        sub_note = "; ".join(spec.plan.substitutions.values())
        # ---- claims
        for c in spec.claims:
            if c.id not in spec.plan.target_claims:
                report.claims.append(ClaimResult(claim_id=c.id, paper_value=c.value, outcome=Outcome.untested, note="not targeted in plan"))
                continue
            rs = by_variant.get(c.method_variant) or by_variant.get("default") or []
            vals = [v for r in rs for v in [_claim_value(r.metrics, c)] if v is not None]
            if not vals:
                report.claims.append(compare.compare_claim(spec, c, None))
                continue
            ours = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else (rs[0].metrics.get(f"{c.metric}_std") if rs else None)
            # tolerance fill from our own series when the analytic rule had nothing (recorded, not edited into the spec)
            tol_override = None
            if c.id not in spec.plan.tolerances:
                fallback_se = std / np.sqrt(len(vals)) if (std is not None and len(vals) > 1) else None
                if fallback_se is None and rs and rs[0].metrics.get("t_stat") and c.metric in ("mean_return", "alpha", "alpha_ff3", "spread", "premium"):
                    fallback_se = abs(ours) / abs(rs[0].metrics["t_stat"])
                tol_override = derived_tolerance(c, fallback_se)
                if tol_override is not None:
                    report.deviations.append(f"tolerance for {c.id} filled from our own SE ({fallback_se:.3g}) at verify time, not frozen")
            sig = None
            if rs and "t_stat" in rs[0].metrics:
                sig = abs(rs[0].metrics["t_stat"]) > 2
            cr = compare.compare_claim(spec, c, ours, std, substitution_note=sub_note, significant=sig,
                                       tolerance_override=tol_override)
            # Compute tier 2: our run is a reduced-scale stand-in. A paper-scale number is not
            # comparable unless the paper reports the same configuration at our scale (the run
            # says so via _intermediates.matched_scale). Report the value, do not call it a match
            # or a mismatch.
            matched = any((r.intermediates or {}).get("matched_scale") for r in rs)
            if spec.plan.compute_tier == 2 and not matched and cr.outcome in (Outcome.match, Outcome.mismatch):
                cr.outcome = Outcome.untested
                cr.note = f"reduced scale (compute tier 2): ours={ours:.4g} is not comparable to the paper-scale value; " + cr.note
            report.claims.append(cr)
        # ---- leakage
        headline = next((c for c in spec.claims if c.priority == "headline"), spec.claims[0] if spec.claims else None)
        base_cfg = spec.config_for_variant(headline.method_variant if headline else "default")
        base_val = None
        if headline:
            rs = by_variant.get(headline.method_variant) or by_variant.get("default") or []
            v = [r.metrics.get(headline.metric) for r in rs if headline.metric in r.metrics]
            base_val = float(np.mean(v)) if v else None
        na = any((r.metrics or {}).get("_shuffle_na") or (r.intermediates or {}).get("shuffle_not_applicable") for r in recs)
        # Decide judgeability BEFORE spending a full run on the shuffled pipeline: a t-stat, a
        # reported null reference, or a metric with a known direction is required.
        base_metrics = next((r.metrics for r in recs if headline and headline.metric in r.metrics), {}) if headline else {}
        judgeable = bool(base_metrics) and ("t_stat" in base_metrics or leakage.null_reference(base_metrics, headline.metric) is not None
                                             or leakage.metric_known(headline.metric))
        if headline and base_val is not None and not na and judgeable:
            report.leakage.append(leakage.shuffle_test(run_fn, base_cfg, headline.metric, base_val))
        elif headline and base_val is not None and not na:
            report.leakage.append(LeakageResult(test="shuffle", passed=None,
                                                detail=f"not judged: headline metric '{headline.metric}' has no t-stat, no reported null reference "
                                                       f"(chance_level / {headline.metric}_random_policy / _baseline), and no known better-direction; shuffled run skipped"))
        elif na:
            report.leakage.append(LeakageResult(test="shuffle", passed=None, detail="not applicable: the pipeline declares no label/reward to shuffle (declared in metrics.json)"))
        if not is_cs:
            try:
                self._require_pipeline()
                timeout = spec.plan.budget.run_timeout_s
                raw = self.sandbox.call("pipeline", "raw_data", {}, timeout_s=timeout)
                build_features = lambda df: self.sandbox.call("pipeline", "build_features", {}, args=[df], timeout_s=timeout)  # noqa: E731
                report.leakage.append(leakage.future_perturbation_test(build_features, raw))
                present = self.sandbox.call("_runner", "has_fns", {"module": "pipeline", "fns": ["pit_audit"]}, timeout_s=timeout)
                if present.get("pit_audit"):
                    ok, n = self.sandbox.call("pipeline", "pit_audit", {}, timeout_s=timeout)
                    report.leakage.append(LeakageResult(test="available_at_audit", passed=ok, detail=f"{n} violations"))
                else:
                    report.leakage.append(LeakageResult(test="available_at_audit", passed=None, detail="pipeline exposes no pit_audit()"))
            except Exception as e:  # noqa: BLE001
                report.leakage.append(LeakageResult(test="future_perturbation", passed=None, detail=f"could not run: {e}"))
        else:
            report.leakage.append(LeakageResult(test="contamination_scan", passed=None, detail="not implemented in MVP: hash-based train/test near-duplicate scan"))
        # ---- descriptive-stats checkpoint from intermediates
        inter = {}
        for r in recs:
            inter.update(r.intermediates or {})
        if spec.descriptive_stats:
            ours = {k: float(v) for k, v in inter.items() if isinstance(v, (int, float))}
            res = ckpt.compare(spec.descriptive_stats, ours, spec.plan.data_tier, family_kb(spec.paper.family.value)["checkpoint"].get("n_firms_rel_tol", 0.25))
            report.deviations.append(f"Table-1 checkpoint ({res.mode}): " + "; ".join(
                f"{r['metric']} paper={r['paper']} ours={r['ours']} gap={r['rel_gap']}" for r in res.rows))
            if res.mode == "gate" and not res.passed:
                report.unexplained.append("Tier A data checkpoint failed: data differs from the paper's Table 1 before modeling")
        # ---- convention grid (one parallel batch; sensitivity, not search)
        if do_grid and headline and base_val is not None:
            t = time.time()
            pts, rng = grid.run_grid(run_fn, base_cfg, spec.ambiguities, headline.metric, base_val,
                                     max_workers=1 if is_cs else 4)
            report.grid, report.grid_range = pts, rng
            self.log(f"grid: {len(pts)} flips in {time.time() - t:.1f}s, range={rng}")
            for cr in report.claims:
                if cr.outcome == Outcome.mismatch and cr.claim_id == headline.id and rng and rng[0] <= headline.value <= rng[1]:
                    best = min((p for p in pts if p.headline_value is not None), key=lambda p: abs(p.headline_value - headline.value))
                    cr.note += (f"; paper value lies inside the convention range; closest flip {best.config_key}={best.value} "
                                f"({best.headline_value:.4g}) has no independent support in the paper -> candidate, not attribution")
                    report.unexplained.append(f"{cr.claim_id}: mismatch under the default conventions; inside the grid range")
        for cr in report.claims:
            if cr.outcome == Outcome.mismatch and not any(s_.id == cr.claim_id and s_.priority == "headline" for s_ in spec.claims):
                report.unexplained.append(f"{cr.claim_id} (secondary): {cr.note}")
        report.ambiguities_used = [{"config_key": a.config_key, "default": a.default, "source": a.source, "sensitivity": a.sensitivity} for a in spec.ambiguities]
        self.stages_done.append("verify")
        return report

    # ------------------------------------------------------------------ stage 5: report (always)
    def stage_report(self, spec: Spec, report: Report, failure: str | None = None) -> Path:
        report.failure = failure
        report.stages_completed = list(self.stages_done)
        report.wall_minutes = self.elapsed_min()
        report.human_minutes = self.human_minutes
        report.cost_usd = self.llm.cost_usd if self.llm else 0.0
        diag: dict[str, Any] = {}
        br = self.work / "build_result.json"
        if br.exists():
            b = json.loads(br.read_text())
            ls = b.get("last_smoke") or {}
            diag["build"] = {k: b.get(k) for k in ("smoke_passed", "turns", "timeouts", "seconds", "blacklist_hits")}
            diag["smoke_problems"] = ls.get("problems")
            diag["scale_probe"] = ls.get("scale_probe")
            diag["shuffle_precheck"] = ls.get("shuffle_precheck")
        if self.runs_dir.exists():
            errs = []
            for rp in sorted(self.runs_dir.glob("*.json")):
                r = json.loads(rp.read_text())
                if r.get("error"):
                    errs.append({"run": rp.stem[:48], "error": r["error"][:160]})
            diag["run_errors"] = errs
        report.reproducibility = {
            "diagnostics": diag,
            "agent_version": __version__, "spec_frozen_hash": spec.plan.frozen_hash, "frozen_at": spec.plan.frozen_at,
            "env_lock_hash": json.loads((self.work / "setup.json").read_text()).get("env_lock_hash") if (self.work / "setup.json").exists() else None,
            "runs": sorted(p.name for p in self.runs_dir.glob("*.json")) if self.runs_dir.exists() else [],
            "work_dir": str(self.work), "command": " ".join(sys.argv),
        }
        spec.save(self.root / "spec.yaml")
        p = report_mod.write(report, spec, self.root)
        self.stages_done.append("report")
        self.log(f"report written: {p}")
        return p

    # ------------------------------------------------------------------ end to end
    def replicate(self, spec: Spec, prebuilt: Path | None = None, do_grid: bool = True) -> Report:
        report = Report(paper_title=spec.paper.title, track=spec.paper.track.value, family=spec.paper.family.value,
                        kind_of_test=spec.plan.kind_of_test, data_tier=spec.plan.data_tier, compute_tier=spec.plan.compute_tier,
                        deviations=[f"{k} -> {v}" for k, v in spec.plan.substitutions.items()])
        failure = None
        try:
            self.stage_setup(spec)
            self.stage_build(spec, prebuilt=prebuilt)
            recs = self.stage_run(spec)
            self.stage_verify(spec, recs, report, do_grid=do_grid)
        except Exception as e:  # noqa: BLE001
            failure = f"{type(e).__name__}: {e}"
            self.log(f"FAILURE: {failure}\n{traceback.format_exc()[-3000:]}")
        finally:
            report.kind_of_test = spec.plan.kind_of_test
            self.stage_report(spec, report, failure)
        return report
