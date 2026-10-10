#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tabular_model_comparison.py
===========================
표형(tabular) 데이터용 예측모델 다중 비교 파이프라인
— KNIH–KSBi Healthcare AI Challenge 2026 「임상 및 유전체 데이터 기반 뇌 위축 바이오마커 예측과 MRI 검사 대상 선별」

무엇을 하나
-----------
* 같은 분할, 같은 전처리, 같은 튜닝 예산(모델당 동일 무작위 탐색 횟수)에서 여러 모델을 비교한다.
* 대상자 단위 중첩 교차검증: 외부 K분할 × R회 반복(OOF 예측) / 내부 CV로 하이퍼파라미터 탐색.
* 분류(위축 양성 여부)와 회귀(연간 위축 속도)를 --task 로 선택.
* 지표: 분류 AUROC 및 AUPRC, Brier, 보정 절편/기울기, 상위 k% 배정 시 민감도 / 회귀 MAE, RMSE, R².
* 모델 간 차이: 같은 대상자의 쌍체 부트스트랩 95% CI.
* 결과: results_summary.csv, pairwise_bootstrap.csv, oof_predictions.csv, summary.md

사용 예
-------
  # 더미 데이터로 동작 확인
  python tabular_model_comparison.py --demo --fast

  # 실제 자료 (CSV/Parquet; 한 행 = 대상자, 열 = 변수)
  python tabular_model_comparison.py --data dev.parquet --id-col T00_ID --target atrophy_pos \
      --task classification --features features_M1G.txt --categorical cats.txt \
      --outer-folds 5 --repeats 2 --inner-folds 3 --n-iter 40 --out results/M1G

  features 파일: 한 줄에 변수명 하나. categorical 파일: 범주형 변수명(한 줄 하나). --group-col 로 근연 집단 열 지정 가능.

주의
----
* 평가군(홀드아웃)은 이 스크립트에 넣지 않는다. 개발군에서만 비교, 선택하고, 선택된 주 모델만 평가군에 한 번 적용한다.
* TabPFN은 설치, 가중치가 있을 때만 실행된다(pip install tabpfn; 연구용 라이선스 확인). 없으면 건너뛴다.
* 사전 고정 선택 규칙: 분류는 개발 CV AUPRC 최고 모델, 차이 0.01 이내면 더 단순, 보정 좋은 모델. 회귀는 MAE 최소.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------------------------
# 옵션 라이브러리: 필요할 때만 불러온다
# 맥에서는 torch(TabPFN)와 LightGBM, XGBoost, CatBoost가 각자 OpenMP 런타임을 가지고 있어 한 프로세스에서
# 함께 계산하면 커널이 죽을 수 있다. 그래서 모듈을 불러올 때 아무것도 올리지 않고, 실제로 계산할 모델의
# 라이브러리만 그때 불러온다. TabPFN은 노트북 커널이 아닌 별도 프로세스(tabpfn_worker.py)에서 계산한다.
# ----------------------------------------------------------------------------------------------
_LIBS: dict = {}


def _lib(name: str):
    """(분류기, 회귀기) 클래스 쌍을 돌려준다. 설치되어 있지 않으면 None."""
    if name not in _LIBS:
        try:
            if name == "lgbm":
                from lightgbm import LGBMClassifier, LGBMRegressor
                _LIBS[name] = (LGBMClassifier, LGBMRegressor)
            elif name == "xgb":
                from xgboost import XGBClassifier, XGBRegressor
                _LIBS[name] = (XGBClassifier, XGBRegressor)
            elif name == "cat":
                from catboost import CatBoostClassifier, CatBoostRegressor
                _LIBS[name] = (CatBoostClassifier, CatBoostRegressor)
            elif name == "tabpfn":
                from tabpfn import TabPFNClassifier, TabPFNRegressor
                _LIBS[name] = (TabPFNClassifier, TabPFNRegressor)
            else:
                _LIBS[name] = None
        except Exception:
            _LIBS[name] = None
    return _LIBS[name]


LABELS = {"enet": "Elastic-net (기준선)", "rf": "Random Forest", "lgbm": "LightGBM", "xgb": "XGBoost",
          "cat": "CatBoost", "svm": "SVM-RBF", "tabpfn": "TabPFN"}


def tabpfn_version_info() -> dict:
    """설치된 tabpfn 패키지 버전과 사용 가능한 모델 버전 목록(이 함수는 torch를 불러오므로 노트북 커널에서는 쓰지 않는다)."""
    if _lib("tabpfn") is None:
        return {"installed": False}
    import tabpfn
    try:
        from tabpfn.constants import ModelVersion
        versions = [v.value for v in ModelVersion]
    except Exception:
        versions = []
    return {"installed": True, "package": getattr(tabpfn, "__version__", "?"), "model_versions": versions}


# ----------------------------------------------------------------------------------------------
# 설정
# ----------------------------------------------------------------------------------------------
@dataclass
class Config:
    task: str = "classification"          # classification | regression
    outer_folds: int = 5
    repeats: int = 2
    inner_folds: int = 3
    n_iter: int = 40                      # 모델당 동일한 무작위 탐색 횟수(예산 통제)
    seed: int = 20260915
    n_jobs: int = -1
    referral_fracs: tuple = (0.10, 0.20, 0.30)  # MRI 배정률 시나리오
    n_boot: int = 1000
    models: list = field(default_factory=lambda: ["enet", "rf", "lgbm", "xgb", "cat", "tabpfn"])
    selection_tol: float = 0.01           # 사전 고정 선택 규칙의 허용 오차
    tabpfn_n_estimators: int = 8          # 패키지 기본값(DEFAULT_N_ESTIMATORS=8)과 같음
    tabpfn_version: str = "v3.5"          # TabPFN-3.5 (tabpfn 9.0.0 이상 필요; 로컬 추론만 사용)
    tabpfn_device: str = "auto"           # auto(애플 GPU가 있으면 MPS), cpu, mps


# ----------------------------------------------------------------------------------------------
# 전처리
# ----------------------------------------------------------------------------------------------
def make_preprocessor(num_cols: list, cat_cols: list, scale: bool, impute: bool) -> ColumnTransformer:
    """트리, GBDT: 결측을 그대로 두고(impute=False) 표준화 없이. 선형, SVM, TabPFN: 대치+지시자+표준화."""
    num_steps = []
    if impute:
        num_steps.append(("imp", SimpleImputer(strategy="median", add_indicator=True)))
    if scale:
        num_steps.append(("sc", StandardScaler()))
    num_pipe = Pipeline(num_steps) if num_steps else "passthrough"
    transformers = [("num", num_pipe, num_cols)]
    if cat_cols:
        cat_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


# ----------------------------------------------------------------------------------------------
# 모델 팩토리: (이름, 추정기, 파라미터 분포, 전처리 옵션)
# ----------------------------------------------------------------------------------------------
def model_zoo(cfg: Config, num_cols: list, cat_cols: list) -> dict:
    clf = cfg.task == "classification"
    zoo = {}
    want = set(cfg.models)

    # Elastic-net 로지스틱 / 선형 회귀 — 해석 가능한 기준선
    if "enet" in want:
        if clf:
            est = LogisticRegression(penalty="elasticnet", solver="saga", max_iter=5000, random_state=cfg.seed)
            params = {"est__C": loguniform(1e-3, 1e2), "est__l1_ratio": uniform(0, 1)}
        else:
            est = ElasticNet(max_iter=20000, random_state=cfg.seed)
            params = {"est__alpha": loguniform(1e-4, 1e1), "est__l1_ratio": uniform(0, 1)}
        zoo["enet"] = dict(label=LABELS["enet"], est=est, params=params, scale=True, impute=True)

    # Random Forest
    if "rf" in want:
        est = (RandomForestClassifier if clf else RandomForestRegressor)(n_estimators=500, n_jobs=cfg.n_jobs,
                                                                        random_state=cfg.seed)
        params = {"est__max_depth": [None, 4, 6, 8, 12, 16], "est__min_samples_leaf": randint(1, 30),
                  "est__max_features": ["sqrt", "log2", 0.3, 0.5, 0.8]}
        zoo["rf"] = dict(label=LABELS["rf"], est=est, params=params, scale=False, impute=True)

    # LightGBM
    if "lgbm" in want and _lib("lgbm"):
        LGBMClassifier, LGBMRegressor = _lib("lgbm")
        est = (LGBMClassifier if clf else LGBMRegressor)(n_estimators=600, verbose=-1, n_jobs=1,
                                                         random_state=cfg.seed)
        params = {"est__learning_rate": loguniform(0.01, 0.2), "est__num_leaves": randint(7, 63),
                  "est__min_child_samples": randint(10, 100), "est__subsample": uniform(0.6, 0.4),
                  "est__subsample_freq": [1], "est__colsample_bytree": uniform(0.5, 0.5),
                  "est__reg_lambda": loguniform(1e-3, 10)}
        zoo["lgbm"] = dict(label=LABELS["lgbm"], est=est, params=params, scale=False, impute=False)

    # XGBoost
    if "xgb" in want and _lib("xgb"):
        XGBClassifier, XGBRegressor = _lib("xgb")
        kw = dict(n_estimators=600, tree_method="hist", n_jobs=1, random_state=cfg.seed)
        est = XGBClassifier(eval_metric="logloss", **kw) if clf else XGBRegressor(**kw)
        params = {"est__learning_rate": loguniform(0.01, 0.2), "est__max_depth": randint(2, 8),
                  "est__min_child_weight": loguniform(1, 30), "est__subsample": uniform(0.6, 0.4),
                  "est__colsample_bytree": uniform(0.5, 0.5), "est__reg_lambda": loguniform(1e-3, 10)}
        zoo["xgb"] = dict(label=LABELS["xgb"], est=est, params=params, scale=False, impute=False)

    # CatBoost (one-hot 통일; 원범주형 처리는 cat_features 옵션으로 확장 가능)
    if "cat" in want and _lib("cat"):
        CatBoostClassifier, CatBoostRegressor = _lib("cat")
        kw = dict(iterations=800, verbose=0, random_seed=cfg.seed, thread_count=1,
                  allow_writing_files=False)
        est = CatBoostClassifier(**kw) if clf else CatBoostRegressor(**kw)
        params = {"est__learning_rate": loguniform(0.01, 0.2), "est__depth": randint(3, 9),
                  "est__l2_leaf_reg": loguniform(1, 30), "est__bagging_temperature": uniform(0, 1)}
        zoo["cat"] = dict(label=LABELS["cat"], est=est, params=params, scale=False, impute=False)

    # SVM (2군)
    if "svm" in want:
        est = SVC(probability=True, random_state=cfg.seed) if clf else SVR()
        params = {"est__C": loguniform(1e-2, 1e2), "est__gamma": loguniform(1e-4, 1e0)}
        zoo["svm"] = dict(label=LABELS["svm"], est=est, params=params, scale=True, impute=True)

    # TabPFN (튜닝 없음; 설치, 라이선스 확인 필요). 모델 버전을 명시해 패키지 기본값 변화에 흔들리지 않게 한다.
    # ignore_pretraining_limits=True: v3 이상은 CPU에서 5,000행 초과 시 기본적으로 오류를 내므로(외부 훈련 폴드 6,400행) 해제한다.
    if "tabpfn" in want and _lib("tabpfn"):
        TabPFNClassifier, TabPFNRegressor = _lib("tabpfn")
        info = tabpfn_version_info()
        if cfg.tabpfn_version in info.get("model_versions", []):
            from tabpfn.constants import ModelVersion
            Cls = TabPFNClassifier if clf else TabPFNRegressor
            est = Cls.create_default_for_version(ModelVersion(cfg.tabpfn_version), n_estimators=cfg.tabpfn_n_estimators,
                                                 random_state=cfg.seed, ignore_pretraining_limits=True,
                                                 device=cfg.tabpfn_device)
            zoo["tabpfn"] = dict(label=f"TabPFN-{cfg.tabpfn_version.lstrip('v')}", est=est, params=None, scale=False, impute=True)
        else:
            print(f"[알림] 설치된 tabpfn {info.get('package')}에는 모델 버전 {cfg.tabpfn_version}이 없습니다 "
                  f"(사용 가능: {info.get('model_versions')}). TabPFN-3.5는 pip install -U \"tabpfn>=9.0.0\" 후 커널을 다시 시작하세요.")

    return zoo


def build_pipeline(spec: dict, num_cols: list, cat_cols: list) -> Pipeline:
    pre = make_preprocessor(num_cols, cat_cols, scale=spec["scale"], impute=spec["impute"])
    return Pipeline([("pre", pre), ("est", spec["est"])])


# ----------------------------------------------------------------------------------------------
# 대상자/근연 집단 단위 분할
# ----------------------------------------------------------------------------------------------
def outer_splits(y: np.ndarray, groups: np.ndarray | None, cfg: Config):
    """반복 r마다 다른 시드로 K분할. groups가 있으면 같은 집단은 같은 폴드."""
    for r in range(cfg.repeats):
        seed = cfg.seed + r
        if groups is None:
            if cfg.task == "classification":
                kf = StratifiedKFold(cfg.outer_folds, shuffle=True, random_state=seed)
                yield from ((r, k, tr, te) for k, (tr, te) in enumerate(kf.split(np.zeros(len(y)), y)))
            else:
                kf = KFold(cfg.outer_folds, shuffle=True, random_state=seed)
                yield from ((r, k, tr, te) for k, (tr, te) in enumerate(kf.split(np.zeros(len(y)))))
        else:
            rng = np.random.default_rng(seed)
            ug = np.unique(groups)
            rng.shuffle(ug)
            fold_of_group = {g: i % cfg.outer_folds for i, g in enumerate(ug)}
            f = np.array([fold_of_group[g] for g in groups])
            for k in range(cfg.outer_folds):
                yield r, k, np.where(f != k)[0], np.where(f == k)[0]


# ----------------------------------------------------------------------------------------------
# 지표
# ----------------------------------------------------------------------------------------------
def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def sensitivity_at_referral(y: np.ndarray, p: np.ndarray, frac: float) -> tuple:
    """상위 frac 비율에 MRI를 배정했을 때 민감도, PPV, 1명 발견당 MRI 건수."""
    n_sel = max(1, int(round(frac * len(y))))
    order = np.argsort(-p, kind="stable")
    sel = order[:n_sel]
    tp = int(y[sel].sum())
    sens = tp / max(1, int(y.sum()))
    ppv = tp / n_sel
    per_case = (n_sel / tp) if tp > 0 else np.nan
    return sens, ppv, per_case


def metrics_for(task: str, y: np.ndarray, p: np.ndarray, fracs) -> dict:
    if task == "classification":
        m = {"AUROC": roc_auc_score(y, p), "AUPRC": average_precision_score(y, p),
             "Brier": brier_score_loss(y, p), "prevalence": float(y.mean())}
        slope, intercept = calibration_slope_intercept(y, p)
        m["cal_slope"], m["cal_intercept"] = slope, intercept
        for f in fracs:
            s, ppv, pc = sensitivity_at_referral(y, p, f)
            m[f"sens@{int(f*100)}%"], m[f"PPV@{int(f*100)}%"], m[f"MRI_per_case@{int(f*100)}%"] = s, ppv, pc
        return m
    return {"MAE": mean_absolute_error(y, p), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": r2_score(y, p)}


def primary_metric(task: str) -> tuple:
    """(지표명, 클수록 좋은지)"""
    return ("AUPRC", True) if task == "classification" else ("MAE", False)


def paired_bootstrap(task: str, y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n_boot: int, seed: int,
                     metric: str) -> tuple:
    """같은 대상자에서 모델 A − B 의 지표 차이와 95% CI."""
    rng = np.random.default_rng(seed)
    n = len(y)
    fn = {"AUROC": roc_auc_score, "AUPRC": average_precision_score,
          "MAE": mean_absolute_error, "RMSE": lambda a, b: np.sqrt(mean_squared_error(a, b))}[metric]
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if task == "classification" and len(np.unique(y[idx])) < 2:
            continue
        diffs.append(fn(y[idx], p_a[idx]) - fn(y[idx], p_b[idx]))
    diffs = np.array(diffs)
    return float(fn(y, p_a) - fn(y, p_b)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# ----------------------------------------------------------------------------------------------
# 실행
# ----------------------------------------------------------------------------------------------
def run(df: pd.DataFrame, id_col: str, target: str, features: list, cat_cols: list, group_col: str | None,
        cfg: Config, out_dir: str, resume: bool = True, compute_only: list | None = None) -> pd.DataFrame:
    """compute_only: 주어지면 저장된 결과를 재사용할 수 없는 모델 중 이 목록에 있는 것만 계산하고 나머지는 빼고 진행한다
    (예: TabPFN 작업자 프로세스에서 GBDT 라이브러리를 불러오지 않기 위해 ["tabpfn"])."""
    os.makedirs(out_dir, exist_ok=True)
    num_cols = [c for c in features if c not in cat_cols]
    X = df[features].copy()
    for c in cat_cols:
        X[c] = X[c].astype("object")
    y = df[target].to_numpy()
    if cfg.task == "classification":
        y = y.astype(int)
    groups = df[group_col].to_numpy() if group_col else None
    ids = df[id_col].to_numpy()

    print(f"[설정] task={cfg.task}  n={len(y)}  p={len(features)}  models={list(cfg.models)}  "
          f"outer={cfg.outer_folds}x{cfg.repeats}  inner={cfg.inner_folds}  n_iter={cfg.n_iter}")

    scoring = "average_precision" if cfg.task == "classification" else "neg_mean_absolute_error"
    splits = list(outer_splits(y, groups, cfg))
    oof = {m: np.full((cfg.repeats, len(y)), np.nan) for m in cfg.models}
    labels = {m: LABELS.get(m, m) for m in cfg.models}
    best_params_log = []
    timing = {}

    # ---- 이어서 실행: 같은 대상자, 같은 분할 설정, 같은 변수로 이미 계산된 모델은 다시 계산하지 않는다
    reused = []
    cfg_path, oof_path = os.path.join(out_dir, "config.json"), os.path.join(out_dir, "oof_predictions.csv")
    if resume and os.path.exists(cfg_path) and os.path.exists(oof_path):
        old = json.load(open(cfg_path, encoding="utf-8"))
        same = all(old.get(k) == v for k, v in [("outer_folds", cfg.outer_folds), ("repeats", cfg.repeats), ("inner_folds", cfg.inner_folds),
                                                 ("n_iter", cfg.n_iter), ("seed", cfg.seed), ("n", int(len(y)))]) and old.get("features") == features
        prev = pd.read_csv(oof_path)
        if same and prev[id_col].astype(str).tolist() == pd.Series(ids).astype(str).tolist():
            prev_sum = pd.read_csv(os.path.join(out_dir, "results_summary.csv")) if os.path.exists(os.path.join(out_dir, "results_summary.csv")) else None
            prev_bp = pd.read_csv(os.path.join(out_dir, "best_params_by_fold.csv")) if os.path.exists(os.path.join(out_dir, "best_params_by_fold.csv")) else None
            for m in cfg.models:
                cols = [f"{m}_rep{r}" for r in range(cfg.repeats)]
                if all(c in prev.columns for c in cols) and prev[cols].notna().all().all():
                    oof[m][:] = prev[cols].to_numpy().T
                    timing[m] = float(prev_sum.set_index("model").loc[m, "train_seconds"]) if prev_sum is not None and m in set(prev_sum.model) else float("nan")
                    if prev_sum is not None and m in set(prev_sum.model) and "label" in prev_sum.columns:
                        labels[m] = str(prev_sum.set_index("model").loc[m, "label"])
                    if prev_bp is not None and "model" in prev_bp.columns:
                        best_params_log += prev_bp[prev_bp.model == m].to_dict("records")
                    reused.append(m)
            if reused:
                print(f"[이어서 실행] 기존 결과 재사용: {reused}")
        elif os.path.exists(oof_path):
            print("[이어서 실행] 기존 결과의 설정이나 대상자가 달라 모두 다시 계산합니다.")

    # 재사용하지 못한 모델만 모델 목록을 만들고(그 라이브러리만 불러옴) 계산한다
    to_compute = [m for m in cfg.models if m not in reused]
    if compute_only is not None:
        dropped = [m for m in to_compute if m not in compute_only]
        if dropped:
            print(f"[알림] 저장된 결과가 없어 이번 실행에서 뺌(compute_only 제한): {dropped}")
        to_compute = [m for m in to_compute if m in compute_only]
    zoo = model_zoo(Config(**{**cfg.__dict__, "models": to_compute}), num_cols, cat_cols) if to_compute else {}
    skipped = [m for m in to_compute if m not in zoo]
    if skipped:
        print(f"[알림] 사용 불가(미설치 또는 버전 없음)로 건너뜀: {skipped}")
    selected = [m for m in cfg.models if m in reused or m in zoo]
    for m in zoo:
        labels[m] = zoo[m]["label"]

    failed = []
    for m in selected:
        if m in reused:
            continue
        spec = zoo[m]
        t0 = time.time()
        try:
          for r, k, tr, te in splits:
            pipe = build_pipeline(spec, num_cols, cat_cols)
            if spec["params"] is None:          # TabPFN 등 튜닝 없는 모델
                model = pipe
            else:
                inner = (StratifiedKFold(cfg.inner_folds, shuffle=True, random_state=cfg.seed + r)
                         if cfg.task == "classification" else KFold(cfg.inner_folds, shuffle=True,
                                                                     random_state=cfg.seed + r))
                model = RandomizedSearchCV(pipe, spec["params"], n_iter=cfg.n_iter, cv=inner, scoring=scoring,
                                           random_state=cfg.seed + r, n_jobs=1 if m == "rf" else cfg.n_jobs,
                                           refit=True)
            model.fit(X.iloc[tr], y[tr])
            if spec["params"] is not None:
                best_params_log.append({"model": m, "repeat": r, "fold": k, **model.best_params_})
            if cfg.task == "classification":
                oof[m][r, te] = model.predict_proba(X.iloc[te])[:, 1]
            else:
                oof[m][r, te] = model.predict(X.iloc[te])
            print(f"  {spec['label']:<22} repeat {r} fold {k}  done ({time.time()-t0:6.0f}s)")
          timing[m] = time.time() - t0
        except Exception as e:                  # 한 모델의 실패가 전체 비교를 멈추지 않도록 한다
            failed.append(m)
            msg = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
            print(f"  [건너뜀] {spec['label']}: {msg}")
            if "License" in type(e).__name__ or "TABPFN_TOKEN" in str(e):
                print(f"      라이선스 동의는 모델 버전마다 따로 필요합니다. https://ux.priorlabs.ai 의 Licenses 탭에서 "
                      f"TabPFN-{cfg.tabpfn_version.lstrip('v')} 라이선스에 동의했는지, TABPFN_TOKEN 키가 같은 계정의 것인지 확인하세요. "
                      "키 없이 진행하려면 models 에서 tabpfn 을 빼면 됩니다.")
            elif "HuggingFace" in type(e).__name__ or "huggingface" in str(e).lower():
                print("      가중치 내려받기(huggingface.co/Prior-Labs) 단계에서 실패했습니다. 인터넷 연결이나 사내망 차단 여부를 확인하세요. "
                      "한 번 내려받으면 캐시에서 읽으므로 이후에는 연결이 필요 없습니다.")
    if failed:
        selected = [m for m in selected if m not in failed]
        for m in failed: oof.pop(m, None)
        if not selected:
            raise RuntimeError("실행 가능한 모델이 없습니다.")

    # ---- 성능 요약: 반복별 지표의 평균/SD + 반복 평균 예측으로 풀링 지표
    rows = []
    for m in selected:
        per_rep = [metrics_for(cfg.task, y, oof[m][r], cfg.referral_fracs) for r in range(cfg.repeats)]
        pooled = metrics_for(cfg.task, y, np.nanmean(oof[m], axis=0), cfg.referral_fracs)
        row = {"model": m, "label": labels[m], "train_seconds": round(timing.get(m, float("nan")), 1)}
        for key in per_rep[0]:
            vals = [d[key] for d in per_rep]
            row[f"{key}_mean"] = float(np.nanmean(vals))
            row[f"{key}_sd"] = float(np.nanstd(vals, ddof=1)) if cfg.repeats > 1 else np.nan
        for key, v in pooled.items():
            row[f"{key}_pooled"] = v
        rows.append(row)
    summary = pd.DataFrame(rows)
    pm, higher = primary_metric(cfg.task)
    summary = summary.sort_values(f"{pm}_mean", ascending=not higher).reset_index(drop=True)

    # ---- 사전 고정 선택 규칙
    best = summary.iloc[0]
    tol_ok = summary[(summary[f"{pm}_mean"] - best[f"{pm}_mean"]).abs() <= cfg.selection_tol]
    simplicity = {"enet": 0, "tabpfn": 1, "rf": 2, "lgbm": 3, "xgb": 3, "cat": 3, "svm": 4}
    if cfg.task == "classification":
        # 허용 오차 내 모델 중 단순성 → 보정 기울기가 1에 가까운 순
        tol_ok = tol_ok.assign(_simp=tol_ok["model"].map(simplicity),
                               _cal=(tol_ok["cal_slope_mean"] - 1).abs()).sort_values(["_simp", "_cal"])
    else:
        tol_ok = tol_ok.assign(_simp=tol_ok["model"].map(simplicity)).sort_values(["_simp"])
    primary = tol_ok.iloc[0]["model"]

    # ---- 쌍체 부트스트랩: 기준선(enet) 대비, 주 모델 대비
    pair_rows = []
    metrics_to_test = ["AUROC", "AUPRC"] if cfg.task == "classification" else ["MAE", "RMSE"]
    ref_models = [x for x in dict.fromkeys(["enet", primary]) if x in selected]
    for ref in ref_models:
        for m in selected:
            if m == ref:
                continue
            for met in metrics_to_test:
                d, lo, hi = paired_bootstrap(cfg.task, y, np.nanmean(oof[m], 0), np.nanmean(oof[ref], 0),
                                             cfg.n_boot, cfg.seed, met)
                pair_rows.append({"model": m, "vs": ref, "metric": met, "diff": d, "ci_lo": lo, "ci_hi": hi})
    pairwise = pd.DataFrame(pair_rows)

    # ---- 저장
    summary.to_csv(os.path.join(out_dir, "results_summary.csv"), index=False)
    pairwise.to_csv(os.path.join(out_dir, "pairwise_bootstrap.csv"), index=False)
    pd.DataFrame(best_params_log).to_csv(os.path.join(out_dir, "best_params_by_fold.csv"), index=False)
    oof_df = pd.DataFrame({id_col: ids, target: y})
    for m in selected:
        for r in range(cfg.repeats):
            oof_df[f"{m}_rep{r}"] = oof[m][r]
    oof_df.to_csv(os.path.join(out_dir, "oof_predictions.csv"), index=False)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({**cfg.__dict__, "features": features, "categorical": cat_cols, "group_col": group_col,
                   "n": int(len(y)), "primary_model": primary}, f, ensure_ascii=False, indent=2, default=str)

    # ---- 마크다운 요약
    show = [f"{pm}_mean", f"{pm}_sd"] + ([f"AUROC_mean", "Brier_mean", "cal_slope_mean", "sens@20%_mean",
                                          "MRI_per_case@20%_mean"] if cfg.task == "classification"
                                         else ["RMSE_mean", "R2_mean"]) + ["train_seconds"]
    def _md(df):  # tabulate가 없으면 고정폭 텍스트로 대체
        try:
            return df.to_markdown(index=False)
        except Exception:
            return "```\n" + df.to_string(index=False) + "\n```"
    lines = [f"# 모델 비교 요약 ({cfg.task}, n={len(y)}, p={len(features)})", "",
             f"외부 {cfg.outer_folds}분할 × {cfg.repeats}회, 내부 {cfg.inner_folds}분할, 모델당 무작위 탐색 {cfg.n_iter}회.",
             f"사전 고정 선택 규칙에 따른 **주 모델: {labels[primary]}** (지표 {pm}, 허용 오차 {cfg.selection_tol}).", "",
             _md(summary[["label"] + show].round(4)), "",
             "## 쌍체 부트스트랩 95% CI (반복 평균 OOF 예측 기준)", "",
             _md(pairwise.round(4)) if len(pairwise) else "(비교 대상 없음)", "",
             "> 이 표는 개발군 결과다. 주 모델만 평가군에 한 번 적용하고, 결과를 본 뒤 주 모델을 바꾸지 않는다."]
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines[:6]))
    print(f"\n[저장] {out_dir}/results_summary.csv, pairwise_bootstrap.csv, oof_predictions.csv, summary.md")
    return summary


# ----------------------------------------------------------------------------------------------
# 더미 데이터 (구조 확인용; 실제 자료와 무관)
# ----------------------------------------------------------------------------------------------
def make_demo(n: int = 2000, seed: int = 0, task: str = "classification") -> tuple:
    rng = np.random.default_rng(seed)
    age = rng.integers(40, 70, n)
    sex = rng.integers(1, 3, n)
    sbp = rng.normal(125 + 0.4 * (age - 55), 15, n)
    bmi = rng.normal(24.5, 3, n)
    htn = (rng.random(n) < 1 / (1 + np.exp(-(sbp - 140) / 10))).astype(int)
    dm = (rng.random(n) < 0.12 + 0.004 * (age - 55)).astype(int)
    lip = rng.integers(0, 2, n)
    smoke = rng.integers(1, 4, n)
    glu = rng.normal(95 + 25 * dm, 12, n)
    prs_hip = rng.normal(0, 1, n)
    prs_t2d = rng.normal(0, 1, n) + 0.5 * dm
    apoe4 = rng.choice([0, 1, 2], n, p=[0.75, 0.22, 0.03])
    latent = (-0.04 * (age - 55) - 0.35 * dm - 0.25 * htn - 0.30 * prs_hip - 0.20 * apoe4
              + 0.10 * (sex == 1) + rng.normal(0, 1, n))
    df = pd.DataFrame(dict(T00_ID=np.arange(n), age=age, sex=sex, sbp=sbp, bmi=bmi, htn=htn, dm=dm, lip=lip,
                           smoke=smoke, glu=glu, prs_hip=prs_hip, prs_t2d=prs_t2d, apoe4=apoe4))
    for c in ["sbp", "bmi", "glu"]:                      # 결측 주입
        df.loc[rng.random(n) < 0.05, c] = np.nan
    if task == "classification":
        df["atrophy_pos"] = (latent < np.quantile(latent, 0.2)).astype(int)
        target = "atrophy_pos"
    else:
        df["atrophy_rate"] = -latent * 0.3 + rng.normal(0, 0.3, n)
        target = "atrophy_rate"
    features = [c for c in df.columns if c not in ("T00_ID", target)]
    return df, target, features, ["sex", "smoke"]


def read_list(path: str | None) -> list:
    if not path:
        return []
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="CSV 또는 Parquet")
    ap.add_argument("--id-col", default="T00_ID")
    ap.add_argument("--target")
    ap.add_argument("--task", default="classification", choices=["classification", "regression"])
    ap.add_argument("--features", help="변수명 목록 파일(한 줄 하나)")
    ap.add_argument("--categorical", help="범주형 변수명 목록 파일")
    ap.add_argument("--group-col", default=None, help="근연 집단 열(같은 집단은 같은 폴드)")
    ap.add_argument("--models", default="enet,rf,lgbm,xgb,cat,tabpfn", help="쉼표 구분: enet,rf,lgbm,xgb,cat,svm,tabpfn")
    ap.add_argument("--outer-folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--inner-folds", type=int, default=3)
    ap.add_argument("--n-iter", type=int, default=40)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--selection-tol", type=float, default=0.01,
                    help="선택 규칙 허용 오차(주 지표 단위; 분류 AUPRC, 회귀 MAE)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--demo", action="store_true", help="더미 데이터로 실행")
    ap.add_argument("--fast", action="store_true", help="빠른 점검(작은 예산)")
    a = ap.parse_args()

    cfg = Config(task=a.task, outer_folds=a.outer_folds, repeats=a.repeats, inner_folds=a.inner_folds,
                 n_iter=a.n_iter, seed=a.seed, n_boot=a.n_boot, models=a.models.split(","),
                 selection_tol=a.selection_tol)
    if a.fast:
        cfg.outer_folds, cfg.repeats, cfg.inner_folds, cfg.n_iter, cfg.n_boot = 3, 1, 2, 4, 200

    if a.demo:
        df, target, features, cats = make_demo(task=a.task)
        id_col = "T00_ID"
        group_col = None
    else:
        if not (a.data and a.target and a.features):
            ap.error("--data, --target, --features 는 필수입니다 (또는 --demo).")
        df = pd.read_parquet(a.data) if a.data.endswith(".parquet") else pd.read_csv(a.data)
        target, features, cats, id_col, group_col = a.target, read_list(a.features), read_list(a.categorical), a.id_col, a.group_col
        missing = [c for c in features + [target, id_col] if c not in df.columns]
        if missing:
            sys.exit(f"자료에 없는 열: {missing}")
    run(df, id_col, target, features, cats, group_col, cfg, a.out)


if __name__ == "__main__":
    main()
