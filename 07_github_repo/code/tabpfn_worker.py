#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tabpfn_worker.py — TabPFN 계산을 노트북 커널과 분리된 프로세스에서 실행한다.

왜 따로 돌리나
  맥에서는 torch(TabPFN)와 LightGBM, XGBoost, CatBoost가 각자 OpenMP 런타임을 가지고 있어, 한 커널 안에서
  함께 계산하면 커널이 죽을 수 있다. 이 작업자는 TabPFN에 필요한 라이브러리만 불러오고, 다른 모델의 결과는
  저장된 OOF 예측을 그대로 재사용한다. 이 프로세스가 죽어도 노트북 커널은 살아 있고, 마지막 출력으로 원인을 볼 수 있다.

사용
  python tabpfn_worker.py check [--device auto|cpu|mps]     # 합성 난수로 단계별 점검(실제 자료 사용 안 함)
  python tabpfn_worker.py run --spec spec.json               # 한 블록의 TabPFN OOF 계산 후 기존 결과에 합침
  python tabpfn_worker.py final --spec spec.json             # 노트북 05용: 개발군 전체 적합, 5분할 교차예측, 평가군과 추적군 예측 저장
"""
import argparse
import faulthandler
import json
import os
import sys
import time

faulthandler.enable()          # 세그폴트나 중단 시 파이썬 스택을 출력
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def step(msg):
    print(f"[단계] {msg}", flush=True)


def check(device: str, version: str):
    t0 = time.time()
    step("1. torch 불러오기")
    import torch
    print(f"    torch {torch.__version__} | MPS 사용 가능 {torch.backends.mps.is_available()} | CUDA {torch.cuda.is_available()}", flush=True)

    step(f"2. torch 기본 연산 (장치 {device})")
    dev = "mps" if (device == "mps" or (device == "auto" and torch.backends.mps.is_available())) else "cpu"
    x = torch.randn(256, 256, device=dev)
    print(f"    {dev} 행렬곱 합계 {float((x @ x).sum()):.1f}", flush=True)

    step("3. tabpfn 불러오기, 모델 버전 확인")
    import tabpfn
    from tabpfn.constants import ModelVersion
    versions = [v.value for v in ModelVersion]
    print(f"    tabpfn {tabpfn.__version__} | 사용 가능 모델 {versions}", flush=True)
    if version not in versions:
        print(f"    [실패] 모델 버전 {version} 없음. pip install -U \"tabpfn>=9.0.0\" 후 커널 재시작", flush=True)
        sys.exit(3)

    import numpy as np
    import tabular_model_comparison as tmc      # 옵션 라이브러리를 필요할 때만 불러오므로 GBDT는 올라가지 않는다
    cols = [f"x{i}" for i in range(35)]
    cfg = tmc.Config(models=["tabpfn"], tabpfn_version=version, tabpfn_device=device)
    rng = np.random.default_rng(0)
    import pandas as pd
    X = pd.DataFrame(rng.normal(size=(8000, 35)), columns=cols)          # 합성 난수, 실제 자료 아님
    y = (X.x0 + rng.normal(size=8000) > 1).astype(int).values

    step("4. 작은 적합 200행 (라이선스 확인, 처음이면 가중치 내려받기)")
    pipe = tmc.build_pipeline(tmc.model_zoo(cfg, cols, [])["tabpfn"], cols, [])
    t = time.time(); pipe.fit(X[:200], y[:200]); pipe.predict_proba(X[200:400])
    est = pipe.named_steps["est"]
    print(f"    완료 {time.time() - t:.0f}초 | 장치 {getattr(est, 'devices_', '?')} | 모델 파일 {os.path.basename(str(est.model_path))}", flush=True)

    step("5. 본 실행 크기 적합 (훈련 6,400행, 예측 1,600행, 35열)")
    t = time.time(); pipe.fit(X[:6400], y[:6400]); pipe.predict_proba(X[6400:]); dt = time.time() - t
    print(f"    완료 {dt:.0f}초 → 주 비교 4블록 × 10폴드 예상 약 {40 * dt / 60:.0f}분", flush=True)
    print(json.dumps({"ok": True, "device": device, "fold_seconds": round(dt, 1), "total_seconds": round(time.time() - t0, 1)}), flush=True)


def run(spec_path: str):
    spec = json.load(open(spec_path, encoding="utf-8"))
    import pandas as pd
    import tabular_model_comparison as tmc
    root = spec["root"]
    der = os.path.join(root, "05_분석결과", "derived")
    B = pd.read_csv(os.path.join(der, "base_analysis.csv"))
    dev = B[B.split == "dev"].reset_index(drop=True)
    gpath = os.path.join(der, "genomic_features.csv")
    if os.path.exists(gpath):                       # 노트북 03과 같은 방식으로 개발군을 만든다
        dev = dev.merge(pd.read_csv(gpath), on="T_ID", how="left")
    for extra in spec.get("extra_csv", []):         # 노트북 07 확장 변수 등(T_ID로 결합)
        dev = dev.merge(pd.read_csv(extra), on="T_ID", how="left")
    task = spec.get("task", "classification"); target = spec.get("target", "atrophy_pos")
    cfg = tmc.Config(task=task, models=spec["models"], tabpfn_version=spec["tabpfn_version"],
                     tabpfn_device=spec["device"], tabpfn_n_estimators=spec.get("tabpfn_n_estimators", 8))
    if spec["fast"]:
        cfg.outer_folds, cfg.repeats, cfg.inner_folds, cfg.n_iter, cfg.n_boot = 3, 1, 2, 4, 200
    step(f"{spec['block']}: TabPFN 계산 (장치 {spec['device']}, {task}, 목표 {target}, 앙상블 {cfg.tabpfn_n_estimators}), 나머지 모델은 저장된 결과 재사용")
    t0 = time.time()
    tmc.run(dev, "T_ID", target, spec["features"], [], None, cfg, spec["out_dir"], resume=True,
            compute_only=["tabpfn"])
    print(json.dumps({"ok": True, "block": spec["block"], "seconds": round(time.time() - t0, 1)}), flush=True)


def final(spec_path: str):
    """노트북 05에서 TabPFN이 주 모델일 때: 개발군 전체로 적합해 평가군, 추적군 예측과 보정용 5분할 교차예측을 저장한다.
    예측 확률만 저장하고(대상자 순서는 입력 파일 순서), 결과값은 쓰지 않는다."""
    spec = json.load(open(spec_path, encoding="utf-8"))
    import pickle
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    import tabular_model_comparison as tmc
    der = os.path.join(spec["root"], "05_분석결과", "derived")
    B = pd.read_csv(os.path.join(der, "base_analysis.csv")); Gf = pd.read_csv(os.path.join(der, "genomic_features.csv"))
    dev = B[B.split == "dev"].reset_index(drop=True).merge(Gf, on="T_ID", how="left")
    ev = B[B.split == "eval"].reset_index(drop=True).merge(Gf, on="T_ID", how="left")
    F1 = pd.read_csv(os.path.join(der, "follow_baseline.csv")); L = pd.read_csv(os.path.join(der, "follow_longitudinal.csv"))
    feats = spec["features"]
    cfg = tmc.Config(task="classification", models=["tabpfn"], seed=spec["seed"], tabpfn_version=spec["tabpfn_version"],
                     tabpfn_device=spec["device"], tabpfn_n_estimators=spec.get("tabpfn_n_estimators", 8))
    zoo = tmc.model_zoo(cfg, feats, [])
    if "tabpfn" not in zoo:
        print("[실패] TabPFN을 만들 수 없습니다.", flush=True); sys.exit(3)
    X, y = dev[feats], dev.atrophy_pos.values
    step(f"{spec['block']}: 개발군 {len(X)}명 전체 적합 후 예측")
    t0 = time.time()
    pipe = tmc.build_pipeline(zoo["tabpfn"], feats, []).fit(X, y)
    pred = {"eval": pipe.predict_proba(ev[feats])[:, 1]}
    ids = {"eval": ev.T_ID.astype(str).tolist()}
    for key, df in [("F1", F1), ("L", L)]:
        if all(c in df.columns for c in feats):
            pred[key] = pipe.predict_proba(df[feats])[:, 1]; ids[key] = df.T00_ID.astype(str).tolist()
    sec = time.time() - t0
    step(f"{spec['block']}: 보정용 5분할 교차예측")
    t1 = time.time()
    fresh = tmc.build_pipeline(tmc.model_zoo(cfg, feats, [])["tabpfn"], feats, [])
    cv_pred = cross_val_predict(fresh, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=spec["seed"]),
                                method="predict_proba", n_jobs=1)[:, 1]
    obj = {"est": None, "best_params": {}, "features": feats, "seconds": round(sec, 1), "cv_seconds": round(time.time() - t1, 1),
           "cv_pred": cv_pred, "pred": pred, "pred_ids": ids, "dev_ids": dev.T_ID.astype(str).tolist(),
           "device": spec["device"], "tabpfn_version": spec["tabpfn_version"]}
    pickle.dump(obj, open(spec["out_path"], "wb"))
    print(json.dumps({"ok": True, "block": spec["block"], "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "run", "final"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--version", default="v3.5")
    ap.add_argument("--spec")
    a = ap.parse_args()
    if a.mode == "check":
        check(a.device, a.version)
    elif a.mode == "run":
        run(a.spec)
    else:
        final(a.spec)
