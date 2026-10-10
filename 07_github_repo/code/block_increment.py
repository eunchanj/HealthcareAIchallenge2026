#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
block_increment.py — 입력 블록 간 증분 가치(쌍체 부트스트랩) 표 생성
노트북 03의 정식 실행 결과(05_분석결과/models/full_*/oof_predictions.csv)를 읽어,
같은 대상자와 같은 분할의 OOF 예측(반복 평균)으로 블록 간 차이와 95% 신뢰구간을 계산한다.
출력: 05_분석결과/tables/table3_block_increment.csv, table2_model_comparison_dev.csv
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / "05_분석결과" / "models"; TAB = ROOT / "05_분석결과" / "tables"; TAB.mkdir(parents=True, exist_ok=True)
SEED, NBOOT = 20260915, 2000
ORDER = ["M0", "M1", "M2", "G", "M1+G", "M2+G", "M1-Dx", "M1-Dx+G"]

def primary(block):
    return json.load(open(M / f"full_{block}" / "config.json", encoding="utf-8"))["primary_model"]

def oof(block, model):
    d = pd.read_csv(M / f"full_{block}" / "oof_predictions.csv")
    cols = [c for c in d.columns if c.startswith(model + "_rep")]
    return d["T_ID"].astype(str).values, d["atrophy_pos"].values, d[cols].mean(axis=1).values

def sens_at(y, p, frac=0.2):
    n = int(round(frac * len(y))); idx = np.argsort(-p, kind="stable")[:n]; return y[idx].sum() / y.sum()

def compare(a, b):
    ida, y, pa = oof(*a); idb, _, pb = oof(*b); assert (ida == idb).all(), "대상자 순서 불일치"
    fns = {"AUROC": roc_auc_score, "AUPRC": average_precision_score, "민감도@20%": sens_at}
    rng = np.random.default_rng(SEED); N = len(y); boots = {k: [] for k in fns}
    for _ in range(NBOOT):
        i = rng.integers(0, N, N)
        for k, f in fns.items(): boots[k].append(f(y[i], pa[i]) - f(y[i], pb[i]))
    row = {}
    for k, f in fns.items():
        row[f"{k} 차이"] = f(y, pa) - f(y, pb)
        row[f"{k} 95%CI 하한"], row[f"{k} 95%CI 상한"] = np.percentile(boots[k], [2.5, 97.5])
    return row

# 표 2: 블록별 모델 비교(개발군 교차검증)
rows = []
for b in ORDER:
    s = pd.read_csv(M / f"full_{b}" / "results_summary.csv"); pm = primary(b)
    for _, r in s.iterrows():
        rows.append({"블록": b, "모델": r["label"], "주 모델": "예" if r["model"] == pm else "",
                     "AUPRC": r["AUPRC_mean"], "AUPRC SD": r["AUPRC_sd"], "AUROC": r["AUROC_mean"], "Brier": r["Brier_mean"],
                     "보정 기울기": r["cal_slope_mean"], "보정 절편": r["cal_intercept_mean"],
                     "민감도@10%": r["sens@10%_mean"], "민감도@20%": r["sens@20%_mean"], "민감도@30%": r["sens@30%_mean"],
                     "1명 발견당 MRI@20%": r["MRI_per_case@20%_mean"]})
T2 = pd.DataFrame(rows); T2.round(4).to_csv(TAB / "table2_model_comparison_dev.csv", index=False, encoding="utf-8-sig")

# 표 3: 블록 간 증분 가치 (주 모델 기준, 같은 알고리즘 민감도 분석 포함)
tests = [("주 비교 1", "진단력이 있을 때 유전체 추가", ("M1+G", primary("M1+G")), ("M1", primary("M1"))),
         ("주 비교 2", "진단력이 없을 때 유전체 추가", ("M1-Dx+G", primary("M1-Dx+G")), ("M1-Dx", primary("M1-Dx"))),
         ("주 비교 2 민감도", "같은 알고리즘(Elastic-net)", ("M1-Dx+G", "enet"), ("M1-Dx", "enet")),
         ("참고", "진단력의 가치", ("M1", primary("M1")), ("M1-Dx", primary("M1-Dx"))),
         ("참고", "임상 정보 대 연령, 성별", ("M1", primary("M1")), ("M0", primary("M0"))),
         ("참고", "혈액검사 추가(같은 알고리즘)", ("M2", primary("M2")), ("M1", primary("M2"))),
         ("참고", "연령, 성별만 있을 때 유전체 추가", ("G", primary("G")), ("M0", primary("M0")))]
rows = []
for kind, lab, a, b in tests:
    rows.append({"구분": kind, "비교": lab, "모형 A": f"{a[0]} [{a[1]}]", "모형 B": f"{b[0]} [{b[1]}]", **compare(a, b)})
T3 = pd.DataFrame(rows); T3.round(4).to_csv(TAB / "table3_block_increment.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print(T3.round(4).to_string(index=False))
print("\n저장:", (TAB / "table2_model_comparison_dev.csv").name, (TAB / "table3_block_increment.csv").name)
