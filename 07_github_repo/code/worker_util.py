# -*- coding: utf-8 -*-
"""worker_util.py — 노트북에서 tabpfn_worker.py를 별도 프로세스로 실행하고 출력을 실시간으로 보여 준다.
작업자 프로세스가 죽어도 노트북 커널은 살아 있고, 종료 코드와 마지막 출력으로 원인을 알려 준다."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

CODE = Path(__file__).resolve().parent
_QUIET = ("Warning", "warnings.warn", "warn(")


def run_worker(args, extra_env=None, show=True):
    """(종료 코드, 출력 줄 목록)을 돌려준다."""
    env = os.environ.copy()
    env.update(extra_env or {})
    env.setdefault("TABPFN_NO_BROWSER", "1")      # 노트북에서는 브라우저 로그인 대신 TABPFN_TOKEN 키를 쓴다
    cmd = [sys.executable, "-u", str(CODE / "tabpfn_worker.py")] + [str(a) for a in args]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=str(CODE))
    out = []
    for line in p.stdout:
        out.append(line)
        if show and not any(q in line for q in _QUIET):
            print(line, end="", flush=True)
    return p.wait(), out


def run_spec(mode, spec, extra_env=None, show=True):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False)
        path = f.name
    try:
        return run_worker([mode, "--spec", path], extra_env, show)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def explain(rc, out):
    txt = "".join(out)
    if rc == 0:
        return "정상 종료"
    if "OMP: Error #15" in txt or "libomp" in txt:
        return "OpenMP 런타임 중복 충돌(맥에서 torch와 다른 라이브러리가 함께 올라갈 때 흔함)"
    if "TabPFNHuggingFace" in txt or "huggingface.co" in txt:
        return "가중치 정보 또는 내려받기 실패(huggingface.co 연결, 사내망 차단 여부 확인)"
    if "TabPFNLicenseError" in txt or "License not yet accepted" in txt:
        return "라이선스 확인 실패(ux.priorlabs.ai Licenses 탭에서 TabPFN-3.5 동의 여부와 TABPFN_TOKEN 확인)"
    if rc == -9:
        return "강제 종료(SIGKILL), 메모리 부족 가능성이 큼"
    if rc == -11:
        return "세그폴트(SIGSEGV), 장치(MPS)나 라이브러리 충돌 가능성"
    if rc == -6:
        return "비정상 중단(SIGABRT)"
    if rc == -10:
        return "버스 오류(SIGBUS)"
    last = [l.strip() for l in out if l.strip()][-3:]
    return "오류 종료: " + " | ".join(last)
