#!/usr/bin/env bash
# 사용법: bash push_to_github.sh <GitHub 저장소 URL> [복사할 로컬 경로]
#   예: bash push_to_github.sh git@github.com:계정/healthcare-ai-challenge-2026.git
# OneDrive 동기화 폴더 안에서 git을 쓰면 .git 객체가 충돌할 수 있어, 먼저 동기화 밖(기본 ~/Projects)으로 복사한 뒤 초기화한다.
set -euo pipefail
URL="${1:?GitHub 저장소 URL을 첫 인수로 주세요 (비공개 저장소여야 합니다)}"
DEST="${2:-$HOME/Projects/healthcare-ai-challenge-2026}"
SRC="$(cd "$(dirname "$0")" && pwd)"
if [ -e "$DEST" ]; then echo "이미 존재: $DEST (다른 경로를 두 번째 인수로 주세요)"; exit 1; fi
mkdir -p "$(dirname "$DEST")"
rsync -a --exclude '.DS_Store' --exclude 'push_to_github.sh' "$SRC/" "$DEST/"
cd "$DEST"
# 올라가면 안 되는 파일이 섞였는지 마지막 점검
if grep -rIl --exclude-dir=.git -e "tabpfn_sk_" -e "K_BASE_00" . >/dev/null 2>&1; then echo "[중단] 키 또는 대상자 ID가 포함된 파일이 있습니다:"; grep -rIl --exclude-dir=.git -e "tabpfn_sk_" -e "K_BASE_00" .; exit 1; fi
git init -b main
git add -A
git -c user.useConfigOnly=true commit -m "뇌 위축 바이오마커 예측 모델: 분석 코드, 공유용 노트북, 집계 결과, 문서" || { echo "git user.name과 user.email을 먼저 설정하세요: git config --global user.name ...; git config --global user.email ..."; exit 1; }
git remote add origin "$URL"
git push -u origin main
echo "완료: $DEST → $URL"
