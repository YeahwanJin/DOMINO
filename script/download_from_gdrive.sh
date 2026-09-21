#!/bin/bash
# =============================================================================
# download_from_gdrive.sh
#
# 대상 서버에서 Google Drive로부터 SmolVLA 학습/평가 필수 파일을 다운로드합니다.
#
# 사전 준비:
#   1. rclone 설치: curl https://rclone.org/install.sh | sudo bash
#   2. rclone 설정: rclone config
#      → New remote → name: gdrive → Storage: Google Drive → 안내 따라 인증
#   3. DOMINO 레포 clone 완료 상태
#   4. 이 스크립트의 RCLONE_REMOTE, GDRIVE_DIR, DOMINO_ROOT 변수 수정
#
# 사용법:
#   bash script/download_from_gdrive.sh
# =============================================================================
set -euo pipefail

# ---- 설정 (수정 필요) ----
RCLONE_REMOTE="gdrive"                        # rclone config에서 설정한 remote 이름
GDRIVE_DIR="DOMINO_transfer"                  # 업로드 시 사용한 Google Drive 폴더명
DOMINO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== DOMINO root: ${DOMINO_ROOT}"
echo "=== Download source: ${RCLONE_REMOTE}:${GDRIVE_DIR}/"
echo ""

# ---- 0. rclone 존재 확인 ----
if ! command -v rclone &>/dev/null; then
    echo "[ERROR] rclone이 설치되어 있지 않습니다."
    echo "  설치: curl https://rclone.org/install.sh | sudo bash"
    echo "  설정: rclone config  (Google Drive remote 추가)"
    exit 1
fi

# ---- 1. data/lerobot_data ----
echo "=== [1/4] data/lerobot_data/ 다운로드 ==="
mkdir -p "${DOMINO_ROOT}/data/lerobot_data"
rclone copy "${RCLONE_REMOTE}:${GDRIVE_DIR}/data/lerobot_data/" \
    "${DOMINO_ROOT}/data/lerobot_data/" \
    --transfers=8 --checkers=16 --progress --fast-list \
    --log-level INFO
echo "  ✓ lerobot_data 완료"

# ---- 2. assets/ ----
echo ""
echo "=== [2/4] assets/ 다운로드 ==="
mkdir -p "${DOMINO_ROOT}/assets"
rclone copy "${RCLONE_REMOTE}:${GDRIVE_DIR}/assets/" \
    "${DOMINO_ROOT}/assets/" \
    --transfers=8 --checkers=16 --progress --fast-list \
    --log-level INFO
echo "  ✓ assets 완료"

# ---- 3. task_config/ ----
echo ""
echo "=== [3/4] task_config/ 다운로드 ==="
mkdir -p "${DOMINO_ROOT}/task_config"
rclone copy "${RCLONE_REMOTE}:${GDRIVE_DIR}/task_config/" \
    "${DOMINO_ROOT}/task_config/" \
    --transfers=4 --progress --fast-list \
    --log-level INFO
echo "  ✓ task_config 완료"

# ---- 4. eval_result/experimental_log.txt ----
echo ""
echo "=== [4/4] eval_result/experimental_log.txt 다운로드 ==="
mkdir -p "${DOMINO_ROOT}/eval_result"
rclone copy "${RCLONE_REMOTE}:${GDRIVE_DIR}/eval_result/" \
    "${DOMINO_ROOT}/eval_result/" \
    --progress --log-level INFO
echo "  ✓ experimental_log.txt 완료"

echo ""
echo "=========================================="
echo "  다운로드 완료!"
echo "  데이터 확인:"
echo "    ls ${DOMINO_ROOT}/data/lerobot_data/ | wc -l   # 35~36개 태스크"
echo "    ls ${DOMINO_ROOT}/assets/                        # objects, embodiments 등"
echo "    cat ${DOMINO_ROOT}/eval_result/experimental_log.txt | head"
echo "=========================================="
