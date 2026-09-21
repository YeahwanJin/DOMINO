#!/bin/bash
# =============================================================================
# upload_to_gdrive.sh
#
# SmolVLA 학습/평가 환경 복제를 위해 필요한 파일들을
# Google Drive에 rclone으로 업로드합니다.
#
# 사전 준비:
#   1. rclone 설치: curl https://rclone.org/install.sh | sudo bash
#   2. rclone 설정: rclone config
#      → New remote → name: gdrive → Storage: Google Drive → 안내 따라 인증
#   3. 이 스크립트의 RCLONE_REMOTE, GDRIVE_DIR 변수 수정
#
# 사용법:
#   bash script/upload_to_gdrive.sh
# =============================================================================
set -euo pipefail

# ---- 설정 (수정 필요) ----
RCLONE_REMOTE="gdrive"                        # rclone config에서 설정한 remote 이름
GDRIVE_DIR="DOMINO_transfer"                  # Google Drive 상의 폴더명
DOMINO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== DOMINO root: ${DOMINO_ROOT}"
echo "=== Upload target: ${RCLONE_REMOTE}:${GDRIVE_DIR}/"
echo ""

# ---- 0. rclone 존재 확인 ----
if ! command -v rclone &>/dev/null; then
    echo "[ERROR] rclone이 설치되어 있지 않습니다."
    echo "  설치: curl https://rclone.org/install.sh | sudo bash"
    echo "  설정: rclone config  (Google Drive remote 추가)"
    exit 1
fi

# ---- 1. data/lerobot_data (가장 큼 — 태스크별로 업로드) ----
echo "=== [1/4] data/lerobot_data/ 업로드 (태스크별) ==="
LEROBOT_DIR="${DOMINO_ROOT}/data/lerobot_data"
if [ -d "${LEROBOT_DIR}" ]; then
    for task_dir in "${LEROBOT_DIR}"/*/; do
        task_name=$(basename "${task_dir}")
        echo "  → ${task_name} ..."
        rclone copy "${task_dir}" "${RCLONE_REMOTE}:${GDRIVE_DIR}/data/lerobot_data/${task_name}/" \
            --transfers=8 --checkers=16 --progress --fast-list \
            --log-level INFO
        echo "  ✓ ${task_name} 완료"
    done
else
    echo "  [WARN] ${LEROBOT_DIR} 가 존재하지 않습니다. 건너뜁니다."
fi

# ---- 2. assets/ (~16GB) ----
echo ""
echo "=== [2/4] assets/ 업로드 ==="
ASSETS_DIR="${DOMINO_ROOT}/assets"
if [ -d "${ASSETS_DIR}" ]; then
    rclone copy "${ASSETS_DIR}" "${RCLONE_REMOTE}:${GDRIVE_DIR}/assets/" \
        --transfers=8 --checkers=16 --progress --fast-list \
        --log-level INFO
    echo "  ✓ assets 완료"
else
    echo "  [WARN] ${ASSETS_DIR} 가 존재하지 않습니다."
fi

# ---- 3. task_config/ (~40KB) ----
echo ""
echo "=== [3/4] task_config/ 업로드 ==="
TASK_CONFIG_DIR="${DOMINO_ROOT}/task_config"
if [ -d "${TASK_CONFIG_DIR}" ]; then
    rclone copy "${TASK_CONFIG_DIR}" "${RCLONE_REMOTE}:${GDRIVE_DIR}/task_config/" \
        --transfers=4 --progress --fast-list \
        --log-level INFO
    echo "  ✓ task_config 완료"
else
    echo "  [WARN] ${TASK_CONFIG_DIR} 가 존재하지 않습니다."
fi

# ---- 4. eval_result/experimental_log.txt ----
echo ""
echo "=== [4/4] eval_result/experimental_log.txt 업로드 ==="
EVAL_LOG="${DOMINO_ROOT}/eval_result/experimental_log.txt"
if [ -f "${EVAL_LOG}" ]; then
    rclone copy "${EVAL_LOG}" "${RCLONE_REMOTE}:${GDRIVE_DIR}/eval_result/" \
        --progress --log-level INFO
    echo "  ✓ experimental_log.txt 완료"
else
    echo "  [WARN] ${EVAL_LOG} 가 존재하지 않습니다."
fi

echo ""
echo "=========================================="
echo "  업로드 완료!"
echo "  Google Drive 경로: ${RCLONE_REMOTE}:${GDRIVE_DIR}/"
echo ""
echo "  대상 서버에서 다운로드:"
echo "    bash script/download_from_gdrive.sh"
echo "=========================================="
