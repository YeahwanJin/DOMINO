#!/bin/bash
# =============================================================================
# upload_to_gdrive.sh
#
# SmolVLA 학습/평가 환경 복제를 위해 필요한 파일들을
# tar.gz로 압축한 뒤 Google Drive에 rclone으로 업로드합니다.
#
# PUMA 전용 캐시(grounding_cache, history_flow_cache)는 제외합니다.
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
STAGING_DIR="${DOMINO_ROOT}/.upload_staging"   # 압축 파일 임시 저장 경로

echo "=== DOMINO root: ${DOMINO_ROOT}"
echo "=== Upload target: ${RCLONE_REMOTE}:${GDRIVE_DIR}/"
echo "=== Staging dir: ${STAGING_DIR}"
echo ""

# ---- 0. 사전 확인 ----
if ! command -v rclone &>/dev/null; then
    echo "[ERROR] rclone이 설치되어 있지 않습니다."
    echo "  설치: curl https://rclone.org/install.sh | sudo bash"
    echo "  설정: rclone config  (Google Drive remote 추가)"
    exit 1
fi

mkdir -p "${STAGING_DIR}"

# ---- 1. data/lerobot_data (태스크별 tar.gz, PUMA 캐시 제외) ----
echo "=== [1/4] data/lerobot_data/ 압축 & 업로드 (태스크별) ==="
LEROBOT_DIR="${DOMINO_ROOT}/data/lerobot_data"
if [ -d "${LEROBOT_DIR}" ]; then
    for task_dir in "${LEROBOT_DIR}"/*/; do
        [ -d "${task_dir}" ] || continue
        task_name=$(basename "${task_dir}")
        archive="${STAGING_DIR}/lerobot_${task_name}.tar.gz"

        echo "  → ${task_name}: 압축 중 (PUMA 캐시 제외) ..."
        tar -czf "${archive}" \
            -C "${LEROBOT_DIR}" \
            --exclude="grounding_cache" \
            --exclude="history_flow_cache" \
            "${task_name}"
        archive_size=$(du -h "${archive}" | cut -f1)
        echo "  → ${task_name}: 업로드 중 (${archive_size}) ..."

        rclone copy "${archive}" "${RCLONE_REMOTE}:${GDRIVE_DIR}/archives/" \
            --progress --log-level INFO
        echo "  ✓ ${task_name} 완료"

        rm -f "${archive}"
    done
else
    echo "  [WARN] ${LEROBOT_DIR} 가 존재하지 않습니다. 건너뜁니다."
fi

# ---- 2. assets/ ----
echo ""
echo "=== [2/4] assets/ 압축 & 업로드 ==="
ASSETS_DIR="${DOMINO_ROOT}/assets"
if [ -d "${ASSETS_DIR}" ]; then
    archive="${STAGING_DIR}/assets.tar.gz"
    echo "  → 압축 중 ..."
    tar -czf "${archive}" -C "${DOMINO_ROOT}" assets
    archive_size=$(du -h "${archive}" | cut -f1)
    echo "  → 업로드 중 (${archive_size}) ..."

    rclone copy "${archive}" "${RCLONE_REMOTE}:${GDRIVE_DIR}/archives/" \
        --progress --log-level INFO
    echo "  ✓ assets 완료"
    rm -f "${archive}"
else
    echo "  [WARN] ${ASSETS_DIR} 가 존재하지 않습니다."
fi

# ---- 3. task_config/ ----
echo ""
echo "=== [3/4] task_config/ 압축 & 업로드 ==="
TASK_CONFIG_DIR="${DOMINO_ROOT}/task_config"
if [ -d "${TASK_CONFIG_DIR}" ]; then
    archive="${STAGING_DIR}/task_config.tar.gz"
    echo "  → 압축 중 ..."
    tar -czf "${archive}" -C "${DOMINO_ROOT}" task_config
    archive_size=$(du -h "${archive}" | cut -f1)
    echo "  → 업로드 중 (${archive_size}) ..."

    rclone copy "${archive}" "${RCLONE_REMOTE}:${GDRIVE_DIR}/archives/" \
        --progress --log-level INFO
    echo "  ✓ task_config 완료"
    rm -f "${archive}"
else
    echo "  [WARN] ${TASK_CONFIG_DIR} 가 존재하지 않습니다."
fi

# ---- 4. eval_result/experimental_log.txt ----
echo ""
echo "=== [4/4] eval_result/experimental_log.txt 업로드 ==="
EVAL_LOG="${DOMINO_ROOT}/eval_result/experimental_log.txt"
if [ -f "${EVAL_LOG}" ]; then
    rclone copy "${EVAL_LOG}" "${RCLONE_REMOTE}:${GDRIVE_DIR}/archives/eval_result/" \
        --progress --log-level INFO
    echo "  ✓ experimental_log.txt 완료"
else
    echo "  [WARN] ${EVAL_LOG} 가 존재하지 않습니다."
fi

# ---- 정리 ----
rmdir "${STAGING_DIR}" 2>/dev/null || true

echo ""
echo "=========================================="
echo "  업로드 완료!"
echo "  Google Drive 경로: ${RCLONE_REMOTE}:${GDRIVE_DIR}/archives/"
echo ""
echo "  대상 서버에서 다운로드:"
echo "    bash script/download_from_gdrive.sh"
echo "=========================================="
