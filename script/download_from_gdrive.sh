#!/bin/bash
# =============================================================================
# download_from_gdrive.sh
#
# Google Drive에서 tar.gz 아카이브를 다운로드하고 DOMINO 프로젝트 경로에 해제합니다.
# upload_to_gdrive.sh로 업로드된 아카이브에 대응합니다.
#
# 사전 준비:
#   1. rclone 설치 및 설정 완료 (upload_to_gdrive.sh 참고)
#   2. 이 스크립트의 RCLONE_REMOTE, GDRIVE_DIR 변수 수정
#
# 사용법:
#   bash script/download_from_gdrive.sh
# =============================================================================
set -euo pipefail

# ---- 설정 (수정 필요) ----
RCLONE_REMOTE="gdrive"                        # rclone config에서 설정한 remote 이름
GDRIVE_DIR="DOMINO_transfer"                  # Google Drive 상의 폴더명
DOMINO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGING_DIR="${DOMINO_ROOT}/.download_staging"

echo "=== DOMINO root: ${DOMINO_ROOT}"
echo "=== Download source: ${RCLONE_REMOTE}:${GDRIVE_DIR}/archives/"
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

# ---- 1. 아카이브 전체 다운로드 ----
echo "=== [1/2] 아카이브 다운로드 ==="
rclone copy "${RCLONE_REMOTE}:${GDRIVE_DIR}/archives/" "${STAGING_DIR}/" \
    --progress --log-level INFO
echo "  ✓ 다운로드 완료"
echo ""

# ---- 2. 아카이브 해제 ----
echo "=== [2/2] 아카이브 해제 ==="

# lerobot_*.tar.gz → data/lerobot_data/
for archive in "${STAGING_DIR}"/lerobot_*.tar.gz; do
    [ -f "${archive}" ] || continue
    task_name=$(basename "${archive}" .tar.gz)
    task_name="${task_name#lerobot_}"
    echo "  → ${task_name} 해제 중 ..."
    mkdir -p "${DOMINO_ROOT}/data/lerobot_data"
    tar -xzf "${archive}" -C "${DOMINO_ROOT}/data/lerobot_data/"
    echo "  ✓ ${task_name} 완료"
    rm -f "${archive}"
done

# assets.tar.gz → assets/
if [ -f "${STAGING_DIR}/assets.tar.gz" ]; then
    echo "  → assets 해제 중 ..."
    tar -xzf "${STAGING_DIR}/assets.tar.gz" -C "${DOMINO_ROOT}/"
    echo "  ✓ assets 완료"
    rm -f "${STAGING_DIR}/assets.tar.gz"
fi

# task_config.tar.gz → task_config/
if [ -f "${STAGING_DIR}/task_config.tar.gz" ]; then
    echo "  → task_config 해제 중 ..."
    tar -xzf "${STAGING_DIR}/task_config.tar.gz" -C "${DOMINO_ROOT}/"
    echo "  ✓ task_config 완료"
    rm -f "${STAGING_DIR}/task_config.tar.gz"
fi

# eval_result/experimental_log.txt (단일 파일, 압축 없음)
if [ -d "${STAGING_DIR}/eval_result" ]; then
    echo "  → eval_result 복사 중 ..."
    mkdir -p "${DOMINO_ROOT}/eval_result"
    cp -r "${STAGING_DIR}/eval_result/"* "${DOMINO_ROOT}/eval_result/"
    echo "  ✓ eval_result 완료"
    rm -rf "${STAGING_DIR}/eval_result"
fi

# ---- 정리 ----
rmdir "${STAGING_DIR}" 2>/dev/null || true

echo ""
echo "=========================================="
echo "  다운로드 & 해제 완료!"
echo "  DOMINO 경로: ${DOMINO_ROOT}/"
echo ""
echo "  해제된 데이터:"
echo "    data/lerobot_data/  (PUMA 캐시 제외)"
echo "    assets/"
echo "    task_config/"
echo "    eval_result/experimental_log.txt"
echo "=========================================="
