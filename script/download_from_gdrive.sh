#!/bin/bash
# =============================================================================
# download_from_gdrive.sh
#
# 대상 서버에서 Google Drive로부터 SmolVLA 학습/평가 필수 파일을 다운로드합니다.
# Google Drive 구조: DOMINO_transfer/archives/ 안에 tar.gz 압축 파일들이 저장되어 있음
#
#   archives/
#   ├── eval_result/                  # 폴더
#   ├── assets.tar.gz                 # → assets/
#   ├── task_config.tar.gz            # → task_config/
#   ├── lerobot_adjust_bottle.tar.gz  # → data/lerobot_data/
#   ├── lerobot_beat_block_ha...      #    (태스크별 압축 파일)
#   └── ...
#
# 사전 준비:
#   1. rclone 설치: curl https://rclone.org/install.sh | sudo bash
#   2. rclone 설정: rclone config
#      → New remote → name: gdrive → Storage: Google Drive → 안내 따라 인증
#   3. DOMINO 레포 clone 완료 상태
#   4. 이 스크립트의 RCLONE_REMOTE, GDRIVE_DIR 변수 수정
#
# 사용법:
#   bash script/download_from_gdrive.sh
# =============================================================================
set -euo pipefail

# ---- 설정 (수정 필요) ----
RCLONE_REMOTE="gdrive"                        # rclone config에서 설정한 remote 이름
GDRIVE_DIR="DOMINO_transfer/archives"         # Google Drive 상의 압축 파일 폴더
DOMINO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="${DOMINO_ROOT}/.download_tmp"

# Google Drive API rate limit 대응 옵션
RCLONE_OPTS="--retries 10 --retries-sleep 30s --low-level-retries 10 --tpslimit 2 --drive-pacer-min-sleep 500ms"

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

# ---- 임시 디렉토리 생성 & 정리 트랩 ----
mkdir -p "${TMP_DIR}"
cleanup() {
    echo ""
    echo "=== 임시 파일 정리 중..."
    rm -rf "${TMP_DIR}"
    echo "  ✓ 정리 완료"
}
trap cleanup EXIT

# ---- 1. 모든 tar.gz + eval_result 한 번에 다운로드 ----
echo "=== [1/2] archives/ 전체 다운로드 (한 번의 rclone 호출) ==="
echo "  Google Drive API rate limit 방지를 위해 단일 연결로 다운로드합니다."
echo ""
rclone copy "${RCLONE_REMOTE}:${GDRIVE_DIR}/" "${TMP_DIR}/" \
    --transfers=4 --checkers=8 --progress --log-level INFO \
    ${RCLONE_OPTS}
echo ""
echo "  ✓ 다운로드 완료"

# ---- 2. 압축 해제 ----
echo ""
echo "=== [2/2] 압축 해제 ==="

# lerobot_*.tar.gz → data/lerobot_data/
echo ""
echo "--- lerobot 태스크 데이터 ---"
mkdir -p "${DOMINO_ROOT}/data/lerobot_data"
LEROBOT_COUNT=0
for archive in "${TMP_DIR}"/lerobot_*.tar.gz; do
    [ -f "${archive}" ] || continue
    fname=$(basename "${archive}")
    echo "  → ${fname} 압축 해제 중..."
    tar xzf "${archive}" -C "${DOMINO_ROOT}/data/lerobot_data/"
    LEROBOT_COUNT=$((LEROBOT_COUNT + 1))
done
echo "  ✓ lerobot 태스크 ${LEROBOT_COUNT}개 완료"

# assets.tar.gz → assets/
echo ""
echo "--- assets ---"
if [ -f "${TMP_DIR}/assets.tar.gz" ]; then
    mkdir -p "${DOMINO_ROOT}/assets"
    echo "  → assets.tar.gz 압축 해제 중..."
    tar xzf "${TMP_DIR}/assets.tar.gz" -C "${DOMINO_ROOT}/assets/"
    echo "  ✓ assets 완료"
else
    echo "  [WARN] assets.tar.gz 를 찾을 수 없습니다."
fi

# task_config.tar.gz → task_config/
echo ""
echo "--- task_config ---"
if [ -f "${TMP_DIR}/task_config.tar.gz" ]; then
    mkdir -p "${DOMINO_ROOT}/task_config"
    echo "  → task_config.tar.gz 압축 해제 중..."
    tar xzf "${TMP_DIR}/task_config.tar.gz" -C "${DOMINO_ROOT}/task_config/"
    echo "  ✓ task_config 완료"
else
    echo "  [WARN] task_config.tar.gz 를 찾을 수 없습니다."
fi

# eval_result/ → eval_result/
echo ""
echo "--- eval_result ---"
if [ -d "${TMP_DIR}/eval_result" ]; then
    mkdir -p "${DOMINO_ROOT}/eval_result"
    cp -r "${TMP_DIR}/eval_result/"* "${DOMINO_ROOT}/eval_result/"
    echo "  ✓ eval_result 완료"
else
    echo "  [WARN] eval_result/ 폴더를 찾을 수 없습니다."
fi

echo ""
echo "=========================================="
echo "  다운로드 완료!"
echo "  데이터 확인:"
echo "    ls ${DOMINO_ROOT}/data/lerobot_data/ | wc -l   # 태스크 수"
echo "    ls ${DOMINO_ROOT}/assets/                        # objects, embodiments 등"
echo "    ls ${DOMINO_ROOT}/task_config/"
echo "    ls ${DOMINO_ROOT}/eval_result/"
echo "=========================================="
