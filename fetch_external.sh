#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/external"

clone_or_update() {
    local repo_url="$1"
    local target_dir="$2"
    local ref="${3:-}"

    if [[ -d "${target_dir}/.git" ]]; then
        if [[ -n "${ref}" ]]; then
            echo "Fetching ${target_dir}"
            git -C "${target_dir}" fetch --tags origin
        elif git -C "${target_dir}" symbolic-ref -q HEAD >/dev/null; then
            echo "Updating ${target_dir}"
            git -C "${target_dir}" pull --ff-only
        else
            echo "Skipping update for ${target_dir}: repository is on a detached HEAD"
        fi
    elif [[ -e "${target_dir}" ]]; then
        echo "Skipping ${target_dir}: path exists but is not a git repository"
    else
        echo "Cloning ${repo_url} -> ${target_dir}"
        git clone "${repo_url}" "${target_dir}"
    fi

    if [[ -n "${ref}" && -d "${target_dir}/.git" ]]; then
        git -C "${target_dir}" checkout "${ref}"
    fi
}

mkdir -p "${EXTERNAL_DIR}"

clone_or_update "https://github.com/xtcpete/rdd" "${EXTERNAL_DIR}/rdd" "cf5192612cb9f0f12f15089b7d544be6a2438221"
clone_or_update "https://github.com/lyp-deeplearning/LiftFeat" "${EXTERNAL_DIR}/LiftFeat"
clone_or_update "https://github.com/SebastianJanampa/LINEA" "${EXTERNAL_DIR}/LINEA"
clone_or_update "https://github.com/fraunhoferhhi/RIPE" "${EXTERNAL_DIR}/RIPE"

pip install poselib
pip install -r "${EXTERNAL_DIR}/LINEA/requirements.txt"

echo "Done."
