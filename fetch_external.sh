#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/external"

clone_or_update() {
    local repo_url="$1"
    local target_dir="$2"

    if [[ -d "${target_dir}/.git" ]]; then
        echo "Updating ${target_dir}"
        git -C "${target_dir}" pull --ff-only
    elif [[ -e "${target_dir}" ]]; then
        echo "Skipping ${target_dir}: path exists but is not a git repository"
    else
        echo "Cloning ${repo_url} -> ${target_dir}"
        git clone "${repo_url}" "${target_dir}"
    fi
}

mkdir -p "${EXTERNAL_DIR}"

clone_or_update "https://github.com/xtcpete/rdd" "${EXTERNAL_DIR}/rdd"
clone_or_update "https://github.com/lyp-deeplearning/LiftFeat" "${EXTERNAL_DIR}/LiftFeat"

# Fix rdd to this version
cd external/rdd; git checkout cf5192612cb9f0f12f15089b7d544be6a2438221 

pip install poselib

echo "Done."
