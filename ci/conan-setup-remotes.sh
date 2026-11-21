#!/usr/bin/env bash
# Configure the Conan remotes VCL builds against: conancenter first, then the
# bfierz/conan-index recipes (docs/adr/0001-conan-with-find-package-seam.md).
#
# With b4z access (VCL_CONAN_B4Z_URL, and a read-only token in
# CONAN_LOGIN_USERNAME_B4Z / CONAN_PASSWORD_B4Z) the index is reached through the
# b4z remote. Without it, the index is cloned at the ref pinned in
# ci/conan-index.ref and registered as a local-recipes-index remote.
#
# Usage: ci/conan-setup-remotes.sh   (conan must be on PATH; rerunnable)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
index_dir="${VCL_CONAN_INDEX_DIR:-${root}/.conan-index}"
index_url="${VCL_CONAN_INDEX_URL:-https://github.com/bfierz/conan-index.git}"
index_ref="$(grep -v '^[[:space:]]*#' "${root}/ci/conan-index.ref" | tr -d '[:space:]')"

# Azure Pipelines leaves `$(NAME)` in place for variables that are not defined; treat as unset.
for v in VCL_CONAN_B4Z_URL CONAN_LOGIN_USERNAME_B4Z CONAN_PASSWORD_B4Z; do
  if [[ "${!v:-}" == '$('* ]]; then unset "$v"; fi
done

# Make sure conancenter exists and keeps the first position.
conan remote add --force conancenter https://center2.conan.io
conan remote update conancenter --index 0

if [[ -n "${VCL_CONAN_B4Z_URL:-}" && -n "${CONAN_PASSWORD_B4Z:-}" ]]; then
  echo "Using the b4z remote"
  conan remote remove vcl-conan-index >/dev/null 2>&1 || true
  conan remote add --force b4z "${VCL_CONAN_B4Z_URL}"
  conan remote login b4z "${CONAN_LOGIN_USERNAME_B4Z:?CONAN_LOGIN_USERNAME_B4Z is required with CONAN_PASSWORD_B4Z}" \
    -p "${CONAN_PASSWORD_B4Z}"
else
  echo "No b4z access, using conan-index at ${index_ref}"
  conan remote remove b4z >/dev/null 2>&1 || true
  if [[ ! -d "${index_dir}/.git" ]]; then
    git clone "${index_url}" "${index_dir}"
  fi
  git -C "${index_dir}" fetch --quiet origin "${index_ref}" 2>/dev/null || git -C "${index_dir}" fetch --quiet origin
  git -C "${index_dir}" checkout --quiet --detach "${index_ref}"
  conan remote add --force --type=local-recipes-index vcl-conan-index "${index_dir}"
fi

conan remote list
