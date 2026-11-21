# Windows counterpart of ci/conan-setup-remotes.sh: conancenter first, then the
# bfierz/conan-index recipes through the b4z remote or a pinned local clone.
# UNVERIFIED on Windows when written; mirrors the bash script.
$ErrorActionPreference = 'Stop'

$root = Resolve-Path (Join-Path $PSScriptRoot '..')
$indexDir = if ($env:VCL_CONAN_INDEX_DIR) { $env:VCL_CONAN_INDEX_DIR } else { Join-Path $root '.conan-index' }
$indexUrl = if ($env:VCL_CONAN_INDEX_URL) { $env:VCL_CONAN_INDEX_URL } else { 'https://github.com/bfierz/conan-index.git' }
$indexRef = (Get-Content (Join-Path $root 'ci/conan-index.ref') | Where-Object { $_ -notmatch '^\s*#' } | Select-Object -First 1).Trim()

function Invoke-Native {
  & $args[0] $args[1..($args.Length - 1)]
  if ($LASTEXITCODE -ne 0) { throw "Command failed: $args" }
}

# Azure Pipelines leaves `$(NAME)` in place for variables that are not defined; treat as unset.
foreach ($v in 'VCL_CONAN_B4Z_URL', 'CONAN_LOGIN_USERNAME_B4Z', 'CONAN_PASSWORD_B4Z') {
  if ([Environment]::GetEnvironmentVariable($v) -like '$(*') { Remove-Item "Env:$v" }
}

Invoke-Native conan remote add --force conancenter https://center2.conan.io
Invoke-Native conan remote update conancenter --index 0

if ($env:VCL_CONAN_B4Z_URL -and $env:CONAN_PASSWORD_B4Z) {
  Write-Host 'Using the b4z remote'
  conan remote remove vcl-conan-index 2>$null
  Invoke-Native conan remote add --force b4z $env:VCL_CONAN_B4Z_URL
  Invoke-Native conan remote login b4z $env:CONAN_LOGIN_USERNAME_B4Z -p $env:CONAN_PASSWORD_B4Z
} else {
  Write-Host "No b4z access, using conan-index at $indexRef"
  conan remote remove b4z 2>$null
  if (-not (Test-Path (Join-Path $indexDir '.git'))) {
    Invoke-Native git clone $indexUrl $indexDir
  }
  git -C $indexDir fetch --quiet origin $indexRef 2>$null
  if ($LASTEXITCODE -ne 0) { Invoke-Native git -C $indexDir fetch --quiet origin }
  Invoke-Native git -C $indexDir checkout --quiet --detach $indexRef
  Invoke-Native conan remote add --force --type=local-recipes-index vcl-conan-index $indexDir
}

Invoke-Native conan remote list
