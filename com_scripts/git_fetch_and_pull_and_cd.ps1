# Ensure we are in the repo directory
$repo = "C:\Users\xiajiakun\Documents\code\utils_come"
if (-not (Test-Path $repo)) { throw "Repo path not found: $repo" }
Set-Location $repo

git fetch origin

$diffCount = [int](git rev-list HEAD...origin/main --count)
if ($diffCount -gt 0) {
  Write-Host "Updates found, pulling..."
  git reset --hard origin/main
} else {
  Write-Host "No updates"
}