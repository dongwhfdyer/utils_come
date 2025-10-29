$repo = "C:\Users\xiajiakun\Documents\code\utils_come"
if (-not (Test-Path $repo)) { throw "Repo path not found: $repo" }
Set-Location $repo

while ($true) {
  # If using $repo: git -C $repo fetch origin
  git fetch origin

  $diffCount = [int](git rev-list HEAD...origin/main --count)
  if ($diffCount -gt 0) {
    Write-Host "Updates found, pulling..."
    # If using $repo: git -C $repo pull
    git reset --hard origin/main
  } else {
    Write-Host "No updates"
  }

  Start-Sleep -Seconds 10
}