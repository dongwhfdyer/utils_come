# Git pull with incremental decryption for Windows
param(
    [string]$RepoPath = "C:\User\xiajiakun\Documents\code\auto_repo"
)

Write-Host "🔄 Starting git pull with incremental decryption..." -ForegroundColor Green

# Change to repo directory
Set-Location $RepoPath

# Git fetch and pull
Write-Host "📥 Fetching and pulling from origin..." -ForegroundColor Yellow
git fetch origin
git pull origin main

# Run incremental decryption
Write-Host "🔓 Running incremental decryption..." -ForegroundColor Yellow
& "$RepoPath\com_scripts\incremental_decrypt.ps1"

Write-Host "✅ Git pull with decryption completed!" -ForegroundColor Green
