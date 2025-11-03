# Incremental decryption for Windows - only decrypt changed .enc files
param(
    [string]$Password = "mySecretPassword123!"
)

Write-Host "🔓 Starting incremental decryption..." -ForegroundColor Green

# Find all .enc files
$encFiles = Get-ChildItem -Path . -Filter "*.enc" -Recurse | Where-Object {
    $_.FullName -notlike "*\.git\*" -and 
    $_.FullName -notlike "*\saved_scripts\*"
}

foreach ($file in $encFiles) {
    $originalFile = $file.FullName -replace "\.enc$", ""
    Write-Host "Decrypting: $($file.Name) -> $(Split-Path $originalFile -Leaf)" -ForegroundColor Yellow
    
    # Read encrypted file
    $encryptedData = [System.IO.File]::ReadAllBytes($file.FullName)
    
    # Decrypt using XOR
    $decryptedData = @()
    for ($i = 0; $i -lt $encryptedData.Length; $i++) {
        $keyChar = $Password[$i % $Password.Length]
        $decryptedByte = $encryptedData[$i] -bxor [int][char]$keyChar
        $decryptedData += $decryptedByte
    }
    
    # Write decrypted file
    [System.IO.File]::WriteAllBytes($originalFile, $decryptedData)
    
    # Remove encrypted file
    Remove-Item $file.FullName
}

Write-Host "✅ Incremental decryption completed!" -ForegroundColor Green
