# Run from any directory. Requires PowerShell 5.1+ and Docker Compose v2.
[CmdletBinding()]
param(
    [ValidateSet('init', 'config', 'up', 'rebuild', 'status', 'logs', 'down', 'pull-model')]
    [string]$Action = 'up',
    [switch]$Gpu,
    [switch]$Bot,
    [switch]$LocalVault,
    [switch]$NoCache
)

$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot
$envPath = Join-Path $projectDir '.env.docker'
$examplePath = Join-Path $projectDir '.env.docker.example'
$utf8 = New-Object System.Text.UTF8Encoding($false)

if (-not (Test-Path -LiteralPath $envPath)) {
    [System.IO.File]::WriteAllText($envPath, [System.IO.File]::ReadAllText($examplePath), $utf8)
}

$envContent = [System.IO.File]::ReadAllText($envPath)
foreach ($keyName in @('VOICEFLOW_API_TOKEN', 'VOICEFLOW_BOT_API_TOKEN')) {
    $tokenMatch = [regex]::Match($envContent, ('(?m)^' + $keyName + '=([^\r\n]*)'))
    if (-not $tokenMatch.Success -or [string]::IsNullOrWhiteSpace($tokenMatch.Groups[1].Value)) {
        $tokenBytes = New-Object byte[] 32
        $random = [System.Security.Cryptography.RandomNumberGenerator]::Create()
        try { $random.GetBytes($tokenBytes) } finally { $random.Dispose() }
        $apiToken = [Convert]::ToBase64String($tokenBytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
        if ($tokenMatch.Success) {
            $envContent = [regex]::Replace($envContent, ('(?m)^' + $keyName + '=[^\r\n]*'), "$keyName=$apiToken")
        } else {
            $envContent += "`n$keyName=$apiToken`n"
        }
    }
}
[System.IO.File]::WriteAllText($envPath, $envContent, $utf8)

$modelsDir = Join-Path $projectDir 'models'
if (-not (Test-Path -LiteralPath $modelsDir)) {
    New-Item -ItemType Directory -Path $modelsDir | Out-Null
}

if ($Action -eq 'init') {
    Write-Host 'Docker configuration is ready: .env.docker. The API key is stored there.'
    exit 0
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Docker is not installed or is not available in PATH.'
}

# Compose gives shell variables precedence over --env-file. Do not accidentally
# use another project's web API key. The bot uses its separate key from env_file.
$previousToken = [Environment]::GetEnvironmentVariable('VOICEFLOW_API_TOKEN', 'Process')
Remove-Item Env:VOICEFLOW_API_TOKEN -ErrorAction SilentlyContinue
try {
    $composeArgs = @('compose', '--project-directory', $projectDir, '--env-file', $envPath,
                     '-f', (Join-Path $projectDir 'docker-compose.yml'))
    if ($Gpu) { $composeArgs += @('-f', (Join-Path $projectDir 'docker-compose.gpu.yml')) }
    if ($LocalVault) { $composeArgs += @('-f', (Join-Path $projectDir 'docker-compose.storage.yml')) }
    if ($Bot) {
        foreach ($name in @('TELEGRAM_BOT_TOKEN', 'BOT_ALLOWED_USER_IDS')) {
            $match = [regex]::Match($envContent, ('(?m)^' + $name + '=([^\r\n]*)'))
            if (-not $match.Success -or [string]::IsNullOrWhiteSpace($match.Groups[1].Value)) {
                throw "Set $name in .env.docker before using -Bot."
            }
        }
        $composeArgs += @('--profile', 'bot')
    }

    function Invoke-Compose([string[]]$CommandArgs) {
        & docker @composeArgs @CommandArgs
        if ($LASTEXITCODE -ne 0) { throw "Docker Compose failed (exit $LASTEXITCODE)." }
    }

    switch ($Action) {
        'config' { Invoke-Compose @('config', '--quiet') }
        'status' { Invoke-Compose @('ps') }
        'logs' { Invoke-Compose @('logs', '-f', '--tail', '100') }
        'down' { Invoke-Compose @('down') }
        'pull-model' {
            $modelMatch = [regex]::Match($envContent, '(?m)^OLLAMA_MODEL=([^\r\n]+)')
            if (-not $modelMatch.Success) { throw 'Set OLLAMA_MODEL in .env.docker.' }
            Invoke-Compose @('up', '-d', 'ollama')
            Invoke-Compose @('exec', '-T', 'ollama', 'ollama', 'pull', $modelMatch.Groups[1].Value.Trim())
        }
        { $_ -in 'up', 'rebuild' } {
            # Separate build/up also supports a clean rebuild without deleting volumes.
            $buildArgs = @('build')
            if ($NoCache) { $buildArgs += '--no-cache' }
            Invoke-Compose $buildArgs
            Invoke-Compose @('up', '-d', '--no-build', '--force-recreate')
            Invoke-Compose @('ps')
        }
    }
} finally {
    if ($null -eq $previousToken) {
        Remove-Item Env:VOICEFLOW_API_TOKEN -ErrorAction SilentlyContinue
    } else {
        $env:VOICEFLOW_API_TOKEN = $previousToken
    }
}
