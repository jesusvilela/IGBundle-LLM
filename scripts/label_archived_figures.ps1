param(
    [Parameter(Mandatory=$true)][string]$Archive,
    [Parameter(Mandatory=$true)][string]$OutputDirectory
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$root = Split-Path $PSScriptRoot -Parent
$audit = Get-Content (Join-Path $root 'docs\benchmark-results\artifact-audit.json') -Raw | ConvertFrom-Json
$labels = @{
    'figure_4_dynamics.png' = 'SYNTHETIC: exponentials + random noise; not recorded training'
    'figure_5_affinity.png' = 'SYNTHETIC: assigned random blocks; not checkpoint affinities'
    'figure_7_svd.png' = 'SYNTHETIC: prescribed power laws; not a measured SVD'
}
foreach ($name in ($labels.Keys | Sort-Object)) {
    $relative = "igbundle-llm/thesis/figures/$name"
    $source = (Resolve-Path (Join-Path $Archive ($relative.Replace('/', '\')))).Path
    $target = [IO.Path]::GetFullPath((Join-Path $OutputDirectory $name))
    if ($source -eq $target) { throw 'Refusing to overwrite the archive source' }
    $record = $audit.images | Where-Object { $_.path -eq $relative }
    if (-not $record -or (Get-FileHash $source -Algorithm SHA256).Hash -ne $record.sha256) {
        throw "Archive image fingerprint changed: $name"
    }
    $image = [Drawing.Image]::FromFile($source)
    try {
        $banner = [int]($image.Width * 0.075)
        $bitmap = New-Object Drawing.Bitmap($image.Width, ($image.Height + $banner))
        try {
            $graphics = [Drawing.Graphics]::FromImage($bitmap)
            $font = New-Object Drawing.Font('Arial', ([single]($image.Width / 65)), [Drawing.FontStyle]::Bold, [Drawing.GraphicsUnit]::Pixel)
            try {
                $graphics.Clear([Drawing.Color]::White)
                $graphics.PageUnit = [Drawing.GraphicsUnit]::Pixel
                $destination = New-Object Drawing.Rectangle(0, $banner, $image.Width, $image.Height)
                $graphics.DrawImage($image, $destination, 0, 0, $image.Width, $image.Height, [Drawing.GraphicsUnit]::Pixel)
                $graphics.FillRectangle([Drawing.Brushes]::LightYellow, 0, 0, $image.Width, $banner)
                $graphics.DrawString($labels[$name], $font, [Drawing.Brushes]::Black, 20, 15)
                $graphics.DrawString('ARCHIVAL ILLUSTRATION - see docs/export-image-audit.md', $font, [Drawing.Brushes]::Black, 20, ([single]($banner / 2)))
                $bitmap.Save($target, [Drawing.Imaging.ImageFormat]::Png)
            } finally { $font.Dispose(); $graphics.Dispose() }
        } finally { $bitmap.Dispose() }
    } finally { $image.Dispose() }
}
