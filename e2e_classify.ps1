$testImg = (Get-ChildItem "E:\BrainTumor\ai-service\dataset\processed\test\glioma" -File)[0]
Write-Host "Test image: $($testImg.FullName)"

$upJson = curl.exe --silent -X POST "http://localhost:5000/api/upload" --form "image=@$($testImg.FullName)"
$up = $upJson | ConvertFrom-Json
$imageId = $up.data.image_id
Write-Host "Upload: success=$($up.success) imageId=$imageId filename=$($up.data.filename)"

if (-not $imageId) { Write-Host "ERROR: no imageId returned"; exit 1 }

# Step 1 — preprocess
$ppJson = curl.exe --silent -X POST "http://localhost:5000/api/preprocess/$imageId"
$pp = $ppJson | ConvertFrom-Json
Write-Host "Preprocess: success=$($pp.success)"

# Step 2 — classify (this is what Detect page calls)
$clJson = curl.exe --silent -X POST "http://localhost:5000/api/classify/$imageId"
$cl = $clJson | ConvertFrom-Json
Write-Host "Classify: success=$($cl.success) class=$($cl.data.predicted_class) conf=$([math]::Round($cl.data.confidence,3)) model=$($cl.data.model_used)"

if ($cl.success) {
    Write-Host ""
    Write-Host "End-to-end pipeline PASSED"
} else {
    Write-Host "ERROR: $($cl.error.message)"
    exit 1
}
