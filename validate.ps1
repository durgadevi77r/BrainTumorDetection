$ErrorActionPreference = "Stop"
$AI = "http://localhost:8000/api/v1"
$BE = "http://localhost:5000"

Write-Host "=== 1. Backend health ==="
$r = Invoke-RestMethod -Uri "$BE/health" -Method GET
Write-Host "Backend: success=$($r.success) status=$($r.status)"

Write-Host "=== 2. AI health ==="
$r = Invoke-RestMethod -Uri "$AI/health" -Method GET
Write-Host "AI: active_model=$($r.active_model) cnn=$($r.models_available.cnn)"

Write-Host "=== 3. AI service login ==="
$loginBody = '{"username":"admin","password":"Admin@123!"}'
$login = Invoke-RestMethod -Uri "$AI/auth/login" -Method POST -ContentType "application/json" -Body $loginBody
$tok = $login.access_token
Write-Host "Login OK: token_type=$($login.token_type)"

Write-Host "=== 4. AI predict (cnn model via curl.exe) ==="
$imgPath = "E:\BrainTumor\ai-service\dataset\processed\test\glioma"
$testImg = Get-ChildItem $imgPath -File -ErrorAction SilentlyContinue | Select-Object -First 1
if ($testImg) {
    $predJson = curl.exe -s -X POST "$AI/predict" `
        -F "image=@$($testImg.FullName);type=image/jpeg" `
        -F "model_name=cnn" `
        -F "generate_gradcam=false"
    $pred = $predJson | ConvertFrom-Json
    Write-Host "Predict: success=$($pred.success) class=$($pred.data.class) conf=$($pred.data.confidence)"
} else {
    Write-Host "SKIP: no test images at $imgPath"
}

Write-Host "=== 5. Dataset validate ==="
$dvBody = '{"min_images_per_class":1}'
$dv = Invoke-RestMethod -Uri "$AI/dataset/validate" -Method POST `
    -ContentType "application/json" `
    -Headers @{ Authorization = "Bearer $tok" } `
    -Body $dvBody
Write-Host "Dataset valid: success=$($dv.success) classes=$($dv.data.classes_found)"

Write-Host "=== 6. Dataset info ==="
try {
    $di = Invoke-RestMethod -Uri "$AI/dataset/info" -Method GET `
        -Headers @{ Authorization = "Bearer $tok" }
    Write-Host "Dataset info: total=$($di.data.total_images)"
} catch {
    Write-Host "Dataset info 404 (not yet prepared, expected)"
}

Write-Host "=== 7. Train experiments list ==="
$exp = Invoke-RestMethod -Uri "$AI/train/experiments" -Method GET `
    -Headers @{ Authorization = "Bearer $tok" }
Write-Host "Experiments: total=$($exp.total)"

Write-Host "=== 8. Models list ==="
$ml = Invoke-RestMethod -Uri "$AI/models" -Method GET `
    -Headers @{ Authorization = "Bearer $tok" }
Write-Host "Models: count=$($ml.data.Count) cache_size=$($ml.cache_stats.size)"

Write-Host "=== 9. Active model (mambavision not trained yet) ==="
try {
    $am = Invoke-RestMethod -Uri "$AI/models/active" -Method GET `
        -Headers @{ Authorization = "Bearer $tok" }
    Write-Host "Active: $($am.data.model_name) available=$($am.data.available)"
} catch {
    Write-Host "Active model 404 (mambavision not yet trained, expected)"
}

Write-Host "=== 10. Dashboard overview ==="
$dash = Invoke-RestMethod -Uri "$AI/dashboard/overview" -Method GET `
    -Headers @{ Authorization = "Bearer $tok" }
Write-Host "Dashboard: success=$($dash.success)"

Write-Host "=== 11. Dashboard system ==="
$sys = Invoke-RestMethod -Uri "$AI/dashboard/system" -Method GET `
    -Headers @{ Authorization = "Bearer $tok" }
Write-Host "System: cpu=$($sys.data.cpu_percent)%"

Write-Host "=== 12. Backend pipeline upload test ==="
$uploadImg = Get-ChildItem "E:\BrainTumor\ai-service\dataset\processed\test\glioma" -File -ErrorAction SilentlyContinue | Select-Object -First 1
if ($uploadImg) {
    $upJson = curl.exe -s -X POST "$BE/api/upload" `
        -F "image=@$($uploadImg.FullName);type=image/jpeg"
    $up = $upJson | ConvertFrom-Json
    Write-Host "Upload: success=$($up.success) imageId=$($up.data.imageId)"
} else {
    Write-Host "SKIP: no images for upload test"
}

Write-Host ""
Write-Host "=== ALL CHECKS COMPLETE ==="
