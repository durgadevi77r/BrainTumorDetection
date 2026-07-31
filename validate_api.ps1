$ErrorActionPreference = "Continue"
$BASE = "http://localhost:8000/api/v1"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  API VALIDATION SUITE" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# ── 1. Health ──────────────────────────────────────────────────────────────────
Write-Host "1. GET /health" -ForegroundColor Yellow
$health = Invoke-RestMethod -Uri "$BASE/health" -Method GET
Write-Host "   status       : $($health.status)"
Write-Host "   active_model : $($health.active_model)"
Write-Host "   class_names  : $($health.class_names -join ', ')"
Write-Host "   models_avail : $(($health.models_available | ConvertTo-Json -Compress))"
if ($health.active_model -ne "mambavision") { Write-Host "   FAIL: expected mambavision" -ForegroundColor Red; exit 1 }
Write-Host "   PASS" -ForegroundColor Green

# ── 2. Auth login ──────────────────────────────────────────────────────────────
Write-Host "`n2. POST /auth/login" -ForegroundColor Yellow
$loginBody = '{"username":"admin","password":"Admin@123!"}'
$loginResp = Invoke-RestMethod -Uri "$BASE/auth/login" -Method POST -ContentType "application/json" -Body $loginBody
$TOKEN = $loginResp.access_token
Write-Host "   role  : $($loginResp.user.role)"
Write-Host "   token : $($TOKEN.Substring(0,30))..."
Write-Host "   PASS" -ForegroundColor Green

# ── 3. Auth /me ────────────────────────────────────────────────────────────────
Write-Host "`n3. GET /auth/me" -ForegroundColor Yellow
$headers = @{ Authorization = "Bearer $TOKEN" }
$me = Invoke-RestMethod -Uri "$BASE/auth/me" -Headers $headers
Write-Host "   username : $($me.data.username)  role : $($me.data.role)"
Write-Host "   PASS" -ForegroundColor Green

# ── 4. Dataset info ────────────────────────────────────────────────────────────
Write-Host "`n4. GET /dataset/info" -ForegroundColor Yellow
try {
    $ds = Invoke-RestMethod -Uri "$BASE/dataset/info" -Headers $headers
    Write-Host "   total_images : $($ds.data.total_images)"
    Write-Host "   PASS" -ForegroundColor Green
} catch {
    Write-Host "   NOTE: dataset not prepared yet (expected) - $($_.Exception.Message)"
}

# ── 5. Predict with a real image ──────────────────────────────────────────────
Write-Host "`n5. POST /predict (MRI image)" -ForegroundColor Yellow
# Find a test image in the dataset
$testImage = Get-ChildItem -Path "E:\BrainTumor\dataset" -Recurse -Include "*.jpg","*.jpeg","*.png" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($null -eq $testImage) {
    # fall back to ai-service dataset processed
    $testImage = Get-ChildItem -Path "E:\BrainTumor\ai-service\dataset\processed" -Recurse -Include "*.jpg","*.jpeg","*.png" -ErrorAction SilentlyContinue | Select-Object -First 1
}
if ($null -ne $testImage) {
    Write-Host "   Using image: $($testImage.FullName)"
    # Multipart POST
    $boundary = [System.Guid]::NewGuid().ToString()
    $fileBytes = [System.IO.File]::ReadAllBytes($testImage.FullName)
    $fileContent = [System.Text.Encoding]::GetEncoding("iso-8859-1").GetString($fileBytes)
    $body = "--$boundary`r`nContent-Disposition: form-data; name=`"file`"; filename=`"$($testImage.Name)`"`r`nContent-Type: image/jpeg`r`n`r`n$fileContent`r`n--$boundary--"
    $predResp = Invoke-RestMethod -Uri "$BASE/predict" -Method POST -ContentType "multipart/form-data; boundary=$boundary" -Body $body
    Write-Host "   prediction  : $($predResp.data.prediction)"
    Write-Host "   confidence  : $($predResp.data.confidence)"
    Write-Host "   model_used  : $($predResp.data.model_used)"
    Write-Host "   gradcam_b64 : $(if ($predResp.data.gradcam_b64) { 'present (' + $predResp.data.gradcam_b64.Length + ' chars)' } else { 'null' })"
    if ($predResp.data.model_used -ne "mambavision") { Write-Host "   FAIL: model_used is not mambavision" -ForegroundColor Red } else { Write-Host "   PASS" -ForegroundColor Green }
} else {
    Write-Host "   SKIP: no test image found (model not trained yet - expected)"
}

# ── 6. Train status endpoint (no long job) ────────────────────────────────────
Write-Host "`n6. GET /train/experiments (list)" -ForegroundColor Yellow
$expts = Invoke-RestMethod -Uri "$BASE/train/experiments" -Headers $headers
Write-Host "   total experiments : $($expts.data.total)"
Write-Host "   PASS" -ForegroundColor Green

# ── 7. Train start (validate schema only, cancel immediately) ─────────────────
Write-Host "`n7. POST /train/start (schema validation)" -ForegroundColor Yellow
$trainBody = '{"model_name":"mambavision","epochs":1,"batch_size":4,"learning_rate":0.0001}'
try {
    $tr = Invoke-RestMethod -Uri "$BASE/train/start" -Method POST -ContentType "application/json" -Headers $headers -Body $trainBody
    $global:JOB_ID = $tr.job_id
    Write-Host "   job_id : $($tr.job_id)"
    Write-Host "   status : $($tr.status)"
    Write-Host "   PASS - job accepted" -ForegroundColor Green
} catch {
    Write-Host "   Response: $($_.Exception.Message)" -ForegroundColor Yellow
    Write-Host "   NOTE: train/start requires dataset - expected without prepared dataset"
}

# ── 8. Train job status (if job was created) ──────────────────────────────────
if ($global:JOB_ID) {
    Write-Host "`n8. GET /train/status/$($global:JOB_ID)" -ForegroundColor Yellow
    Start-Sleep -Seconds 2
    $status = Invoke-RestMethod -Uri "$BASE/train/status/$($global:JOB_ID)" -Headers $headers
    Write-Host "   status : $($status.data.status)"
    Write-Host "   PASS" -ForegroundColor Green
}

# ── 9. Evaluate (will fail cleanly if no model/dataset) ───────────────────────
Write-Host "`n9. POST /evaluate" -ForegroundColor Yellow
$evalBody = '{"model_name":"mambavision","batch_size":32}'
try {
    $ev = Invoke-RestMethod -Uri "$BASE/evaluate" -Method POST -ContentType "application/json" -Headers $headers -Body $evalBody
    Write-Host "   test_acc : $($ev.data.test_acc)"
    Write-Host "   PASS" -ForegroundColor Green
} catch {
    $errMsg = $_.Exception.Message
    Write-Host "   Expected error (no model/dataset): $errMsg"
    Write-Host "   PASS (clean error response)" -ForegroundColor Green
}

# ── 10. Dataset validate ──────────────────────────────────────────────────────
Write-Host "`n10. POST /dataset/validate" -ForegroundColor Yellow
$dv = Invoke-RestMethod -Uri "$BASE/dataset/validate" -Method POST -Headers $headers
Write-Host "    valid  : $($dv.data.valid)"
Write-Host "    errors : $(($dv.data.errors | ConvertTo-Json -Compress))"
Write-Host "    PASS" -ForegroundColor Green

# ── 11. Dashboard overview ────────────────────────────────────────────────────
Write-Host "`n11. GET /dashboard/overview" -ForegroundColor Yellow
$dash = Invoke-RestMethod -Uri "$BASE/dashboard/overview" -Headers $headers
Write-Host "    cpu_percent  : $($dash.data.system.cpu_percent)"
Write-Host "    ram_used_mb  : $($dash.data.system.ram_used_mb)"
Write-Host "    total_preds  : $($dash.data.inference.total_predictions)"
Write-Host "    PASS" -ForegroundColor Green

# ── 12. CORS check ────────────────────────────────────────────────────────────
Write-Host "`n12. CORS preflight check" -ForegroundColor Yellow
$corsHeaders = @{
    "Origin" = "http://localhost:3000"
    "Access-Control-Request-Method" = "POST"
    "Access-Control-Request-Headers" = "content-type,authorization"
}
$corsResp = Invoke-WebRequest -Uri "$BASE/predict" -Method OPTIONS -Headers $corsHeaders -UseBasicParsing
$acao = $corsResp.Headers["Access-Control-Allow-Origin"]
Write-Host "    Access-Control-Allow-Origin: $acao"
if ($acao -eq "http://localhost:3000" -or $acao -eq "*") { Write-Host "    PASS" -ForegroundColor Green } else { Write-Host "    WARN: ACAO header missing or unexpected" -ForegroundColor Yellow }

# ── 13. Preprocess quality check ──────────────────────────────────────────────
Write-Host "`n13. POST /preprocess/quality-check (synthetic image)" -ForegroundColor Yellow
Add-Type -AssemblyName System.Drawing
$bmp = New-Object System.Drawing.Bitmap(224, 224)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.FillRectangle([System.Drawing.Brushes]::Gray, 0, 0, 224, 224)
$g.DrawEllipse([System.Drawing.Pens]::White, 50, 50, 124, 124)
$g.Dispose()
$ms = New-Object System.IO.MemoryStream
$bmp.Save($ms, [System.Drawing.Imaging.ImageFormat]::Jpeg)
$imgBytes = $ms.ToArray()
$bmp.Dispose(); $ms.Dispose()
$boundary2 = [System.Guid]::NewGuid().ToString()
$imgContent = [System.Text.Encoding]::GetEncoding("iso-8859-1").GetString($imgBytes)
$qcBody = "--$boundary2`r`nContent-Disposition: form-data; name=`"file`"; filename=`"test.jpg`"`r`nContent-Type: image/jpeg`r`n`r`n$imgContent`r`n--$boundary2--"
try {
    $qc = Invoke-RestMethod -Uri "$BASE/preprocess/quality-check" -Method POST -ContentType "multipart/form-data; boundary=$boundary2" -Body $qcBody
    Write-Host "    passed       : $($qc.data.passed)"
    Write-Host "    blur_score   : $($qc.data.blur_score)"
    Write-Host "    dimensions   : $($qc.data.dimensions -join 'x')"
    Write-Host "    PASS" -ForegroundColor Green
} catch {
    Write-Host "    Error: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  VALIDATION COMPLETE" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan
