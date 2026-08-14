$AI = "http://localhost:8000/api/v1"
$BE = "http://localhost:5000"
$sep = "=" * 55

Write-Host $sep
Write-Host "  BRAIN TUMOR DETECTION - FULL SYSTEM OUTPUT"
Write-Host $sep

# ---- 1. AI SERVICE HEALTH ----------------------------------------
Write-Host ""
Write-Host "[ 1 ] AI SERVICE HEALTH"
$ai = Invoke-RestMethod -Uri "$AI/health"
Write-Host "  status       : $($ai.status)"
Write-Host "  active_model : $($ai.active_model)"
Write-Host "  environment  : $($ai.environment)"
Write-Host "  python       : $($ai.python_version)"
Write-Host "  image_size   : $($ai.image_size) x $($ai.image_size)"
Write-Host "  classes      : $($ai.class_names -join ', ')"
Write-Host "  cnn_trained  : $($ai.models_available.cnn)"
Write-Host "  mambavision  : $($ai.models_available.mambavision)"

# ---- 2. BACKEND HEALTH -------------------------------------------
Write-Host ""
Write-Host "[ 2 ] BACKEND HEALTH"
$be = Invoke-RestMethod -Uri "$BE/health"
Write-Host "  status       : $($be.status)"
Write-Host "  service      : $($be.service)"
Write-Host "  version      : $($be.version)"

# ---- 3. AUTH -----------------------------------------------------
Write-Host ""
Write-Host "[ 3 ] AUTH - login as admin"
$body = '{"username":"admin","password":"Admin@123!"}'
$login = Invoke-RestMethod -Uri "$AI/auth/login" -Method POST -ContentType "application/json" -Body $body
$tok = $login.access_token
Write-Host "  token_type   : $($login.token_type)  OK"
$hdr = @{ Authorization = "Bearer $tok" }

# ---- 4. GLCM FEATURE EXTRACTION ----------------------------------
Write-Host ""
Write-Host "[ 4 ] GLCM FEATURE EXTRACTION"
$img = (Get-ChildItem "E:\BrainTumor\ai-service\dataset\processed\test\glioma" -File)[0]
Write-Host "  file         : $($img.Name)"
$glcmRaw = curl.exe --silent -X POST "$AI/glcm" --form "image=@$($img.FullName)"
$glcm = $glcmRaw | ConvertFrom-Json
Write-Host "  entropy      : $($glcm.data.entropy)"
Write-Host "  correlation  : $($glcm.data.correlation)"
Write-Host "  energy       : $($glcm.data.energy)"
Write-Host "  contrast     : $($glcm.data.contrast)"
Write-Host "  mean         : $($glcm.data.mean)"
Write-Host "  std_dev      : $($glcm.data.std_dev)"
Write-Host "  variance     : $($glcm.data.variance)"

# ---- 5. FULL BACKEND PIPELINE ------------------------------------
Write-Host ""
Write-Host "[ 5 ] FULL BACKEND PIPELINE"
$upRaw = curl.exe --silent -X POST "$BE/api/upload" --form "image=@$($img.FullName)"
$up = $upRaw | ConvertFrom-Json
$iid = $up.data.image_id
Write-Host "  upload       : success=$($up.success)  id=$iid"

$ppRaw = curl.exe --silent -X POST "$BE/api/preprocess/$iid"
$pp = $ppRaw | ConvertFrom-Json
Write-Host "  preprocess   : success=$($pp.success)  time=$($pp.data.computational_time_ms)ms"

$clRaw = curl.exe --silent -X POST "$BE/api/classify/$iid"
$cl = $clRaw | ConvertFrom-Json
Write-Host "  classify     : success=$($cl.success)"
Write-Host "  predicted    : $($cl.data.predicted_class)"
Write-Host "  confidence   : $([math]::Round($cl.data.confidence * 100, 2))%"
Write-Host "  model_used   : $($cl.data.model_used)"
Write-Host "  time         : $($cl.data.computational_time_ms) ms"

# ---- 6. RESULTS + GLCM FROM DATABASE ----------------------------
Write-Host ""
Write-Host "[ 6 ] RESULTS + GLCM FROM DATABASE"
$res = Invoke-RestMethod -Uri "$BE/api/results/$iid"
Write-Host "  pipeline_complete : $($res.data.pipeline_complete)"
$f = $res.data.features
Write-Host "  GLCM entropy      : $($f.entropy)"
Write-Host "  GLCM correlation  : $($f.correlation)"
Write-Host "  GLCM energy       : $($f.energy)"
Write-Host "  GLCM contrast     : $($f.contrast)"
Write-Host "  GLCM mean         : $($f.mean)"
Write-Host "  GLCM std_dev      : $($f.std_dev)"
Write-Host "  GLCM variance     : $($f.variance)"
Write-Host "  class             : $($res.data.result.predicted_class)"
Write-Host "  confidence        : $([math]::Round($res.data.result.confidence * 100, 2))%"
Write-Host "  model_used        : $($res.data.result.model_used)"

# ---- 7. MODEL METRICS -------------------------------------------
Write-Host ""
Write-Host "[ 7 ] MODEL METRICS"
$met = Invoke-RestMethod -Uri "$BE/api/metrics"
Write-Host "  accuracy     : $($met.data.accuracy)%"
Write-Host "  sensitivity  : $($met.data.sensitivity)%"
Write-Host "  specificity  : $($met.data.specificity)%"
Write-Host "  psnr         : $($met.data.psnr) dB"

# ---- 8. MODEL COMPARISON ----------------------------------------
Write-Host ""
Write-Host "[ 8 ] MODEL COMPARISON"
$cmp = Invoke-RestMethod -Uri "$BE/api/compare"
Write-Host "  source       : $($cmp.data.source)"
Write-Host "  models       : $($cmp.data.models -join ' | ')"
Write-Host "  accuracy     : $($cmp.data.metrics.accuracy -join ' | ')%"
Write-Host "  sensitivity  : $($cmp.data.metrics.sensitivity -join ' | ')%"
Write-Host "  specificity  : $($cmp.data.metrics.specificity -join ' | ')%"

# ---- 9. DATASET VALIDATE ----------------------------------------
Write-Host ""
Write-Host "[ 9 ] DATASET VALIDATE"
$dvBody = '{"min_images_per_class":1}'
$dv = Invoke-RestMethod -Uri "$AI/dataset/validate" -Method POST -ContentType "application/json" -Headers $hdr -Body $dvBody
Write-Host "  is_valid     : $($dv.success)"
Write-Host "  classes      : $($dv.data.classes_found -join ', ')"
Write-Host "  total_images : $($dv.data.total_images)"
$dv.data.class_counts.PSObject.Properties | ForEach-Object {
    Write-Host "    $($_.Name.PadRight(14)): $($_.Value) images"
}

# ---- 10. DASHBOARD SYSTEM ----------------------------------------
Write-Host ""
Write-Host "[ 10 ] DASHBOARD - SYSTEM"
$sys = Invoke-RestMethod -Uri "$AI/dashboard/system" -Headers $hdr
Write-Host "  cpu_percent  : $($sys.data.cpu_percent)%"
Write-Host "  ram_percent  : $($sys.data.ram_percent)%"
Write-Host "  ram_used_mb  : $([math]::Round($sys.data.ram_used_mb, 1)) MB"
Write-Host "  disk_percent : $($sys.data.disk_percent)%"
Write-Host "  gpu_available: $($sys.data.gpu_available)"
Write-Host "  uptime_s     : $([math]::Round($sys.data.uptime_seconds, 0)) s"

# ---- 11. DASHBOARD INFERENCE -------------------------------------
Write-Host ""
Write-Host "[ 11 ] DASHBOARD - INFERENCE"
$inf = Invoke-RestMethod -Uri "$AI/dashboard/inference" -Headers $hdr
Write-Host "  total_preds  : $($inf.data.total_predictions)"
Write-Host "  avg_latency  : $($inf.data.avg_latency_ms) ms"
Write-Host "  success_rate : $($inf.data.success_rate)"

Write-Host ""
Write-Host $sep
Write-Host "  DONE - open http://localhost:3000 in your browser"
Write-Host $sep
