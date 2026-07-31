$testImg = (Get-ChildItem "E:\BrainTumor\ai-service\dataset\processed\test\glioma" -File)[0]
Write-Host "Testing GLCM with: $($testImg.FullName)"
$r = curl.exe --silent -X POST "http://localhost:8000/api/v1/glcm" --form "image=@$($testImg.FullName)"
Write-Host "GLCM response: $r"
