# virtual environement

tdsp2env\Scripts\Activate   

# code to re run
uvicorn app:app --reload


# sample local api testing code
Invoke-RestMethod -Uri "http://127.0.0.1:8000/run?task=hi" -Method Post -ContentType "application/json"

# to get output in json format for view also
Invoke-WebRequest -Uri "http://127.0.0.1:8000/run?task=Calculate the result of this Google Sheets formula: =SUM(ARRAY_CONSTRAIN(SEQUENCE(100, 100, 10, 8), 1, 10)" -Method Post).Content        


$boundary = [System.Guid]::NewGuid().ToString()
$LF = "`r`n"

$body = (
    "--$boundary",
    'Content-Disposition: form-data; name="question"',
    "",
    "Calculate the result of this Google Sheets formula: =SUM(ARRAY_CONSTRAIN(SEQUENCE(100, 100, 8, 8), 1, 10))",
    "--$boundary--"
) -join $LF

$response = Invoke-WebRequest `
  -Uri "http://localhost:8000/api/" `
  -Method POST `
  -ContentType "multipart/form-data; boundary=$boundary" `
  -Body $body

Write-Host $response.Content


curl.exe -X POST "http://localhost:8000/api/" `
>>   -H "Content-Type: multipart/form-data" `
>>   -F "question=Let's make sure you can write formulas in Google Sheets. Type this formula into Google Sheets. (It won't work in Excel) =SUM(ARRAY_CONSTRAIN(SEQUENCE(100, 100, 8, 8), 1, 10)) What is the result?"