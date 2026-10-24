# =============================================================================
# Nakshatra Lunar Rover — .NET SerialPort (COM7) <-> HTTP Bridge (127.0.0.1:8081)
# =============================================================================
$ErrorActionPreference = "Continue"

$port = New-Object System.IO.Ports.SerialPort "COM7", 115200, "None", 8, "One"
$port.DtrEnable = $true
$port.RtsEnable = $true
$port.ReadTimeout = 2500
$port.WriteTimeout = 1500
$port.Open()
Start-Sleep -Milliseconds 400
$port.DiscardInBuffer()
Write-Host "[COM7 Bridge] Connected to ESP32-S3 OV7670 on COM7!"

$listener = New-Object System.Net.HttpListener
$listener.Prefixes.Add("http://127.0.0.1:8081/")
$listener.Start()
Write-Host "[COM7 Bridge] Listening on http://127.0.0.1:8081/ (/capture, /status, /cmd)"

$latestJpg = $null
$latestStatus = '{"board":"ESP32-S3-PRIMARY-OV7670","camera_ok":true,"camera_pid":118}'
$lastCapTime = [DateTime]::MinValue

while ($listener.IsListening) {
  try {
    $ctx = $listener.GetContext()
    $req = $ctx.Request
    $res = $ctx.Response

    $res.Headers.Add("Access-Control-Allow-Origin", "*")
    $res.Headers.Add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    $res.Headers.Add("Access-Control-Allow-Headers", "Content-Type")
    $res.Headers.Add("Cache-Control", "no-store, no-cache, must-revalidate")

    if ($req.HttpMethod -eq "OPTIONS") {
      $res.StatusCode = 204
      $res.Close()
      continue
    }

    $path = $req.Url.AbsolutePath

    if ($path -eq "/capture") {
      $now = [DateTime]::UtcNow
      if (($null -eq $latestJpg) -or (($now - $lastCapTime).TotalMilliseconds -gt 120)) {
        $port.DiscardInBuffer()
        $port.WriteLine("CAPTURE")
        $line = $port.ReadLine().Trim()
        if ($line.StartsWith("JPG64:")) {
          $latestJpg = [Convert]::FromBase64String($line.Substring(6))
          $lastCapTime = $now
        }
      }
      if ($null -ne $latestJpg) {
        $res.StatusCode = 200
        $res.ContentType = "image/jpeg"
        $res.ContentLength64 = $latestJpg.Length
        $res.OutputStream.Write($latestJpg, 0, $latestJpg.Length)
      } else {
        $res.StatusCode = 503
      }
      $res.Close()
      continue
    }

    if ($path -eq "/status") {
      $port.DiscardInBuffer()
      $port.WriteLine("STATUS")
      $sLine = $port.ReadLine().Trim()
      if ($sLine.StartsWith("{")) {
        $latestStatus = $sLine
      }
      $buf = [Text.Encoding]::UTF8.GetBytes($latestStatus)
      $res.StatusCode = 200
      $res.ContentType = "application/json"
      $res.ContentLength64 = $buf.Length
      $res.OutputStream.Write($buf, 0, $buf.Length)
      $res.Close()
      continue
    }

    if ($path -eq "/cmd") {
      $qs = $req.QueryString
      $l = if ($qs["l"]) { $qs["l"] } else { "0.0" }
      $r = if ($qs["r"]) { $qs["r"] } else { "0.0" }
      $wp = if ($qs["wp"]) { $qs["wp"] } else { "1" }
      $m = if ($qs["mode"]) { $qs["mode"] } else { "auto" }
      $port.DiscardInBuffer()
      $port.WriteLine("CMD $l $r $wp $m")
      $sLine = $port.ReadLine().Trim()
      if ($sLine.StartsWith("{")) {
        $latestStatus = $sLine
      }
      $buf = [Text.Encoding]::UTF8.GetBytes($latestStatus)
      $res.StatusCode = 200
      $res.ContentType = "application/json"
      $res.ContentLength64 = $buf.Length
      $res.OutputStream.Write($buf, 0, $buf.Length)
      $res.Close()
      continue
    }

    $msg = [Text.Encoding]::UTF8.GetBytes("ESP32-S3 COM7 Bridge Active")
    $res.StatusCode = 200
    $res.ContentType = "text/plain"
    $res.ContentLength64 = $msg.Length
    $res.OutputStream.Write($msg, 0, $msg.Length)
    $res.Close()
  } catch {
    try { $ctx.Response.Close() } catch {}
  }
}
