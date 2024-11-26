[cmdletbinding()] Param()

$ErrorActionPreference="Stop"


Write-Verbose "downloading NetworkDirect DDK.."
Invoke-WebRequest -Uri "https://download.microsoft.com/download/5/A/E/5AEA3C34-32A1-4A70-9622-F9734E92981F/NetworkDirect_DDK.zip" -OutFile "NetworkDirect_DDK.zip"
Write-Verbose "done"

Write-Verbose "extracting NetworkDirect DDK.."
$wd=$PWD.Path; & { Add-Type -A "System.IO.Compression.FileSystem"; [IO.Compression.ZipFile]::ExtractToDirectory("$wd\NetworkDirect_DDK.zip", "$wd"); }
Write-Verbose "done"

Write-Verbose "moving NetworkDirect headers.."
move NetDirect\include\* include\windows
Write-Verbose "done"

Write-Verbose "downloading efawin files.."
git clone https://github.com/aws/efawin
Write-Verbose "done"

Write-Verbose "copying efawin files.."
xcopy /s efawin\interface\* prov\efa\src\windows\
Write-Verbose "done"
