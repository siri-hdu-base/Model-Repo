param([string[]]$DocumentNames = @('数据生成与评估方案', '代码说明'))
# Windows 缺少配套 LibreOffice 时，用 Word 只读打开新建文档并导出供视觉检查的 PDF。
# 不打开、不保存任何用户已有的论文或数据文件。
$ErrorActionPreference = 'Stop'
$documentFolder = $PSScriptRoot
$renderFolder = Join-Path $documentFolder 'qa'
New-Item -ItemType Directory -Path $renderFolder -Force | Out-Null
$wordInstance = New-Object -ComObject Word.Application
$wordInstance.Visible = $false
$wordInstance.DisplayAlerts = 0
try {
    foreach ($documentName in $DocumentNames) {
        $inputPath = Join-Path $documentFolder ($documentName + '.docx')
        $outputPath = Join-Path $renderFolder ($documentName + '.pdf')
        $document = $wordInstance.Documents.Open($inputPath, $false, $true)
        try {
            $document.Repaginate()
            $document.ExportAsFixedFormat($outputPath, 17)
            Write-Output $outputPath
        } finally {
            $document.Close(0)
            [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($document)
        }
    }
} finally {
    $wordInstance.Quit()
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($wordInstance)
}
