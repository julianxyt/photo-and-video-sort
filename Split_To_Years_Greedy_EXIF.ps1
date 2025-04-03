function Invoke-MoveToYear {
    param (
        [string]$housingDir,
        [string]$targetDir,
        [string]$year,
        [System.IO.FileInfo]$file
    )
    # Create the target directory if it doesn't exist
    if (-not (Test-Path -Path $targetDir)) {
        Write-Host "Creating directory: $targetDir"
        New-Item -Path $targetDir -ItemType Directory | Out-Null
    }
    # Move the file to the target directory
    $destination = Join-Path -Path $targetDir -ChildPath $file.Name
    Write-Host "Moving file to: $destination"

    if (Test-Path -Path $destination) {
        Write-Warning "File already exists at destination: $destination. Recycling..."
        $moveTo = Join-Path -Path "$housingDir\Recycle Bin" -ChildPath $file.Name
        try {
            Move-Item -Path $file.FullName -Destination $moveTo -ErrorAction Stop
        } catch {
            Invoke-Remove -file $file -moveTo $moveTo -desination $destination
        }
    } else {
        # Move the file to the destination
        Move-Item -Path $file.FullName -Destination $destination
        Write-Host "$( $file.FullName ) --> $destination" -ForegroundColor Green
    }
}

function Invoke-Remove {
    param (
        [System.IO.FileInfo]$file,
        [string]$moveTo
    )
    $sourceHash = Get-FileHash -Path $file.FullName -Algorithm SHA256
    $destHash = Get-FileHash -Path $moveTo -Algorithm SHA256
    if ($sourceHash.Hash -eq $destHash.Hash) {
        Write-Output "Duplicate detected. Deleting: $($file.FullName)"
        Remove-Item -Path $file.FullName
    } else {
        Write-Output "File with same name exists but has different content. Keeping both."
        $newName = "{0}({1}){2}" -f $file.BaseName, (Get-Random), $file.Extension
        $destination = Join-Path -Path $( Split-Path -Path $moveTo ) -ChildPath $newName
        Move-Item -Path $file.FullName -Destination $destination
    }
}

function Get-TargetDir {
    param (
        [string]$year,
        [string]$type
    )
    Write-Host "Year extracted: $year"
    $childFolder = "$year $type"
    # Move to relevant folder
    $targetDir = Join-Path -Path $housingDir -ChildPath $childFolder
    return $targetDir
}

function Invoke-Sort {
    param (
        [string]$housingDir,
        [System.IO.FileInfo]$file
    )
    # Define the video file extensions
    $videoextensions = @('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')
    $photoextensions = @('.jpg', '.jpeg', '.heic', '.png', '.nef')
    $rawextensions = @('.raf')
    $livephotoextensions = @('.mp')
    $totalextensions = $videoextensions + $photoextensions + $livephotoextensions + $rawextensions
    if (-not (Test-Path -Path "$housingDir\Recycle Bin")) {
        Write-Host "Creating directory: $housingDir\Recycle Bin"
        New-Item -Path "$housingDir\Recycle Bin" -ItemType Directory | Out-Null
    }
    # Loop through each file
    foreach ($file in $files) {
        #### Guard statements for non-year files
        if (!($totalextensions -contains $file.Extension.ToLower())) {   
            $targetDir = Join-Path -Path $housingDir -ChildPath "Metadata"
            Invoke-MoveToYear -housingDir $housingDir -file $file -year $year -targetDir $targetDir
            continue
        }
        if ($livephotoextensions -contains $file.Extension.ToLower()) {         
            $targetDir = Join-Path -Path $housingDir -ChildPath "Live Photos"
            Invoke-MoveToYear -housingDir $housingDir -file $file -year $year -targetDir $targetDir
            continue
        }
        if ($file.Name -match "^Screenshot") {
            $targetDir = Join-Path -Path $housingDir -ChildPath "Screenshots"
            Invoke-MoveToYear -housingDir $housingDir -file $file -year $year -targetDir $targetDir
            continue
        }
        if ($file.Name -match "^FB|received") {
            $targetDir = Join-Path -Path $housingDir -ChildPath "Facebook"
            Invoke-MoveToYear -housingDir $housingDir -file $file -year $year -targetDir $targetDir
            continue
        }
        # Determine Photo or Video
        if ($photoextensions -contains $file.Extension.ToLower()) { 
            $type = "Photos" 
        } elseif ($videoextensions -contains $file.Extension.ToLower()) { 
            $type = "Videos" 
        } elseif ($rawextensions -contains $file.Extension.ToLower()) { 
            $type = "RAWs" 
        } else {
            $type = "Other" 
        }
        # Attempt to extract the year from the exif
        try {
            Write-Host "Processing file: $($file.Name)"
            $exifData = & $exifToolPath -j $file.FullName
            # Convert JSON output to PowerShell object for easier processing
            $exifObject = $exifData | ConvertFrom-Json
            # Access specific EXIF properties (e.g., DateTimeOriginal for date taken)
            Write-Host "Original Time of Create: $( $exifObject.DateTimeOriginal )"
            $year = ($exifObject.DateTimeOriginal).Substring(0, 4)
            $targetDir = Get-TargetDir -year $year -type $type
            Invoke-MoveToYear -housingDir $housingDir -file $file -year $year -targetDir $targetDir
        } catch {
            # Failed - go to Name Pattern Match
            Write-Host "File does not have System.photo.DateTaken, using Name Pattern Matching: $($file.Name)" -ForegroundColor Cyan
            if ($file.Name -match "(20[0-9]{2})([0][0-9]|[1][0-2])([0-3][0-9])") {
                $year = $matches[1]
                $targetDir = Get-TargetDir -year $year -type $type
                Invoke-MoveToYear -housingDir $housingDir -file $file -year $year -targetDir $targetDir
            } else {
                    Write-Host "File name does not have with a valid year: $($file.FullName)"
                    $targetDir = Get-TargetDir -year "Unclassified" -type $type
                    Invoke-MoveToYear -housingDir $housingDir -file $file -year $year -targetDir $targetDir
                # try {
                #     $year = ($file.LastWriteTime).Year
                #     $targetDir = Get-TargetDir -year $year -type $type
                #     Invoke-MoveToYear -housingDir $housingDir -file $file -year $year -targetDir $targetDir
                # } catch {
                #     Write-Host "File name does not have with a valid year: $($file.FullName)"
                #     $targetDir = Get-TargetDir -year "Unclassified" -type $type
                #     Invoke-MoveToYear -housingDir $housingDir -file $file -year $year -targetDir $targetDir
                # }
            }
        }
    }
}

$env:EXIFTOOLPATH = 'C:\Program Files\exiftool-12.97_64\exiftool.exe'
# Go to the dir where all your photos/videos are
$unsortedDir = $pwd
# Define the directory to house the new folder destinations
# $housingDir = Split-Path -Path $pwd -Parent
$housingDir = $pwd

# Get all files in the source directory
#$files = Get-ChildItem -Path $unsortedDir -File
$files = Get-ChildItem -Path $unsortedDir -File -Recurse

Invoke-Sort -files $files -housingDir $housingDir