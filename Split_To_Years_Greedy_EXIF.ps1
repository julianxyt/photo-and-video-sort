function Invoke-MoveToYear {
    param (
        [string]$parentDir,
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
        $move_to = Join-Path -Path "$parentDir\Recycle Bin" -ChildPath $file.Name
        try {
            Move-Item -Path $file.FullName -Destination $move_to -ErrorAction Stop
        } catch {
            Remove-Item -Path $file.FullName
        }
    } else {
        # Move the file to the destination
        Move-Item -Path $file.FullName -Destination $destination
        Write-Host "$( $file.FullName ) --> $destination" -ForegroundColor Green
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
    $targetDir = Join-Path -Path $parentDir -ChildPath $childFolder
    return $targetDir
}

function Invoke-Sort {
    param (
        [string]$parentDir,
        [System.IO.FileInfo]$file
    )
    # Define the video file extensions
    $videoextensions = @('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')
    $photoextensions = @('.jpg', '.jpeg', '.heic', '.png')
    $livephotoextensions = @('.mp')
    $totalextensions = $videoextensions + $photoextensions + $livephotoextensions
    if (-not (Test-Path -Path "$parentDir\Recycle Bin")) {
        Write-Host "Creating directory: $parentDir\Recycle Bin"
        New-Item -Path "$parentDir\Recycle Bin" -ItemType Directory | Out-Null
    }
    # Loop through each file
    foreach ($file in $files) {
        #### Guard statements for non-year files
        if (!($totalextensions -contains $file.Extension.ToLower())) {   
            $targetDir = Join-Path -Path $parentDir -ChildPath "Metadata"
            Invoke-MoveToYear -parentDir $parentDir -file $file -year $year -targetDir $targetDir
            continue
        }
        if ($livephotoextensions -contains $file.Extension.ToLower()) {         
            $targetDir = Join-Path -Path $parentDir -ChildPath "Live Photos"
            Invoke-MoveToYear -parentDir $parentDir -file $file -year $year -targetDir $targetDir
            continue
        }
        if ($file.Name -match "^Screenshot") {
            $targetDir = Join-Path -Path $parentDir -ChildPath "Screenshots"
            Invoke-MoveToYear -parentDir $parentDir -file $file -year $year -targetDir $targetDir
            continue
        }
        if ($file.Name -match "^FB|received") {
            $targetDir = Join-Path -Path $parentDir -ChildPath "Facebook"
            Invoke-MoveToYear -parentDir $parentDir -file $file -year $year -targetDir $targetDir
            continue
        }
        # Determine Photo or Video
        if ($photoextensions -contains $file.Extension.ToLower()) { 
            $type = "Photos" 
        } elseif ($videoextensions -contains $file.Extension.ToLower()) { 
            $type = "Videos" 
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
            Invoke-MoveToYear -parentDir $parentDir -file $file -year $year -targetDir $targetDir
        } catch {
            # Failed - go to Name Pattern Match
            Write-Host "File does not have System.photo.DateTaken, using Name Pattern Matching: $($file.Name)" -ForegroundColor Cyan
            if ($file.Name -match "(20[0-9]{2})([0][0-9]|[1][0-2])([0-3][0-9])") {
                $year = $matches[1]
                $targetDir = Get-TargetDir -year $year -type $type
                Invoke-MoveToYear -parentDir $parentDir -file $file -year $year -targetDir $targetDir
            } else {
                try {
                    $year = ($file.LastWriteTime).Year
                    $targetDir = Get-TargetDir -year $year -type $type
                    Invoke-MoveToYear -parentDir $parentDir -file $file -year $year -targetDir $targetDir
                } catch {
                    Write-Host "File name does not have with a valid year: $($file.FullName)"
                    $targetDir = Get-TargetDir -year "Unclassified" -type $type
                    Invoke-MoveToYear -parentDir $parentDir -file $file -year $year -targetDir $targetDir
                }
            }
        }
    }
}

$env:EXIFTOOLPATH = 'C:\Program Files\exiftool-12.97_64\exiftool.exe'
# Define the source directory where the images and videos are located
$workingDir = $pwd
$parentDir = Split-Path -Path $pwd -Parent

# Get all files in the source directory
#$files = Get-ChildItem -Path $workingDir -File
$files = Get-ChildItem -Path $workingDir -File -Recurse

Invoke-Sort -files $files -parentDir $parentDir