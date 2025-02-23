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
        Write-Warning "File already exists at destination: $destination. Skipping move and auto-deleting."
        $mock_bin = "$parentDir/Recycle Bin"
        Move-Item -Path $file.FullName -Destination "$( Join-Path -Path $mock_bin -ChildPath $file.Name )"
    } else {
        # Move the file to the destination
        Move-Item -Path $file.FullName -Destination $destination
        Write-Host "$( $file.FullName ) --> moved to $destination" -ForegroundColor Green
    }
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
        # Attempt to extract the year from the exif
        try {
            Write-Host "Processing file: $($file.Name)"
            $exifData = & $exifToolPath -j $file.FullName
            # Convert JSON output to PowerShell object for easier processing
            $exifObject = $exifData | ConvertFrom-Json
            # Access specific EXIF properties (e.g., DateTimeOriginal for date taken)
            Write-Host "Original Time of Create: $( $exifObject.DateTimeOriginal )"
            $year = ($exifObject.DateTimeOriginal).Substring(0, 4)
            Write-Host "Year extracted: $year"
            # Move to relevant folder
            if ($videoextensions -contains $file.Extension.ToLower()) { $targetDir = Join-Path -Path $parentDir -ChildPath "$year Videos" } 
            elseif ($photoextensions -contains $file.Extension.ToLower()) { $targetDir = Join-Path -Path $parentDir -ChildPath "$year Photos" }
            Invoke-MoveToYear -parentDir $parentDir -file $file -year $year -targetDir $targetDir
        } catch {
            # Failed - go to Name Pattern Match
            Write-Host "File does not have System.photo.DateTaken, using Name Pattern Matching: $($file.Name)" -ForegroundColor Cyan
            if ($file.Name -match "(20[0-9]{2})([0][0-9]|[1][0-2])([0-3][0-9])") {
                $year = $matches[1]
                Write-Host "Year extracted: $year"
                # Move to relevant folder
                if ($videoextensions -contains $file.Extension.ToLower()) { $targetDir = Join-Path -Path $parentDir -ChildPath "$year Videos" } 
                elseif ($photoextensions -contains $file.Extension.ToLower()) { $targetDir = Join-Path -Path $parentDir -ChildPath "$year Photos" }
                Invoke-MoveToYear -parentDir $parentDir -file $file -year $year -targetDir $targetDir
            } else {
                Write-Host "File name does not have with a valid year: $($file.FullName)"
                if ($videoextensions -contains $file.Extension.ToLower()) { $targetDir = Join-Path -Path $parentDir -ChildPath "Unclassified Videos" } 
                elseif ($photoextensions -contains $file.Extension.ToLower()) { $targetDir = Join-Path -Path $parentDir -ChildPath "Unclassified Photos" }
                else { $targetDir = Join-Path -Path $parentDir -ChildPath "Unclassified" }
                Invoke-MoveToYear -parentDir $parentDir -file $file -year $year -targetDir $targetDir
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