function Invoke-MoveToYear {
    param (
        [string]$sourceDir,
        [string]$targetDir,
        [string]$year,
        [System.IO.FileInfo]$file
    )
    sleep 2
    # Create the target directory if it doesn't exist
    if (-not (Test-Path -Path $targetDir)) {
        Write-Output "Creating directory: $targetDir"
        New-Item -Path $targetDir -ItemType Directory | Out-Null
    }

    # Move the file to the target directory
    $destination = Join-Path -Path $targetDir -ChildPath $file.Name
    Write-Output "Moving file to: $destination"

    if (Test-Path -Path $destination) {
        Write-Output "File already exists at destination: $destination. Skipping move and auto-deleting."
        Remove-Item -Path $file.FullName -whatif 
        break
    } else {
        # Move the file to the destination
        Move-Item -Path $file.FullName -Destination $destination -whatif
        Write-Output "File moved successfully." 
    }
}

function Invoke-Sort {
    param (
        [string]$sourceDir,
        [System.IO.FileInfo]$file
    )
    # Define the video file extensions
    $videoextensions = @('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')
    $photoextensions = @('.jpg', '.jpeg', '.heic', '.png')
    $livephotoextensions = @('.mp')
    $totalextensions = $videoextensions + $photoextensions + $livephotoextensions
    # Loop through each file
    foreach ($file in $files) {
        echo "$( $file.FullName )"
        echo "$( $file.Extension.ToLower() )"
        #### Guard statements for non-year files
        if (!($totalextensions -contains $file.Extension.ToLower())) {   
            $targetDir = Join-Path -Path $sourceDir -ChildPath "Metadata"
            Invoke-MoveToYear -sourceDir $sourceDir -file $file -year $year -targetDir $targetDir
            continue
        }
        if ($livephotoextensions -contains $file.Extension.ToLower()) {         
            $targetDir = Join-Path -Path $sourceDir -ChildPath "live photos"
            Invoke-MoveToYear -sourceDir $sourceDir -file $file -year $year -targetDir $targetDir
            continue
        }
        if ($file.Name -match "^Screenshot") {
            $targetDir = Join-Path -Path $sourceDir -ChildPath "Screenshots"
            Invoke-MoveToYear -sourceDir $sourceDir -file $file -year $year -targetDir $targetDir
            continue
        }
        if ($file.Name -match "^FB|received") {
            $targetDir = Join-Path -Path $sourceDir -ChildPath "Facebook"
            Invoke-MoveToYear -sourceDir $sourceDir -file $file -year $year -targetDir $targetDir
            continue
        }
        # Attempt to extract the year from the exif
        try {
            Write-Output "Processing file: $($file.Name)"
            $exifData = & $exifToolPath -j $file.FullName
            # Convert JSON output to PowerShell object for easier processing
            $exifObject = $exifData | ConvertFrom-Json
            # Access specific EXIF properties (e.g., DateTimeOriginal for date taken)
            Write-Output "Original Time of Create: $( $exifObject.DateTimeOriginal )"
            $year = ($exifObject.DateTimeOriginal).Substring(0, 4)
            Write-Output "Year extracted: $year"
            # Move to relevant folder
            if ($videoextensions -contains $file.Extension.ToLower()) { $targetDir = Join-Path -Path $sourceDir -ChildPath "$year videos" } 
            elseif ($photoextensions -contains $file.Extension.ToLower()) { $targetDir = Join-Path -Path $sourceDir -ChildPath "$year photos" }
            Invoke-MoveToYear -sourceDir $sourceDir -file $file -year $year -targetDir $targetDir
        } catch {
            # Failed - go to Name Pattern Match
            Write-Warning "File does not have System.photo.DateTaken, using Name Pattern Matching: $($file.Name)"
            if ($file.Name -match "(20[0-9]{2})([0][0-9]|[1][0-2])([0-3][0-9])") {
                $year = $matches[1]
                Write-Output "Year extracted: $year"
                # Move to relevant folder
                if ($videoextensions -contains $file.Extension.ToLower()) { $targetDir = Join-Path -Path $sourceDir -ChildPath "$year videos" } 
                elseif ($photoextensions -contains $file.Extension.ToLower()) { $targetDir = Join-Path -Path $sourceDir -ChildPath "$year photos" }
                Invoke-MoveToYear -sourceDir $sourceDir -file $file -year $year -targetDir $targetDir
            } else {
                Write-Output "File name does not have with a valid year: $($file.FullName)"
                if ($videoextensions -contains $file.Extension.ToLower()) { $targetDir = Join-Path -Path $sourceDir -ChildPath "Unclassified videos" } 
                elseif ($photoextensions -contains $file.Extension.ToLower()) { $targetDir = Join-Path -Path $sourceDir -ChildPath "Unclassified photos" }
                else { $targetDir = Join-Path -Path $sourceDir -ChildPath "Unclassified" }
                Invoke-MoveToYear -sourceDir $sourceDir -file $file -year $year -targetDir $targetDir
            }
        }
    }
}

$env:EXIFTOOLPATH = 'C:\Program Files\exiftool-12.97_64\exiftool.exe'
# Define the source directory where the images and videos are located
$workingDir = $pwd
$sourceDir = Split-Path -Path $pwd -Parent

# Get all files in the source directory
#$files = Get-ChildItem -Path $workingDir -File
$files = Get-ChildItem -Path $workingDir -File -Recurse

Invoke-Sort -files $files -sourceDir $sourceDir