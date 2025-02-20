function Invoke-MoveToYear {
    param (
        [string]$sourceDir,
        [string]$year,
        [System.IO.FileInfo]$file
    )
    # Define the video file extensions
    $videoExtensions = @('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')
    $photoExtensions = @('.jpg', '.jpeg', '.heic', '.png')

    # Check if the file is a video based on its extension
    if ($videoExtensions -contains $file.Extension.ToLower()) {
        # Define the target directory for videos
        $targetDir = Join-Path -Path $sourceDir -ChildPath "$year Videos"
        Write-Output "File is a video. Target directory: $targetDir"
    } elseif ($photoExtensions -contains $file.Extension.ToLower()) {
        # Define the target directory for photos
        $targetDir = Join-Path -Path $sourceDir -ChildPath "$year Photos"
        Write-Output "File is a photo. Target directory: $targetDir"
    } else {
        $targetDir = Join-Path -Path $sourceDir -ChildPath "Metadata"
    }

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
        Remove-Item -Path $file.FullName
    } else {
        # Move the file to the destination
        Move-Item -Path $file.FullName -Destination $destination
        Write-Output "File moved successfully."
    }
}

$exifToolPath = 'C:\Program Files\exiftool-12.97_64\exiftool.exe'
# Define the source directory where the images and videos are located
$sourceDir = $pwd

# Define the video file extensions
$videoExtensions = @('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')
$photoExtensions = @('.jpg', '.jpeg', '.heic', '.png')

# Get all files in the source directory
$files = Get-ChildItem -Path $sourceDir -File -Recurse

# Loop through each file
foreach ($file in $files) {
    Write-Output "Processing file: $($file.Name)"
    # Extract the year from the file name (assuming it starts with the year)
    try {
        $exifData = & $exifToolPath -j $file.FullName
        # Convert JSON output to PowerShell object for easier processing
        $exifObject = $exifData | ConvertFrom-Json
        # Access specific EXIF properties (e.g., DateTimeOriginal for date taken)
        Write-Output "Original Time of Create: $( $exifObject.DateTimeOriginal )"
        $year = ($exifObject.DateTimeOriginal).Substring(0, 4)
        Write-Output "Year extracted: $year"
        Invoke-MoveToYear -sourceDir $sourceDir -file $file -year $year
    } catch {
        #Failed - go to Name Pattern Match
        Write-Warning "File does not have System.Photo.DateTaken, using Name Pattern Matching: $($file.Name)"
        if ($file.Name -match "(20[0-9]{2})([0][0-9]|[1][0-2])") {
            $year = $matches[1]
            Write-Output "Year extracted: $year"
            Invoke-MoveToYear -sourceDir $sourceDir -file $file -year $year
        } else {
            Write-Output "File name does not have with a valid year: $($file.Name)"
            Continue
        }
    }
}

########################### Write-Output "Files have been moved successfully."
