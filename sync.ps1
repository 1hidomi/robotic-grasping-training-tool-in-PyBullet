# PowerShell script to automatically stage, commit and push changes to GitHub
param(
    [string]$Message = "Update project code and models"
)

Write-Host "Syncing changes to GitHub..." -ForegroundColor Cyan
git add .
$status = git status --porcelain
if ($status) {
    git commit -m $Message
    git push origin main
    Write-Host "Successfully synced to GitHub!" -ForegroundColor Green
} else {
    Write-Host "No changes detected. Working tree clean." -ForegroundColor Yellow
}
