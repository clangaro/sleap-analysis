@echo off
REM Re-encode an MP4 with dense keyframes so SLEAP's MediaVideo backend can
REM seek into it without crashing. The original is left untouched; output is
REM written next to it with a "_fixed" suffix.
REM
REM Usage: scripts\reencode_video.bat path\to\video.mp4

if "%~1"=="" (
    echo Usage: reencode_video.bat path\to\video.mp4
    exit /b 1
)

set "IN=%~1"
set "OUT=%~dpn1_fixed%~x1"

ffmpeg -i "%IN%" -c:v libx264 -g 30 -keyint_min 30 -pix_fmt yuv420p -an "%OUT%"
echo.
echo Re-encoded video: %OUT%
