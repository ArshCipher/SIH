@echo off
title PlantUML to PNG Converter - SIH 2025
color 0A

echo.
echo ===============================================
echo    SIH 2025 - PlantUML to PNG Converter
echo ===============================================
echo.

REM Create presentation_images directory
if not exist "presentation_images" mkdir presentation_images

echo 🎨 Converting PlantUML diagrams to PNG...
echo.

REM Try Python conversion first
echo 📝 Attempting Python conversion...
python convert_to_png.py

REM Check if PNG files were created
if exist "presentation_images\*.png" (
    echo.
    echo ✅ SUCCESS! PNG files created in presentation_images folder
    echo.
    echo 📁 Files created:
    dir /b presentation_images\*.png
    echo.
    echo 🎯 Your diagrams are ready for SIH 2025 presentation!
    echo.
) else (
    echo.
    echo ⚠️ Python conversion failed. Trying alternative methods...
    echo.
    
    REM Try online PlantUML service
    echo 🌐 You can also convert online at:
    echo    http://www.plantuml.com/plantuml/
    echo.
    echo 📋 PlantUML files to convert:
    echo    diagrams\simple_architecture.puml
    echo    diagrams\ultra_simple_architecture.puml
    echo    diagrams\simple_flow.puml
    echo.
    echo 💡 Manual steps:
    echo 1. Go to http://www.plantuml.com/plantuml/
    echo 2. Copy-paste each .puml file content
    echo 3. Click "Submit" to generate PNG
    echo 4. Right-click and "Save image as" to presentation_images folder
    echo.
)

echo 📋 Presentation Tips:
echo • Use simple_architecture.png for Technical Approach slide
echo • Use ultra_simple_architecture.png for Solution Overview
echo • Use simple_flow.png for Live Demo explanation
echo.

pause