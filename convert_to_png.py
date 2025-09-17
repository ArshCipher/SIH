"""
PlantUML to PNG Converter for SIH 2025 Presentation

This script converts PlantUML diagram files to PNG images for use in presentations.
Requires plantuml.jar or online PlantUML service.
"""

import os
import subprocess
import requests
import base64
import zlib
from pathlib import Path

class PlantUMLConverter:
    def __init__(self):
        self.diagrams_dir = Path("diagrams")
        self.output_dir = Path("presentation_images")
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)
        
        # PlantUML files to convert
        self.puml_files = [
            "simple_architecture.puml",
            "ultra_simple_architecture.puml", 
            "simple_flow.puml"
        ]

    def encode_plantuml(self, plantuml_text):
        """Encode PlantUML text for online service"""
        # Remove @startuml and @enduml tags for online service
        lines = plantuml_text.strip().split('\n')
        if lines[0].startswith('@startuml'):
            lines = lines[1:]
        if lines[-1].startswith('@enduml'):
            lines = lines[:-1]
        
        clean_text = '\n'.join(lines)
        
        # Compress and encode
        compressed = zlib.compress(clean_text.encode('utf-8'))
        encoded = base64.b64encode(compressed).decode('ascii')
        
        # PlantUML URL encoding
        plantuml_alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_'
        base64_alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
        
        # Simple character replacement for PlantUML encoding
        result = encoded.translate(str.maketrans(base64_alphabet, plantuml_alphabet))
        return result

    def convert_via_online_service(self, puml_file):
        """Convert PlantUML file to PNG using online service"""
        try:
            # Read PlantUML file
            file_path = self.diagrams_dir / puml_file
            with open(file_path, 'r', encoding='utf-8') as f:
                plantuml_text = f.read()
            
            print(f"📄 Converting {puml_file}...")
            
            # Use PlantUML online service
            url = "http://www.plantuml.com/plantuml/png/"
            
            # Simple approach: use text parameter
            response = requests.post(url, data={'text': plantuml_text}, timeout=30)
            
            if response.status_code == 200:
                # Save PNG file
                output_file = self.output_dir / f"{puml_file.replace('.puml', '.png')}"
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                print(f"✅ Successfully created {output_file}")
                return True
            else:
                print(f"❌ Failed to convert {puml_file}: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error converting {puml_file}: {e}")
            return False

    def convert_via_local_plantuml(self, puml_file):
        """Convert PlantUML file to PNG using local plantuml.jar"""
        try:
            file_path = self.diagrams_dir / puml_file
            
            # Try to find plantuml.jar
            plantuml_jar = None
            possible_locations = [
                "plantuml.jar",
                "tools/plantuml.jar",
                "../plantuml.jar",
                "C:/tools/plantuml.jar"
            ]
            
            for location in possible_locations:
                if os.path.exists(location):
                    plantuml_jar = location
                    break
            
            if not plantuml_jar:
                print("⚠️ plantuml.jar not found, trying online service...")
                return self.convert_via_online_service(puml_file)
            
            # Run PlantUML
            cmd = [
                "java", "-jar", plantuml_jar,
                "-tpng",  # PNG output
                "-o", str(self.output_dir.absolute()),  # Output directory
                str(file_path.absolute())  # Input file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                output_file = self.output_dir / f"{puml_file.replace('.puml', '.png')}"
                print(f"✅ Successfully created {output_file}")
                return True
            else:
                print(f"❌ PlantUML error: {result.stderr}")
                return self.convert_via_online_service(puml_file)
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout converting {puml_file}")
            return False
        except Exception as e:
            print(f"❌ Error with local PlantUML: {e}")
            return self.convert_via_online_service(puml_file)

    def create_simple_png_alternative(self, puml_file):
        """Create a simple text-based PNG as fallback"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Read the PlantUML file to extract title
            file_path = self.diagrams_dir / puml_file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract title
            title = "Medical AI Architecture"
            for line in content.split('\n'):
                if line.strip().startswith('title'):
                    title = line.replace('title', '').strip()
                    break
            
            # Create simple diagram image
            img = Image.new('RGB', (800, 600), color='white')
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.truetype("arial.ttf", 24)
                title_font = ImageFont.truetype("arial.ttf", 32)
            except:
                font = ImageFont.load_default()
                title_font = ImageFont.load_default()
            
            # Draw title
            draw.text((50, 50), title, fill='black', font=title_font)
            
            # Draw simple architecture boxes
            if "ultra_simple" in puml_file:
                draw.rectangle([50, 150, 200, 250], outline='blue', width=2)
                draw.text((60, 190), "Rural User", fill='black', font=font)
                
                draw.rectangle([300, 150, 450, 250], outline='green', width=2)
                draw.text((310, 190), "Medical AI", fill='black', font=font)
                
                draw.rectangle([550, 150, 700, 250], outline='red', width=2)
                draw.text((560, 190), "Government", fill='black', font=font)
                
                # Draw arrows
                draw.line([200, 200, 300, 200], fill='black', width=2)
                draw.line([450, 200, 550, 200], fill='black', width=2)
            
            # Save image
            output_file = self.output_dir / f"{puml_file.replace('.puml', '.png')}"
            img.save(output_file)
            print(f"✅ Created simple alternative: {output_file}")
            return True
            
        except ImportError:
            print("❌ PIL not available for fallback images")
            return False
        except Exception as e:
            print(f"❌ Error creating fallback image: {e}")
            return False

    def convert_all_diagrams(self):
        """Convert all PlantUML diagrams to PNG"""
        print("🎨 Starting PlantUML to PNG conversion...")
        print("=" * 50)
        
        success_count = 0
        
        for puml_file in self.puml_files:
            file_path = self.diagrams_dir / puml_file
            
            if not file_path.exists():
                print(f"⚠️ File not found: {file_path}")
                continue
            
            # Try local PlantUML first, then online service
            if self.convert_via_local_plantuml(puml_file):
                success_count += 1
            elif self.create_simple_png_alternative(puml_file):
                success_count += 1
                print("  📝 Used simple fallback diagram")
        
        print("=" * 50)
        print(f"🎯 Conversion complete: {success_count}/{len(self.puml_files)} diagrams converted")
        
        if success_count > 0:
            print(f"📁 PNG files saved in: {self.output_dir.absolute()}")
            print("\n📋 Files created:")
            for png_file in self.output_dir.glob("*.png"):
                print(f"  ✅ {png_file.name}")
        
        return success_count

    def create_presentation_ready_images(self):
        """Create high-quality presentation images with descriptions"""
        
        # Create a summary document
        summary_file = self.output_dir / "diagram_descriptions.md"
        
        descriptions = {
            "simple_architecture.png": """
# Simple Architecture Diagram
**Use for:** Technical Approach slide
**Shows:** 4 main components - User Access, AI Brain, Government, Data
**Key Message:** Clean, organized system architecture
**Duration:** 30 seconds explanation
            """,
            "ultra_simple_architecture.png": """
# Ultra Simple Overview 
**Use for:** Opening slide or solution overview
**Shows:** Linear flow from Rural User to Government Systems
**Key Message:** Simple user journey with powerful backend
**Duration:** 20 seconds explanation
            """,
            "simple_flow.png": """
# Process Flow Diagram
**Use for:** Live demo explanation
**Shows:** Step-by-step user interaction with decision points
**Key Message:** Fast, intelligent medical assistance
**Duration:** 1 minute walkthrough
            """
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("# SIH 2025 - Presentation Diagrams\n\n")
            for filename, description in descriptions.items():
                f.write(description + "\n")
        
        print(f"📋 Created diagram guide: {summary_file}")

def main():
    """Main conversion function"""
    converter = PlantUMLConverter()
    
    print("🏥 SIH 2025 - PlantUML to PNG Converter")
    print("Converting diagrams for presentation...")
    
    # Convert diagrams
    success_count = converter.convert_all_diagrams()
    
    if success_count > 0:
        # Create presentation guide
        converter.create_presentation_ready_images()
        
        print("\n🎉 CONVERSION SUCCESSFUL!")
        print("💡 Your PNG diagrams are ready for SIH 2025 presentation!")
        print("\n🎯 Next steps:")
        print("1. Check the 'presentation_images' folder")
        print("2. Insert PNG files into your presentation slides")
        print("3. Use the diagram_descriptions.md for timing guidance")
        
    else:
        print("\n⚠️ No diagrams were converted successfully.")
        print("💡 Alternative options:")
        print("1. Install Java and download plantuml.jar")
        print("2. Use online PlantUML editor: http://www.plantuml.com/plantuml/")
        print("3. Use the text descriptions to create slides manually")

if __name__ == "__main__":
    main()