import os
from PIL import Image

# Update this path to your test data directory
test_dir = r"C:\code\unlearn\data\test_set" 

for filename in os.listdir(test_dir):
    if filename.lower().endswith(('.png', '.jpg', '.fits')):
        img_path = os.path.join(test_dir, filename)
        print(f"Opening: {filename}")
        with Image.open(img_path) as img:
            img.show()
            # Wait for input to continue to the next image
            input("Press Enter to see the next image...")