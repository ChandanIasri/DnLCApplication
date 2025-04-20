from PIL import Image

# Load your PNG image
img = Image.open("icon.png")

# Save as .ico (you can specify multiple sizes)
img.save("icon.ico", format='ICO', sizes=[(256, 256)])
