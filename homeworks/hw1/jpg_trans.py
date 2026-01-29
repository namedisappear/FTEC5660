import os
import base64
from PIL import Image
import io

folder = r"H:\desk\python\ai_agent\HWagent\pictures" + "\\"
picture_name = "1.jpg"
picture_path = folder + picture_name

# 图像预处理
def process_image(image_path, max_size=512):
    img = Image.open(image_path)
    img.thumbnail((max_size, max_size))
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


print(process_image(picture_path))
