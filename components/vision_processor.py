import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from PIL import Image
import base64
from io import BytesIO

def encode_image(image_path):
    """Encode image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_image_with_vision(image_path: str, api_key: str) -> str:
    """Use Gemini Vision to describe the image."""
    model = ChatGoogleGenerativeAI(model="gemini-pro-vision", google_api_key=api_key)
    image_data = encode_image(image_path)
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Describe this image in detail, including any text visible."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
        ]
    )
    response = model.invoke([message])
    return response.content
