

# =================================================
# NECESSARY INSTALLATION
# =================================================

"""
!pip install fastapi uvicorn whisper transformers gradio pyngrok pyyaml
!pip install ffmpeg-python
"""

# =================================================
# SETTING NGROK
# =================================================

import yaml

def get_ngrok_auth_token(yaml_path='ngrok.yml'):
    with open(yaml_path, 'r') as file:
        ngrok_config = yaml.safe_load(file)
    return ngrok_config.get('authtoken', None)

# Import necessary libraries
from pyngrok import ngrok

import os

def start_ngrok():
    ngrok_auth_token = get_ngrok_auth_token()
    if ngrok_auth_token:
        ngrok.set_auth_token(ngrok_auth_token)
    public_url = ngrok.connect(port=8000)
    print(f"ngrok public URL: {public_url}")
    return public_url

public_url = start_ngrok()


# =================================================
# CREATING FASTAPI APP
# =================================================

# app.py
from fastapi import FastAPI, File, UploadFile
import whisper
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from io import BytesIO
import base64


# !pip install git+https://github.com/openai/whisper.git

app = FastAPI()

# Load Whisper model
whisper_model = whisper.load_model("base")

# Load CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    # Save uploaded video
    video_content = await file.read()

    # Process audio with Whisper
    audio_path = "temp_audio.wav"
    with open(audio_path, "wb") as audio_file:
        audio_file.write(video_content)
    transcription = whisper_model.transcribe(audio_path)["text"]

    # Process image with CLIP
    image_path = "temp_image.jpg"
    with open(image_path, "wb") as image_file:
        image_file.write(video_content)
    image = Image.open(image_path)
    inputs = clip_processor(images=image, return_tensors="pt")
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    visual_description = probs.topk(5).values.tolist()

    return {
        "transcription": transcription,
        "visual_description": visual_description
    }
# =================================================
# RUNNING FASTAPI SERVER
# =================================================

import uvicorn
import threading

def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Run the FastAPI server in a background thread
threading.Thread(target=run_fastapi).start()


# =================================================
# CREATING INTERFACE WITH GRADIO
# =================================================

import gradio as gr
import requests

def process_video(file):
    url = "http://localhost:8000/upload/"   # upload-video
    files = {'file': file}
    response = requests.post(url, files=files)
    result = response.json()
    return result["transcription"], result["visual_description"]

# Define Gradio interface
iface = gr.Interface(
    fn=process_video,
    inputs=gr.inputs.Video(type="file"),
    outputs=[
        gr.outputs.Textbox(label="Transcription"),
        gr.outputs.Textbox(label="Visual Description")
    ],
    title="Video Upload and Analysis",
    description="Upload a video to get its transcription and visual description."
)

iface.launch(share=True)



