from fastapi import FastAPI, File, UploadFile, Response
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io
import numpy as np

app = FastAPI()

# --- KONFIGURASI MODEL CLIP ---
MODEL_NAME = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading CLIP Model: {MODEL_NAME} on {device}...")
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
print("Model Loaded Successfully!")

# --- DEFINISI KELAS DARI NOTEBOOK ---
class_labels = {
    "Falling": "a person is falling down",
    "Holding_weapon": "a person is holding a weapon, such as a gun",
    "Punching": "a person is punching another person",
    "Running": "a motion-blurred shot of someone sprinting fast",
    "Kicking": "a person is kicking another person",
    "Normal": "a person standing normally, walking peacefully, no violence" # Tambahan untuk baseline
}

# Urutan label untuk prediksi
labels_list = list(class_labels.keys())
text_prompts = list(class_labels.values())

@app.get("/")
def home():
    return {"status": "Violence Detection System Ready", "model": "CLIP-ViT-Base"}

@app.post("/detect_stream")
async def detect_stream(file: UploadFile = File(...)):
    # 1. Baca gambar
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 2. Preprocess & Prediksi (CLIP Zero-Shot)
    inputs = processor(
        text=text_prompts,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image # image-text similarity score
        probs = logits_per_image.softmax(dim=1) # konversi ke probabilitas

    # 3. Ambil Hasil
    # Pindahkan ke CPU untuk diproses numpy/python
    probs_np = probs.cpu().numpy()[0] 
    pred_idx = np.argmax(probs_np)
    pred_label = labels_list[pred_idx]
    pred_conf = float(probs_np[pred_idx])
    
    # 4. Interpretasi Status (Logic Bisnis)
    status = "SAFE"
    message = f"{pred_label} ({pred_conf:.1%})"
    
    # Mapping Label ke Status
    DANGER_CLASSES = ["Punching", "Kicking", "Holding_weapon"]
    WARNING_CLASSES = ["Falling", "Running"]
    
    if pred_label in DANGER_CLASSES:
        if pred_conf > 0.6: # Threshold agar tidak false alarm
            status = "DANGER"
            message = f"VIOLENCE: {pred_label} ({pred_conf:.0%})"
        else:
            # Jika confidence rendah, anggap warning/safe
            status = "WARNING" 
            message = f"Suspect: {pred_label}?"
            
    elif pred_label in WARNING_CLASSES:
        status = "WARNING"
        message = f"Alert: {pred_label}"
        
    else: # Normal
        status = "SAFE"
        message = "Monitoring..."

    # 5. Return Original Image
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    return Response(content=img_byte_arr.getvalue(), media_type="image/jpeg", headers={
        "X-Detection-Status": status,
        "X-Detection-Message": message,
        "X-Confidence": str(pred_conf),
        "X-Class": pred_label
    })
