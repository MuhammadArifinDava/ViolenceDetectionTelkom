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

# --- DEFINISI KELAS DARI NOTEBOOK (SPECIFIC & REFINED) ---
# Prompt Engineering: Tetap detail tapi spesifik per aksi
class_labels = {
    "Punching": "a photo of a person punching, hitting, or striking another person with a fist",
    "Kicking": "a photo of a person kicking or stomping on another person",
    "Weapon": "a photo of a person holding a gun, knife, pistol, or dangerous weapon",
    "Running": "a photo of a person running away fast, escaping, or sprinting",
    "Falling": "a photo of a person falling down to the ground, collapsing, or tripping",
    "Normal": "a photo of people walking, standing, sitting, or talking normally without violence"
}

# Urutan label untuk prediksi
labels_list = list(class_labels.keys())
text_prompts = list(class_labels.values())

@app.get("/")
def home():
    return {"status": "Violence Detection System Ready", "model": "CLIP-ViT-Base (Specific Actions)"}

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
    probs_np = probs.cpu().numpy()[0]
    
    # DEBUG: Print semua probabilitas ke terminal
    print("\n--- Detection Result ---")
    for label, score in zip(labels_list, probs_np):
        print(f"{label}: {score:.4f}")
        
    pred_idx = np.argmax(probs_np)
    pred_label = labels_list[pred_idx]
    pred_conf = float(probs_np[pred_idx])
    
    # 4. Interpretasi Status (Logic Bisnis yang Lebih Detail)
    status = "SAFE"
    message = f"{pred_label} ({pred_conf:.1%})"
    
    # Kategori Bahaya Tinggi
    if pred_label in ["Punching", "Kicking", "Weapon"]:
        if pred_conf > 0.50: 
            status = "DANGER"
            message = f"VIOLENCE: {pred_label} ({pred_conf:.0%})"
        else:
            status = "WARNING"
            message = f"Suspect: {pred_label} ({pred_conf:.0%})"
    
    # Kategori Peringatan
    elif pred_label in ["Running", "Falling"]:
        status = "WARNING"
        message = f"Alert: {pred_label} ({pred_conf:.0%})"
            
    else: # Normal
        status = "SAFE"
        message = "Normal Activity"

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
