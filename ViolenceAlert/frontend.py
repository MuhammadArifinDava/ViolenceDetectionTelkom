import streamlit as st
import requests
from PIL import Image
import io
import time
from streamlit_extras.stylable_container import stylable_container

# Konfigurasi Halaman
st.set_page_config(
    page_title="🛡️ Sentinel AI - Violence Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    
    * { font-family: 'Roboto', sans-serif; }
    
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
    }
    
    h1 {
        background: linear-gradient(90deg, #ff00cc, #333399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    .status-badge {
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        font-size: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    .status-safe {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        border: 2px solid #96c93d;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #f093fb 0%, #fee140 100%);
        color: white;
        border: 2px solid #fee140;
    }
    
    .status-danger {
        background: linear-gradient(135deg, #cb2d3e, #ef473a);
        color: white;
        animation: pulse 1s infinite;
        border: 2px solid #ff0000;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Sentinel AI - Violence Detection System")
st.markdown("### Real-time CCTV Monitoring & Zero-Shot Classification")

# Sidebar
st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("---")
st.sidebar.markdown("#### Model Info")
st.sidebar.info("Model: CLIP (ViT-Base)")
st.sidebar.markdown("**Detected Classes:**")
st.sidebar.markdown("- 👊 Punching\n- 🦶 Kicking\n- 🔫 Weapon\n- 🏃 Running\n- 🤸 Falling\n- 🚶 Normal")

st.sidebar.markdown("#### Input Source")
input_source = st.sidebar.radio("Select Input:", ["Webcam", "Upload Video"])

if 'streaming' not in st.session_state:
    st.session_state.streaming = False

if input_source == "Webcam":
    col_btn1, col_btn2 = st.sidebar.columns(2)
    with col_btn1:
        with stylable_container(
            key="start_btn",
            css_styles="""
                button {
                    background-color: #00b09b;
                    color: white;
                }
                """
        ):
            if st.button("▶ START", use_container_width=True):
                st.session_state.streaming = True
                st.rerun()
    with col_btn2:
        with stylable_container(
            key="stop_btn",
            css_styles="""
                button {
                    background-color: #ef473a;
                    color: white;
                }
                """
        ):
            if st.button("⏹ STOP", use_container_width=True):
                st.session_state.streaming = False
                st.rerun()
else:
    # Logic untuk Upload Video
    uploaded_file = st.sidebar.file_uploader("Upload Video File (mp4/avi)", type=['mp4', 'avi', 'mov'])
    if uploaded_file:
        st.session_state.video_file = uploaded_file
    else:
        st.session_state.streaming = False # Stop jika tidak ada file

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Status")
status_ph = st.sidebar.empty()
conf_ph = st.sidebar.empty()

# Layout
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card"><h4>📹 Live Feed</h4></div>', unsafe_allow_html=True)
    cam_placeholder = st.empty()

with col2:
    st.markdown('<div class="card"><h4>🤖 AI Analysis</h4></div>', unsafe_allow_html=True)
    result_placeholder = st.empty()
    alert_placeholder = st.empty()

# Main Loop
if st.session_state.streaming or (input_source == "Upload Video" and 'video_file' in st.session_state):
    import cv2
    import numpy as np
    import concurrent.futures
    import tempfile
    
    if input_source == "Webcam":
        cap = cv2.VideoCapture(0)
        # Set resolusi kamera agar tidak terlalu besar (beban di Streamlit)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        # Handle Uploaded Video
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(st.session_state.video_file.read())
        cap = cv2.VideoCapture(tfile.name)
    
    if not cap.isOpened():
        st.error("❌ Camera not found!")
    else:
        # FPS Control Variables
        inference_interval = 0.5 # Deteksi setiap 0.5 detik
        last_inference_time = 0
        
        # Placeholder & Executor
        last_status = "SAFE"
        last_message = "Safe"
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if input_source == "Upload Video":
                     st.info("End of Video")
                     break
                else:
                     st.warning("Failed to read frame")
                     break
            
            # 1. Resize Frame untuk Display (Kunci anti-lag di Streamlit Loop)
            # Mengirim gambar 1080p lewat websocket Streamlit itu berat. 
            # Kita resize ke 640px lebar agar ringan.
            display_frame = cv2.resize(frame, (640, 480))
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            cam_placeholder.image(frame_rgb, channels="RGB", width=640)
            
            # 2. Logic AI Inference (Asynchronous / Non-Blocking)
            current_time = time.time()
            if current_time - last_inference_time > inference_interval:
                # Cek apakah request sebelumnya sudah selesai (agar tidak menumpuk)
                if future is None or future.done():
                    last_inference_time = current_time
                    
                    # Ambil hasil dari future sebelumnya jika ada
                    if future and future.done():
                        try:
                            result = future.result()
                            if result:
                                status, message, img_content = result
                                
                                # Update UI Hasil (Hanya update jika ada hasil baru)
                                if img_content:
                                    result_img = Image.open(io.BytesIO(img_content))
                                    result_placeholder.image(result_img, caption=f"AI Analyzed ({status})", width=640)
                                
                                if status == "DANGER":
                                    alert_placeholder.markdown(f"""
                                        <div class="status-badge status-danger">
                                            🚨 {message} 🚨
                                        </div>
                                    """, unsafe_allow_html=True)
                                    status_ph.error("VIOLENCE DETECTED")
                                elif status == "WARNING":
                                    alert_placeholder.markdown(f"""
                                        <div class="status-badge status-warning">
                                            ⚠️ {message}
                                        </div>
                                    """, unsafe_allow_html=True)
                                    status_ph.warning("WARNING")
                                else:
                                    alert_placeholder.markdown(f"""
                                        <div class="status-badge status-safe">
                                            ✅ {message}
                                        </div>
                                    """, unsafe_allow_html=True)
                                    status_ph.success("SAFE")
                        except Exception as e:
                            print(f"Async Error: {e}")

                    # Siapkan frame untuk request baru (Resize 224x224 untuk VGG)
                    frame_ai = cv2.resize(frame, (224, 224))
                    _, buffer = cv2.imencode('.jpg', frame_ai)
                    frame_bytes = io.BytesIO(buffer).getvalue()
                    
                    # Fungsi untuk dijalankan di thread terpisah
                    def send_frame(img_bytes):
                        try:
                            files = {"file": ("frame.jpg", img_bytes, "image/jpeg")}
                            response = requests.post("http://127.0.0.1:8001/detect_stream", files=files, timeout=2.0)
                            if response.status_code == 200:
                                return (
                                    response.headers.get("X-Detection-Status", "SAFE"),
                                    response.headers.get("X-Detection-Message", "Safe"),
                                    response.content
                                )
                        except Exception:
                            return None
                        return None

                    # Submit ke executor
                    future = executor.submit(send_frame, frame_bytes)
            
            # Sedikit sleep untuk melepas resource CPU
            time.sleep(0.01)
        
        cap.release()
        executor.shutdown(wait=False) 
else:
    cam_placeholder.info("System Standby. Press START to monitor.")
    result_placeholder.image("https://via.placeholder.com/640x480/000000/ffffff?text=AI+Ready", use_container_width=True)

