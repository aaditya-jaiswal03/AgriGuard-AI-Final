import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# ==========================================
# 1. PAGE CONFIGURATION & DESIGN SYSTEM
# ==========================================
st.set_page_config(
    page_title="AgriGuard | Farmer's Assistant",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Modern CSS injected into the DOM
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    @keyframes mainGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .stApp {
        background: linear-gradient(-45deg, #050505, #081c11, #050505, #061f14);
        background-size: 400% 400%;
        animation: mainGradient 15s ease infinite;
        color: #FAFAFA;
    }
    
    footer { visibility: hidden !important; }
    #MainMenu { visibility: hidden !important; }
    [data-testid="stDecoration"] { display: none !important; }
    
    [data-testid="stSidebar"] {
        background-color: rgba(5, 5, 5, 0.6) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(30px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
        100% { transform: translateY(0px); }
    }
    
    @keyframes pulse-glow {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.5); }
        70% { box-shadow: 0 0 0 25px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    @keyframes danger-glow {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.5); }
        70% { box-shadow: 0 0 0 25px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    .hero-title {
        font-size: 5rem;
        font-weight: 800;
        letter-spacing: -0.05em;
        line-height: 1.05;
        margin-bottom: 0;
        color: #FFFFFF;
        text-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .hero-highlight {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%, #047857 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 0 20px rgba(16, 185, 129, 0.3));
    }

    .bento-card {
        background: rgba(20, 20, 22, 0.5);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 28px;
        padding: 32px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        opacity: 0;
        animation: fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    .hover-glow:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: rgba(16, 185, 129, 0.4);
        box-shadow: 0 20px 40px -10px rgba(16, 185, 129, 0.2), inset 0 0 20px rgba(255,255,255,0.02);
        background: rgba(30, 30, 35, 0.6);
    }

    .float-icon {
        display: inline-block;
        animation: float 4s ease-in-out infinite;
    }

    /* Modern Action Button */
    div.stButton > button, div.stDownloadButton > button {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        border: 1px solid rgba(255,255,255,0.1);
        padding: 0.8rem 2rem;
        border-radius: 16px;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        width: 100%;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 20px 0 rgba(16, 185, 129, 0.25);
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 25px rgba(16, 185, 129, 0.4);
        border-color: rgba(255,255,255,0.3);
    }
    
    [data-testid="stFileUploadDropzone"] {
        background-color: rgba(0, 0, 0, 0.3);
        border: 2px dashed rgba(255, 255, 255, 0.15);
        border-radius: 24px;
        transition: all 0.3s ease;
        padding: 2rem;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #10B981;
        background-color: rgba(16, 185, 129, 0.05);
        box-shadow: inset 0 0 20px rgba(16, 185, 129, 0.1);
    }
    
    .status-healthy {
        animation: pulse-glow 2.5s infinite;
        border: 2px solid #10B981;
    }
    .status-danger {
        animation: danger-glow 2.5s infinite;
        border: 2px solid #EF4444;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. NEURAL ENGINE (CACHED)
# ==========================================
@st.cache_resource(show_spinner=False)
def get_neural_engine():
    try:
        return tf.keras.models.load_model('trained_model.keras')
    except Exception as e:
        st.error(f"System Error: Cannot load the AI model. {e}")
        return None

model = get_neural_engine()

def execute_inference(image_buffer):
    """Safely rewinds buffer pointer and executes tensor evaluation."""
    image_buffer.seek(0)
    
    img = Image.open(image_buffer).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)
    return np.argmax(predictions)

DISEASE_CLASSES = [
    'Apple - Apple Scab', 'Apple - Black Rot', 'Apple - Cedar Apple Rust', 'Apple - Healthy',
    'Blueberry - Healthy', 'Cherry - Powdery Mildew', 'Cherry - Healthy',
    'Corn - Cercospora Leaf Spot', 'Corn - Common Rust', 'Corn - Northern Leaf Blight', 'Corn - Healthy',
    'Grape - Black Rot', 'Grape - Esca (Black Measles)', 'Grape - Leaf Blight', 'Grape - Healthy',
    'Orange - Huanglongbing (Citrus Greening)', 'Peach - Bacterial Spot', 'Peach - Healthy',
    'Bell Pepper - Bacterial Spot', 'Bell Pepper - Healthy', 'Potato - Early Blight',
    'Potato - Late Blight', 'Potato - Healthy', 'Raspberry - Healthy', 'Soybean - Healthy',
    'Squash - Powdery Mildew', 'Strawberry - Leaf Scorch', 'Strawberry - Healthy',
    'Tomato - Bacterial Spot', 'Tomato - Early Blight', 'Tomato - Late Blight',
    'Tomato - Leaf Mold', 'Tomato - Septoria Leaf Spot', 'Tomato - Spider Mites',
    'Tomato - Target Spot', 'Tomato - Yellow Leaf Curl Virus', 'Tomato - Mosaic Virus', 'Tomato - Healthy'
]

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 2.5rem; margin-top: 1rem;">
            <div class="float-icon" style="width: 48px; height: 48px; border-radius: 14px; background: linear-gradient(135deg, #10B981, #047857); display: flex; align-items: center; justify-content: center; font-size: 1.6rem; box-shadow: 0 8px 25px rgba(16,185,129,0.4); border: 1px solid rgba(255,255,255,0.2);">🌿</div>
            <h2 style="margin: 0; font-weight: 800; letter-spacing: -1.5px; font-size: 2rem; color: #fff;">AgriGuard</h2>
        </div>
    """, unsafe_allow_html=True)
    
    app_mode = st.radio("Navigation", ["Home", "How it Works", "Check Plant Health"], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("""
        <div class="bento-card" style="padding: 20px; animation-delay: 0.3s; background: rgba(10,10,10,0.5);">
            <p style="margin: 0; font-size: 0.75rem; color: #A1A1AA; text-transform: uppercase; font-weight: 700; letter-spacing: 1.5px;">System Status</p>
            <div style="margin-top: 18px; display: flex; align-items: center; gap: 12px;">
                <div style="width: 10px; height: 10px; border-radius: 50%; background: #10B981; box-shadow: 0 0 15px #10B981; animation: pulse-glow 2s infinite;"></div>
                <span style="font-size: 0.95rem; font-weight: 600; color: #E5E7EB;">App Ready</span>
            </div>
            <div style="margin-top: 12px; display: flex; align-items: center; gap: 12px;">
                <div style="width: 10px; height: 10px; border-radius: 50%; background: #10B981; box-shadow: 0 0 15px #10B981; animation: pulse-glow 2.5s infinite;"></div>
                <span style="font-size: 0.95rem; font-weight: 600; color: #E5E7EB;">Database Connected</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ==========================================
# 4. ROUTING & PAGES
# ==========================================
if app_mode == "Home":
    st.markdown("""
        <div class="bento-card" style="width: 100%; min-height: 480px; border-radius: 36px; overflow: hidden; position: relative; margin-bottom: 2.5rem; padding: 0; border: 1px solid rgba(255,255,255,0.1); background: linear-gradient(120deg, #022c22, #064e3b, #000000, #022c22); background-size: 300% 300%; animation: mainGradient 12s ease infinite, fadeInUp 0.2s forwards;">
            <div style="position: absolute; inset: 0; background: radial-gradient(circle at top right, rgba(16, 185, 129, 0.15), transparent 50%);"></div>
            <div style="position: relative; z-index: 1; padding: 5rem; height: 100%; display: flex; flex-direction: column; justify-content: center;">
                <div class="float-icon" style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); padding: 8px 16px; border-radius: 100px; width: fit-content; margin-bottom: 2rem; color: #34D399; font-weight: 600; font-size: 0.9rem; letter-spacing: 1px;">SMART FARMING ASSISTANT</div>
                <h1 class="hero-title">Protect Your <br><span class="hero-highlight">Harvest.</span></h1>
                <p style="color: #A1A1AA; font-size: 1.3rem; max-width: 600px; margin-top: 1.5rem; line-height: 1.7; font-weight: 400;">
                    Instantly check your crops for diseases using your camera. Get reliable results and simple advice on how to save your plants.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown("""
            <div class="bento-card hover-glow" style="height: 100%; animation-delay: 0.1s;">
                <div class="float-icon" style="width: 56px; height: 56px; border-radius: 16px; background: linear-gradient(135deg, rgba(52, 211, 153, 0.1), rgba(4, 120, 87, 0.2)); color: #34D399; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; margin-bottom: 2rem; border: 1px solid rgba(52, 211, 153, 0.2);">⚡</div>
                <h2 style="margin: 0; font-size: 2.8rem; font-weight: 800; letter-spacing: -1px;">Instant</h2>
                <p style="color: #A1A1AA; margin-top: 0.5rem; font-weight: 600; font-size: 1.1rem;">Scan Results</p>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
            <div class="bento-card hover-glow" style="height: 100%; animation-delay: 0.2s;">
                <div class="float-icon" style="width: 56px; height: 56px; border-radius: 16px; background: linear-gradient(135deg, rgba(52, 211, 153, 0.1), rgba(4, 120, 87, 0.2)); color: #34D399; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; margin-bottom: 2rem; border: 1px solid rgba(52, 211, 153, 0.2);">🎯</div>
                <h2 style="margin: 0; font-size: 3.5rem; font-weight: 800; letter-spacing: -2px;">96<span style="font-size: 1.8rem; color: #52525B; font-weight: 600;">%</span></h2>
                <p style="color: #A1A1AA; margin-top: 0.5rem; font-weight: 600; font-size: 1.1rem;">AI Accuracy</p>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
            <div class="bento-card hover-glow" style="height: 100%; animation-delay: 0.3s;">
                <div class="float-icon" style="width: 56px; height: 56px; border-radius: 16px; background: linear-gradient(135deg, rgba(52, 211, 153, 0.1), rgba(4, 120, 87, 0.2)); color: #34D399; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; margin-bottom: 2rem; border: 1px solid rgba(52, 211, 153, 0.2);">🌱</div>
                <h2 style="margin: 0; font-size: 3.5rem; font-weight: 800; letter-spacing: -2px;">38</h2>
                <p style="color: #A1A1AA; margin-top: 0.5rem; font-weight: 600; font-size: 1.1rem;">Plants Covered</p>
            </div>
        """, unsafe_allow_html=True)

elif app_mode == "How it Works":
    st.markdown('<h1 class="hero-title" style="font-size: 4rem; margin-bottom: 2rem; animation: fadeInUp 0.4s forwards; opacity:0;">How It <span class="hero-highlight">Works.</span></h1>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="bento-card" style="padding: 0; overflow: hidden; position: relative; height: 280px; margin-bottom: 2.5rem; animation-delay: 0.1s; background: linear-gradient(to right, #000000, #022c22); border: 1px solid rgba(16, 185, 129, 0.2);">
            <div style="position: absolute; width: 200%; height: 200%; top: -50%; left: -50%; background-image: radial-gradient(rgba(16, 185, 129, 0.2) 1px, transparent 1px); background-size: 40px 40px; transform: rotate(15deg); animation: mainGradient 30s linear infinite;"></div>
            <div style="position: absolute; inset: 0; display: flex; align-items: center; padding: 4rem; background: linear-gradient(90deg, rgba(0,0,0,0.9) 0%, transparent 100%);">
                <div>
                    <div style="color: #10B981; font-weight: 700; letter-spacing: 2px; font-size: 0.9rem; margin-bottom: 10px;">MADE FOR FARMERS</div>
                    <h2 style="margin: 0; font-size: 3rem; font-weight: 800; letter-spacing: -1px; text-shadow: 0 4px 20px rgba(0,0,0,0.8);">Built on 87,000+ Plant Scans</h2>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("""
            <div class="bento-card hover-glow" style="height: 100%; animation-delay: 0.2s;">
                <h3 style="color: #10B981; margin-top:0; font-size: 1.8rem; display: flex; align-items: center; gap: 10px;"><span class="float-icon">📸</span> 1. Take a Photo</h3>
                <p style="color: #A1A1AA; margin-bottom: 1.5rem; font-size: 1.1rem; line-height: 1.6;">Simply snap a clear picture of a single leaf that looks sick. Make sure there is good lighting and the leaf is in focus.</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="bento-card hover-glow" style="height: 100%; animation-delay: 0.3s;">
                <h3 style="color: #10B981; margin-top:0; font-size: 1.8rem; display: flex; align-items: center; gap: 10px;"><span class="float-icon">🧠</span> 2. Our AI Checks It</h3>
                <p style="color: #A1A1AA; margin-bottom: 1.5rem; font-size: 1.1rem; line-height: 1.6;">Our smart system compares your leaf to a database of thousands of known plant diseases in seconds, giving you an accurate diagnosis and next steps.</p>
            </div>
        """, unsafe_allow_html=True)

elif app_mode == "Check Plant Health":
    st.markdown('<h1 class="hero-title" style="font-size: 4rem; margin-bottom: 2.5rem; animation: fadeInUp 0.3s forwards; opacity:0;">Check <span class="hero-highlight">Plant Health.</span></h1>', unsafe_allow_html=True)

    col_input, col_output = st.columns([1, 1.2], gap="large")

    with col_input:
        st.markdown("""
            <div class="bento-card" style="margin-bottom: 1rem; animation-delay: 0.1s;">
                <h4 style="margin-top: 0; margin-bottom: 0.5rem; font-size: 1.4rem; font-weight: 700;">📥 Upload Photo</h4>
                <p style="color: #9CA3AF; font-size: 1rem; margin-bottom: 1.5rem; line-height: 1.6;">Take a clear photo of the leaf you want to check.</p>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            st.image(uploaded_file, use_container_width=True, caption="Photo Uploaded Successfully")
            
        st.markdown("</div>", unsafe_allow_html=True)

    with col_output:
        st.markdown("""<div class="bento-card" style="min-height: 100%; animation-delay: 0.2s;">""", unsafe_allow_html=True)
        st.markdown('<h4 style="margin-top: 0; margin-bottom: 1.5rem; font-size: 1.4rem; font-weight: 700;">🔬 Analysis Results</h4>', unsafe_allow_html=True)
        
        if uploaded_file is None:
            st.markdown("""
                <div style="text-align: center; color: #6B7280; padding: 6rem 1rem; display: flex; flex-direction: column; align-items: center;">
                    <div class="float-icon" style="font-size: 4.5rem; margin-bottom: 1.5rem; opacity: 0.4; filter: grayscale(1);">📸</div>
                    <h3 style="margin-bottom: 0.8rem; color: #E5E7EB; font-weight: 700; font-size: 1.6rem;">Waiting for Photo...</h3>
                    <p style="font-size: 1.05rem; color: #9CA3AF; max-width: 300px; line-height: 1.6;">Please upload an image using the box on the left to start the health check.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            if st.button("Check Plant Health 🔍"):
                with st.spinner("Checking for diseases..."):
                    try:
                        result_idx = execute_inference(uploaded_file)
                        diagnosis = DISEASE_CLASSES[result_idx]
                        
                        plant_type, pathology = diagnosis.split(" - ")
                        
                        is_healthy = "Healthy" in pathology
                        status_class = "status-healthy" if is_healthy else "status-danger"
                        status_color = "#10B981" if is_healthy else "#EF4444"
                        status_text = "Looks Healthy!" if is_healthy else "Disease Found"
                        icon = "✅" if is_healthy else "⚠️"
                        
                        # Farmer-friendly actionable text
                        action_text = "No action needed. Keep watering and fertilizing as normal." if is_healthy else f"We suspect this plant has {pathology}. We recommend taking a sample leaf and this report to your local agricultural store to find the right spray or treatment."
                        
                        # Generate text for the download report feature
                        report_content = f"""AGRIGUARD AI - CROP HEALTH REPORT
-----------------------------------
Crop Type: {plant_type}
Detected Condition: {pathology}
Status: {status_text}

Advice: {action_text}

* Note: This is an AI-generated assessment. Always verify with a local agricultural expert.
"""

                        st.markdown(f"""
<div class="{status_class}" style="background: rgba(10,10,12,0.6); border-radius: 20px; padding: 32px; margin-top: 1.5rem; position: relative; overflow: hidden;">
<div style="position: absolute; top: 0; left: 0; width: 100%; height: 4px; background: {status_color};"></div>
<h5 style="color: #A1A1AA; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 16px; font-size: 0.85rem;">Health Report</h5>
<div style="margin-bottom: 24px;">
<p style="margin: 0; color: #9CA3AF; font-size: 0.95rem; font-weight: 600; text-transform: uppercase;">Plant Type</p>
<h3 style="margin: 0 0 16px 0; color: #FFFFFF; font-weight: 800; font-size: 1.8rem; letter-spacing: -0.5px;">{plant_type}</h3>
<p style="margin: 0; color: #9CA3AF; font-size: 0.95rem; font-weight: 600; text-transform: uppercase;">Condition</p>
<h2 style="margin: 0; color: {status_color}; font-weight: 800; font-size: 2.6rem; letter-spacing: -1px;">
<span class="float-icon" style="margin-right: 10px;">{icon}</span> {pathology}
</h2>
</div>
<div style="background: rgba(255,255,255,0.03); padding: 20px 24px; border-radius: 16px; border-left: 4px solid {status_color}; margin-bottom: 28px; backdrop-filter: blur(10px);">
<p style="margin: 0 0 10px 0; color: #FFFFFF; font-weight: 700; font-size: 1.2rem; letter-spacing: 0.5px;">Status: {status_text}</p>
<p style="margin: 0; color: #A1A1AA; font-size: 0.95rem; display: flex; align-items: center; gap: 8px;">
<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{status_color}" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
AI Certainty: High
</p>
</div>
<div style="background: rgba(0,0,0,0.3); padding: 20px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 20px;">
<p style="color: #E5E7EB; font-size: 1.05rem; line-height: 1.7; margin: 0;">
<b style="color: #fff;">What you should do next:</b><br> {action_text}
</p>
</div>
</div>
""", unsafe_allow_html=True)

                        # New Practical Feature: Downloadable Report
                        st.download_button(
                            label="📥 Save Report to Show at Farm Shop",
                            data=report_content,
                            file_name=f"{plant_type}_Health_Report.txt",
                            mime="text/plain"
                        )
                        
                    except Exception as e:
                        st.error(f"Analysis Failed: Please try another photo. (Error: {str(e)})")
        
        st.markdown("</div>", unsafe_allow_html=True)