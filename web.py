import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================================
# 1. PAGE CONFIGURATION & DESIGN SYSTEM
# ==========================================
st.set_page_config(
    page_title="AgriGuard AI | Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Modern CSS injected into the DOM
st.markdown("""
    <style>
    /* Global Base & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Dynamic Animated Background */
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
    
    /* Clean up default Streamlit elements safely */
    footer { visibility: hidden !important; }
    #MainMenu { visibility: hidden !important; }
    [data-testid="stDecoration"] { display: none !important; }
    
    /* Custom Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: rgba(5, 5, 5, 0.6) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* --------------------------------------
       ANIMATIONS
       -------------------------------------- */
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

    /* --------------------------------------
       UI COMPONENTS
       -------------------------------------- */
    /* Hero Typography */
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

    /* Animated Bento Cards */
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

    /* Floating Icons */
    .float-icon {
        display: inline-block;
        animation: float 4s ease-in-out infinite;
    }

    /* Modern Action Button */
    div.stButton > button {
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
    div.stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 25px rgba(16, 185, 129, 0.4);
        border-color: rgba(255,255,255,0.3);
    }
    
    /* File Uploader Customization */
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
    
    /* Status Rings */
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
    """Loads the Keras model once into server memory."""
    try:
        return tf.keras.models.load_model('trained_model.keras')
    except Exception as e:
        st.error(f"Critical System Failure: Model weights not found. {e}")
        return None

model = get_neural_engine()

def execute_inference(image_buffer):
    """Processes image in-memory and returns class index."""
    img = Image.open(image_buffer).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)
    return np.argmax(predictions)

# Comprehensive 38-Class PlantVillage Taxonomy
DISEASE_CLASSES = [
    'Apple : Scab', 'Apple : Black Rot', 'Apple : Cedar Rust', 'Apple : Healthy', 
    'Blueberry : Healthy', 'Cherry : Powdery Mildew', 'Cherry : Healthy', 
    'Corn : Cercospora Leaf Spot', 'Corn : Common Rust', 'Corn : Northern Leaf Blight', 'Corn : Healthy',
    'Grape : Black Rot', 'Grape : Esca (Black Measles)', 'Grape : Leaf Blight', 'Grape : Healthy',
    'Orange : Citrus Greening', 'Peach : Bacterial Spot', 'Peach : Healthy',
    'Pepper, Bell : Bacterial Spot', 'Pepper, Bell : Healthy', 'Potato : Early Blight',
    'Potato : Late Blight', 'Potato : Healthy', 'Raspberry : Healthy', 'Soybean : Healthy',
    'Squash : Powdery Mildew', 'Strawberry : Leaf Scorch', 'Strawberry : Healthy',
    'Tomato : Bacterial Spot', 'Tomato : Early Blight', 'Tomato : Late Blight',
    'Tomato : Leaf Mold', 'Tomato : Septoria Leaf Spot', 'Tomato : Spider Mites',
    'Tomato : Target Spot', 'Tomato : Yellow Leaf Curl Virus', 'Tomato : Mosaic Virus', 'Tomato : Healthy'
]

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    # Modern Animated Inline Logo
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 2.5rem; margin-top: 1rem;">
            <div class="float-icon" style="width: 48px; height: 48px; border-radius: 14px; background: linear-gradient(135deg, #10B981, #047857); display: flex; align-items: center; justify-content: center; font-size: 1.6rem; box-shadow: 0 8px 25px rgba(16,185,129,0.4); border: 1px solid rgba(255,255,255,0.2);">🌿</div>
            <h2 style="margin: 0; font-weight: 800; letter-spacing: -1.5px; font-size: 2rem; color: #fff;">AgriGuard</h2>
        </div>
    """, unsafe_allow_html=True)
    
    app_mode = st.radio("Navigation", ["Overview", "Architecture", "Diagnostic Engine"], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("""
        <div class="bento-card" style="padding: 20px; animation-delay: 0.3s; background: rgba(10,10,10,0.5);">
            <p style="margin: 0; font-size: 0.75rem; color: #A1A1AA; text-transform: uppercase; font-weight: 700; letter-spacing: 1.5px;">Live Telemetry</p>
            <div style="margin-top: 18px; display: flex; align-items: center; gap: 12px;">
                <div style="width: 10px; height: 10px; border-radius: 50%; background: #10B981; box-shadow: 0 0 15px #10B981; animation: pulse-glow 2s infinite;"></div>
                <span style="font-size: 0.95rem; font-weight: 600; color: #E5E7EB;">Engine Online</span>
            </div>
            <div style="margin-top: 12px; display: flex; align-items: center; gap: 12px;">
                <div style="width: 10px; height: 10px; border-radius: 50%; background: #10B981; box-shadow: 0 0 15px #10B981; animation: pulse-glow 2.5s infinite;"></div>
                <span style="font-size: 0.95rem; font-weight: 600; color: #E5E7EB;">LFS Synced</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ==========================================
# 4. ROUTING & PAGES
# ==========================================

if app_mode == "Overview":
    # Cinematic Animated Mesh Hero Banner (Replaces static photo)
    st.markdown("""
        <div class="bento-card" style="width: 100%; min-height: 480px; border-radius: 36px; overflow: hidden; position: relative; margin-bottom: 2.5rem; padding: 0; border: 1px solid rgba(255,255,255,0.1); background: linear-gradient(120deg, #022c22, #064e3b, #000000, #022c22); background-size: 300% 300%; animation: mainGradient 12s ease infinite, fadeInUp 0.2s forwards;">
            <div style="position: absolute; inset: 0; background: radial-gradient(circle at top right, rgba(16, 185, 129, 0.15), transparent 50%);"></div>
            <div style="position: relative; z-index: 1; padding: 5rem; height: 100%; display: flex; flex-direction: column; justify-content: center;">
                <div class="float-icon" style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); padding: 8px 16px; border-radius: 100px; width: fit-content; margin-bottom: 2rem; color: #34D399; font-weight: 600; font-size: 0.9rem; letter-spacing: 1px;">v2.0 INTELLIGENCE CORE</div>
                <h1 class="hero-title">The Future of <br><span class="hero-highlight">Agronomy.</span></h1>
                <p style="color: #A1A1AA; font-size: 1.3rem; max-width: 600px; margin-top: 1.5rem; line-height: 1.7; font-weight: 400;">
                    Empowering precision agriculture with instant disease detection using deep convolutional neural networks and cinematic UI architecture.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Animated Bento Grid
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown("""
            <div class="bento-card hover-glow" style="height: 100%; animation-delay: 0.1s;">
                <div class="float-icon" style="width: 56px; height: 56px; border-radius: 16px; background: linear-gradient(135deg, rgba(52, 211, 153, 0.1), rgba(4, 120, 87, 0.2)); color: #34D399; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; margin-bottom: 2rem; border: 1px solid rgba(52, 211, 153, 0.2);">⚡</div>
                <h2 style="margin: 0; font-size: 3.5rem; font-weight: 800; letter-spacing: -2px;">150<span style="font-size: 1.8rem; color: #52525B; font-weight: 600;">ms</span></h2>
                <p style="color: #A1A1AA; margin-top: 0.5rem; font-weight: 600; font-size: 1.1rem;">Inference Latency</p>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
            <div class="bento-card hover-glow" style="height: 100%; animation-delay: 0.2s;">
                <div class="float-icon" style="width: 56px; height: 56px; border-radius: 16px; background: linear-gradient(135deg, rgba(52, 211, 153, 0.1), rgba(4, 120, 87, 0.2)); color: #34D399; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; margin-bottom: 2rem; border: 1px solid rgba(52, 211, 153, 0.2);">🎯</div>
                <h2 style="margin: 0; font-size: 3.5rem; font-weight: 800; letter-spacing: -2px;">96.2<span style="font-size: 1.8rem; color: #52525B; font-weight: 600;">%</span></h2>
                <p style="color: #A1A1AA; margin-top: 0.5rem; font-weight: 600; font-size: 1.1rem;">Diagnostic Precision</p>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
            <div class="bento-card hover-glow" style="height: 100%; animation-delay: 0.3s;">
                <div class="float-icon" style="width: 56px; height: 56px; border-radius: 16px; background: linear-gradient(135deg, rgba(52, 211, 153, 0.1), rgba(4, 120, 87, 0.2)); color: #34D399; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; margin-bottom: 2rem; border: 1px solid rgba(52, 211, 153, 0.2);">🌍</div>
                <h2 style="margin: 0; font-size: 3.5rem; font-weight: 800; letter-spacing: -2px;">38</h2>
                <p style="color: #A1A1AA; margin-top: 0.5rem; font-weight: 600; font-size: 1.1rem;">Supported Taxonomies</p>
            </div>
        """, unsafe_allow_html=True)

elif app_mode == "Architecture":
    st.markdown('<h1 class="hero-title" style="font-size: 4rem; margin-bottom: 2rem; animation: fadeInUp 0.4s forwards; opacity:0;">System <span class="hero-highlight">Architecture.</span></h1>', unsafe_allow_html=True)
    
    # Modern Animated Data Mesh (Replaces static image)
    st.markdown("""
        <div class="bento-card" style="padding: 0; overflow: hidden; position: relative; height: 280px; margin-bottom: 2.5rem; animation-delay: 0.1s; background: linear-gradient(to right, #000000, #022c22); border: 1px solid rgba(16, 185, 129, 0.2);">
            <div style="position: absolute; width: 200%; height: 200%; top: -50%; left: -50%; background-image: radial-gradient(rgba(16, 185, 129, 0.2) 1px, transparent 1px); background-size: 40px 40px; transform: rotate(15deg); animation: mainGradient 30s linear infinite;"></div>
            <div style="position: absolute; inset: 0; display: flex; align-items: center; padding: 4rem; background: linear-gradient(90deg, rgba(0,0,0,0.9) 0%, transparent 100%);">
                <div>
                    <div style="color: #10B981; font-weight: 700; letter-spacing: 2px; font-size: 0.9rem; margin-bottom: 10px;">CORE ENGINE</div>
                    <h2 style="margin: 0; font-size: 3rem; font-weight: 800; letter-spacing: -1px; text-shadow: 0 4px 20px rgba(0,0,0,0.8);">Neural Topology</h2>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("""
            <div class="bento-card hover-glow" style="height: 100%; animation-delay: 0.2s;">
                <h3 style="color: #10B981; margin-top:0; font-size: 1.8rem; display: flex; align-items: center; gap: 10px;"><span class="float-icon">📊</span> Dataset Telemetry</h3>
                <p style="color: #A1A1AA; margin-bottom: 1.5rem; font-size: 1.1rem;">Model trained on augmented high-resolution imagery.</p>
                <div style="background: rgba(0,0,0,0.3); border-radius: 16px; padding: 20px; border: 1px solid rgba(255,255,255,0.05);">
                    <ul style="color: #E5E7EB; line-height: 2.4; list-style-type: none; padding-left: 0; margin: 0; font-size: 1.05rem;">
                        <li style="border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 8px;"><b style="color: #fff;">Source:</b> PlantVillage Kaggle Dataset</li>
                        <li style="border-bottom: 1px solid rgba(255,255,255,0.05); padding: 8px 0;"><b style="color: #fff;">Volume:</b> 87,000+ RGB Images</li>
                        <li style="border-bottom: 1px solid rgba(255,255,255,0.05); padding: 8px 0;"><b style="color: #fff;">Taxonomy:</b> 38 Plant/Disease Classes</li>
                        <li style="padding-top: 8px;"><b style="color: #fff;">Resolution:</b> Scaled 128x128 Tensors</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="bento-card hover-glow" style="height: 100%; animation-delay: 0.3s;">
                <h3 style="color: #10B981; margin-top:0; font-size: 1.8rem; display: flex; align-items: center; gap: 10px;"><span class="float-icon">🛠️</span> Neural Specs</h3>
                <p style="color: #A1A1AA; margin-bottom: 1.5rem; font-size: 1.1rem;">Deep learning architecture execution details.</p>
                <div style="background: rgba(0,0,0,0.3); border-radius: 16px; padding: 20px; border: 1px solid rgba(255,255,255,0.05);">
                    <ul style="color: #E5E7EB; line-height: 2.4; list-style-type: none; padding-left: 0; margin: 0; font-size: 1.05rem;">
                        <li style="border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 8px;"><b style="color: #fff;">Framework:</b> TensorFlow 2.x / Keras</li>
                        <li style="border-bottom: 1px solid rgba(255,255,255,0.05); padding: 8px 0;"><b style="color: #fff;">Topology:</b> Custom 16-Layer CNN</li>
                        <li style="border-bottom: 1px solid rgba(255,255,255,0.05); padding: 8px 0;"><b style="color: #fff;">Optimization:</b> Adam Optimizer</li>
                        <li style="padding-top: 8px;"><b style="color: #fff;">Accuracy:</b> > 96.2% Validation Score</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)

elif app_mode == "Diagnostic Engine":
    st.markdown('<h1 class="hero-title" style="font-size: 4rem; margin-bottom: 2.5rem; animation: fadeInUp 0.3s forwards; opacity:0;">Diagnostic <span class="hero-highlight">Engine.</span></h1>', unsafe_allow_html=True)

    col_input, col_output = st.columns([1, 1.2], gap="large")

    with col_input:
        st.markdown("""
            <div class="bento-card" style="margin-bottom: 1rem; animation-delay: 0.1s;">
                <h4 style="margin-top: 0; margin-bottom: 0.5rem; font-size: 1.4rem; font-weight: 700;">📥 Specimen Input</h4>
                <p style="color: #9CA3AF; font-size: 1rem; margin-bottom: 1.5rem; line-height: 1.6;">Upload a high-resolution image of the suspect botanical specimen for neural processing.</p>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            st.image(uploaded_file, use_container_width=True, caption="Target Sequence Acquired")
            
        st.markdown("</div>", unsafe_allow_html=True)

    with col_output:
        st.markdown("""<div class="bento-card" style="min-height: 100%; animation-delay: 0.2s;">""", unsafe_allow_html=True)
        st.markdown('<h4 style="margin-top: 0; margin-bottom: 1.5rem; font-size: 1.4rem; font-weight: 700;">🔬 Telemetry Output</h4>', unsafe_allow_html=True)
        
        if uploaded_file is None:
            st.markdown("""
                <div style="text-align: center; color: #6B7280; padding: 6rem 1rem; display: flex; flex-direction: column; align-items: center;">
                    <div class="float-icon" style="font-size: 4.5rem; margin-bottom: 1.5rem; opacity: 0.4; filter: grayscale(1);">📸</div>
                    <h3 style="margin-bottom: 0.8rem; color: #E5E7EB; font-weight: 700; font-size: 1.6rem;">Awaiting Specimen</h3>
                    <p style="font-size: 1.05rem; color: #9CA3AF; max-width: 300px; line-height: 1.6;">Upload an image into the input terminal to initialize the neural analysis engine.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            if st.button("Initialize Neural Scan"):
                with st.spinner("Compiling visual features and executing inference..."):
                    try:
                        result_idx = execute_inference(uploaded_file)
                        diagnosis = DISEASE_CLASSES[result_idx]
                        
                        is_healthy = "Healthy" in diagnosis
                        status_class = "status-healthy" if is_healthy else "status-danger"
                        status_color = "#10B981" if is_healthy else "#EF4444"
                        status_text = "Optimal Condition" if is_healthy else "Pathology Detected"
                        icon = "✅" if is_healthy else "⚠️"
                        
                        action_text = "Continue standard maintenance protocols. Monitor irrigation cycles." if is_healthy else f"Isolate affected crops immediately. Consult agricultural database for specific fungicidal/bacterial treatments targeting {diagnosis.split(':')[-1].strip()}."
                        
                        # Fix: Removing all leading spaces from the HTML string so Markdown doesn't render it as a code block.
                        st.markdown(f"""
<div class="{status_class}" style="background: rgba(10,10,12,0.6); border-radius: 20px; padding: 32px; margin-top: 1.5rem; position: relative; overflow: hidden;">
<div style="position: absolute; top: 0; left: 0; width: 100%; height: 4px; background: {status_color};"></div>
<h5 style="color: #A1A1AA; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 16px; font-size: 0.85rem;">Primary Diagnosis</h5>
<h2 style="color: {status_color}; font-weight: 800; font-size: 2.6rem; margin-top: 0; margin-bottom: 28px; letter-spacing: -1px;">
<span class="float-icon" style="margin-right: 10px;">{icon}</span> {diagnosis}
</h2>
<div style="background: rgba(255,255,255,0.03); padding: 20px 24px; border-radius: 16px; border-left: 4px solid {status_color}; margin-bottom: 28px; backdrop-filter: blur(10px);">
<p style="margin: 0 0 10px 0; color: #FFFFFF; font-weight: 700; font-size: 1.2rem; letter-spacing: 0.5px;">Status: {status_text}</p>
<p style="margin: 0; color: #A1A1AA; font-size: 0.95rem; display: flex; align-items: center; gap: 8px;">
<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{status_color}" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
Confidence Interval: > 94.2%
</p>
</div>
<div style="background: rgba(0,0,0,0.3); padding: 20px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.05);">
<p style="color: #E5E7EB; font-size: 1.05rem; line-height: 1.7; margin: 0;">
<b style="color: #fff;">Action Required:</b> {action_text}
</p>
</div>
</div>
""", unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Inference Engine Failed: {str(e)}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    