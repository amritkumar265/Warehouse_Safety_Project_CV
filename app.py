import streamlit as st

st.set_page_config(
    page_title = "VisionSentinel",
    page_icon="🛡️",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,0.2);
    }

    .main-header {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .sub-header {
        color: #888888;
        font-size: 0.95rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.2);
        padding: 15px;
        border-radius: 10px;
    }
</style>
""" , unsafe_allow_html =True
)

with st.sidebar:
    st.markdown("# 🛡️ VisionSentinel")
    st.caption("Warehouse Safety Intelligence")
    st.divider()

    st.markdown("**SYSTEM STATUS**")
    st.success("Application Online")
    st.info("CV Engine: Integration Pending")

    st.divider()
    st.caption("VisionSentinel v0.1.0")
    st.caption("Warehouse Safety Monitoring Platform")


# Main dashboard
st.markdown(
    '<p class="main-header">Warehouse Safety Dashboard</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-header">Intelligent video analytics and safety monitoring</p>',
    unsafe_allow_html=True
)

st.divider()

# KPI placeholders
col1, col2, col3, col4 = st.columns(4)

col1.metric("Registered Cameras", "0")
col2.metric("Videos Processed", "0")
col3.metric("Safety Events", "0")
col4.metric("System FPS", "—")

st.divider()

# Main content
st.subheader("Welcome to VisionSentinel")

st.write(
    "Your centralized warehouse safety monitoring platform. "
    "Configure cameras, analyze CCTV footage, track objects "
    "and investigate safety events."
)

left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.subheader("📹 Camera Management")
        st.write("Register CCTV cameras and configure operational safety zones.")
        if st.button("Open Camera Registry", use_container_width=True):
            st.switch_page("pages/camera_registry.py")

    with st.container(border=True):
        st.subheader("🔍 Video Analysis")
        st.write("Upload warehouse footage and run computer vision analysis.")
        if st.button("Open Video Analysis", use_container_width=True):
            st.switch_page("pages/video_analysis.py")

with right:
    with st.container(border=True):
        st.subheader("🚦 Detection & Tracking")
        st.write("Inspect object detection, fusion and multi-object tracking.")
        if st.button("Open Detection Lab", use_container_width=True):
            st.switch_page("pages/detection_lab.py")

    with st.container(border=True):
        st.subheader("⚠️ Safety Intelligence")
        st.write("Analyze near-misses, obstructions and PPE compliance.")
        if st.button("Open Safety Audit", use_container_width=True):
            st.switch_page("pages/safety_audit.py")

st.divider()

st.caption(
    "VisionSentinel | Built with Streamlit, OpenCV, PyTorch and Computer Vision"
)



