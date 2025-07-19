import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
import pandas as pd
import base64
import plotly.graph_objects as go
import requests
from streamlit_extras.let_it_rain import rain


# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb.h5')

# Helper functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    return sequence.pad_sequences([encoded_review], maxlen=500)


def get_base64_of_bin_file(bin_file_path):
    with open(bin_file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def get_base64_of_bin_file(bin_file_path):
    with open(bin_file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Read & encode once
img_data = get_base64_of_bin_file("assets/bgimage2.jpg")



def load_lottieurl(url: str):
    """Fetch Lottie JSON from a URL and return as Python dict."""
    try:
        r = requests.get(url)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"⚠️ Could not load animation: {e}")
        return None

st.set_page_config(page_title="IMDB Sentiment Analysis", 
                   page_icon="🎬", 
                   layout="centered",
                    initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* Sidebar text styling */
    section[data-testid="stSidebar"] {
        color: #0a0a0a !important;  /* Deep black for sharpness */
        font-weight: 600 !important;  /* Bolder font */
        font-size: 16px !important;
        text-shadow: none !important;
        -webkit-font-smoothing: antialiased !important;
        -moz-osx-font-smoothing: grayscale !important;
    }

    /* Sidebar caption */
    section[data-testid="stSidebar"] .st-emotion-cache-1kyxreq {
        color: #222 !important;
        font-weight: 500 !important;
        font-size: 14px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("⚙️ App Settings")
threshold = st.sidebar.slider("Positive Threshold", 
                              min_value=0.0, 
                              max_value=1.0, 
                              value=0.5, 
                              step=0.01,
                              help="Reviews with score above this threshold are labeled Positive.")

st.sidebar.caption("👆 Adjust how strict the classification is. Lower values make the model more permissive.")

dark_mode = st.sidebar.toggle("🌙 Enable Dark Mode", 
                              value=False)

st.sidebar.markdown(
    """
    <hr style='border: none; height: 2px; background-color: rgba(0, 0, 0, 0.2); margin: 20px 0;' />
    """,
    unsafe_allow_html=True
)


if dark_mode:
     # Apply simple dark mode styling
     st.markdown("""
        <style>
        .stApp {
            background-color: #121212 !important;
            color: #DDDDDD;
        }
        h1 { color: #90CAF9; }
        .stMetricLabel, .stMetricValue { color: #FFFFFF; }
        .stTextArea textarea, .stTextInput input {
            background-color: #222 !important;
            color: #eee !important;
        }
        
        /* Make label text white */
        label, .stTextArea label, .stTextInput label {
        color: #ffffff !important;
}

        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                linear-gradient(rgba(0,0,0,0.28), rgba(0,0,0,0.28)),
                url("data:image/jpeg;base64,{img_data}") no-repeat center center fixed;
            background-size: cover;
            filter: brightness(1.15) contrast(1.1);
            color: #ffffff;
        }}

        .stApp::before {{
            content: "";
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            pointer-events: none;
            background-image: url("data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iMTIiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yLi8yMDAwL3N2ZyI+PHJlY3Qgd2lkdGg9IjEyIiBoZWlnaHQ9IjEyIiBmaWxsPSJub25lIi8+PHJlY3QgZmlsbD0iI2ZmZiIgb3BhY2l0eT0iMC4xIiB3aWR0aD0iMSIgaGVpZ2h0PSIxIi8+PC9zdmc+");
            mix-blend-mode: overlay;
            opacity: 1;
            z-index: -1;
        }}

        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
            text-shadow: 0 2px 6px rgba(0, 0, 0, 0.8);
        }}

        .stTextArea textarea, .stTextInput input {{
            background-color: rgba(255,255,255,0.9) !important;
            color: #000 !important;
            border-radius: 8px;
        }}

        label, .stTextArea label, .stTextInput label {{
            color: #ffffff !important;
        }}

        </style>
        """,
        unsafe_allow_html=True
)



st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("""
        Enter a movie review below and click **🔍 Classify**.  Adjust the threshold in the sidebar to tune sensitivity.
        """)

# User input text area
user_input = st.text_area("✏️ Write your review here", 
                          height=150,
                          placeholder="e.g., This movie was fascinating and beautifully shot.")

# Style the Classify button
st.markdown(
    """
    <style>
    div.stButton > button {
        background: linear-gradient(135deg, #00C9A7, #00B4D8);
        color: white;
        font-weight: 600;
        font-size: 17px;
        padding: 10px 28px;
        border: none;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        transition: all 0.3s ease-in-out;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #3DD598, #90F1EF);
        transform: scale(1.05);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Warning button styling
st.markdown(
    """
    <style>
    /* Improve st.warning visibility */
    .stAlert[data-testid="stAlert"] {
        background-color: #FFF3CD !important;  /* light yellow */
        color: #5A3E00 !important;             /* dark text for contrast */
        border-left: 6px solid #FFC107 !important;
        box-shadow: 0 0 10px rgba(0,0,0,0.2);
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Success and error box styling
st.markdown("""
<style>
/* Success box */
.stAlert[data-testid="stAlert-success"], 
.stAlert[data-testid="stAlert-error"] {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 12px 16px !important;      /* equal top/bottom padding */
    margin: 8px 0 !important;           /* space around alert boxes */
    border-radius: 6px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
}

/* Background & border colors */
.stAlert[data-testid="stAlert-success"] {
    background-color: #D1FADF !important;
    color: #065F46 !important;
    border-left: 6px solid #22C55E !important;
}
.stAlert[data-testid="stAlert-error"] {
    background-color: #FEE2E2 !important;
    color: #7F1D1D !important;
    border-left: 6px solid #EF4444 !important;
}

/* Center and style the text within */
.stAlert div[data-testid="stMarkdownContainer"] {
    width: 100% !important;
    text-align: left !important;
    margin: 0 !important;
    padding: 0 !important;
}
.stAlert div[data-testid="stMarkdownContainer"] p {
    font-size: 22px !important;
    font-weight: 700 !important;
    margin: 0 !important;
    line-height: 1.2 !important;        /* tight line-height for vertical centering */
}
</style>
""", unsafe_allow_html=True)



# Main classification logic
# Classify button
if st.button('🔍 Classify'):
    if not user_input.strip():
        st.warning("Please enter some text to classify.")
    else:
        # Predict and store in session state
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)
        score = float(prediction[0][0])
        positive_conf = score
        negative_conf = 1 - score
        sentiment = 'Positive' if score > threshold else 'Negative'

        if sentiment == "Positive":
            st.success(f"👍 Sentiment: **{sentiment}**")
            st.balloons()

            # Toast
            st.toast("Positive sentiment detected!", icon="✅")

        else:
            st.error(f"👎 Sentiment: **{sentiment}**")


            # Toast
            st.toast("Negative sentiment detected.", icon="❌")


        # Store in session state
        st.session_state['sentiment'] = sentiment
        st.session_state['score'] = score
        st.session_state['positive_conf'] = positive_conf
        st.session_state['negative_conf'] = negative_conf


# Check if results are in session state and display them
if 'sentiment' in st.session_state:
    sentiment = st.session_state['sentiment']
    score = st.session_state['score']
    positive_conf = st.session_state['positive_conf']
    negative_conf = st.session_state['negative_conf']

    # Score and confidence
    st.markdown(f"""
        <div style='margin-bottom:1rem'>
            <strong>Model Score:</strong> {score:.3f} (Threshold = {threshold:.2f})<br>
            <strong>Confidence:</strong> {positive_conf*100:.1f}% positive, {negative_conf*100:.1f}% negative
        </div>
    """, unsafe_allow_html=True)

    # Graph selector and plot
    graph_option = st.selectbox("📈 Choose Confidence Visualization", ["Radial Gauge","Pie Chart"])

    st.subheader("🔎 Confidence Overview")

    if graph_option == "Radial Gauge":
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=positive_conf * 100,
            delta={'reference': 50},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#22C55E"},
                'steps': [
                    {'range': [0, 50], 'color': '#FCA5A5'},
                    {'range': [50, 100], 'color': '#BBF7D0'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold * 100
                }
            },
            title={'text': "Positive Confidence (%)"}
        ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)

    elif graph_option == "Pie Chart":
        fig = go.Figure(data=[go.Pie(
            labels=["Positive", "Negative"],
            values=[positive_conf, negative_conf],
            hole=0.5,
            marker=dict(colors=["#22C55E", "#F37843"]),
            textinfo="label+percent",
            textfont_size=16
        )])
        fig.update_layout(
            showlegend=False,
            height=280,
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        st.plotly_chart(fig, use_container_width=True)

    
st.markdown("""
<hr style='border: none; height: 2px; background-color: rgba(255, 255, 255, 0.5); margin: 25px 0;' />

<div style='text-align: left; color: white; font-size: 14px;'>
    © 2025 <strong>Tanuj Hinduja</strong> • 
    <a href='https://github.com/TanujHinduja' target='_blank' style='color: #FFDE59; text-decoration: none;'>GitHub</a> • 
    <a href='mailto:tanujhinduja54@gmail.com' style='color: #FFDE59; text-decoration: none;'>Email</a>
</div>
""", unsafe_allow_html=True)
