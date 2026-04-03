import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="AI Detector", page_icon="🧬", layout="centered")

st.markdown("""
<style>

/* Background with subtle pattern */
.stApp {
    background: linear-gradient(135deg, #ede9fe, #ddd6fe);
    background-image: 
        linear-gradient(135deg, rgba(139,92,246,0.1) 1px, transparent 1px),
        linear-gradient(45deg, rgba(139,92,246,0.1) 1px, transparent 1px);
    background-size: 40px 40px;
}

/* Title */
h1 {
    color: #6d28d9;
}

/* Paragraph */
p {
    color: #5b21b6;
}

/* Text area styling */
textarea {
    background-color: #f5f3ff !important;
    color: black !important;
    border: 2px solid #8b5cf6 !important;
    border-radius: 12px !important;
}

/* Button */
.stButton>button {
    background-color: #8b5cf6;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #6d28d9;
}

/* Result box */
.result-box {
    padding: 15px;
    border-radius: 12px;
    margin-top: 10px;
}

.ai-box {
    background-color: #fee2e2;
    border: 2px solid #dc2626;
}

.human-box {
    background-color: #dcfce7;
    border: 2px solid #16a34a;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align: center;'>🧬 AI Content Detector</h1>
<p style='text-align: center;'>Smart detection with confidence analysis</p>
""", unsafe_allow_html=True)

st.sidebar.title("⚙️ Settings")

threshold = st.sidebar.slider(
    "AI Confidence Threshold (%)",
    50, 100, 70
)

st.sidebar.markdown("Adjust sensitivity of AI detection")

df = pd.read_csv("ai_vs_human_content_v2_20000.csv")
df = df[['content', 'label']]
df['label'] = df['label'].map({'human': 0, 'ai': 1})

@st.cache_resource
def train_model():
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(df['content'])
    y = df['label']

    model = LogisticRegression()
    model.fit(X, y)

    return vectorizer, model

vectorizer, model = train_model()

st.write("")

user_input = st.text_area(
    "📝 Paste your content below:",
    height=220,
    placeholder="Type or paste text here..."
)


if user_input:
    word_count = len(user_input.split())
    char_count = len(user_input)

    st.caption(f"📊 Words: {word_count} | Characters: {char_count}")


if st.button("🔍 Analyze Text"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            input_vector = vectorizer.transform([user_input])

            prediction = model.predict(input_vector)
            probability = model.predict_proba(input_vector)
            confidence = round(max(probability[0]) * 100, 2)

        st.write("---")

       
        if confidence >= threshold:
            result = "AI Generated"
            box_class = "ai-box"
            icon = "🤖"
        else:
            result = "Human Written"
            box_class = "human-box"
            icon = "👤"

        st.markdown(f"""
        <div class='result-box {box_class}'>
        <h3>{icon} {result}</h3>
        <p>Confidence: <b>{confidence}%</b></p>
        </div>
        """, unsafe_allow_html=True)