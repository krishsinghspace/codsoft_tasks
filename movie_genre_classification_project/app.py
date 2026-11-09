"""Optional Streamlit interface: streamlit run app.py"""
from pathlib import Path
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "movie_genre_model.joblib"

st.set_page_config(page_title="Movie Genre Classifier", page_icon="🎬", layout="centered")
st.title("🎬 Movie Genre Classification")
st.write("Predict a movie genre from its title and plot summary using TF-IDF and a Linear SVM.")
if not MODEL_PATH.exists():
    st.warning("The trained model was not found. Run `python train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)
title = st.text_input("Movie title (optional)", placeholder="e.g., The Lost Expedition")
plot = st.text_area("Plot summary", height=180, placeholder="Describe the story, characters, and main events...")
if st.button("Predict genre", type="primary"):
    text = (title + " " + plot).strip()
    if not text:
        st.error("Enter a plot summary first.")
    else:
        prediction = model.predict([text])[0]
        st.success(f"Predicted genre: **{prediction.title()}**")
        st.caption("This is a model prediction, not a guaranteed official classification.")
