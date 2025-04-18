import streamlit as st
st.set_page_config(page_title="Instagram Predictor", layout="centered")

import pandas as pd
import numpy as np
import re
from textblob import TextBlob
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import datetime
import plotly.express as px
from fpdf import FPDF
import instaloader
from instaloader import Post

# --- Session State for Login ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# --- Token-based Access Control ---
VALID_TOKENS = {"demo123", "ajixyz", "insta2025"}

if not st.session_state.logged_in:
    st.title("Welcome to Instanzee")
    st.subheader("🔐 Enter Access Token")
    st.markdown("Enter a valid token to continue. Use `demo123` to try it out or just click on Login Button")
    token = st.text_input("Token", value="demo123", type="password")
    if st.button("🔓 Login"):
        if token in VALID_TOKENS:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("❌ Invalid token. Please try again.")
    st.stop()

# --- Logout Button ---
st.sidebar.button("🔒 Logout", on_click=lambda: st.session_state.update({"logged_in": False}))

# --- Load and train models ---
@st.cache_resource
def load_models():
    df = pd.read_csv("Instagram - Posts.csv")
    df = df.dropna(subset=["description", "likes", "followers", "date_posted", "content_type"])

    df["caption_length"] = df["description"].apply(len)
    df["hashtag_count"] = df["hashtags"].apply(lambda x: len(eval(x)) if isinstance(x, str) else 0)
    df["polarity"] = df["description"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df["subjectivity"] = df["description"].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
    df["post_hour"] = pd.to_datetime(df["date_posted"]).dt.hour
    df["weekday"] = pd.to_datetime(df["date_posted"]).dt.day_name()
    df["month"] = pd.to_datetime(df["date_posted"]).dt.month
    df["location"] = df["location"].fillna("Unknown")

    categorical_cols = ["content_type", "weekday", "location"]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)

    features = ["caption_length", "hashtag_count", "polarity", "subjectivity", "post_hour", "month", "followers"]
    X = pd.concat([df[features], encoded_df], axis=1)
    y_likes = df["likes"]
    y_comments = df["num_comments"].fillna(0)

    model_likes = RandomForestRegressor(n_estimators=100, random_state=42)
    model_likes.fit(X, y_likes)

    model_comments = RandomForestRegressor(n_estimators=100, random_state=42)
    model_comments.fit(X, y_comments)

    best_day = df.groupby("weekday")["likes"].mean().idxmax()
    return model_likes, model_comments, encoder, encoder.get_feature_names_out(categorical_cols), categorical_cols, best_day

model_likes, model_comments, encoder, encoded_feature_names, cat_cols, best_day = load_models()

# --- Streamlit UI ---
st.title("📸 Instagram Post Performance Predictor")
tab1, tab2 = st.tabs(["📝 Create New Post", "🔗 Analyze IG Link"])
with tab1:
    st.subheader("\U0001F9EA Simulate a New Instagram Post")
    caption = st.text_area("Caption")
    hashtags = st.text_input("Hashtags", placeholder="#fun #travel")
    followers = st.number_input("Follower Count", value=1000)
    post_hour = st.slider("Posting Hour", 0, 23, 12)
    post_date = st.date_input("Post Date", value=datetime.date.today())
    content_type = st.selectbox("Content Type", ["Reel", "Video", "Carousel"])
    location = st.text_input("Location", placeholder="e.g. Mumbai")

    if st.button("\U0001F52E Predict Performance"):
        caption_len = len(caption)
        hashtag_count = len([tag for tag in hashtags.split() if tag.startswith("#")])
        polarity = TextBlob(caption).sentiment.polarity
        subjectivity = TextBlob(caption).sentiment.subjectivity
        weekday = pd.to_datetime(post_date).day_name()
        month = pd.to_datetime(post_date).month
        location = location if location else "Unknown"

        cat_input = pd.DataFrame([[content_type, weekday, location]], columns=cat_cols)
        encoded_input = encoder.transform(cat_input)
        encoded_df = pd.DataFrame(encoded_input, columns=encoded_feature_names)

        input_df = pd.DataFrame([{ "caption_length": caption_len, "hashtag_count": hashtag_count, "polarity": polarity,
            "subjectivity": subjectivity, "post_hour": post_hour, "month": month, "followers": followers }])
        final_input = pd.concat([input_df, encoded_df], axis=1)

        predicted_likes = int(model_likes.predict(final_input)[0])
        predicted_comments = int(model_comments.predict(final_input)[0])
        engagement_rate = (predicted_likes / followers) * 100 if followers else 0

        st.subheader("\U0001F4CA Predicted Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("\U0001F44D Likes", f"{predicted_likes:,}")
        col2.metric("\U0001F4AC Comments", f"{predicted_comments:,}")
        col3.metric("\U0001F4C8 Engagement", f"{engagement_rate:.2f}%")

        fig = px.bar(x=["Likes", "Comments"], y=[predicted_likes, predicted_comments],
                     labels={"x": "Metric", "y": "Count"},
                     title="Predicted Engagement Breakdown")
        st.plotly_chart(fig)

        sentiment_text = "Neutral"
        if polarity > 0.1:
            sentiment_text = "Positive"
        elif polarity < -0.1:
            sentiment_text = "Negative"

        st.markdown("### \U0001F9E0 Sentiment Analysis")
        st.markdown(f"- Sentiment: `{sentiment_text}`")
        st.markdown(f"- Polarity: `{polarity:.2f}`")
        st.markdown(f"- Subjectivity: `{subjectivity:.2f}`")
        st.info(f"\U0001F4C5 Best day to post: **{best_day}**")

        # Export prediction
        result_df = pd.DataFrame([{ "Caption": caption, "Hashtag Count": hashtag_count, "Followers": followers,
            "Post Type": content_type, "Location": location, "Polarity": polarity, "Subjectivity": subjectivity,
            "Predicted Likes": predicted_likes, "Predicted Comments": predicted_comments,
            "Engagement Rate": engagement_rate }])

        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name="instagram_prediction.csv", mime='text/csv')

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Instagram Post Prediction Report", ln=True, align="C")
        for col in result_df.columns:
            pdf.cell(200, 10, txt=f"{col}: {result_df[col][0]}", ln=True)
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        st.download_button("Download PDF", data=pdf_bytes, file_name="instagram_prediction.pdf", mime='application/pdf')

with tab2:
    st.subheader("🔍 Analyze Public Instagram Post Link")
    post_link = st.text_input("Paste IG Post URL")
    ig_user = st.text_input("Your Instagram Username", placeholder="For login (required)")
    ig_pass = st.text_input("Your Instagram Password", type="password")

    if st.button("📥 Fetch & Analyze Post") and post_link and ig_user and ig_pass:
        try:
            shortcode = post_link.strip("/").split("/")[-1]
            st.info(f"Extracted shortcode: `{shortcode}`")

            L = instaloader.Instaloader()
            L.login(ig_user, ig_pass)
            post = Post.from_shortcode(L.context, shortcode)

            caption = post.caption or ""
            caption_len = len(caption)
            hashtag_count = len(re.findall(r"#\w+", caption))
            polarity = TextBlob(caption).sentiment.polarity
            subjectivity = TextBlob(caption).sentiment.subjectivity
            post_hour = post.date_utc.hour
            post_date = post.date_utc
            weekday = post_date.strftime("%A")
            month = post_date.month
            content_type = post.typename  # Possible: GraphImage, GraphVideo, GraphSidecar
            followers = 5000  # Or estimate/default

            location = post.location.name if post.location else "Unknown"

            # Map Instagram typename to content_type
            if "Video" in content_type:
                content_type = "Video"
            elif "Sidecar" in content_type:
                content_type = "Carousel"
            else:
                content_type = "Reel"

            # Encode categorical values
            cat_input = pd.DataFrame([[content_type, weekday, location]], columns=cat_cols)
            encoded_input = encoder.transform(cat_input)
            encoded_df = pd.DataFrame(encoded_input, columns=encoded_feature_names)

            # Combine features
            input_df = pd.DataFrame([{
                "caption_length": caption_len,
                "hashtag_count": hashtag_count,
                "polarity": polarity,
                "subjectivity": subjectivity,
                "post_hour": post_hour,
                "month": month,
                "followers": followers
            }])
            final_input = pd.concat([input_df, encoded_df], axis=1)

            # Predict
            predicted_likes = int(model_likes.predict(final_input)[0])
            predicted_comments = int(model_comments.predict(final_input)[0])
            engagement_rate = (predicted_likes / followers) * 100 if followers else 0

            st.subheader("📊 Predicted Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("👍 Likes", f"{predicted_likes:,}")
            col2.metric("💬 Comments", f"{predicted_comments:,}")
            col3.metric("📈 Engagement", f"{engagement_rate:.2f}%")

            st.markdown("### 🧠 Sentiment Analysis")
            st.markdown(f"- Sentiment: `{('Positive' if polarity > 0.1 else 'Negative' if polarity < -0.1 else 'Neutral')}`")
            st.markdown(f"- Polarity: `{polarity:.2f}`")
            st.markdown(f"- Subjectivity: `{subjectivity:.2f}`")

            st.info(f"📅 Based on your data, best day to post: **{best_day}**")

        except Exception as e:
            st.error(f"❌ Failed to fetch post: {e}")