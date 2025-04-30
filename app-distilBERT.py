
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- キャッシュして読み込み ---
@st.cache_resource
def load_model_and_tokenizer():
    model = AutoModelForSequenceClassification.from_pretrained("./saved_model")
    tokenizer = AutoTokenizer.from_pretrained("./saved_model")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# --- 推論関数 ---
def predict_popularity(headline, threshold=0.35):
    # トークナイズ
    inputs = tokenizer(headline, return_tensors="pt", padding="max_length", truncation=True, max_length=40)

    # 推論
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        score = probs[0][1].item()  # ラベル1（ヒット）の確率

    # 判定
    if score >= threshold:
        return f"🎯 ヒットの可能性が高いです！（確率: {score:.2%}）"
    else:
        return f"📉 ヒットの可能性は低めです…（確率: {score:.2%}）"

# --- Streamlit UI ---
st.title("📰 Blueskyバズ予測ツール")

st.markdown("#### 📝 アプリの概要")
st.write("""
これはあなたのBluesky投稿が「バズるかどうか」を予測するアプリです。  
LINEヤフーが公開しているDistilBERTモデルを用いて、ヒット予測モデルにより判定を行います。  
（モデルは20250430)
""")

headline = st.text_input("記事の見出しを入力してください（最大32文字）")

if st.button("予測する"):
    result = predict_popularity(headline)
    st.success(result)
