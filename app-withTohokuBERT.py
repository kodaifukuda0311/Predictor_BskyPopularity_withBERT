
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- モデルとトークナイザの読み込み ---
@st.cache_resource
@st.cache_resource
def load_model_and_tokenizer():
    model = AutoModelForSequenceClassification.from_pretrained(
        "kodaifukuda0311/BERT-bskypopularity-predictor", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "kodaifukuda0311/BERT-bskypopularity-predictor", trust_remote_code=True
    )
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# --- 予測関数 ---
def predict(headline, threshold=0.4):
    # 推論モード＆入力トークナイズ
    model.eval()
    encoded = tokenizer(
        headline,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=40
    )

    # 入力を model と同じデバイスへ移動
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    # 推論
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        score = probs[0][1].item()  # ラベル1の確率

    return score

# --- Streamlit UI ---
st.title("📰 Blueskyバズ予測アプリ")

st.markdown("""
見出しを打ち込むだけで、Blueskyの投稿が**バズるかどうか**を90%の精度で予測します！  
東北大学が公開しているBERTを使い、ファインチューニングしたものです。
（20250618更新）
""")

headline = st.text_input("見出しを入力してください（最大32文字）")

if st.button("予測する"):
    score = predict(headline, threshold=0.4)
    if score >= 0.4:
        st.success(f"🎯 いいね！ヒットする可能性が高いです！")
    else:
        st.warning(f"📉 ごめんね、ヒットする可能性は低いです…")
