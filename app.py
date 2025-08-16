
from dotenv import load_dotenv

load_dotenv()

import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# =========================
# コア処理関数
# =========================
def run_llm(user_text: str, role_choice: str) -> str:
    """
    引数:
        user_text    : 入力テキスト（ユーザー質問）
        role_choice  : ラジオボタンの選択値（"A" or "B" を含む文字列を想定）

    戻り値:
        LLMの回答テキスト（str）
    """
    # 選択に応じたシステムメッセージ
    if str(role_choice).startswith("A"):
        system_message = (
            "あなたはAの領域に詳しい『健康の専門家』です。"
            "日本語で、エビデンスに基づいたわかりやすい助言を与えてください。"
            "診断行為は避け、一般的な健康アドバイスとセルフケア、受診の目安を示してください。"
        )
    else:
        system_message = (
            "あなたはBの領域に詳しい『教育の専門家』です。"
            "日本語で、学習者の理解を助ける構造化された説明と、年齢やレベルに応じた実践的ステップを提示してください。"
            "根拠（理論・研究・事例）を短く補足してください。"
        )

    # ChatOpenAI セットアップ
    # OPENAI_API_KEY は環境変数 or Streamlit Secrets で設定しておく
    #   - Streamlit Cloud: [Project] → [Settings] → [Secrets] に OPENAI_API_KEY を登録
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
    )

    # プロンプト（選択ロールに合わせたシステムメッセージを使用）
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", "{user_input}"),
        ]
    )

    # チェーン（Prompt → LLM）
    chain = prompt | llm

    # 実行
    result = chain.invoke({"user_input": user_text})
    # result は BaseMessage。通常は .content にテキストが入る
    return getattr(result, "content", str(result))


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="LangChain × LLM デモ", page_icon="🤖", layout="centered")
st.title("🤖 LangChain × LLM フォーム送信デモ")

# 概要と操作方法（ユーザー向け説明）
st.markdown(
    """
### このアプリについて
- 1つの入力フォームに質問や文章を入力し、送信すると、選択した専門家の視点でLLM（大規模言語モデル）が回答します。  
- LangChain を用いて、フォームの入力をそのままプロンプトとして LLM へ渡しています。

### 使い方
1. **専門家の種類** をラジオボタンから選びます（A: 健康の専門家 / B: 教育の専門家）。  
2. **入力フォーム** に質問や文章を入力します。  
3. **送信** ボタンを押すと、LLM の回答が画面下に表示されます。

> ※ Streamlit Community Cloud にデプロイする場合は、`runtime.txt` に `python-3.11` と記載してください。  
> ※ APIキー（`OPENAI_API_KEY`）は、環境変数または Streamlit の Secrets に設定してください。
"""
)

# APIキー確認（未設定時は警告）
if not (os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)):
    st.warning("環境変数または Secrets に OPENAI_API_KEY が設定されていません。実行前に設定してください。")

# --- 専門家ロール選択 ---
role_choice = st.radio(
    "LLMの振る舞い（専門家の種類）を選択してください：",
    options=["A（Aの領域の健康の専門家）", "B（Bの領域の教育の専門家）"],
    horizontal=True,
)

# 入力フォーム
with st.form("query_form"):
    user_text = st.text_area(
        "入力フォーム：",
        placeholder="ここに質問や文章を入力して送信してください。",
        height=150,
    )
    submitted = st.form_submit_button("送信")

# 送信時の実行（関数 run_llm を利用）
if submitted:
    if not user_text.strip():
        st.error("テキストを入力してください。")
    else:
        with st.spinner("LLMに問い合わせ中…"):
            try:
                answer = run_llm(user_text=user_text, role_choice=role_choice)
                st.subheader("回答結果")
                st.markdown(answer)
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
