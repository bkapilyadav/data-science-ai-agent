import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import openai
import traceback
import contextlib

# --- OPENAI API KEY FROM SECRETS ---
openai_api_key = st.secrets["OPENAI_API_KEY"]
client = openai.OpenAI(api_key=openai_api_key)

# --- SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None

# --- FILE UPLOAD ---
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    st.session_state.uploaded_df = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded!")

# --- CHAT UI ---
st.title("🧑‍💻 Data Science AI Copilot Agent")
st.write("Ask any data science question about your uploaded data. The agent will analyze, visualize, and explain—no extra steps!")

user_input = st.text_input("You:", key="user_input")
if st.button("Send", key="send_btn"):
    st.session_state.chat_history.append(("user", user_input))

    # --- LLM RESPONSE ---
    df_context = ""
    if st.session_state.uploaded_df is not None:
        df = st.session_state.uploaded_df
        df_context = f"\nThe dataframe 'df' has columns: {', '.join(df.columns)}.\n"
        df_context += f"Here are the first few rows:\n{df.head(3).to_csv(index=False)}\n"

    prompt = f"""
You are a professional Data Science AI Agent.
{df_context}
Your task is to autonomously analyze the user's data-related question using the dataframe 'df' and deliver a comprehensive, actionable report.
STRICT INSTRUCTIONS:
- Always generate a single, complete Python code block that answers ALL parts of the user's question using the dataframe 'df'.
- The code must print all relevant results (numbers, tables, lists, etc.) and generate all relevant visualizations (matplotlib or seaborn) in one go.
- DO NOT explain, DO NOT describe steps, DO NOT ask follow-up questions, DO NOT output markdown, DO NOT output text, DO NOT output anything except the code block.
- DO NOT output any text before or after the code block.
- The code block must be fully self-contained and executable as-is.
- If the user asks a theory question (not related to the data), answer conversationally.
User's message: {user_input}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=1800
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"LLM Error: {e}"

    st.session_state.chat_history.append(("ai", answer))

# --- DISPLAY CHAT HISTORY ---
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Copilot:**")
        # If the answer is code, try to execute it
        if msg.startswith("```python"):
            code = msg.strip("```python").strip("```")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                try:
                    local_vars = {}
                    if st.session_state.uploaded_df is not None:
                        local_vars["df"] = st.session_state.uploaded_df
                    exec(code, {}, local_vars)
                except Exception as e:
                    st.error(f"Code Error: {e}\n{traceback.format_exc()}")
            st.text(output.getvalue())
            st.pyplot(plt)
            plt.clf()
        else:
            st.markdown(msg)

# --- DOWNLOAD DATA ---
if st.session_state.uploaded_df is not None:
    csv = st.session_state.uploaded_df.to_csv(index=False).encode()
    st.sidebar.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name="your_data.csv",
        mime="text/csv"
    )
