# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import openai
import re

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="GeoSci-Agent 🌍",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌍 GeoSci-Agent")
st.caption("Built with Python, pandas, statsmodels, Streamlit, and LLM APIs")

st.markdown("""
### 🤖 AI-driven Scientific Data Analysis Assistant

Upload your CSV / Excel data and perform analysis using **natural language**:

- 📊 Exploratory Data Analysis (EDA)
- 🔗 Correlation Analysis
- 📈 Regression Modeling
- 📉 Distribution Visualization

💡 Features:
- Smart column recognition (case / underscore insensitive)
- Built-in statistical tools (VIF, partial correlation)
- Optional LLM-powered code generation

🚀 Use cases: Environmental Science / Public Health / Finance / General Data Science
""")

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("OpenAI API Key (optional)", type="password")

    st.markdown("---")
    st.header("📂 Upload Data")
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx", "xls"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
            st.success(f"Loaded: {uploaded_file.name}")
            st.dataframe(df.head(3))
            st.session_state['data'] = df
        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        st.info("Upload a dataset to begin.")

# -------------------- Chat Init --------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content":
         "Hi! I'm 🌍 **GeoSci-Agent** 🤖\n\n"
         "Try the following:\n\n"
         "🔍 **EDA / Data Quality**\n"
         "→ e.g., check missing values\n\n"
         "🔗 **Correlation Analysis**\n"
         "→ analyze correlation between SST and CHL\n\n"
         "📈 **Regression**\n"
         "→ use SST, PAR to predict N2_fixation\n\n"
         "📊 **Distribution**\n"
         "→ plot distribution of temperature\n\n"
         "🧠 **Advanced (LLM)**\n"
         "→ ask anything to generate custom analysis code\n\n"
         "💡 Column names are matched flexibly"
         }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------- Utils --------------------
def normalize(text):
    return re.sub(r'[_\s\-]+', '', str(text).lower())

def extract_columns(prompt, df):
    prompt_n = normalize(prompt)
    found = []

    for col in df.columns:
        col_n = normalize(col)

        if col_n in prompt_n:
            found.append(col)
            continue

        tokens = re.split(r'[_\s]', col.lower())
        if any(t in prompt.lower() for t in tokens if len(t) > 2):
            found.append(col)

    return list(dict.fromkeys(found))

# -------------------- Built-in Analysis --------------------
def builtin_missing(df):
    miss = df.isnull().sum()
    miss = miss[miss > 0]

    if not miss.empty:
        st.dataframe(miss.rename("Missing Count"))
        st.warning("Missing values detected")
    else:
        st.success("No missing values")

def partial_corr(df, x, y, controls):
    df_clean = df[[x, y] + controls].dropna()
    Xc = sm.add_constant(df_clean[controls])

    rx = sm.OLS(df_clean[x], Xc).fit().resid
    ry = sm.OLS(df_clean[y], Xc).fit().resid

    return np.corrcoef(rx, ry)[0,1]

def compute_vif(X):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    Xc = sm.add_constant(X)

    data = []
    for i in range(1, Xc.shape[1]):
        data.append({
            "feature": X.columns[i-1],
            "VIF": round(variance_inflation_factor(Xc.values, i), 2)
        })
    return pd.DataFrame(data)

def builtin_correlation(df, prompt):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cols = extract_columns(prompt, df)
    use = cols if len(cols) >= 2 else num_cols

    if len(use) < 2:
        st.warning("Need at least 2 variables")
        return

    corr = df[use].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    target = use[-1]
    st.markdown(f"### 🎯 Correlation with `{target}`")

    for c, v in corr[target].drop(target).items():
        st.write(f"{c} ↔ {target}: {v:.2f}")

    # Partial corr
    st.markdown("### 🧪 Partial Correlation")
    if len(use) >= 3:
        try:
            pc = partial_corr(df, use[0], target, use[1:-1])
            st.info(f"Partial corr: {pc:.2f}")
        except:
            st.info("Failed to compute")

    # DAG hint
    st.markdown("### 🧭 Causal Hint (Heuristic)")
    for c in use[:-1]:
        val = corr.loc[c, target]
        if abs(val) > 0.3:
            st.write(f"{c} → {target} ? (corr={val:.2f})")

def builtin_regression(df, prompt):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cols = extract_columns(prompt, df)

    if len(cols) < 2:
        cols = num_cols[:2]

    Y = cols[-1]
    X = cols[:-1]

    data = df[X + [Y]].dropna()
    Xc = sm.add_constant(data[X])
    Yc = data[Y]

    model = sm.OLS(Yc, Xc).fit()

    st.dataframe(pd.read_html(model.summary().tables[1].as_html())[0])

    st.markdown("### 📊 VIF")
    st.dataframe(compute_vif(data[X]))

def builtin_distribution(df, prompt):
    cols = extract_columns(prompt, df)
    if not cols:
        return

    col = cols[0]
    fig, ax = plt.subplots()
    sns.histplot(df[col].dropna(), kde=True, ax=ax)
    st.pyplot(fig)

# -------------------- LLM --------------------
def generate_code(prompt, df, api_key):
    if not api_key:
        return None

    client = openai.OpenAI(api_key=api_key)

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )

    return res.choices[0].message.content

# -------------------- Chat --------------------
if prompt := st.chat_input("Enter your analysis request..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role":"user","content":prompt})

    if 'data' not in st.session_state:
        st.warning("Upload data first")
        st.stop()

    df = st.session_state['data']

    with st.chat_message("assistant"):
        if "missing" in prompt:
            builtin_missing(df)
        elif "corr" in prompt:
            builtin_correlation(df, prompt)
        elif "reg" in prompt or "predict" in prompt:
            builtin_regression(df, prompt)
        elif "distribution" in prompt:
            builtin_distribution(df, prompt)
        else:
            if api_key:
                code = generate_code(prompt, df, api_key)
                st.code(code)
            else:
                st.warning("No API key provided")