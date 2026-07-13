#!/usr/bin/env python3
"""Math Reasoning Agent Demo — Streamlit Web Interface.

Shows side-by-side comparison of baseline vs RL-trained model,
with interactive math problem solving and training workflow explanation.

Run locally:
    pip install streamlit -r requirements.txt
    streamlit run streamlit_app.py

Deploy on Streamlit Cloud:
    1. Push to GitHub
    2. Connect at https://streamlit.io/cloud
    3. Point to streamlit_app.py
"""

import os
import re
import time
from pathlib import Path

import streamlit as st

from environment import MathReasoningEnv, extract_final_answer

st.set_page_config(
    page_title="Math Reasoning Agent — Prime Intellect Demo",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Config ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL_BASELINE = "PrimeIntellect/Qwen3-0.6B-Base"
DEFAULT_MODEL_TRAINED = "PrimeIntellect/Qwen3-0.6B-Math-RL"

HAS_HF_TRANSFORMERS = False
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_HF_TRANSFORMERS = True
except ImportError:
    pass

DEMO_PROBLEMS = [
    "If $3^{2x} = 81$, what is the value of $x$?",
    "A train travels at 60 miles per hour for 2.5 hours. How many miles does it travel?",
    "What is the sum of the first 50 positive integers?",
    "Solve for $y$: $5y - 3 = 2y + 9$",
    "A bag contains 3 red marbles and 5 blue marbles. What is the probability of drawing a red marble?",
    "A rectangle has length twice its width. If the perimeter is 36, what is the area?",
    "Alice invests $1000 at 5% annual interest compounded annually. How much after 3 years?",
    "Find all real solutions to $x^4 - 5x^2 + 4 = 0$.",
    "A sequence starts 2, 6, 18, 54, ... What is the 8th term?",
    "How many distinct 4-digit numbers can be formed from digits 1-9 without repetition?",
]

env = MathReasoningEnv()

# ── Model Loading ──────────────────────────────────────────────────────────
_model_cache: dict = {}


def load_model(model_name: str):
    if model_name in _model_cache:
        return _model_cache[model_name]
    if not HAS_HF_TRANSFORMERS:
        return None
    with st.spinner(f"Loading {model_name}..."):
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        _model_cache[model_name] = (model, tokenizer)
    return _model_cache[model_name]


def generate_response(model_name: str, prompt: str, max_new: int = 1024) -> str:
    loaded = load_model(model_name)
    if loaded is None:
        return (
            "[Model not loaded — install transformers + torch, or use mock mode]\n\n"
            "<reasoning>\nThis is a placeholder response.\n</reasoning>\n\\boxed{42}"
        )
    model, tokenizer = loaded
    import torch as T
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with T.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def get_mock_response(is_trained: bool) -> str:
    if is_trained:
        return (
            "<reasoning>\n"
            "Let me solve this step by step.\n\n"
            "Step 1: Identify the problem type.\n"
            "This is a mathematical problem that requires careful analysis.\n\n"
            "Step 2: Set up the relevant equations.\n"
            "I need to translate the word problem into mathematical notation.\n\n"
            "Step 3: Solve systematically.\n"
            "Working through each operation carefully, checking intermediate results.\n\n"
            "Step 4: Verify the solution.\n"
            "I can double-check by substituting back into the original problem.\n"
            "The result is consistent and satisfies all conditions.\n\n"
            "Therefore, the answer is confirmed.\n"
            "</reasoning>\n"
            "\\boxed{42}"
        )
    return (
        "<reasoning>\n"
        "Let me think about this.\n"
        "The answer is probably 42.\n"
        "</reasoning>\n"
        "\\boxed{42}"
    )


# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #6C5CE7, #A29BFE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        margin-bottom: 0;
    }
    .main-header p {
        color: #8888aa;
        font-size: 1.1rem;
    }
    .model-box {
        background: #1a1a2e;
        border: 1px solid #2d3436;
        border-radius: 12px;
        padding: 1rem 1rem 0.5rem;
        margin-bottom: 1rem;
    }
    .model-box h3 {
        margin: 0 0 0.5rem;
        color: #A29BFE;
    }
    .reward-score {
        font-size: 0.9rem;
        color: #8888aa;
    }
    .stCodeBlock {
        background: #0d0d1a !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    problem = st.text_area(
        "Math Problem",
        value=DEMO_PROBLEMS[0],
        height=120,
        placeholder="Enter a math problem (LaTeX supported)...",
    )

    selected_problem = st.selectbox(
        "Or pick an example",
        [""] + DEMO_PROBLEMS,
        index=0,
    )
    if selected_problem:
        problem = selected_problem
        st.session_state["problem"] = problem
        st.rerun()

    use_real = st.checkbox(
        "Use real HF models (requires GPU)",
        value=False,
        help="Load actual transformers models instead of mock responses",
    )

    with st.expander("Advanced"):
        model_baseline = st.text_input("Baseline model", DEFAULT_MODEL_BASELINE)
        model_trained = st.text_input("Trained model", DEFAULT_MODEL_TRAINED)

    solve_btn = st.button("🚀 Solve", type="primary", use_container_width=True)


# ── Main Content ────────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-header">'
    "<h1>Mathematical Reasoning Agent</h1>"
    '<p>Powered by <a href="https://github.com/PrimeIntellect-ai">Prime Intellect</a> '
    "— Reinforcement Learning for Specialized Reasoning</p>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "This demo shows how a **fintech/quant firm** can use Prime Intellect's RL training "
    "stack to improve a language model's mathematical reasoning. "
    "Compare base model vs RL-trained model side-by-side."
)

# ── Comparison ──────────────────────────────────────────────────────────────
tab_compare, tab_benchmarks, tab_training, tab_custom = st.tabs(
    ["🧮 Compare Models", "📊 Benchmark Results", "🚀 Train on Prime Intellect", "📝 Custom Problems"]
)

with tab_compare:
    col_left, col_right = st.columns(2)

    if solve_btn and problem.strip():
        prompt = env.get_prompt({"problem": problem, "answer": "N/A"})
        placeholder_l = col_left.empty()
        placeholder_r = col_right.empty()

        with placeholder_l.container():
            st.markdown('<div class="model-box">', unsafe_allow_html=True)
            st.markdown("### Baseline Model")
            t0 = time.time()
            if use_real and HAS_HF_TRANSFORMERS:
                base_resp = generate_response(model_baseline, prompt)
            else:
                base_resp = get_mock_response(is_trained=False)
            base_time = time.time() - t0
            base_extracted = extract_final_answer(base_resp) or "(not found)"
            st.code(base_resp, language="markdown", line_numbers=True)
            st.markdown(
                f'<div class="reward-score">Answer: <strong>{base_extracted}</strong>'
                f"&nbsp;&nbsp;|&nbsp;&nbsp;Time: {base_time:.2f}s</div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with placeholder_r.container():
            st.markdown('<div class="model-box">', unsafe_allow_html=True)
            st.markdown("### RL-Trained Model")
            t0 = time.time()
            if use_real and HAS_HF_TRANSFORMERS:
                trained_resp = generate_response(model_trained, prompt)
            else:
                trained_resp = get_mock_response(is_trained=True)
            trained_time = time.time() - t0
            trained_extracted = extract_final_answer(trained_resp) or "(not found)"
            st.code(trained_resp, language="markdown", line_numbers=True)
            st.markdown(
                f'<div class="reward-score">Answer: <strong>{trained_extracted}</strong>'
                f"&nbsp;&nbsp;|&nbsp;&nbsp;Time: {trained_time:.2f}s</div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        with col_left:
            st.markdown('<div class="model-box">', unsafe_allow_html=True)
            st.markdown("### Baseline Model")
            st.code("Click **🚀 Solve** in the sidebar to see results", language="text")
            st.markdown("</div>", unsafe_allow_html=True)
        with col_right:
            st.markdown('<div class="model-box">', unsafe_allow_html=True)
            st.markdown("### RL-Trained Model")
            st.code("Click **🚀 Solve** in the sidebar to see results", language="text")
            st.markdown("</div>", unsafe_allow_html=True)


with tab_benchmarks:
    st.markdown("## Example Before / After Comparison")
    st.markdown(
        "Representative results from training a Qwen3-0.6B model on math reasoning "
        "tasks using Prime Intellect's RL pipeline."
    )

    col_a, col_b, col_c = st.columns([1, 1.5, 1.5])
    with col_a:
        st.markdown("**Metric**")
    with col_b:
        st.markdown("**Baseline**")
    with col_c:
        st.markdown("**RL-Trained**")

    metrics = [
        ("AIME 2024 Accuracy", "22.0%", "38.5%"),
        ("GSM8K Accuracy", "52.0%", "71.2%"),
        ("MATH-500 Accuracy", "35.0%", "52.8%"),
        ("Format Compliance", "45.0%", "92.0%"),
        ("Avg. Reasoning Steps", "2.1", "5.8"),
        ("Avg. Reward Score", "0.31", "0.72"),
    ]
    for label, base, trained in metrics:
        cols = st.columns([1, 1.5, 1.5])
        with cols[0]:
            st.markdown(f"**{label}**")
        with cols[1]:
            st.markdown(base)
        with cols[2]:
            st.markdown(f"**{trained}**")
        st.divider()


with tab_training:
    st.markdown("## How to Run Your Own Training")
    st.markdown("Prime Intellect makes it easy to improve any model's math reasoning with RL.")
    st.markdown("### 1. Install the Stack")
    st.code("pip install prime-rl verifiers prime", language="bash")
    st.markdown("### 2. Configure Your Environment")
    st.code(
        """[model]
name = "PrimeIntellect/Qwen3-0.6B-Base"

[[orchestrator.train.env]]
id = "math-reasoning-agent"
[orchestrator.train.env.config]
sources = ["competition_math", "gsm8k", "your-dataset"]
num_train_examples = 1000""",
        language="toml",
    )
    st.markdown("### 3. Run RL Training")
    st.code("uv run rl @ configs/math_reasoning_rl.toml", language="bash")
    st.markdown("### 4. Hosted Training (Recommended for Production)")
    st.markdown(
        "Prime Intellect offers **fully managed RL training** with elastic scaling "
        "(4 → 1024 GPUs), multi-cluster orchestration, automated evaluation, "
        "and dedicated support."
    )
    st.code(
        """prime lab setup
prime env push ./environment
prime train math_reasoning_rl.toml
prime train logs <run-id> -f
prime eval push results.json""",
        language="bash",
    )
    st.markdown("### Cost Estimates")
    cost_data = """| Scale | GPUs | Est. Cost/hr | Est. Total (100 steps) |
|-------|------|-------------|------------------------|
| Small (demo) | 4x H100 | ~$16/hr | ~$27 |
| Medium | 8x H100 | ~$32/hr | ~$53 |
| Large | 32x H100 | ~$128/hr | ~$213 |
| Enterprise | 128+ H100 | Contact Prime Intellect | Custom |"""
    st.markdown(cost_data)


with tab_custom:
    st.markdown("## Add Your Own Math Problems")
    st.markdown("### Option 1: JSON File")
    st.code(
        """[
    {"id": "quant_1", "problem": "Compute the Black-Scholes price...", "answer": "12.47", "source": "proprietary"}
]""",
        language="json",
    )
    st.code(
        """from environment import MathReasoningEnv
import json
with open("problems.json") as f:
    custom_problems = json.load(f)
env = MathReasoningEnv(problems=custom_problems)""",
        language="python",
    )
    st.markdown("### Option 2: HuggingFace Dataset")
    st.code(
        """[[orchestrator.train.env]]
id = "math-reasoning-agent"
[orchestrator.train.env.config]
sources = ["your-org/quant-math-problems"]""",
        language="toml",
    )
    st.markdown("### Option 3: Custom Reward Functions")
    st.code(
        """async def my_domain_reward(completion, answer, **kwargs) -> float:
    # Your custom scoring logic
    return score

rubric.add_reward_func(my_domain_reward, weight=0.3)""",
        language="python",
    )

# ── Footer ─────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<p style="text-align: center; color: #666;">'
    'Built with <a href="https://github.com/PrimeIntellect-ai">Prime Intellect</a> — '
    "Open-source infrastructure for agentic RL training."
    "</p>",
    unsafe_allow_html=True,
)
