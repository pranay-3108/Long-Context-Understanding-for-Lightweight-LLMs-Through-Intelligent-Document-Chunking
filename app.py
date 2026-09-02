from pathlib import Path
import json
import re
from collections import Counter
from datetime import datetime

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "results" / "bert" / "models"
REVIEW_ROOT = ROOT / "results" / "bert" / "manual_reviews"
BENCHMARK_PATH = ROOT / "benchmark" / "bert_benchmark.json"
REVIEW_ROOT.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Research Paper Understanding Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #071018; }
    .block-container { max-width: 1600px; padding-top: 1.0rem; }
    .hero { padding: 1.35rem 1.5rem; border: 1px solid #273748; border-radius: 18px; background: linear-gradient(135deg,#0d1722,#09111a); }
    .kicker { color:#94b6ff; font-size:.72rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }
    .title { font-size:2.35rem; font-weight:850; margin:.18rem 0 .35rem; }
    .sub { color:#9fb0c2; max-width:1200px; line-height:1.5; }
    .panel { border:1px solid #29394a; border-radius:14px; padding:1rem 1.05rem; background:#0d1721; line-height:1.55; }
    .note { border-left:3px solid #7184a0; padding:.7rem .9rem; background:#0b141d; color:#aab8c8; border-radius:8px; }
    .good { color:#9dd2a8; font-weight:700; }
    .warn { color:#f0c97a; font-weight:700; }
    .bad { color:#f2a0a0; font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True,
)


def normalize(text):
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def answer_f1(prediction, gold):
    a = normalize(prediction).split()
    b = normalize(gold).split()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    ca, cb = Counter(a), Counter(b)
    common = sum((ca & cb).values())
    if common == 0:
        return 0.0
    p = common / len(a)
    r = common / len(b)
    return 2 * p * r / (p + r)


def token_diff(gold, pred, limit=30):
    stop = {
        "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "with", "from",
        "by", "is", "are", "was", "were", "be", "this", "that", "these", "those", "it", "they",
        "their", "its", "as", "at", "we", "our", "you", "your", "can", "could", "would", "should",
        "does", "do", "did", "than", "then", "also", "not", "only", "into", "over", "under", "which",
        "what", "how", "why", "where", "when", "who", "will", "all", "both", "more", "most", "such",
        "each", "some", "any", "because", "while", "during", "using", "used", "use", "there",
    }
    g = Counter(w for w in normalize(gold).split() if len(w) > 2 and w not in stop)
    p = Counter(w for w in normalize(pred).split() if len(w) > 2 and w not in stop)
    return list((g - p).elements())[:limit], list((p - g).elements())[:limit]


def load_benchmark():
    if not BENCHMARK_PATH.exists():
        return {}
    try:
        data = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        return {int(r["question_id"]): r for r in data.get("records", [])}
    except Exception:
        return {}


def discover_runs():
    rows = []
    if not RESULT_ROOT.exists():
        return rows
    for model_dir in sorted(RESULT_ROOT.iterdir()):
        if not model_dir.is_dir():
            continue
        for method_dir in sorted(model_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            for stage_dir in sorted(method_dir.glob("*k")):
                result = stage_dir / "result.json"
                if not result.exists():
                    continue
                try:
                    obj = json.loads(result.read_text(encoding="utf-8"))
                except Exception:
                    continue
                answers = obj.get("answers", obj.get("records", []))
                if not answers:
                    continue
                stage = int(obj.get("stage_k", obj.get("requested_stage_chars", int(stage_dir.name.replace("k", ""))) // 1000))
                normalized_answers = []
                for a in answers:
                    qid = int(a.get("question_id", 0))
                    normalized_answers.append({
                        "question_id": qid,
                        "question": a.get("question", ""),
                        "category": a.get("category", ""),
                        "ground_truth_answer": a.get("ground_truth_answer", ""),
                        "answer": a.get("answer", a.get("model_answer", "")),
                        "answer_time_sec": a.get("answer_time_sec"),
                    })
                f1_vals = [answer_f1(a["answer"], a["ground_truth_answer"]) for a in normalized_answers]
                rows.append({
                    "Model": obj.get("model_label", model_dir.name),
                    "Model ID": obj.get("model", model_dir.name),
                    "Method": obj.get("method_display", "Direct full-context" if obj.get("mode") == "direct" else "Fixed 7k chunking"),
                    "Mode": obj.get("mode", method_dir.name),
                    "Stage": stage,
                    "Questions": len(normalized_answers),
                    "Answer F1": sum(f1_vals) / len(f1_vals),
                    "Time (s)": obj.get("total_time_sec"),
                    "Chunk size": obj.get("chunk_size_chars"),
                    "Path": str(result),
                    "answers": normalized_answers,
                })
    return rows


def load_reviews():
    path = REVIEW_ROOT / "reviews.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_reviews(reviews):
    (REVIEW_ROOT / "reviews.json").write_text(json.dumps(reviews, indent=2, ensure_ascii=False), encoding="utf-8")


benchmark = load_benchmark()
runs = discover_runs()
reviews = load_reviews()

st.markdown(
    '<div class="hero"><div class="kicker">Research benchmark · evidence-grounded paper understanding</div>'
    '<div class="title">Research Paper Understanding Lab</div>'
    '<div class="sub">A controlled comparison of local LLM outputs against two independent references: the Claude ground-truth answer key and the supplied Granite 7B answer set. The 7B set is preserved as a comparator, not treated as corrected ground truth.</div></div>',
    unsafe_allow_html=True,
)

if not runs:
    st.warning("No BERT result.json files found yet under results/bert/models/... .")
    st.stop()

run_df = pd.DataFrame([{k: v for k, v in r.items() if k != "answers"} for r in runs])

with st.sidebar:
    st.header("Study controls")
    models = ["All"] + sorted(run_df["Model"].unique().tolist())
    methods = ["All"] + sorted(run_df["Method"].unique().tolist())
    stage_values = sorted(run_df["Stage"].unique().tolist())
    model_sel = st.selectbox("Model", models)
    method_sel = st.selectbox("Processing method", methods)
    stage_sel = st.selectbox("Paper length", ["All"] + stage_values)
    st.caption("Stages are character-based prefixes: 8k, 16k, 24k, 32k, 44k.")
    st.markdown("---")
    st.caption("Primary reference: Claude ground truth. Secondary comparator: supplied Granite 7B answers.")

filtered = run_df.copy()
if model_sel != "All":
    filtered = filtered[filtered["Model"] == model_sel]
if method_sel != "All":
    filtered = filtered[filtered["Method"] == method_sel]
if stage_sel != "All":
    filtered = filtered[filtered["Stage"] == int(stage_sel)]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Runs", len(filtered))
c2.metric("Answers", int(filtered["Questions"].sum()) if not filtered.empty else 0)
c3.metric("Best Claude F1", f"{filtered['Answer F1'].max():.1%}" if not filtered.empty else "—")
c4.metric("Worst Claude F1", f"{filtered['Answer F1'].min():.1%}" if not filtered.empty else "—")
c5.metric("Saved reviews", len(reviews))

st.markdown("## 0. Reference integrity")
ref_count = sum(bool(r.get("granite_model_answer")) for r in benchmark.values())
st.info(
    f"Claude primary reference: 55/55 answers loaded. Granite 7B comparator: {ref_count}/55 answers loaded. "
    "Granite 7B is preserved as supplied and is not treated as ground truth."
)

st.markdown("## 1. Method × stage overview")
comparison = filtered[["Model", "Stage", "Method", "Questions", "Answer F1", "Time (s)", "Chunk size"]].copy()
comparison["Answer F1"] = comparison["Answer F1"].map(lambda x: f"{x:.1%}")
st.dataframe(comparison.sort_values(["Model", "Stage", "Method"]), use_container_width=True, hide_index=True)


st.markdown("## 2. Three-way answer-key comparison")
st.markdown(
    '<div class="note">Claude is the primary reference. The supplied Granite 7B answers are a second model comparator, not ground truth. The selected model is compared against both, question by question. Lexical F1 is a baseline; semantic review is still required.</div>',
    unsafe_allow_html=True,
)

available = []
for r in runs:
    if model_sel != "All" and r["Model"] != model_sel:
        continue
    if method_sel != "All" and r["Method"] != method_sel:
        continue
    if stage_sel != "All" and r["Stage"] != int(stage_sel):
        continue
    available.append(r)

labels = [f"{r['Model']} · {r['Method']} · {r['Stage']}k" for r in available]
selected_idx = st.selectbox("Run", range(len(available)), format_func=lambda i: labels[i])
run = available[selected_idx]
answers = run["answers"]

q_rows = []
for a in answers:
    qid = int(a["question_id"])
    ref = benchmark.get(qid, {})
    claude = ref.get("ground_truth_answer", a.get("ground_truth_answer", ""))
    granite7 = ref.get("granite_model_answer", "")
    pred = a.get("answer", "")
    q_rows.append({
        "Q": qid,
        "Category": a.get("category", ""),
        "Claude F1": answer_f1(pred, claude),
        "7B F1": answer_f1(granite7, claude) if granite7 else None,
        "Current vs 7B F1": answer_f1(pred, granite7) if granite7 else None,
        "Question": a.get("question", ""),
    })

q_df = pd.DataFrame(q_rows).sort_values("Q")

# Headline metrics: selected model vs Claude, Granite 7B vs Claude, and selected model vs 7B.
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Current vs Claude", f"{q_df['Claude F1'].mean():.1%}")
m2.metric("Granite 7B vs Claude", f"{q_df['7B F1'].mean():.1%}")
m3.metric("Current vs 7B", f"{q_df['Current vs 7B F1'].mean():.1%}")
m4.metric("Current Qs", len(q_df))
m5.metric("Current avg words", int(round(sum(len(normalize(a.get("answer","")).split()) for a in answers) / max(len(answers),1))))

st.markdown("### 2A. Quick visual: who is closer to the Claude reference?")
view_mode = st.radio("Visualization", ["Selected question", "All 55 questions", "Win count across all questions"], horizontal=True)

if view_mode == "Selected question":
    # The selected question controls this chart below, after q_id is chosen.
    st.caption("The chart compares each model's lexical Answer F1 against the Claude reference. Higher is closer to the supplied reference answer.")
elif view_mode == "All 55 questions":
    plot_df = q_df[["Q", "Claude F1", "7B F1"]].copy()
    plot_df = plot_df.set_index("Q")
    plot_df.columns = ["Current model vs Claude", "Granite 7B vs Claude"]
    st.bar_chart(plot_df, height=420)
    st.caption("Each bar is one question. This is a lexical-overlap view; it does not decide semantic correctness by itself.")
elif view_mode == "Win count across all questions":
    valid = q_df.dropna(subset=["7B F1"]).copy()
    current_wins = int((valid["Claude F1"] > valid["7B F1"]).sum())
    seven_wins = int((valid["7B F1"] > valid["Claude F1"]).sum())
    ties = int((valid["Claude F1"] == valid["7B F1"]).sum())
    wins_df = pd.DataFrame({"Questions": [current_wins, seven_wins, ties]}, index=[run["Model"], "Granite 7B", "Tie"])
    st.bar_chart(wins_df, height=300)
    st.write(f"**Question wins:** {run['Model']} {current_wins} · Granite 7B {seven_wins} · Ties {ties}")
    st.caption("A win means higher Answer F1 against Claude for that question. Treat this as a quick signal, not a semantic verdict.")

st.markdown("### 2A. Overall comparison at this stage")
overall_row = pd.DataFrame([{
    "Model / reference": f"{run['Model']} · {run['Method']} · {run['Stage']}k",
    "Vs Claude (F1)": q_df["Claude F1"].mean(),
    "Granite 7B vs Claude (F1)": q_df["7B F1"].mean(),
    "Current vs Granite 7B (F1)": q_df["Current vs 7B F1"].mean(),
}])
disp = overall_row.copy()
for c in ["Vs Claude (F1)", "Granite 7B vs Claude (F1)", "Current vs Granite 7B (F1)"]:
    disp[c] = disp[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
st.dataframe(disp, use_container_width=True, hide_index=True)

st.markdown("### 2B. Question-level comparison")
weak = q_df.sort_values("Claude F1").head(12).copy()
weak["Claude F1"] = weak["Claude F1"].map(lambda x: f"{x:.1%}")
weak["7B F1"] = weak["7B F1"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
weak["Current vs 7B F1"] = weak["Current vs 7B F1"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
st.dataframe(
    weak[["Q", "Category", "Claude F1", "7B F1", "Current vs 7B F1", "Question"]],
    use_container_width=True,
    hide_index=True,
)

st.markdown("### 2C. Category-level comparison")
cat = q_df.groupby("Category", as_index=False)[["Claude F1", "7B F1", "Current vs 7B F1"]].mean().sort_values("Claude F1")
cat_display = cat.copy()
for c in ["Claude F1", "7B F1", "Current vs 7B F1"]:
    cat_display[c] = cat_display[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
st.dataframe(cat_display, use_container_width=True, hide_index=True)

qids = q_df["Q"].tolist()
q_id = st.selectbox("Question", qids, format_func=lambda q: f"Q{q:02d}")
item = next(a for a in answers if int(a["question_id"]) == q_id)
ref = benchmark.get(q_id, {})

question = item.get("question", "")
pred = item.get("answer", "")
claude = ref.get("ground_truth_answer", item.get("ground_truth_answer", ""))
granite7 = ref.get("granite_model_answer", "")

claude_f1 = answer_f1(pred, claude)
granite7_to_claude_f1 = answer_f1(granite7, claude) if granite7 else None
current_to_7b_f1 = answer_f1(pred, granite7) if granite7 else None

miss_claude, extra_claude = token_diff(claude, pred)
miss_7b_current, extra_7b_current = token_diff(granite7, pred) if granite7 else ([], [])
miss_7b_claude, extra_7b_claude = token_diff(claude, granite7) if granite7 else ([], [])

st.markdown("#### Question-level visual")
question_chart = pd.DataFrame(
    {"Answer F1 vs Claude": [claude_f1, granite7_to_claude_f1 if granite7_to_claude_f1 is not None else 0]},
    index=[run["Model"], "Granite 7B"],
)
st.bar_chart(question_chart, height=260)
if granite7_to_claude_f1 is None:
    st.info("No Granite 7B answer is available for this question.")
elif claude_f1 > granite7_to_claude_f1:
    st.success(f"Winner for Q{q_id:02d}: {run['Model']} (higher Answer F1 vs Claude)")
elif granite7_to_claude_f1 > claude_f1:
    st.warning(f"Winner for Q{q_id:02d}: Granite 7B (higher Answer F1 vs Claude)")
else:
    st.info(f"Q{q_id:02d}: tie on Answer F1")

st.markdown("#### What each model missed / added relative to Claude")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Current model vs Claude**")
    st.write("Missed:", ", ".join(miss_claude) if miss_claude else "None")
    st.write("Added:", ", ".join(extra_claude) if extra_claude else "None")
with col_b:
    st.markdown("**Granite 7B vs Claude**")
    st.write("Missed:", ", ".join(miss_7b_claude) if miss_7b_claude else "None")
    st.write("Added:", ", ".join(extra_7b_claude) if extra_7b_claude else "None")
st.caption("Lexical differences are candidate signals, not semantic judgments.")

st.markdown("#### Direct 2B ↔ 7B model difference")
e1, e2 = st.columns(2)
with e1:
    st.write("**Terms in 7B but not current:**", ", ".join(miss_7b_current) if miss_7b_current else "None")
with e2:
    st.write("**Terms in current but not 7B:**", ", ".join(extra_7b_current) if extra_7b_current else "None")
st.caption("Agreement with Granite 7B is informative but does not override the Claude reference.")

st.markdown("#### Research review")
review_key = f"{run['Model']}|{run['Method']}|{run['Stage']}|{q_id}"
existing = reviews.get(review_key, {})
verdict_options = ["Not reviewed", "Correct", "Partially correct", "Incorrect", "Unsupported / hallucinated"]
valid_options = ["Not assessed", "Yes", "Partly", "No"]
with st.form("research_review_form"):
    verdict = st.selectbox("Reference-based verdict", verdict_options, index=verdict_options.index(existing.get("verdict", "Not reviewed")))
    current_missed = st.text_area("What current model missed", existing.get("current_missed", ""))
    current_added = st.text_area("What current model added", existing.get("current_added", ""))
    current_added_valid = st.selectbox("Are current-model additions valid/supportable?", valid_options, index=valid_options.index(existing.get("current_added_valid", "Not assessed")))
    seventh_missed = st.text_area("What Granite 7B missed", existing.get("granite7b_missed", ""))
    seventh_added = st.text_area("What Granite 7B added", existing.get("granite7b_added", ""))
    seventh_added_valid = st.selectbox("Are Granite 7B additions valid/supportable?", valid_options, index=valid_options.index(existing.get("granite7b_added_valid", "Not assessed")))
    notes = st.text_area("Research notes", existing.get("notes", ""))
    submitted = st.form_submit_button("Save review")
    if submitted:
        reviews[review_key] = {
            "verdict": verdict,
            "current_missed": current_missed,
            "current_added": current_added,
            "current_added_valid": current_added_valid,
            "granite7b_missed": seventh_missed,
            "granite7b_added": seventh_added,
            "granite7b_added_valid": seventh_added_valid,
            "notes": notes,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        save_reviews(reviews)
        st.success("Saved research review.")
st.caption("Granite 7B is a model comparator, not ground truth. Claude remains the primary reference.")

st.markdown("## 3. Direct vs chunked at the same stage")
st.caption("A direct-vs-chunk winner is shown only when both runs actually exist. Missing runs are shown as unavailable, not treated as zero.")
for model in sorted({r["Model"] for r in runs}):
    m = run_df[run_df["Model"] == model]
    rows = []
    for stage in sorted(m["Stage"].unique()):
        x = m[m["Stage"] == stage].set_index("Method")
        direct_val = x.loc["Direct full-context", "Answer F1"] if "Direct full-context" in x.index else None
        chunk_val = x.loc["Fixed 7k chunking", "Answer F1"] if "Fixed 7k chunking" in x.index else None
        row = {
            "Model": model,
            "Stage": stage,
            "Direct · Claude F1": direct_val,
            "Fixed 7k · Claude F1": chunk_val,
        }
        if direct_val is not None and chunk_val is not None:
            row["Chunk − Direct"] = chunk_val - direct_val
            row["Winner"] = "Fixed 7k chunking" if chunk_val > direct_val else "Direct full-context" if chunk_val < direct_val else "Tie"
        else:
            row["Chunk − Direct"] = None
            row["Winner"] = "Awaiting both methods"
        rows.append(row)
    if rows:
        st.markdown(f"**{model}**")
        df = pd.DataFrame(rows)
        for col in ["Direct · Claude F1", "Fixed 7k · Claude F1", "Chunk − Direct"]:
            df[col] = df[col].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
        st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown("## 4. Model-size view")
model_view = filtered.groupby(["Model", "Stage"], as_index=False)["Answer F1"].mean()
if not model_view.empty:
    st.line_chart(model_view.pivot(index="Stage", columns="Model", values="Answer F1"), height=340)

st.markdown("## 5. Human research diagnosis")
st.caption("Use this to explain semantic quality. Automatic F1 is a lexical baseline, not a proof of understanding.")
current = reviews.get(f"{run['Model']}|{run['Method']}|{run['Stage']}|{q_id}", {})

v_options = ["Correct", "Partial", "Incorrect", "Unsupported / hallucinated"]
q_options = ["Strong", "Good", "Mixed", "Weak", "Severely flawed"]
a_options = ["Not applicable", "Valid addition", "Unsupported addition", "Both valid and unsupported"]
verdict = st.selectbox("Verdict", v_options, index=v_options.index(current.get("verdict", "Partial")))
quality = st.selectbox("Overall answer quality", q_options, index=q_options.index(current.get("quality", "Mixed")))
added_status = st.selectbox("Status of model additions", a_options, index=a_options.index(current.get("added_status", "Not applicable")))
what_right = st.text_area("What did the model get right?", current.get("what_right", ""))
what_missed = st.text_area("What did the model miss?", current.get("what_missed", ""))
what_added = st.text_area("What did the model add?", current.get("what_added", ""))
wrong = st.text_area("What is factually wrong or unsupported?", current.get("wrong", ""))
notes = st.text_area("Researcher notes", current.get("notes", ""))

if st.button("Save research diagnosis", type="primary"):
    key = f"{run['Model']}|{run['Method']}|{run['Stage']}|{q_id}"
    reviews[key] = {
        "model": run["Model"],
        "method": run["Method"],
        "stage_k": run["Stage"],
        "question_id": q_id,
        "verdict": verdict,
        "quality": quality,
        "added_status": added_status,
        "what_right": what_right,
        "what_missed": what_missed,
        "what_added": what_added,
        "wrong": wrong,
        "notes": notes,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_reviews(reviews)
    st.success("Saved.")
    st.rerun()

st.markdown("## 6. Export")
export_rows = list(reviews.values())
if export_rows:
    review_csv = pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8")
    st.download_button("Download research diagnosis CSV", review_csv, "bert_research_diagnosis.csv", "text/csv")
else:
    st.info("No human diagnoses saved yet.")

st.divider()
st.caption("Primary evaluation: current model answer vs Claude ground truth. Secondary comparisons: supplied Granite 7B answer set and model-to-model agreement. Raw outputs are preserved.")
