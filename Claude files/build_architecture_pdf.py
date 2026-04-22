"""Generate the interview-reference architecture PDF for Project 3.

Dual-lens format: plain-language summaries for non-technical readers,
technical drill-downs for architects. Output: architecture_interview.pdf.
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "architecture_interview.pdf"

styles = getSampleStyleSheet()

NAVY = colors.HexColor("#0b2340")
GREY = colors.HexColor("#555555")
LIGHT_GREY = colors.HexColor("#888888")
CREAM = colors.HexColor("#f7f5ee")
PALE_BLUE = colors.HexColor("#eef3fa")
CODE_BG = colors.HexColor("#f4f4f4")

TITLE = ParagraphStyle(
    "Title", parent=styles["Title"], fontSize=22, leading=26, spaceAfter=2,
    textColor=colors.HexColor("#1a1a1a"),
)
SUBTITLE = ParagraphStyle(
    "Subtitle", parent=styles["Normal"], fontSize=11, leading=14, spaceAfter=16,
    textColor=GREY,
)
H1 = ParagraphStyle(
    "H1", parent=styles["Heading1"], fontSize=15, leading=19, spaceBefore=16,
    spaceAfter=6, textColor=NAVY,
)
H2 = ParagraphStyle(
    "H2", parent=styles["Heading2"], fontSize=12, leading=15, spaceBefore=10,
    spaceAfter=4, textColor=NAVY,
)
H3 = ParagraphStyle(
    "H3", parent=styles["Heading3"], fontSize=10.5, leading=13, spaceBefore=6,
    spaceAfter=2, textColor=NAVY, fontName="Helvetica-Bold",
)
BODY = ParagraphStyle(
    "Body", parent=styles["BodyText"], fontSize=10, leading=14, spaceAfter=6,
    alignment=TA_JUSTIFY,
)
BULLET = ParagraphStyle(
    "Bullet", parent=BODY, leftIndent=12, spaceAfter=2, alignment=TA_LEFT,
)
PLAIN = ParagraphStyle(
    "Plain", parent=BODY, backColor=CREAM, borderPadding=8, leftIndent=4,
    rightIndent=4, spaceBefore=4, spaceAfter=8,
)
TECH = ParagraphStyle(
    "Tech", parent=BODY, backColor=PALE_BLUE, borderPadding=8, leftIndent=4,
    rightIndent=4, spaceBefore=4, spaceAfter=8,
)
CODE = ParagraphStyle(
    "Code", parent=styles["Code"], fontSize=8.5, leading=11, leftIndent=8,
    rightIndent=8, spaceBefore=4, spaceAfter=6, backColor=CODE_BG,
    textColor=colors.HexColor("#222222"), borderPadding=6,
)
NOTE = ParagraphStyle(
    "Note", parent=BODY, fontSize=9, leading=12, textColor=GREY,
    spaceAfter=10,
)
LEGEND = ParagraphStyle(
    "Legend", parent=BODY, fontSize=9, leading=12, textColor=GREY,
    spaceBefore=2, spaceAfter=10, alignment=TA_LEFT,
)


def p(text: str, style: ParagraphStyle = BODY):
    return Paragraph(text, style)


def plain(text: str):
    return p("<b>Plain English.</b> " + text, PLAIN)


def tech(text: str):
    return p("<b>Technical detail.</b> " + text, TECH)


def ul(items: list[str], style: ParagraphStyle = BULLET):
    return ListFlowable(
        [ListItem(p(t, style), leftIndent=10, value="bullet") for t in items],
        bulletType="bullet", start="•", leftIndent=14, bulletFontSize=9,
    )


def kv_table(rows: list[list[str]], col_widths=None):
    t = Table(rows, colWidths=col_widths or [4.5 * cm, 12 * cm])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f0f0")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def metric_table(rows: list[list[str]], col_widths=None):
    t = Table(rows, colWidths=col_widths or [4.5 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f7f7")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def ascii_diagram(text: str) -> Paragraph:
    esc = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(
        f"<font face='Courier'>{esc.replace(chr(10), '<br/>')}</font>", CODE,
    )


story: list = []


def add_section(title: str, body_blocks: list):
    story.append(p(title, H1))
    story.extend(body_blocks)


# ═══ Title + orientation ═══
story += [
    p("ML Payment Recovery Engine", TITLE),
    p("System architecture — interview reference", SUBTITLE),
    p(
        "Project 3 · Ivan Antonov · 2026-04-22 · Live: "
        "<b>https://ivanantonov.com/recovery/</b>",
        LEGEND,
    ),
    p("How to read this document", H2),
    p(
        "This document is written for two audiences at once. Every major "
        "section contains two blocks:"
    ),
    ul([
        "<b>Plain English</b> (cream background) — what this is, for "
        "non-technical readers such as product managers, business stakeholders, "
        "or interviewers from non-ML backgrounds",
        "<b>Technical detail</b> (blue background) — the engineering, for "
        "architects and ML practitioners",
    ]),
    p(
        "The plain-English blocks can be read on their own as a coherent "
        "product narrative. The technical blocks add depth without repeating "
        "business context. Tables and diagrams sit outside the blocks because "
        "both audiences benefit from them."
    ),
]

# ═══ Executive summary ═══
add_section("Executive summary", [
    p("What it is", H2),
    p(
        "A decision-support service for declined card transactions. When a "
        "merchant's payment processor rejects a card, the service scores the "
        "chance of recovery if the merchant retries, combines that score with "
        "the merchant's economics (margin, retry cost, cardholder friction), "
        "and returns a single recommendation — <i>retry</i> or "
        "<i>do_not_retry</i> — with a plain-English explanation."
    ),
    p("Why it exists", H2),
    p(
        "Card-not-present merchants lose 10–15% of approved volume to "
        "preventable declines. Blanket-retry policies waste processing fees "
        "and irritate cardholders; rule-only policies retry too aggressively "
        "on declines that will never recover. The system replaces rule-based "
        "retries with an expected-value decision so every retry is defensible "
        "in dollar terms, not pattern-matching."
    ),
    p("What's shipped", H2),
    p(
        "A LightGBM classifier (test AUC 0.732, test F1 0.331) with isotonic "
        "calibration, served by a FastAPI app behind Caddy on a DigitalOcean "
        "droplet. Decision policy tuned against validation dollar net value, "
        "not classification accuracy. SHAP-backed explanations available on "
        "every prediction. Champion/challenger reconciliation, drift "
        "monitoring, and MLflow experiment tracking all live alongside."
    ),
    p("What was deliberately cut", H2),
    p(
        "Kafka, Redis dedup, Terraform, Great Expectations, live A/B, "
        "GitHub Actions CI/CD, Prometheus, and Grafana. Each is narrated as "
        "\"next step at threshold X\" rather than built as architecture "
        "LARPing at demo scale."
    ),
]),

# ═══ The problem in plain English ═══
add_section("1. The problem", [
    plain(
        "A merchant sells something online. A customer tries to pay, and the "
        "card is declined — maybe the issuer flagged it, maybe the network "
        "was briefly overloaded, maybe the card was close to its limit at the "
        "exact moment of purchase. About one in ten of those declines will "
        "actually succeed if the merchant tries again a moment later. The "
        "other nine will not, and each failed retry costs the merchant 10–15 "
        "cents in processing fees plus the subtler cost of annoying the "
        "customer. Today most merchants either retry every decline (expensive "
        "and annoying) or retry none (money left on the table). This system "
        "decides, per decline, whether it's worth retrying — with reasoning."
    ),
    tech(
        "The problem is framed as a binary classification on declined card "
        "transactions, with the positive class being "
        "<i>target_recovered_by_retry = True</i>. The positive rate in the "
        "synthetic training set is 14.7%. The decision boundary is not set at "
        "0.5 (standard classification) or at Youden's J statistic (standard "
        "imbalanced classification) but at whichever probability maximizes "
        "realized dollar net value on the validation set, given an explicit "
        "cost model. This reframes the ML task away from accuracy maximization "
        "toward expected-utility decision theory, which matters because the "
        "interviewer response to \"our F1 is 0.33\" is very different from "
        "\"our policy recovers $28,400 on the test set while the rules "
        "baseline recovers $28,435 with 14 percentage points more retries.\""
    ),
]),

# ═══ Business economics ═══
add_section("2. The economics", [
    plain(
        "Every retry has four numbers attached: (1) how likely the retry is "
        "to succeed, (2) how much margin the merchant earns if it does, "
        "(3) how much the retry costs in fees, and (4) how much cardholder "
        "goodwill we burn if it fails. The model estimates the first number; "
        "the other three are set by the merchant. The decision is simply "
        "whether the expected recovery covers the expected cost."
    ),
    tech(
        "The expected-value formula used in the decision engine is:"
    ),
    ascii_diagram(
        "E[net_value] = p_recovery × amount_usd × margin_rate\n"
        "              − retry_cost_usd\n"
        "              − friction_cost_usd\n"
        "\n"
        "decision = retry   if   p_recovery ≥ threshold  AND  E[net_value] > 0\n"
        "         = do_not_retry  otherwise"
    ),
    p("Default values used in the shipped service:"),
    kv_table([
        ["Retry cost", "$0.12 per attempt (covers processor fee + gateway + overhead)"],
        ["Margin rate", "35% of amount_usd (representative CNP SaaS / ecom margin)"],
        ["Friction cost", "$0.03 per retry (stylized cardholder-annoyance cost)"],
        ["Threshold", "0.05 — chosen to maximize validation realized net value"],
    ], col_widths=[4 * cm, 12 * cm]),
    p(
        "These values live in <i>project3_runtime.py</i> as "
        "<i>DEFAULT_RETRY_COST_USD</i>, <i>DEFAULT_MARGIN_RATE</i>, "
        "<i>DEFAULT_FRICTION_COST_USD</i>, and <i>DEFAULT_THRESHOLD</i>. "
        "They're intentionally constants right now; the planned next step is "
        "a per-merchant configuration endpoint so a merchant with 15% margins "
        "can override the defaults at request time."
    ),
]),

# ═══ User journey ═══
add_section("3. Who uses this and how", [
    plain(
        "Three kinds of people interact with the service. First, the "
        "merchant's engineering team — they call the API whenever a payment "
        "is declined. Second, merchant operations analysts — they use the "
        "explanations to understand why certain declines should be retried "
        "and others abandoned. Third, compliance and audit reviewers — they "
        "check the model card, the segment-level fairness metrics, and the "
        "explanation quality before any live rollout."
    ),
    tech(
        "Primary integration is a synchronous POST from the merchant's "
        "payment-gateway handler to <i>/recovery/predict</i>. Latency target "
        "P50 &lt;50ms including Pydantic validation and SHAP-backed top-3 "
        "drivers. The model-card endpoint is a pull interface for governance "
        "review — returning JSON that an auditor can cite or a CI pipeline "
        "can gate against. Segment metrics and drift reports are consumed "
        "offline rather than through the API."
    ),
    p("Three call-sites in the real-world deployment:", H3),
    ul([
        "<b>At decline time</b>: the merchant's payment handler calls "
        "<i>/predict</i> the moment an authorization is rejected; the "
        "decision is returned inline, the merchant either enqueues a retry "
        "or marks the transaction as failed",
        "<b>In a bulk reconciliation job</b>: overnight, the merchant batch-"
        "scores the day's declines for post-mortem (would our current rule "
        "have matched ML? how much net value did we leave?)",
        "<b>From an analyst tool</b>: the chatbot in Project 1 wraps "
        "<i>/predict</i> as a tool call, so a non-engineer can ask \"should "
        "I retry this transaction?\" and get a reasoned answer",
    ]),
]),

# ═══ System architecture ═══
add_section("4. System architecture", [
    plain(
        "The system is one web service on one server, with a reverse proxy "
        "in front. The web service loads the trained model into memory when "
        "it starts, then waits for requests. Each request gets checked "
        "against a list of allowed fields, scored by the model, combined "
        "with the business economics, and returned with a short explanation. "
        "A Redis instance sits nearby for future caching but isn't needed "
        "at current scale."
    ),
    tech(
        "Integration pattern is Pattern 1 — merchant-direct SaaS, synchronous "
        "request/response. No Kafka, no callback webhooks. Single-process "
        "FastAPI + uvicorn container behind Caddy for TLS termination and "
        "path rewriting. All expensive cold-start work (model load, feature "
        "policy, reference frame, category maps, SHAP explainer) is hoisted "
        "to startup via lru-cached accessors in <i>project3_runtime.py</i>. "
        "Per-request work is O(inference) — one booster predict + optional "
        "contribution extraction."
    ),
    ascii_diagram(
        "╔══════════════════════════════════════════════════════════════════════╗\n"
        "║                        Public internet (HTTPS)                       ║\n"
        "╚══════════════════════════════════════════════════════════════════════╝\n"
        "                                 │\n"
        "                                 ▼\n"
        "                 ┌──────────────────────────────┐\n"
        "                 │  Caddy reverse proxy (TLS)   │\n"
        "                 │  ivanantonov.com/recovery/*  │\n"
        "                 │  handle_path /recovery/* →   │\n"
        "                 │      reverse_proxy :8091     │\n"
        "                 └───────────────┬──────────────┘\n"
        "                                 │ 127.0.0.1\n"
        "                                 ▼\n"
        "    ┌────────────────────────────────────────────────────────────┐\n"
        "    │ Docker container: FastAPI + uvicorn (single process)      │\n"
        "    │                                                           │\n"
        "    │  ┌──────────────────┐     ┌───────────────────────────┐   │\n"
        "    │  │ Startup phase    │     │ Request phase             │   │\n"
        "    │  │ (once, lru_cache)│     │ (per POST /predict)       │   │\n"
        "    │  │                  │     │                           │   │\n"
        "    │  │ • load_model     │     │ 1. Pydantic validation    │   │\n"
        "    │  │ • feature_policy │     │    (78-feature allowlist) │   │\n"
        "    │  │ • metadata       │     │ 2. Feature prep + encoding│   │\n"
        "    │  │ • reference_frame│     │ 3. booster.predict        │   │\n"
        "    │  │ • category_maps  │     │ 4. Decision engine (EV)   │   │\n"
        "    │  │ • shap_explainer │     │ 5. Top-3 drivers          │   │\n"
        "    │  │ • model_card     │     │ 6. Structured JSON log    │   │\n"
        "    │  └──────────────────┘     └───────────────────────────┘   │\n"
        "    │           │                        │                     │\n"
        "    │           ▼                        ▼                     │\n"
        "    │     in-memory artifacts     JSON response to caller      │\n"
        "    └────────────────────────────────────────────────────────────┘\n"
        "                                 │\n"
        "                                 ▼\n"
        "              ┌──────────────────────────────────┐\n"
        "              │ Redis 7-alpine (reserved)        │\n"
        "              │ future: feature cache + dedup    │\n"
        "              └──────────────────────────────────┘"
    ),
    p(
        "The diagram shows two clear phases. <b>Cold start</b> happens once "
        "when the container boots: six lru-cached accessors fire in sequence "
        "(see <i>app.py:62-77</i> <i>startup_event</i>). This front-loads "
        "every expensive step so that the per-request path is as thin as "
        "possible. <b>Per request</b>, the six-step pipeline runs in under "
        "50ms for the fast path and under 200ms for the SHAP-enriched path."
    ),
]),

# ═══ Data layer ═══
add_section("5. Data layer", [
    plain(
        "The training data is a synthetic set of about 12,000 card declines. "
        "Each row is one decline event with roughly 80 pieces of information "
        "attached (amount, currency, country, processor, response code, and "
        "so on). The data is split into three chunks in time order — oldest "
        "for training, middle for tuning, newest for final testing — so the "
        "model never sees the future during training."
    ),
    tech(
        "Dataset: ~12,300 synthetic card-decline events, one row per original "
        "decline. Target is <i>target_recovered_by_retry</i> at 14.7% "
        "positive. Split is temporal 70/15/15 train/val/test, sorted on the "
        "event timestamp to prevent any forward-looking leakage (a row from "
        "July cannot inform a row from June)."
    ),
    p("Feature policy is a governance primitive, not a convenience:", H3),
    ul([
        "<b>78 allowed features</b> in <i>project3_feature_policy.json</i>, "
        "split into <i>numeric_columns</i>, <i>boolean_columns</i>, "
        "<i>categorical_columns</i>",
        "<b>Allowlist enforcement</b> at the API boundary — any request with "
        "a field outside the allowlist returns HTTP 422 before any inference "
        "runs (<i>app.py:108-111</i>)",
        "<b>Temporal safety</b> — only fields knowable at original decline "
        "time are in the policy; no retry outcomes, settlement state, or "
        "chargeback signals",
        "<b>No PII</b> — no PAN, no CVV, no email, no direct cardholder "
        "identifiers; the closest to identity is <i>merchant_country</i> and "
        "<i>card_country</i>",
    ]),
    p("Feature categories (of 78):", H3),
    kv_table([
        ["Merchant context", "vertical, MCC, country, archetype, processor, routing reason"],
        ["Transaction", "amount, amount_usd, currency, cross-border flag"],
        ["Card", "brand, type, country, funding source, token presence"],
        ["Authorization", "response code, response message, decline bucket, soft-decline flag"],
        ["3DS / SCA", "requested, outcome, version, flow, ECI, exemption"],
        ["Latency", "auth, 3DS, scheme (ms)"],
        ["Risk", "risk score, fraud flag, model version, skip flag"],
        ["Geography", "billing / shipping / IP / issuer country"],
        ["Temporal", "event hour, day of week, month, weekend flag"],
    ], col_widths=[4 * cm, 12 * cm]),
]),

# ═══ Model ═══
add_section("6. Model", [
    plain(
        "The model is a \"tree ensemble\" — think of it as a panel of "
        "several hundred very short decision trees that each cast a weighted "
        "vote on whether a given decline will recover. Each tree looks at "
        "maybe five or six features and draws a simple boundary; the ensemble "
        "is the majority-weighted opinion. Tree ensembles are the right tool "
        "for tabular data with mixed types (numbers and categories) and "
        "moderate dataset sizes. Neural nets tend to overfit here; linear "
        "models can't capture the interactions."
    ),
    tech(
        "LightGBM binary classifier, best iteration 66 (early-stopped on "
        "validation AUC). Hyperparameters tuned for tabular with ~12K rows "
        "and high-cardinality categoricals. Logistic regression v1 (plain) "
        "and v2 (target-encoded) serve as baselines and as the challenger in "
        "reconciliation. Training pipeline in <i>train_project3_lightgbm.py</i> "
        "with MLflow tracking wired via a <i>maybe_mlflow_run</i> context "
        "manager that no-ops if mlflow isn't installed."
    ),
    p("Final test-set performance (1,845 declines):", H3),
    metric_table([
        ["Model", "AUC", "F1", "Log loss", "Brier"],
        ["LightGBM (champion)", "0.732", "0.331", "0.350", "0.111"],
        ["Logistic v1 (baseline)", "0.741", "0.336", "—", "—"],
        ["Logistic v2 (target-encoded)", "0.729", "0.331", "—", "—"],
        ["LightGBM + isotonic", "0.720", "—", "0.358", "0.110"],
    ]),
    p(
        "Logistic v1 has a marginally higher test AUC (0.741 vs 0.732). The "
        "champion is still LightGBM because the selection criterion is "
        "<b>validation dollar net value</b>, not test AUC — and "
        "champion/challenger reconciliation showed the tree model finds "
        "nonlinear recovery signal in high-risk-orchestrator and soft-decline "
        "cohorts that the logistic misses. This is the honest interview "
        "answer to \"how do you compare models?\""
    ),
    p("Segment stability on the test set:", H3),
    kv_table([
        ["merchant_country=US", "AUC 0.704, F1 0.301 — largest cohort, weakest lift"],
        ["merchant_country=DE", "AUC 0.761, F1 0.326"],
        ["merchant_country=FR", "AUC 0.729, F1 0.411"],
        ["merchant_country=IN", "AUC 0.767, F1 0.441"],
        ["merchant_vertical=saas", "AUC 0.797, F1 0.250 — strong ranking, weak precision"],
        ["processor_name=global-acquirer-a", "AUC 0.730, F1 0.358"],
    ], col_widths=[5 * cm, 11 * cm]),
    p(
        "Variance is material. The model card treats this as the key gating "
        "finding before any live rollout: cohort-level monitoring and a "
        "fallback-to-rules policy for cohorts where the model under-performs "
        "rules must be in place. An unattended single-policy deployment "
        "would be irresponsible."
    ),
]),

# ═══ Calibration ═══
add_section("7. Calibration", [
    plain(
        "Models that rank well don't necessarily produce probabilities that "
        "<i>mean</i> what they look like. A raw model score of 0.30 might "
        "actually be right about 8% of the time, not 30%. Calibration is a "
        "short post-processing step that maps raw scores to actual "
        "probabilities. This matters when downstream systems treat the score "
        "as a probability — e.g., a merchant's risk team sets a rule \"retry "
        "only when probability &gt; 40%\" and that 40% needs to be a real 40%."
    ),
    tech(
        "Isotonic regression fitted on validation predictions, applied to "
        "test. Isotonic is preferred over Platt scaling for tree ensembles "
        "because the score distribution is non-sigmoidal. Switchable at "
        "serve time — calibrated probabilities are available behind a config "
        "flag rather than the default, since ranking-purity metrics (AUC) "
        "drop slightly post-calibration. Report at "
        "<i>project3_lightgbm_calibration_report.md</i>."
    ),
    kv_table([
        ["Test AUC", "0.732 → 0.720 (post-calibration drop, expected)"],
        ["Test Brier", "0.111 → 0.110 (better probability quality)"],
        ["Test log loss", "0.350 → 0.358 (slight regression — ranking-driven)"],
        ["Bucket 1 raw mean", "0.007 → calibrated 0.081 vs event rate 0.081 (exact)"],
        ["Bucket 5 raw mean", "0.103 → calibrated 0.424 vs event rate 0.317"],
    ], col_widths=[4 * cm, 12 * cm]),
]),

# ═══ Decision engine ═══
add_section("8. Decision engine", [
    plain(
        "The decision engine is a three-line function. Given a probability "
        "and an amount, it computes the expected dollar value of retrying "
        "and returns either \"retry\" or \"do_not_retry.\" Simple math on "
        "top of the model, but this layer is what makes the output a product "
        "instead of a research artifact."
    ),
    tech(
        "Implemented in <i>project3_runtime.py</i> as two pure functions: "
        "<i>expected_retry_value(prob, amount_usd)</i> and "
        "<i>decision_from_prob_and_value(prob, exp_value, threshold)</i>. "
        "Both are numpy-vectorized so the same code serves one request or a "
        "batch-score job. The threshold is looked up at request time via "
        "<i>decision_threshold()</i> which reads "
        "<i>model_metadata['decision_threshold']</i>."
    ),
    p("Policy comparison on the test set (1,845 declines):", H3),
    metric_table([
        ["Policy", "Retry rate", "Recovered $", "Net value", "Wasted retries"],
        ["ML policy", "68.5%", "$81,602", "$28,400", "$122"],
        ["Rules policy", "82.6%", "$81,810", "$28,435", "$153"],
        ["Δ (ML − Rules)", "−14.1 pp", "−$208", "−$34", "−$31"],
    ], col_widths=[3.5 * cm, 3 * cm, 3 * cm, 3 * cm, 3.5 * cm]),
    p(
        "On the test set the ML policy retries 261 fewer transactions while "
        "recovering only 4 fewer declines, for a $34 net-value gap. Framed "
        "correctly in an interview: the ML policy sacrifices $34 in test "
        "recovery to avoid 261 unnecessary retries. At 100× scale the "
        "customer-fatigue savings (and the downstream cost of eroded trust) "
        "dominate the $34."
    ),
]),

# ═══ Explainability ═══
add_section("9. Explainability", [
    plain(
        "Every prediction comes with the three reasons the model gave that "
        "decision, in plain English. Example: \"insufficient-funds decline "
        "on a high-risk processor with a soft-decline flag usually doesn't "
        "recover on immediate retry.\" This is what makes the service "
        "defensible to regulators, auditors, and a merchant's risk team — "
        "not just \"the model said so.\""
    ),
    tech(
        "Two explanation layers, chosen by a query-parameter at request time:"
    ),
    ul([
        "<b>Fast path</b> (default, <i>include_explanation=false</i>): "
        "inference-time LightGBM contribution values via "
        "<i>booster.predict(X, pred_contrib=True)</i>. No measurable latency "
        "cost — contribution extraction is a memory-read on the tree traversal "
        "already done for prediction. Returns top-3 feature contributions",
        "<b>Slow path</b> (<i>include_explanation=true</i>): full SHAP "
        "TreeExplainer computed once at startup, queried per request. "
        "Slower by ~100ms but more rigorous — satisfies \"can you prove the "
        "explanation is faithful to the model?\" interview probe",
    ]),
    p("Business-phrase translator", H3),
    p(
        "Raw SHAP values are numeric (\"feature <i>response_code</i> value "
        "<i>51</i> contributed −0.34 to the logit\"). That's useless to a "
        "merchant ops analyst. The translator "
        "(<i>business_phrase()</i> in <i>project3_runtime.py</i>) "
        "maps feature-value pairs to natural-language phrases: \"insufficient "
        "funds decline,\" \"high-risk processor cohort,\" \"first-try 3DS "
        "frictionless,\" etc. Templates exist for the nine features that "
        "appear in the top-10 global importance list; anything else falls "
        "back to a generic \"feature X shifted the score\" phrase."
    ),
    p(
        "This hand-written layer is a deliberate product choice: merchant-"
        "facing copy doesn't leak model internals, doesn't confuse "
        "non-technical readers, and remains consistent across runs. It's the "
        "layer that turns SHAP from a research tool into a merchant feature."
    ),
]),

# ═══ Serving stack ═══
add_section("10. Serving stack", [
    plain(
        "The service is a small Python web application. It listens on a "
        "port, accepts JSON requests, runs the model, and returns JSON "
        "responses. Packaged in a Docker container so it runs the same way "
        "on any machine."
    ),
    tech(
        "FastAPI + uvicorn, Python 3.12, single-process. Three endpoints — "
        "<i>GET /health</i>, <i>GET /model-card</i>, <i>POST /predict</i> — "
        "all in <i>app.py</i>. Pydantic request model "
        "dynamically generated from the 78-feature allowlist via "
        "<i>pydantic.create_model</i> (<i>app.py:51-55</i>) so the schema "
        "never drifts from the policy file."
    ),
    p("Sample request and response:", H3),
    ascii_diagram(
        "POST /recovery/predict?include_explanation=true&explanation_depth=3\n"
        "Content-Type: application/json\n"
        "\n"
        "{\n"
        "  \"amount_usd\": 5834.24,\n"
        "  \"merchant_country\": \"US\",\n"
        "  \"processor_name\": \"global-acquirer-a\",\n"
        "  \"response_code\": \"51\",\n"
        "  \"is_soft_decline\": true,\n"
        "  \"three_ds_outcome\": \"authenticated\",\n"
        "  \"risk_score\": 42,\n"
        "  ... (73 more fields from allowlist) ...\n"
        "}\n"
        "\n"
        "200 OK\n"
        "{\n"
        "  \"request_id\": \"a1b2c3d4e5f6\",\n"
        "  \"latency_ms\": 14,\n"
        "  \"recommended_action\": \"retry\",\n"
        "  \"recovery_probability\": 0.315,\n"
        "  \"expected_value_usd\": 643.97,\n"
        "  \"decision_threshold\": 0.05,\n"
        "  \"top_drivers\": [\n"
        "    {\"feature\": \"response_code\", \"value\": \"51\",\n"
        "     \"contribution\": +0.28,\n"
        "     \"explanation\": \"soft decline from insufficient funds —\n"
        "       often recovers on immediate retry\"},\n"
        "    {\"feature\": \"processor_name\", \"value\": \"global-acquirer-a\",\n"
        "     \"contribution\": +0.12,\n"
        "     \"explanation\": \"processor cohort with stable retry recovery\"},\n"
        "    {\"feature\": \"amount_usd\", \"value\": 5834.24,\n"
        "     \"contribution\": +0.08,\n"
        "     \"explanation\": \"mid-band amount; neither too small to bother\n"
        "       nor large enough to flag fraud\"}\n"
        "  ],\n"
        "  \"model_version\": \"v1\"\n"
        "}"
    ),
    p("Startup flow (once per container boot):", H3),
    ul([
        "<i>load_model()</i> — <i>lgb.Booster(model_file=\"project3_lightgbm_model.txt\")</i>, ~50ms",
        "<i>load_reference_frame()</i> — 11 MB CSV into a pandas DataFrame for category maps",
        "<i>categorical_category_maps()</i> — build the stable ordering for each categorical",
        "<i>top_global_features()</i> — read SHAP global-importance CSV",
        "<i>build_model_card()</i> — prebuild the JSON returned by <i>/model-card</i>",
        "<i>load_shap_explainer()</i> — construct SHAP TreeExplainer bound to the booster",
    ]),
    p(
        "All six are <i>functools.lru_cache(maxsize=1)</i> so they're idempotent "
        "and can be called from tests without state leakage "
        "(<i>project3_runtime.py:38-80</i>)."
    ),
]),

# ═══ Governance at the boundary ═══
add_section("11. Governance at the API boundary", [
    plain(
        "Before the model ever sees a request, the service checks that "
        "every field in the request is on an approved list. Anything extra "
        "is rejected with a helpful error message. This is the \"airport "
        "security\" layer — it makes sure nothing unexpected reaches the "
        "model."
    ),
    tech(
        "Three-layer validation pipeline before <i>predict_one</i> runs:"
    ),
    ul([
        "<b>Pydantic schema</b> — dynamically constructed with "
        "<i>extra=\"forbid\"</i>, so unknown keys are rejected at the "
        "framework level with field-level error messages",
        "<b>Allowlist re-check</b> — <i>app.py:108-111</i> cross-checks the "
        "parsed payload against the feature policy allowlist in case the "
        "Pydantic schema was bypassed in testing or downstream tooling "
        "generated the schema from a stale source",
        "<b>Type coercion</b> — categorical features are mapped to their "
        "known category lists at prediction time; unseen values fall back to "
        "<i>MISSING</i> rather than crashing",
    ]),
    p(
        "The allowlist is also the governance artifact. When a new feature "
        "is proposed, the policy file must be updated; that change is "
        "reviewable independently of any model-training change. Pull "
        "requests touching the allowlist go through a stricter review than "
        "pull requests touching hyperparameters — by design."
    ),
]),

# ═══ Observability ═══
add_section("12. Observability and logging", [
    plain(
        "The service writes a short structured log line for every request — "
        "request ID, how long it took, what the model decided, which version "
        "of the model answered. These logs are the foundation for "
        "monitoring, incident response, and post-hoc analysis."
    ),
    tech(
        "Structured JSON logging via a custom <i>JSONFormatter</i> "
        "(<i>app.py:27-38</i>) attached to the root logger. Every /predict "
        "call emits one line with: <i>ts</i>, <i>level</i>, <i>msg</i>, "
        "<i>request_id</i> (from <i>X-Request-ID</i> header or generated "
        "uuid4 hex prefix), <i>latency_ms</i>, <i>prediction</i> "
        "(<i>action|explain=bool</i>), <i>model_version</i>, <i>path</i>."
    ),
    p("Observability gaps narrated (not built):", H3),
    ul([
        "<b>Prometheus metrics</b> — would expose request counts, latency "
        "histograms, decision counts by label, drift indicator freshness",
        "<b>Grafana dashboards</b> — would consume Prometheus, add cohort "
        "panels and alerting",
        "<b>Trace propagation</b> — <i>X-Request-ID</i> header is respected "
        "but no OTEL / Jaeger integration; narrated as \"add at first "
        "cross-service call\"",
    ]),
    p(
        "At current scale (demo traffic) a single container's stdout is "
        "fine. The scaling trigger for adding Prometheus is when there's "
        "real merchant traffic and someone needs an SLO, not before."
    ),
]),

# ═══ MLOps ═══
add_section("13. MLOps surface", [
    plain(
        "Beyond the live service, four standard ML operational capabilities "
        "are in place — experiment tracking, calibration, drift detection, "
        "and champion/challenger comparison. Each one exists as a running "
        "artifact, not a slide. This is the interview story for \"how do "
        "you operate ML responsibly?\""
    ),
    tech(
        "Each capability is intentionally scoped for a one-person demo, not "
        "a production ML team:"
    ),
    p("Experiment tracking — MLflow", H3),
    p(
        "File-store backend at "
        "<i>C:/Users/ivana/.codex/memories/project3-mlruns/</i>. 6 FINISHED "
        "runs across the 3 model families (baseline_v1, baseline_v2, "
        "lightgbm) captured over 2 training campaigns. Each run records 11 "
        "parameters (num_leaves, learning_rate, decision_threshold, etc.), "
        "14 metrics (train/val/test AUC, F1, Brier, precision, recall, "
        "log-loss, predicted positive rate), and 5 artifacts including the "
        "serialized model. Training scripts use a "
        "<i>maybe_mlflow_run</i> context manager that no-ops when mlflow "
        "isn't installed — so training is runnable in any environment."
    ),
    p("Drift monitoring", H3),
    p(
        "<i>drift_monitor.py</i> computes Population Stability Index and "
        "Kolmogorov-Smirnov two-sample statistics on numeric features, "
        "Jensen-Shannon divergence on categoricals, against a reference "
        "distribution. Designed as an offline / cron-worthy job; not inline "
        "with inference. Current PSI values from the latest reference pull: "
        "highest is <i>event_month</i> at 0.0264 (well under the 0.2 yellow "
        "flag threshold), <i>mastercard_advice_code</i> at 0.0065, "
        "everything else under 0.003."
    ),
    p("Champion/challenger reconciliation", H3),
    p(
        "<i>reconcile_project3_challenger.py</i> scores all 1,845 test rows "
        "with both champion (LightGBM) and challenger (baseline_v2 "
        "target-encoded logistic) and emits a side-by-side CSV. Mean "
        "absolute probability delta 0.084; 4.77% of rows diverge by &gt;0.20. "
        "Largest disagreements concentrate in high-risk-orchestrator "
        "processor declines with <i>response_code=51</i> — a useful review "
        "cohort for feature-engineering work."
    ),
    p("Model card as API", H3),
    p(
        "The model card is not a static markdown file alone — "
        "<i>GET /recovery/model-card</i> returns JSON with training-data "
        "description, test and validation metrics, segment breakdowns, "
        "limitations, ethical considerations, and PII posture. An audit "
        "pipeline or downstream service can cite or gate on this endpoint."
    ),
]),

# ═══ Deployment topology ═══
add_section("14. Deployment topology", [
    plain(
        "The service runs in a Docker container on a rented cloud server. "
        "A reverse proxy handles HTTPS and routes traffic to the container. "
        "A small shell script redeploys new versions."
    ),
    tech(
        "DigitalOcean droplet at 209.38.71.25 shared with three other "
        "services (portfolio, OpenClaw, Jobsearch dashboard). Docker "
        "container orchestrated via compose with a production overlay that "
        "binds the API to 127.0.0.1:8091 so nothing is directly exposed."
    ),
    p("Container layout", H3),
    ascii_diagram(
        "docker-compose.yml (base):          docker-compose.prod.yml (overlay):\n"
        "services:                            services:\n"
        "  api:                                 api:\n"
        "    build: .                             ports:\n"
        "    ports: [\"8000:8000\"]                 - \"127.0.0.1:8091:8000\"\n"
        "    environment:                         environment:\n"
        "      PROJECT3_PORT: \"8000\"                PROJECT3_PORT: \"8000\"\n"
        "    depends_on: [redis]\n"
        "  redis:\n"
        "    image: redis:7-alpine\n"
        "    ports: [\"6379:6379\"]"
    ),
    p("Caddy route configuration (in <i>/etc/caddy/Caddyfile</i>):", H3),
    ascii_diagram(
        "handle /recovery {\n"
        "    redir /recovery/ permanent\n"
        "}\n"
        "handle_path /recovery/* {\n"
        "    reverse_proxy localhost:8091\n"
        "}"
    ),
    p("Deploy flow — <i>deploy_project3.sh</i>", H3),
    p(
        "Five-step script: (1) mkdir on droplet at <i>/opt/project3-recovery</i>, "
        "(2) tar local folder excluding caches, pycache, and mlruns, pipe "
        "through SSH and untar remote, (3) <i>docker compose up -d --build</i> "
        "with the prod overlay, (4) 30-attempt polling health check against "
        "<i>https://ivanantonov.com/recovery/health</i> with 2s backoff, "
        "(5) exit non-zero if health never succeeds. Caddy reload is a "
        "first-time-setup step, not on every deploy."
    ),
    p("Live verification", H3),
    ascii_diagram(
        "$ curl https://ivanantonov.com/recovery/health\n"
        "\n"
        "{ \"ok\": true,\n"
        "  \"model_loaded\": true,\n"
        "  \"model_version\": \"v1\",\n"
        "  \"decision_threshold\": 0.05,\n"
        "  \"feature_count\": 78 }"
    ),
]),

# ═══ Failure modes ═══
add_section("15. Failure modes and defenses", [
    plain(
        "Every production service needs to answer \"what breaks this, and "
        "what happens when it does?\" The table below lists the realistic "
        "failure scenarios for this service and how it handles each."
    ),
    tech(
        "Failure-mode analysis isolated to the deployed surface area. Items "
        "flagged <i>planned</i> are acknowledged gaps rather than "
        "implemented defenses — each has a concrete design rather than hand-"
        "waving."
    ),
    Table([
        ["Failure mode", "Today's behavior", "Planned defense"],
        ["Model file missing / corrupt",
         "Container fails fast at startup via lru_cache on load_model",
         "Health probe catches it; orchestrator holds back traffic"],
        ["Request with unknown fields",
         "HTTP 422 before any inference (allowlist + Pydantic extra=forbid)",
         "Already handled"],
        ["Unseen categorical value at request time",
         "Falls back to MISSING category, prediction still runs",
         "Log rate alert if MISSING-rate exceeds baseline"],
        ["Concurrent training + serving of new model",
         "Not supported — serialized model is loaded once at startup",
         "Blue/green container rollout with health gate"],
        ["Upstream processor name drift",
         "Drift monitor shows JS divergence on processor_name",
         "Weekly cron + Slack alert on JS > 0.2"],
        ["Prediction latency spike",
         "Structured log captures latency_ms per request",
         "Prometheus + SLO alert (planned)"],
        ["Misconfigured economics",
         "Constants in project3_runtime.py — change requires redeploy",
         "Per-merchant config endpoint (planned next)"],
        ["Caddy / TLS expiry",
         "Caddy auto-renews via ACME; failure pages user",
         "Monitoring on cert expiry date (planned)"],
    ], colWidths=[4.5 * cm, 6 * cm, 6 * cm], style=TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f7f7")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ])),
]),

# ═══ Trade-offs cut ═══
add_section("16. Explicit trade-offs and what was cut", [
    plain(
        "A big part of engineering judgment is what you choose <i>not</i> "
        "to build. The team-of-one context and demo-scale traffic both cut "
        "in the same direction: skip infrastructure that would be "
        "architecture theatre at this scale, and narrate the scaling path "
        "instead. The table below is the cut list with rationale."
    ),
    tech(
        "Approximate effort saved: 11 engineering-weeks across Sessions 7, "
        "7.5, 9, 10, 11, 12. Re-invested into (a) calibration, (b) drift "
        "monitoring, (c) champion/challenger, (d) deploy automation, "
        "(e) this documentation. Each cut has a scaling-trigger note so the "
        "answer to \"when would you add this?\" is concrete, not vague."
    ),
    Table([
        ["Component", "Why cut", "Narrated replacement"],
        ["Kafka consumer", "~10 TPS, 1 consumer, 1-person team — no backpressure, fanout, or replay need",
         "Sync HTTP; add Kafka audit bus at 100 TPS or PSP-embedded mode"],
        ["Callback webhook", "Merchants hate building receivers",
         "Decision returned in POST response"],
        ["Redis dedup", "No fire-and-forget path means no dedup need",
         "Reserved as a feature cache for historical merchant features"],
        ["Terraform", "Single container on existing droplet",
         "scp + docker-compose; IaC justified at 3+ services per droplet"],
        ["Great Expectations", "One dataset, one schema",
         "Pre-retrain schema + value-range checks (planned inline)"],
        ["Live A/B", "Solo project, no live traffic to split",
         "Shadow-scoring batch reconciliation against challenger"],
        ["CI/CD (GH Actions)", "No PR flow, solo maintainer",
         "Local pytest + manual eval-gate; add at first co-maintainer"],
        ["Prometheus + Grafana", "No production load means no SLO to monitor",
         "Structured JSON logs; add when real merchant traffic begins"],
    ], colWidths=[3.8 * cm, 6 * cm, 6.7 * cm], style=TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f7f7")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ])),
]),

# ═══ Scaling narrative ═══
add_section("17. Scaling narrative", [
    plain(
        "The question \"how does this scale?\" has three separate answers "
        "depending on what's growing — traffic volume, deployment model, or "
        "feature sophistication. Each has a concrete trigger and a concrete "
        "architectural step."
    ),
    tech(
        "Scaling is treated as three orthogonal axes rather than a single "
        "\"bigger\" arrow:"
    ),
    p("Axis 1 — traffic volume", H3),
    p(
        "At ~100 TPS or when introducing a second ML consumer (shadow-scoring "
        "or a retention model reusing the same feature stream), decouple "
        "ingestion from prediction by adding a Kafka audit bus. The API "
        "handler publishes each scored prediction to <i>audit.predictions</i>; "
        "downstream consumers subscribe. The model and decision engine are "
        "untouched — this is an ingress-side change."
    ),
    p("Axis 2 — deployment topology (PSP-embedded)", H3),
    p(
        "For a PSP like Yuno, Adyen, or Borealis, the merchant-direct API "
        "becomes irrelevant — the PSP already has the decline stream. Replace "
        "the webhook ingestor with a Kafka consumer on the PSP's existing "
        "<i>payment.declined</i> topic. Same ML core, same decision engine, "
        "different ingress. The case-study deliberately picks the "
        "merchant-direct variant because it's the simpler story; the "
        "PSP-embedded variant is the realistic commercial path."
    ),
    p("Axis 3 — feature sophistication", H3),
    p(
        "Today's features are all knowable at request time (amount, country, "
        "processor, response code, etc.). The next lift requires historical "
        "features — merchant decline-rate trailing-30d, issuer authorization "
        "trend, cardholder retry-success history. These require a feature "
        "store with a Redis cache in front. Latency budget at 100 TPS: "
        "5–10ms cache lookup plus inference. No model retraining required to "
        "add — the training pipeline is already designed to accept historical "
        "features via the allowlist."
    ),
]),

# ═══ Interview FAQ ═══
add_section("18. Anticipated interview probes", [
    p("<b>Q1. Why LightGBM over XGBoost, CatBoost, or a neural net?</b>", H3),
    p(
        "Tabular data, small dataset (~12K rows), mixed numeric and "
        "categorical features, native missing-value handling, fast training, "
        "and the SHAP explainer is mature. XGBoost is a near-tie — I'd accept "
        "\"I used XGBoost instead\" as equivalent. CatBoost's categorical "
        "encoder is clever but the accuracy improvement on this dataset "
        "didn't justify the dependency. Neural nets are the wrong tool at "
        "12K rows; they'd overfit before they converged."
    ),
    p("<b>Q2. How do you know the decision threshold is right?</b>", H3),
    p(
        "The threshold is chosen to maximize validation realized dollar net "
        "value — not F1, not Youden's J, not precision. The objective "
        "function has explicit retry-cost, margin-rate, and friction-cost "
        "assumptions. Those assumptions are the product knob: if a merchant "
        "has 15% margins instead of 35%, the threshold naturally shifts "
        "upward. That's by design, not a bug."
    ),
    p("<b>Q3. Why synchronous HTTP instead of an event stream?</b>", H3),
    p(
        "At 10 TPS with one consumer, Kafka's usual justifications "
        "(backpressure, fanout, replay, team decoupling) don't apply. "
        "Building it would be architecture LARPing — the interview red flag. "
        "The scaling trigger is 100 TPS or a second consumer; at that point "
        "a Kafka audit bus decouples ingestion from prediction without "
        "touching the ML core."
    ),
    p("<b>Q4. How do you handle model drift?</b>", H3),
    p(
        "Offline drift monitor computes PSI and Kolmogorov-Smirnov on "
        "numeric features, Jensen-Shannon divergence on categoricals, "
        "against a reference distribution. Yellow-flag thresholds: PSI > 0.2, "
        "KS p-value < 0.05 on stable features, JS > 0.2 on top categoricals. "
        "Production rollout would run this weekly, alert on threshold "
        "breach, and gate retraining on champion-beats-challenger on a "
        "held-out set."
    ),
    p("<b>Q5. How do you explain predictions to non-technical users?</b>", H3),
    p(
        "Two layers. The fast path returns inference-time contributions — "
        "top-3 features the model used — in &lt;1ms. The slow path runs "
        "proper SHAP TreeExplainer. Both feed into a hand-written "
        "business-phrase translator that converts "
        "\"<i>response_code=51</i> contributed +0.28\" into \"soft-decline "
        "from insufficient funds — often recovers on immediate retry.\" The "
        "translator is the reason this is a product and not a research "
        "artifact."
    ),
    p("<b>Q6. What's your governance story?</b>", H3),
    p(
        "Four controls, reviewable independently: (1) a 78-feature allowlist "
        "enforced at the API boundary, versioned in <i>feature_policy.json</i>; "
        "(2) a machine-readable model card at <i>GET /model-card</i> an "
        "auditor can cite or a CI pipeline can gate against; (3) per-"
        "prediction SHAP explanations for the \"why\" question; (4) segment-"
        "level metrics published so cohort weakness is visible before "
        "rollout. The system is explicitly decision-support, not autonomous "
        "action — a human or a rules engine approves the retry, the model "
        "ranks the queue."
    ),
    p("<b>Q7. What are the model's biggest weaknesses?</b>", H3),
    p(
        "Three. First, synthetic data — no real-world distribution shift "
        "validated against live traffic. Second, segment variance — the US "
        "cohort (largest) has the weakest lift, so a one-size-fits-all "
        "policy is irresponsible without cohort-level monitoring. Third, "
        "fixed economics — retry cost, margin, and friction are constants "
        "right now; different merchants have different values. Each weakness "
        "has a next-step in the roadmap."
    ),
    p("<b>Q8. What would you build next?</b>", H3),
    p(
        "Three things in priority order: (1) a configurable economics "
        "endpoint so merchants can set their own margin/friction/retry-cost "
        "at request time; (2) online learning from retry outcomes — closing "
        "the loop between prediction and ground-truth label; (3) a PSP-"
        "embedded variant that consumes <i>payment.declined</i> from the "
        "PSP's event bus. Item 3 is the realistic go-to-market because "
        "merchants don't buy standalone retry services — PSPs embed them."
    ),
    p("<b>Q9. What was the hardest part?</b>", H3),
    p(
        "Not the model. The model converged in a handful of training runs. "
        "The hardest part was the decision-engine calibration — turning a "
        "probability into a defensible retry decision required committing "
        "to specific dollar values for margin, retry cost, and friction "
        "cost, and that commitment shapes every downstream metric. The "
        "interview-relevant lesson is that in ML products the economic "
        "modeling is often harder than the statistical modeling, and should "
        "be worked on with the same rigor."
    ),
    p("<b>Q10. If you had another two weeks, what would you do?</b>", H3),
    p(
        "Add a shadow A/B mode to the live service — every prediction is "
        "scored by both champion and challenger, the champion's answer is "
        "returned, both are logged, and a nightly reconciliation job "
        "produces a drift-aware lift estimate. This turns "
        "champion/challenger from a one-off batch comparison into a "
        "continuous operational control. Two weeks would also cover "
        "Prometheus instrumentation and a deploy-gate on test-set metrics "
        "in CI."
    ),
]),

# ═══ Glossary ═══
add_section("19. Glossary", [
    p(
        "Quick reference for readers unfamiliar with the ML or payments "
        "vocabulary used above."
    ),
    kv_table([
        ["AUC", "Area under ROC curve. A ranking-quality measure; 0.5 is random, 1.0 is perfect. 0.73 is moderate lift on an imbalanced problem"],
        ["Brier score", "Mean squared error on probabilities. Lower is better-calibrated"],
        ["Calibration", "Post-processing that makes predicted probabilities match observed frequencies"],
        ["CNP", "Card-not-present. Online / phone transactions; higher decline rates than in-person"],
        ["Decline bucket", "Categorization of decline reason: soft (recoverable — insufficient funds, velocity), hard (unrecoverable — stolen card, closed account)"],
        ["Expected value", "Probability-weighted average of outcomes; the decision criterion"],
        ["F1", "Harmonic mean of precision and recall; balances false positives and false negatives"],
        ["Feature policy", "Explicit allowlist of model input features; anything outside returns HTTP 422"],
        ["Friction cost", "Dollarized cost of annoying a cardholder with an unnecessary retry"],
        ["Isotonic regression", "Non-parametric calibration method; used here over Platt because tree outputs aren't sigmoidal"],
        ["KS test", "Kolmogorov-Smirnov two-sample test; detects distribution shift in numeric features"],
        ["LightGBM", "Gradient-boosted decision tree library. Fast, tabular, native categorical support"],
        ["MCC", "Merchant Category Code. 4-digit classification of merchant business type"],
        ["MLflow", "Experiment tracker. Logs parameters, metrics, artifacts per training run"],
        ["PSI", "Population Stability Index. Drift metric; PSI > 0.2 is a yellow flag"],
        ["PSP", "Payment Service Provider. Orchestrates payments across processors (Yuno, Adyen, Stripe, etc.)"],
        ["Response code", "Issuer's coded reason for decline. 51 = insufficient funds, 05 = do not honor, 91 = issuer unavailable"],
        ["SHAP", "SHapley Additive exPlanations. Rigorous attribution of a model's prediction to individual feature contributions"],
        ["Soft decline", "Decline the issuer would likely reverse on retry (funds, velocity). Contrast with hard decline"],
        ["Target encoding", "Categorical encoding that replaces a category with the mean target value for that category"],
        ["3DS", "3-D Secure. Authentication layer for CNP cards; reduces fraud liability"],
        ["TPS", "Transactions per second. Throughput measure"],
    ], col_widths=[3.5 * cm, 12.5 * cm]),
]),

# ═══ Closing ═══
story += [
    Spacer(1, 14),
    p(
        "All artifacts referenced live in "
        "<i>~/OneDrive/Everything/Pet projects/AI tools/Payments bots/Project 3/Claude files/</i>. "
        "Model serving: <i>app.py</i> and <i>project3_runtime.py</i>. Product "
        "spec: <i>ml_payment_recovery_engine.md</i>. Model card: "
        "<i>model_card.md</i>. Evaluation reports: "
        "<i>project3_*_evaluation.md</i> and <i>project3_*_report.md</i>",
        NOTE,
    ),
]


def build() -> None:
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title="ML Payment Recovery Engine — System Architecture",
        author="Ivan Antonov",
    )

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(LIGHT_GREY)
        canvas.drawString(
            2 * cm, 1 * cm,
            "ML Payment Recovery Engine · Project 3 · Ivan Antonov",
        )
        canvas.drawRightString(
            A4[0] - 2 * cm, 1 * cm, f"Page {doc_.page}",
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"written: {OUTPUT}  ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    build()
