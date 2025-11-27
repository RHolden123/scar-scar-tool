import datetime
from typing import List, Dict, Tuple
from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st


# ---------- ALDEN (EXACT TABLE-BASED IMPLEMENTATION) ----------

def alden_latency_score(
    start_date: datetime.date,
    onset_date: datetime.date,
    rechallenge_category: str,
) -> int:
    """
    Latency component per ALDEN table.

    If there was previous SJS/TEN to the SAME drug ("pos_disease_drug"),
    the latency windows are shifted (1–4 days = +3; 5–56 days = +1).
    Otherwise default windows: 5–28 (+3), 29–56 (+2), 1–4 (+1), >56 (−1),
    drug given on index day (day 0) = −3.
    """
    latency_days = (onset_date - start_date).days
    prev_same_sjs = rechallenge_category == "pos_disease_drug"

    if prev_same_sjs:
        # If previous reaction to same drug:
        # "suggestive +3" from 1–4 days and "probable +1" from 5–56 days.
        if 1 <= latency_days <= 4:
            return 3
        elif 5 <= latency_days <= 56:
            return 1
        elif latency_days > 56:
            return -1
        else:
            # <1 day or drug administered on index day
            return -3
    else:
        if 5 <= latency_days <= 28:
            return 3
        elif 29 <= latency_days <= 56:
            return 2
        elif 1 <= latency_days <= 4:
            return 1
        elif latency_days > 56:
            return -1
        else:
            # <1 day or drug administered first on index day
            return -3


def alden_presence_score(
    stop_date: datetime.date,
    onset_date: datetime.date,
    half_life_days: float,
    impaired_clearance_or_interactions: bool,
) -> int:
    """
    Probability drug was present in system at onset.
    Uses 5 elimination half-lives rule as per ALDEN table.
    """
    if stop_date >= onset_date:
        # Drug given up to or beyond index day
        return 0  # "Definitive" 0

    days_since_stop = (onset_date - stop_date).days
    threshold = 5.0 * max(half_life_days, 0.01)

    if days_since_stop <= threshold:
        return 0  # "Definitive" +0

    # Stopped >5 half-lives before onset
    if impaired_clearance_or_interactions:
        return -1  # "Doubtful"
    else:
        return -3  # "Excluded"


def alden_rechallenge_score(rechallenge_category: str) -> int:
    """
    Prechallenge / rechallenge component per ALDEN.
    """
    mapping = {
        "pos_disease_drug": 4,     # Positive specifically for disease AND drug
        "pos_disease_or_drug": 2,  # Positive for disease OR drug
        "pos_nonspecific": 1,      # Positive non-specifically
        "unknown": 0,              # Unknown/not performed
        "negative": -2,            # Previous exposure with no reaction
    }
    return mapping.get(rechallenge_category, 0)


def alden_dechallenge_score(dechallenge_status: str) -> int:
    """
    Dechallenge component:
    - Neutral (drug stopped or unknown) = 0
    - Negative (drug not stopped without worsening) = −2
    """
    if dechallenge_status == "negative":
        return -2
    return 0


def alden_notoriety_score(notoriety_level: str) -> int:
    """
    Drug notoriety per ALDEN table.
    """
    mapping = {
        "strong": 3,   # strongly associated high-risk drug
        "associated": 2,
        "suspect": 1,
        "unknown": 0,
        "not_suspect": -1,
    }
    return mapping.get(notoriety_level, 0)


def alden_alternative_etiology_score(other_possible_etiology: bool) -> int:
    """
    Other possible aetiologies component: if there is at least one other drug
    with intermediate score >3, ALDEN subtracts 1.
    """
    return -1 if other_possible_etiology else 0


def compute_alden_score_exact(
    start_date: datetime.date,
    stop_date: datetime.date,
    onset_date: datetime.date,
    half_life_days: float,
    impaired_clearance_or_interactions: bool,
    rechallenge_category: str,
    dechallenge_status: str,
    notoriety_level: str,
    other_possible_etiology: bool,
) -> Tuple[int, str, Dict[str, int]]:
    """
    Full ALDEN score (−12 to +10) based exactly on the published table.
    """
    component_scores: Dict[str, int] = {}

    component_scores["latency"] = alden_latency_score(
        start_date=start_date,
        onset_date=onset_date,
        rechallenge_category=rechallenge_category,
    )

    component_scores["presence_in_system"] = alden_presence_score(
        stop_date=stop_date,
        onset_date=onset_date,
        half_life_days=half_life_days,
        impaired_clearance_or_interactions=impaired_clearance_or_interactions,
    )

    component_scores["prechallenge_rechallenge"] = alden_rechallenge_score(
        rechallenge_category
    )

    component_scores["dechallenge"] = alden_dechallenge_score(dechallenge_status)

    component_scores["notoriety"] = alden_notoriety_score(notoriety_level)

    component_scores["alternative_etiology"] = alden_alternative_etiology_score(
        other_possible_etiology
    )

    total_score = sum(component_scores.values())

    if total_score >= 6:
        category = "Very probable"
    elif 4 <= total_score <= 5:
        category = "Probable"
    elif 2 <= total_score <= 3:
        category = "Possible"
    elif 0 <= total_score <= 1:
        category = "Unlikely"
    else:
        category = "Very unlikely"

    return total_score, category, component_scores


# ---------- RegiSCAR DRESS (ALIGNED WITH VALIDATION DESCRIPTION) ----------

def compute_regiscar_dress_score_exact(
    fever_38_5: str,               # "yes" or "no_unknown"
    lymph_nodes: bool,
    eos_category: str,             # "none", "0.7-1.49", ">=1.5"
    atypical_lymphocytes: bool,
    skin_suggestive: str,          # "yes", "no", "unknown"
    skin_extent_50: str,           # "<50", ">=50"
    biopsy_result: str,            # "not_done", "suggestive", "not_suggestive"
    organ_involvement_count: int,
    blood_tests_negative: bool,    # ≥3 tests negative, none positive
    resolution_under_15d: bool,
) -> Tuple[int, str, Dict[str, int]]:
    """
    RegiSCAR DRESS scoring as per validation description:
    - Skin: eruption suggesting DRESS = +1; extent >50% = +1;
      non-suggestive eruption gets a −1 deduction.
    - Lymph nodes ≥2 sites, >1 cm = +1.
    - Eosinophilia: 0.70–1.49×10^9/L = +1; ≥1.5×10^9/L = +2.
    - Atypical lymphocytes present = +1.
    - Organ involvement: each organ = +1.
    - Blood tests: 0 positive and ≥3 negative (hepatitis, Mycoplasma, Chlamydia, ANA,
      cultures) = +1.
    - Deductions: −1 each for fever <38.5°C, eruption not suggestive of DRESS,
      biopsy not suggestive of DRESS, and resolution <15 days.
    """
    comp: Dict[str, int] = {}
    total = 0

    # Fever (deduction only)
    if fever_38_5 == "yes":
        comp["fever"] = 0
    else:
        comp["fever"] = -1
    total += comp["fever"]

    # Lymph nodes
    comp["lymph_nodes"] = 1 if lymph_nodes else 0
    total += comp["lymph_nodes"]

    # Eosinophilia
    if eos_category == "0.7-1.49":
        comp["eosinophilia"] = 1
    elif eos_category == ">=1.5":
        comp["eosinophilia"] = 2
    else:
        comp["eosinophilia"] = 0
    total += comp["eosinophilia"]

    # Atypical lymphocytes
    comp["atypical_lymphocytes"] = 1 if atypical_lymphocytes else 0
    total += comp["atypical_lymphocytes"]

    # Skin: suggestive +1; extent >50% +1; non-suggestive −1
    if skin_suggestive == "yes":
        comp["skin_suggestive"] = 1
    elif skin_suggestive == "no":
        comp["skin_suggestive"] = -1
    else:
        comp["skin_suggestive"] = 0
    total += comp["skin_suggestive"]

    if skin_extent_50 == ">=50":
        comp["skin_extent"] = 1
    else:
        comp["skin_extent"] = 0
    total += comp["skin_extent"]

    # Biopsy
    if biopsy_result == "suggestive":
        comp["biopsy"] = 1
    elif biopsy_result == "not_suggestive":
        comp["biopsy"] = -1
    else:
        comp["biopsy"] = 0
    total += comp["biopsy"]

    # Organ involvement: each organ +1
    comp["organ_involvement"] = max(0, int(organ_involvement_count))
    total += comp["organ_involvement"]

    # Blood tests block
    comp["blood_tests"] = 1 if blood_tests_negative else 0
    total += comp["blood_tests"]

    # Resolution <15 days (deduction)
    if resolution_under_15d:
        comp["resolution_under_15d"] = -1
    else:
        comp["resolution_under_15d"] = 0
    total += comp["resolution_under_15d"]

    # Classification
    if total < 2:
        classification = "No DRESS"
    elif 2 <= total <= 3:
        classification = "Possible DRESS"
    elif 4 <= total <= 5:
        classification = "Probable DRESS"
    else:
        classification = "Definite DRESS"

    return total, classification, comp


# ---------- STREAMLIT APP ----------

st.set_page_config(page_title="SCAR Drug Timeline & Scoring Tool", layout="wide")

st.title("SCAR Drug Timeline & Scoring Tool")
st.caption(
    "Prototype tool for SJS/TEN (ALDEN) and DRESS (RegiSCAR). "
    "Uses published scoring tables; still requires clinical judgement."
)

if "drugs" not in st.session_state:
    st.session_state["drugs"] = []  # list of dicts


# ---- Case-level information ----
st.header("1. Case information")

col_case1, col_case2 = st.columns(2)

with col_case1:
    patient_id = st.text_input("Patient ID / Initials", value="")

with col_case2:
    onset_date = st.date_input(
        "SCAR onset date (index day)", value=datetime.date.today()
    )

scar_type = st.selectbox(
    "SCAR phenotype (for scoring)",
    options=["SJS/TEN", "DRESS"],
    index=0,
)

enable_alden = scar_type == "SJS/TEN"
enable_regiscar = scar_type == "DRESS"

st.markdown("---")

# ---- Drug entry form ----
st.header("2. Drugs to assess")

with st.form("add_drug_form", clear_on_submit=True):
    st.subheader("Add a new drug")

    c1, c2, c3 = st.columns(3)

    with c1:
        drug_name = st.text_input("Drug name")
        category = st.selectbox(
            "Drug category (for colour only)",
            options=[
                "Antimicrobial",
                "Anticonvulsant",
                "Allopurinol",
                "NSAID",
                "Antihypertensive",
                "Diuretic",
                "Cardiovascular - other",
                "Opioid",
                "Psychotropic",
                "Chemotherapy",
                "Biologic",
                "IV contrast",
                "Other",
            ],
            index=0,
        )

    with c2:
        start_date = st.date_input(
            "Start date",
            value=onset_date - datetime.timedelta(days=7),
            key="start_date_input",
        )
        stop_date = st.date_input(
            "Stop date (or estimated)",
            value=onset_date + datetime.timedelta(days=1),
            key="stop_date_input",
        )

    with c3:
        half_life_days = st.number_input(
            "Elimination half-life (days)",
            min_value=0.01,
            max_value=60.0,
            value=1.0,
            step=0.25,
            help="Approximate elimination half-life in days "
                 "(used for presence-at-onset component in ALDEN).",
        )
        impaired_clearance_or_interactions = st.checkbox(
            "Renal/hepatic impairment or strong interactions?",
            value=False,
            help="If yes AND drug stopped >5 half-lives before onset, "
                 "ALDEN presence score is −1 instead of −3.",
        )

    st.markdown("**Pre-/Re-challenge, dechallenge, notoriety and alternatives (ALDEN)**")

    c4, c5, c6 = st.columns(3)

    with c4:
        rechallenge_choice = st.selectbox(
            "Prechallenge/rechallenge",
            options=[
                ("Unknown / no info", "unknown"),
                ("Previous SJS/TEN to this drug", "pos_disease_drug"),
                ("Previous SJS/TEN to similar drug OR other reaction to this drug",
                 "pos_disease_or_drug"),
                ("Other ADR to similar drug", "pos_nonspecific"),
                ("Previous exposure without any reaction", "negative"),
            ],
            format_func=lambda x: x[0],
        )
        rechallenge_category = rechallenge_choice[1]

    with c5:
        dechallenge_choice = st.selectbox(
            "Dechallenge status",
            options=[
                ("Drug stopped or unknown", "neutral"),
                ("Drug not stopped, no worsening", "negative"),
            ],
            format_func=lambda x: x[0],
        )
        dechallenge_status = dechallenge_choice[1]

        notoriety_choice = st.selectbox(
            "Drug notoriety for SJS/TEN",
            options=[
                ("Strongly associated (high-risk)", "strong"),
                ("Associated (proven risk)", "associated"),
                ("Suspect / under surveillance", "suspect"),
                ("Unknown", "unknown"),
                ("Not suspect / no evidence", "not_suspect"),
            ],
            format_func=lambda x: x[0],
        )
        notoriety_level = notoriety_choice[1]

    with c6:
        other_possible_etiology = st.checkbox(
            "Alternative aetiology likely (other drug with high ALDEN)?",
            value=False,
            help="Tick if there is at least one other drug with an "
                 "intermediate ALDEN score >3.",
        )

    submitted = st.form_submit_button("Add drug to list")

    if submitted:
        if drug_name.strip() == "":
            st.error("Please enter a drug name.")
        elif stop_date < start_date:
            st.error("Stop date cannot be before start date.")
        else:
            st.session_state["drugs"].append(
                {
                    "drug_name": drug_name.strip(),
                    "category": category,
                    "start_date": start_date,
                    "stop_date": stop_date,
                    "half_life_days": float(half_life_days),
                    "impaired_clearance_or_interactions": impaired_clearance_or_interactions,
                    "rechallenge_category": rechallenge_category,
                    "dechallenge_status": dechallenge_status,
                    "notoriety_level": notoriety_level,
                    "other_possible_etiology": other_possible_etiology,
                }
            )
            st.success(f"Added {drug_name.strip()}.")


if st.session_state["drugs"]:
    if st.button("Clear all drugs"):
        st.session_state["drugs"] = []


# ---- Show drugs, ALDEN (if SJS/TEN), and Gantt ----
if st.session_state["drugs"]:
    st.subheader("Drug list")

    df_drugs = pd.DataFrame(st.session_state["drugs"])

    if enable_alden:
        alden_scores = []
        alden_categories = []

        for row in st.session_state["drugs"]:
            total_score, category, _ = compute_alden_score_exact(
                start_date=row["start_date"],
                stop_date=row["stop_date"],
                onset_date=onset_date,
                half_life_days=row["half_life_days"],
                impaired_clearance_or_interactions=row[
                    "impaired_clearance_or_interactions"
                ],
                rechallenge_category=row["rechallenge_category"],
                dechallenge_status=row["dechallenge_status"],
                notoriety_level=row["notoriety_level"],
                other_possible_etiology=row["other_possible_etiology"],
            )
            alden_scores.append(total_score)
            alden_categories.append(category)

        df_drugs["ALDEN_score"] = alden_scores
        df_drugs["ALDEN_category"] = alden_categories

    display_df = df_drugs.copy()
    display_df["start_date"] = pd.to_datetime(display_df["start_date"]).dt.strftime(
        "%d/%m/%Y"
    )
    display_df["stop_date"] = pd.to_datetime(display_df["stop_date"]).dt.strftime(
        "%d/%m/%Y"
    )

    cols_to_show = [
        "drug_name",
        "category",
        "start_date",
        "stop_date",
        "half_life_days",
    ]
    if enable_alden:
        cols_to_show += ["ALDEN_score", "ALDEN_category"]

    st.dataframe(display_df[cols_to_show], use_container_width=True)

    # ---- Gantt chart ----
    st.subheader("Drug timeline (Gantt chart)")

    gantt_df = df_drugs.copy()
    gantt_df["Start"] = gantt_df["start_date"].apply(
        lambda d: datetime.datetime.combine(d, datetime.time.min)
    )
    gantt_df["Finish"] = gantt_df["stop_date"].apply(
        lambda d: datetime.datetime.combine(d, datetime.time.min)
    )

    hover_fields = ["category"]
    if enable_alden:
        hover_fields += ["ALDEN_score", "ALDEN_category"]

    fig = px.timeline(
        gantt_df,
        x_start="Start",
        x_end="Finish",
        y="drug_name",
        color="category",
        hover_data=hover_fields,
    )

    # Onset vertical line and annotation (separated to avoid datetime + int bug)
    onset_dt = datetime.datetime.combine(onset_date, datetime.time.min)
    fig.add_vline(
        x=onset_dt,
        line_width=2,
        line_dash="dash",
    )
    fig.add_annotation(
        x=onset_dt,
        y=1,
        xref="x",
        yref="paper",
        text="Onset",
        showarrow=False,
        yanchor="bottom",
    )

    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(
        tickformat="%d/%m/%Y",
        showgrid=True,
    )
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=30, b=20))

    st.plotly_chart(fig, use_container_width=True)

    # ---- PDF export of timeline ----
    pdf_bytes = None
    try:
        import plotly.io as pio
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.utils import ImageReader

        img_bytes = pio.to_image(fig, format="png", scale=2)

        pdf_buffer = BytesIO()
        page_size = landscape(A4)
        c = canvas.Canvas(pdf_buffer, pagesize=page_size)
        width, height = page_size

        img_stream = BytesIO(img_bytes)
        image = ImageReader(img_stream)
        img_w, img_h = image.getSize()

        max_w = width - 40
        max_h = height - 40
        scale = min(max_w / img_w, max_h / img_h)
        new_w = img_w * scale
        new_h = img_h * scale

        x = (width - new_w) / 2
        y = (height - new_h) / 2

        c.drawImage(image, x, y, width=new_w, height=new_h)
        c.showPage()
        c.save()
        pdf_buffer.seek(0)
        pdf_bytes = pdf_buffer.getvalue()
    except Exception:
        # On Streamlit Cloud or environments without Kaleido/Chrome or ReportLab support
        st.info(
            "PDF export of the timeline is not available in this environment. "
            "If running locally, install 'kaleido' and 'reportlab' via "
            "`pip install kaleido reportlab`."
        )
        pdf_bytes = None

    if pdf_bytes:
        filename_base = patient_id or "scar_case"
        st.download_button(
            label="Download timeline as PDF",
            data=pdf_bytes,
            file_name=f"{filename_base}_timeline.pdf",
            mime="application/pdf",
        )

st.markdown("---")

# ---- RegiSCAR DRESS scoring ----
st.header("3. RegiSCAR DRESS scoring (only for DRESS phenotype)")

if not enable_regiscar:
    st.info("Select **DRESS** as SCAR phenotype above to enable RegiSCAR scoring.")
else:
    st.markdown(
        "Uses RegiSCAR scoring as per validation description "
        "(skin, fever, eosinophilia, organs, biopsy, blood tests, deductions)."
    )

    with st.form("regiscar_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            fever_sel = st.selectbox(
                "Fever ≥38.5°C at any point?",
                options=[("Yes", "yes"), ("No / Unknown", "no_unknown")],
                format_func=lambda x: x[0],
            )
            fever_38_5 = fever_sel[1]

            lymph_nodes = st.checkbox(
                "Enlarged lymph nodes at ≥2 sites (>1 cm)?",
                value=False,
            )

            eos_sel = st.selectbox(
                "Peak eosinophils (×10⁹/L)",
                options=[
                    ("<0.70 or none", "none"),
                    ("0.70–1.49", "0.7-1.49"),
                    ("≥1.50", ">=1.5"),
                ],
                format_func=lambda x: x[0],
            )
            eos_category = eos_sel[1]

        with c2:
            atypical_lymphocytes = st.checkbox(
                "Atypical lymphocytes present on blood film?",
                value=False,
            )

            skin_suggestive_sel = st.selectbox(
                "Eruption suggests DRESS (infiltrated, facial oedema, etc.)?",
                options=[
                    ("Yes", "yes"),
                    ("No (non-suggestive)", "no"),
                    ("Unknown", "unknown"),
                ],
                format_func=lambda x: x[0],
            )
            skin_suggestive = skin_suggestive_sel[1]

            skin_extent_sel = st.selectbox(
                "Extent of rash",
                options=[
                    ("<50% BSA", "<50"),
                    (">=50% BSA", ">=50"),
                ],
                format_func=lambda x: x[0],
            )
            skin_extent_50 = skin_extent_sel[1]

        with c3:
            biopsy_sel = st.selectbox(
                "Skin biopsy result",
                options=[
                    ("Not done / unknown", "not_done"),
                    ("Compatible with DRESS", "suggestive"),
                    ("Not compatible with DRESS", "not_suggestive"),
                ],
                format_func=lambda x: x[0],
            )
            biopsy_result = biopsy_sel[1]

            organ_involvement_count = st.number_input(
                "Number of involved internal organs "
                "(liver, kidney, lung, heart, etc.)",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
            )

            blood_tests_negative = st.checkbox(
                "Viral/autoimmune/infectious workup: 0 positive, ≥3 negative?",
                value=False,
                help="E.g. hepatitis, Mycoplasma, Chlamydia, ANA, cultures.",
            )

            resolution_under_15d = st.checkbox(
                "Resolution <15 days from onset?",
                value=False,
            )

        regiscar_submitted = st.form_submit_button("Compute RegiSCAR DRESS score")

    if enable_regiscar and regiscar_submitted:
        total_regiscar, class_regiscar, comp = compute_regiscar_dress_score_exact(
            fever_38_5=fever_38_5,
            lymph_nodes=bool(lymph_nodes),
            eos_category=eos_category,
            atypical_lymphocytes=bool(atypical_lymphocytes),
            skin_suggestive=skin_suggestive,
            skin_extent_50=skin_extent_50,
            biopsy_result=biopsy_result,
            organ_involvement_count=int(organ_involvement_count),
            blood_tests_negative=bool(blood_tests_negative),
            resolution_under_15d=bool(resolution_under_15d),
        )

        st.subheader("RegiSCAR DRESS result")
        st.markdown(f"**Total score:** {total_regiscar}")
        st.markdown(f"**Classification:** {class_regiscar}")

        with st.expander("Show component scores"):
            comp_df = pd.DataFrame(
                [{"item": k, "score": v} for k, v in comp.items()]
            ).set_index("item")
            st.table(comp_df)
