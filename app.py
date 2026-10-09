
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
from io import BytesIO
from datetime import datetime
import numpy as np
import pandas as pd

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Road Damage AI",
    page_icon="🚧",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "best (3).pt"
# ---------------- PROFESSIONAL LIGHT/DARK-BLUE DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background: #f2f6fc;
    color: #20344d;
}
[data-testid="stHeader"] {
    background: #f2f6fc;
}
[data-testid="stSidebar"] {
    background: #e3edf8;
    border-right: 1px solid #c5d6e8;
}
.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.hero {
    background: linear-gradient(120deg, #14365b, #23658b);
    padding: 30px;
    border-radius: 18px;
    margin-bottom: 24px;
    border: 1px solid #427da5;
}
.hero h1 {
    color: #ffffff !important;
    font-size: 34px;
    margin: 0 0 8px 0;
}
.hero p {
    color: #e8f3ff !important;
    font-size: 16px;
    margin: 0;
}
.section-title {
    color: #173e65 !important;
    font-size: 22px;
    font-weight: 750;
    margin: 18px 0 12px 0;
}
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #d4e0ed;
    border-radius: 14px;
    padding: 17px;
    box-shadow: 0 3px 12px rgba(20, 54, 91, 0.05);
}
div[data-testid="stMetricLabel"] {
    color: #526981 !important;
}
div[data-testid="stMetricValue"] {
    color: #163d64 !important;
}
.stButton > button {
    background: #1769aa;
    color: white;
    border: 1px solid #1769aa;
    border-radius: 10px;
    min-height: 45px;
    font-weight: 700;
}
.stButton > button:hover {
    background: #10558d;
    color: white;
    border-color: #10558d;
}
div[data-testid="stDownloadButton"] button {
    background: #e1f2ff;
    color: #174b73;
    border: 1px solid #9bc8e7;
    border-radius: 10px;
    min-height: 43px;
    font-weight: 650;
}
div[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 1px dashed #9db5cd;
    border-radius: 12px;
    padding: 12px;
}
div[data-testid="stFileUploader"] label,
div[data-testid="stFileUploader"] small {
    color: #263f59 !important;
}
.stMarkdown, .stCaption, label, li {
    color: #263f59;
}
h1, h2, h3, h4, h5 {
    color: #173e65;
}
hr {
    border-color: #cfdeed;
}
.small-note {
    color: #536b82;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Model file nahi mili. 'best (3).pt' ko app.py ke saath "
            "GitHub ke same folder mein upload karein."
        )
    return YOLO(str(MODEL_PATH))

# ---------------- CREATE PDF REPORT ----------------

def create_pdf_report(analysis):
    # PDF library sirf report banate waqt load hogi.
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer,
        Image as PDFImage, Table, TableStyle
    )

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=38,
        leftMargin=38,
        topMargin=35,
        bottomMargin=35
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="RoadTitle",
        parent=styles["Title"],
        fontSize=21,
        leading=27,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#173e65"),
        spaceAfter=12
    ))
    styles.add(ParagraphStyle(
        name="RoadHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#1769aa"),
        spaceBefore=14,
        spaceAfter=8
    ))

    detections = analysis["detections"]
    summary = analysis["summary"]
    count = len(detections)
    average = (
        sum(item["confidence_value"] for item in detections) / count
        if count else 0
    )

    story = [
        Paragraph("ROAD DAMAGE AI REPORT", styles["RoadTitle"]),
        Paragraph(
            "Generated: " + datetime.now().strftime("%d %B %Y, %I:%M %p"),
            styles["Normal"]
        ),
        Spacer(1, 14),
        Paragraph("Analysis Summary", styles["RoadHeading"])
    ]

    summary_table_data = [
        ["Metric", "Result"],
        ["Total detections", str(count)],
        ["Different damage classes", str(len(summary))],
        ["Average detection confidence", f"{average:.2%}"],
        ["Confidence threshold used", f'{analysis["threshold"]:.2f}']
    ]

    summary_table = Table(summary_table_data, colWidths=[3.4 * inch, 2.6 * inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173e65")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#eff6fc")),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#20344d")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c6d6e6")),
        ("PADDING", (0, 0), (-1, -1), 7)
    ]))
    story.append(summary_table)

    story.append(Paragraph("Damage Class Breakdown", styles["RoadHeading"]))

    class_data = [["Damage class", "Count", "Avg. confidence"]]
    for item in summary:
        class_data.append([
            item["damage_class"],
            str(item["count"]),
            f'{item["average_confidence"]:.2%}'
        ])

    if not summary:
        class_data.append(["No detections", "0", "N/A"])

    class_table = Table(
        class_data,
        colWidths=[3.2 * inch, 1.0 * inch, 1.8 * inch],
        repeatRows=1
    )
    class_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1769aa")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#eff6fc")),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#20344d")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c6d6e6")),
        ("PADDING", (0, 0), (-1, -1), 7)
    ]))
    story.append(class_table)

    story.append(Paragraph("Detected Image", styles["RoadHeading"]))

    detected_image = Image.open(
        BytesIO(analysis["image_bytes"])
    ).convert("RGB")
    detected_image.thumbnail((1100, 650))

    image_buffer = BytesIO()
    detected_image.save(image_buffer, format="JPEG", quality=88)
    image_buffer.seek(0)

    pdf_image = PDFImage(
        image_buffer,
        width=6.4 * inch,
        height=6.4 * inch * detected_image.height / detected_image.width
    )
    story.append(pdf_image)

    story.append(Paragraph("Individual Detections", styles["RoadHeading"]))

    detail_data = [["#", "Damage class", "Confidence"]]
    for index, item in enumerate(detections[:100], start=1):
        detail_data.append([
            str(index),
            item["damage_class"],
            f'{item["confidence_value"]:.2%}'
        ])

    if not detections:
        detail_data.append(["-", "No damage detected", "-"])

    detail_table = Table(
        detail_data,
        colWidths=[0.5 * inch, 3.8 * inch, 1.7 * inch],
        repeatRows=1
    )
    detail_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173e65")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#eff6fc")),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#20344d")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c6d6e6")),
        ("PADDING", (0, 0), (-1, -1), 6)
    ]))
    story.append(detail_table)

    if len(detections) > 100:
        story.append(Spacer(1, 8))
        story.append(Paragraph(
            "Note: Is PDF mein pehli 100 individual detections hain. "
            "Summary mein tamam detections shamil hain.",
            styles["Normal"]
        ))

    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "Disclaimer: Confidence model ka estimate hai, guaranteed accuracy "
        "nahi. Is report ko professional road inspection ka replacement "
        "na samjhein.",
        styles["Normal"]
    ))

    document.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown("## ROAD DAMAGE AI")
    st.caption("Computer Vision • YOLO")
    st.divider()

    st.markdown("### Detection Settings")
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.75,
        value=0.10,
        step=0.05,
        help=(
            "Agar detections nahi aa rahi to 0.05 try karein. "
            "Threshold kam karne se false detections bhi aa sakti hain."
        )
    )

    st.markdown("---")
    st.markdown("### How to use")
    st.write("1. Road image upload karein.")
    st.write("2. Run Detection dabayein.")
    st.write("3. Image aur damage summary dekhein.")
    st.write("4. Image ya PDF report download karein.")

    st.divider()
    st.caption("Model file: best (3).pt")

# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>Road Damage Intelligence</h1>
    <p>AI-powered road crack and surface damage analysis</p>
</div>
""", unsafe_allow_html=True)

# ---------------- MODEL CHECK ----------------

try:
    model = load_model()
except Exception as error:
    st.error(f"Model load nahi hua: {error}")
    st.info(
        "Check karein ke 'best (3).pt' file, app.py ke same folder mein "
        "GitHub par uploaded hai."
    )
    st.stop()

# ---------------- UPLOAD IMAGE ----------------

st.markdown(
    '<div class="section-title">Upload Road Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Road ki JPG, JPEG ya PNG image upload karein",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Detection shuru karne ke liye road image upload karein.")
    st.markdown(
        '<p class="small-note">Supported formats: JPG, JPEG, PNG</p>',
        unsafe_allow_html=True
    )
    st.stop()

try:
    original_bytes = uploaded_file.getvalue()
    image = Image.open(BytesIO(original_bytes)).convert("RGB")
except Exception:
    st.error("Image open nahi ho saki. Doosri JPG ya PNG image try karein.")
    st.stop()

# Nayi image upload hone par purana result remove karein
if st.session_state.get("upload_signature") != (
    uploaded_file.name, len(original_bytes), original_bytes[:100]
):
    st.session_state["upload_signature"] = (
        uploaded_file.name, len(original_bytes), original_bytes[:100]
    )
    st.session_state.pop("analysis", None)

left, right = st.columns(2)

with left:
    st.markdown("#### Original Image")
    st.image(image, use_container_width=True)

with right:
    st.markdown("#### Image Details")
    st.metric("Width", f"{image.width} px")
    st.metric("Height", f"{image.height} px")
    st.caption(f"Filename: {uploaded_file.name}")

# ---------------- RUN DETECTION ----------------

if st.button("RUN AI DAMAGE DETECTION", use_container_width=True):
    with st.spinner("AI model road image analyse kar raha hai..."):
        try:
            results = model.predict(
                source=np.array(image),
                conf=confidence,
                verbose=False
            )
            result = results[0]

            # Ultralytics plot result BGR mein hota hai; Streamlit ke liye RGB
            annotated_rgb = result.plot()[:, :, ::-1]
            annotated_image = Image.fromarray(annotated_rgb)

            image_buffer = BytesIO()
            annotated_image.save(image_buffer, format="PNG")

            detections = []

            for box in result.boxes:
                class_id = int(box.cls[0].item())
                class_name = str(model.names[class_id])
                conf_value = float(box.conf[0].item())

                detections.append({
                    "damage_class": class_name,
                    "confidence_value": conf_value
                })

            summary = []
            if detections:
                detection_df = pd.DataFrame(detections)
                for class_name, group in detection_df.groupby("damage_class"):
                    summary.append({
                        "damage_class": class_name,
                        "count": len(group),
                        "average_confidence": float(
                            group["confidence_value"].mean()
                        )
                    })

                summary.sort(
                    key=lambda item: item["count"],
                    reverse=True
                )

            st.session_state["analysis"] = {
                "image_bytes": image_buffer.getvalue(),
                "detections": detections,
                "summary": summary,
                "threshold": confidence,
                "filename": uploaded_file.name
            }

        except Exception as error:
            st.error(f"Detection mein error aaya: {error}")

# ---------------- DISPLAY RESULTS ----------------

analysis = st.session_state.get("analysis")

if analysis is not None:
    detections = analysis["detections"]
    summary = analysis["summary"]
    total = len(detections)

    average_confidence = (
        sum(item["confidence_value"] for item in detections) / total
        if total else 0
    )

    st.divider()
    st.markdown(
        '<div class="section-title">Detection Dashboard</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Detections", total)
    col2.metric("Damage Classes", len(summary))
    col3.metric("Average Confidence", f"{average_confidence:.2%}")

    st.caption(
        f"Threshold used for this result: {analysis['threshold']:.2f}. "
        "Threshold change karne ke baad dobara Run Detection dabayein."
    )

    st.markdown("#### AI Detection Result")
    st.image(
        analysis["image_bytes"],
        caption="Predicted road damage boxes",
        use_container_width=True
    )

    st.markdown("#### Damage Class Analysis")

    if summary:
        chart_col, table_col = st.columns([1, 1])

        summary_df = pd.DataFrame(summary)
        display_df = summary_df.copy()
        display_df["Average Confidence"] = display_df[
            "average_confidence"
        ].map(lambda value: f"{value:.2%}")

        display_df = display_df.rename(columns={
            "damage_class": "Damage Class",
            "count": "Count"
        })
        display_df = display_df[
            ["Damage Class", "Count", "Average Confidence"]
        ]

        with chart_col:
            st.markdown("##### Detections by Class")
            st.bar_chart(
                summary_df.set_index("damage_class")["count"]
            )

        with table_col:
            st.markdown("##### Class Summary")
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

        st.markdown("#### Individual Detections")
        detail_df = pd.DataFrame([
            {
                "Damage Class": item["damage_class"],
                "Confidence": f"{item['confidence_value']:.2%}"
            }
            for item in detections
        ])
        st.dataframe(
            detail_df,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.warning(
            "Is threshold par koi damage detect nahi hua. Sidebar mein "
            "threshold 0.05 karein aur RUN AI DAMAGE DETECTION dobara dabayein. "
            "Agar phir bhi result nahi aata, model is image ko identify nahi "
            "kar pa raha ho sakta hai."
        )

    st.divider()
    st.markdown(
        '<div class="section-title">Download Results</div>',
        unsafe_allow_html=True
    )

    download_col1, download_col2 = st.columns(2)

    with download_col1:
        st.download_button(
            label="Download Annotated Image",
            data=analysis["image_bytes"],
            file_name="road_damage_detected.png",
            mime="image/png",
            use_container_width=True
        )

    with download_col2:
        try:
            pdf_data = create_pdf_report(analysis)
            st.download_button(
                label="Download PDF Report",
                data=pdf_data,
                file_name="road_damage_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as error:
            st.error(
                "PDF report nahi bani. requirements.txt mein reportlab "
                f"check karein. Details: {error}"
            )

st.divider()
st.markdown(
    '<p class="small-note">Road Damage AI • YOLO Computer Vision Project</p>',
    unsafe_allow_html=True
)
