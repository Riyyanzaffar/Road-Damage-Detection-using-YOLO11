
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
from io import BytesIO
from datetime import datetime
import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as PDFImage,
    Table, TableStyle, KeepTogether
)

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Road Damage AI",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #071426 0%, #0b1930 55%, #102541 100%);
    color: #f1f5f9;
}
[data-testid="stSidebar"] {
    background-color: #081323;
    border-right: 1px solid #223b59;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}
.hero {
    background: linear-gradient(120deg, #102b4c, #123d61);
    padding: 28px;
    border-radius: 18px;
    border: 1px solid #285477;
    margin-bottom: 24px;
}
.hero h1 {
    color: #ffffff;
    font-size: 34px;
    margin-bottom: 8px;
}
.hero p {
    color: #c5d7e9;
    font-size: 16px;
}
.section-title {
    color: #dbeafe;
    font-size: 21px;
    font-weight: 700;
    margin: 15px 0;
}
div[data-testid="stMetric"] {
    background: #10233b;
    border: 1px solid #254564;
    padding: 18px;
    border-radius: 14px;
}
div[data-testid="stMetricLabel"] {
    color: #b7cbe0;
}
.stButton > button {
    background: #1685e8;
    color: white;
    border: 0;
    border-radius: 10px;
    min-height: 45px;
    font-weight: 700;
}
.stButton > button:hover {
    background: #086bc2;
    color: white;
    border: 1px solid #64b5ff;
}
div[data-testid="stDownloadButton"] button {
    background: #123b36;
    color: #d1fae5;
    border: 1px solid #247d68;
    border-radius: 10px;
    min-height: 42px;
}
div[data-testid="stFileUploader"] {
    background: #0d2036;
    padding: 14px;
    border: 1px dashed #426586;
    border-radius: 12px;
}
hr {
    border-color: #25415f;
}
.small-note {
    color: #9fb4cb;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

MODEL_PATH = Path(__file__).parent / "best (3).pt"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file nahi mili: {MODEL_PATH.name}. "
            "Is file ko app.py ke saath same folder mein upload karein."
        )
    return YOLO(str(MODEL_PATH))

# ---------------- PDF REPORT ----------------

def create_pdf_report(analysis):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=38,
        leftMargin=38,
        topMargin=38,
        bottomMargin=38
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontSize=23,
        leading=28,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#12365a"),
        spaceAfter=10
    ))
    styles.add(ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#155e75"),
        spaceBefore=14,
        spaceAfter=8
    ))

    story = [
        Paragraph("ROAD DAMAGE DETECTION REPORT", styles["ReportTitle"]),
        Paragraph(
            f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
            styles["Normal"]
        ),
        Spacer(1, 14),
        Paragraph("Analysis Summary", styles["SectionHeading"])
    ]

    detections = analysis["detections"]
    total = len(detections)
    avg_conf = (
        sum(item["Confidence Value"] for item in detections) / total
        if total else 0
    )

    summary_data = [
        ["Metric", "Result"],
        ["Total detections", str(total)],
        ["Damage classes found", str(len(analysis["summary"]))],
        ["Average confidence", f"{avg_conf:.2%}"]
    ]

    summary_table = Table(summary_data, colWidths=[250, 230])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12365a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#b7c8d8")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f1f6fa")),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#15283b")),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)

    story.append(Paragraph("Damage Class Breakdown", styles["SectionHeading"]))

    class_data = [["Damage Class", "Count", "Average Confidence"]]
    for item in analysis["summary"]:
        class_data.append([
            item["Damage Class"],
            str(item["Count"]),
            f'{item["Average Confidence"]:.2%}'
        ])

    if not analysis["summary"]:
        class_data.append(["No damage detected", "0", "N/A"])

    class_table = Table(class_data, colWidths=[230, 80, 170], repeatRows=1)
    class_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#155e75")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#b7c8d8")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f1f6fa")),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#15283b")),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(class_table)

    story.append(Paragraph("Detected Road Image", styles["SectionHeading"]))

    detected_image = Image.open(BytesIO(analysis["image_bytes"])).convert("RGB")
    image_buffer = BytesIO()
    detected_image.save(image_buffer, format="JPEG", quality=88)
    image_buffer.seek(0)

    max_width = 500
    image_height = max_width * detected_image.height / detected_image.width
    pdf_image = PDFImage(
        image_buffer,
        width=max_width,
        height=image_height
    )
    story.append(pdf_image)

    story.append(Paragraph("Detection Details", styles["SectionHeading"]))

    detail_data = [["#", "Damage Class", "Confidence"]]
    for index, item in enumerate(detections[:100], start=1):
        detail_data.append([
            str(index),
            item["Damage Class"],
            f'{item["Confidence Value"]:.2%}'
        ])

    if not detections:
        detail_data.append(["-", "No damage detected", "-"])

    detail_table = Table(
        detail_data,
        colWidths=[40, 300, 140],
        repeatRows=1
    )
    detail_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12365a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#b7c8d8")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f1f6fa")),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#15283b")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(detail_table)

    if len(detections) > 100:
        story.append(Spacer(1, 8))
        story.append(Paragraph(
            "Note: Report mein pehli 100 detections ki details dikhayi gayi hain. "
            "Summary mein tamam detections shamil hain.",
            styles["Normal"]
        ))

    story.append(Spacer(1, 16))
    story.append(Paragraph(
        "Note: Confidence model ka estimate hai, guaranteed accuracy ya "
        "professional road-safety inspection ka replacement nahi.",
        styles["Italic"]
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown("## ROAD AI")
    st.caption("Computer Vision • YOLO")
    st.divider()

    st.markdown("### Detection Settings")
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.75,
        value=0.10,
        step=0.05,
        help="Kam threshold par zyada detections aa sakti hain, lekin false detections bhi barh sakti hain."
    )

    st.divider()
    st.markdown("### How to use")
    st.write("1. Road image upload karein.")
    st.write("2. Detect button dabayein.")
    st.write("3. Results aur damage summary dekhein.")
    st.write("4. Image ya PDF report download karein.")

    st.divider()
    st.caption("Model: best (3).pt")
    st.caption("Detection quality image aur training data par depend karti hai.")

# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>Road Damage Intelligence</h1>
    <p>AI-powered road crack and surface damage analysis</p>
</div>
""", unsafe_allow_html=True)

try:
    model = load_model()
except Exception as error:
    st.error(f"Model load nahi hua: {error}")
    st.info(
        "Confirm karein ke best (3).pt aur app.py GitHub repository ke "
        "same folder mein hain."
    )
    st.stop()

# ---------------- IMAGE UPLOAD ----------------

st.markdown('<div class="section-title">Upload Road Image</div>',
            unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Road ki JPG, JPEG ya PNG image select karein",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Shuru karne ke liye road ki image upload karein.")
    st.markdown("""
    <div class="small-note">
    Supported formats: JPG, JPEG, PNG
    </div>
    """, unsafe_allow_html=True)
    st.stop()

original_bytes = uploaded_file.getvalue()
image = Image.open(BytesIO(original_bytes)).convert("RGB")

# New image upload hone par purana result clear karein
if st.session_state.get("current_upload") != original_bytes:
    st.session_state["current_upload"] = original_bytes
    st.session_state.pop("analysis", None)

preview_left, preview_right = st.columns([1, 1])

with preview_left:
    st.markdown("#### Original Image")
    st.image(image, use_container_width=True)

with preview_right:
    st.markdown("#### Image Information")
    st.metric("Image Width", f"{image.width} px")
    st.metric("Image Height", f"{image.height} px")
    st.caption(f"File: {uploaded_file.name}")

if st.button("RUN AI DAMAGE DETECTION", use_container_width=True):
    with st.spinner("AI model image analyse kar raha hai..."):
        try:
            results = model.predict(
                source=np.array(image),
                conf=confidence,
                verbose=False
            )

            result = results[0]
            annotated_bgr = result.plot()
            annotated_rgb = annotated_bgr[:, :, ::-1]

            annotated_image = Image.fromarray(annotated_rgb)
            image_buffer = BytesIO()
            annotated_image.save(image_buffer, format="PNG")

            detection_rows = []

            for box in result.boxes:
                class_id = int(box.cls[0].item())
                class_name = model.names[class_id]
                conf_value = float(box.conf[0].item())

                detection_rows.append({
                    "Damage Class": str(class_name),
                    "Confidence": f"{conf_value:.2%}",
                    "Confidence Value": conf_value
                })

            summary_rows = []
            if detection_rows:
                summary_df = pd.DataFrame(detection_rows)

                for class_name, group in summary_df.groupby("Damage Class"):
                    summary_rows.append({
                        "Damage Class": class_name,
                        "Count": len(group),
                        "Average Confidence": group["Confidence Value"].mean()
                    })

                summary_rows.sort(
                    key=lambda row: row["Count"],
                    reverse=True
                )

            st.session_state["analysis"] = {
                "image_bytes": image_buffer.getvalue(),
                "detections": detection_rows,
                "summary": summary_rows,
                "filename": uploaded_file.name,
                "threshold": confidence
            }

        except Exception as error:
            st.error(f"Detection mein error aaya: {error}")

# ---------------- DISPLAY RESULTS ----------------

analysis = st.session_state.get("analysis")

if analysis is not None:
    detections = analysis["detections"]
    total_detections = len(detections)

    average_confidence = (
        sum(row["Confidence Value"] for row in detections) / total_detections
        if total_detections else 0
    )

    st.divider()
    st.markdown('<div class="section-title">Detection Dashboard</div>',
                unsafe_allow_html=True)

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric("Total Detections", total_detections)
    metric2.metric("Damage Classes", len(analysis["summary"]))
    metric3.metric("Average Confidence", f"{average_confidence:.2%}")

    st.markdown("#### AI Detection Result")
    st.image(
        analysis["image_bytes"],
        caption="Road image with predicted damage boxes",
        use_container_width=True
    )

    st.markdown("#### Damage Class Analysis")

    if analysis["summary"]:
        summary_df = pd.DataFrame(analysis["summary"])
        summary_df["Average Confidence"] = summary_df[
            "Average Confidence"
        ].map(lambda value: f"{value:.2%}")

        table_col, chart_col = st.columns([1, 1])

        with table_col:
            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True
            )

        with chart_col:
            chart_data = pd.DataFrame(analysis["summary"])
            st.bar_chart(
                chart_data.set_index("Damage Class")["Count"]
            )

        st.markdown("#### Individual Detections")
        details_df = pd.DataFrame(detections)
        details_df = details_df.drop(columns=["Confidence Value"])
        st.dataframe(
            details_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning(
            "Is threshold par koi damage detect nahi hua. "
            "Threshold kam karke ya doosri image ke saath try karein."
        )

    st.divider()
    st.markdown('<div class="section-title">Download Results</div>',
                unsafe_allow_html=True)

    download_col1, download_col2 = st.columns(2)

    with download_col1:
        st.download_button(
            "Download Annotated Image",
            data=analysis["image_bytes"],
            file_name="road_damage_detected.png",
            mime="image/png",
            use_container_width=True
        )

    with download_col2:
        try:
            pdf_bytes = create_pdf_report(analysis)
            st.download_button(
                "Download PDF Report",
                data=pdf_bytes,
                file_name="road_damage_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as error:
            st.error(f"PDF report generate nahi hui: {error}")

st.divider()
st.markdown(
    '<p class="small-note">Road Damage AI • YOLO Computer Vision Project</p>',
    unsafe_allow_html=True
)
