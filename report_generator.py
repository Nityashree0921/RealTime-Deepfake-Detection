from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

styles = getSampleStyleSheet()


def generate_report(prediction, confidence, image_path=None):
    os.makedirs(REPORTS_DIR, exist_ok=True)

    filename = os.path.join(
        REPORTS_DIR,
        datetime.now().strftime("%Y%m%d_%H%M%S") + ".pdf"
    )

    doc = SimpleDocTemplate(filename)
    story = []

    story.append(Paragraph("<b>AI Deepfake Detection Report</b>", styles["Title"]))
    story.append(Paragraph(f"Date : {datetime.now().strftime('%d-%m-%Y')}", styles["Normal"]))
    story.append(Paragraph(f"Time : {datetime.now().strftime('%H:%M:%S')}", styles["Normal"]))
    story.append(Paragraph(f"Prediction : <b>{prediction}</b>", styles["Normal"]))
    story.append(Paragraph(f"Confidence : {confidence:.2f}%", styles["Normal"]))

    if image_path:
        story.append(Paragraph(f"Screenshot : {image_path}", styles["Normal"]))

    doc.build(story)

    print("PDF Report Saved:", filename)
    return filename