from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

styles = getSampleStyleSheet()


def generate_report(prediction, confidence, image_path=None, file_path=None, modality=None, duration=None):
    os.makedirs(REPORTS_DIR, exist_ok=True)

    filename = os.path.join(
        REPORTS_DIR,
        datetime.now().strftime("%Y%m%d_%H%M%S") + ".pdf"
    )

    doc = SimpleDocTemplate(filename)
    story = []

    # Infer modality if not provided
    if not modality:
        if str(prediction).startswith("AUDIO-"):
            modality = "AUDIO"
        elif str(prediction).startswith("VIDEO-"):
            modality = "VIDEO"
        elif str(prediction).startswith("IMAGE-"):
            modality = "IMAGE"
        else:
            modality = "MULTIMODAL / WEBCAM"

    story.append(Paragraph("<b>AI Deepfake Detection Report</b>", styles["Title"]))
    story.append(Paragraph(f"<b>Modality</b> : {modality}", styles["Normal"]))
    story.append(Paragraph(f"Date : {datetime.now().strftime('%d-%m-%Y')}", styles["Normal"]))
    story.append(Paragraph(f"Time : {datetime.now().strftime('%H:%M:%S')}", styles["Normal"]))
    story.append(Paragraph(f"Prediction : <b>{prediction}</b>", styles["Normal"]))
    story.append(Paragraph(f"Confidence : {confidence:.2f}%", styles["Normal"]))

    if file_path:
        story.append(Paragraph(f"Analyzed File : {file_path}", styles["Normal"]))
    if duration:
        story.append(Paragraph(f"Audio Duration : {duration:.2f} seconds", styles["Normal"]))
    if image_path:
        story.append(Paragraph(f"Screenshot : {image_path}", styles["Normal"]))

    doc.build(story)

    print("PDF Report Saved:", filename)
    return filename