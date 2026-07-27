import smtplib
from email.message import EmailMessage
import os

SENDER_EMAIL = "shanbhagmadhusudan251@gmail.com"
APP_PASSWORD = "agag aafa vayq mzss"
RECEIVER_EMAIL = "nityashreeshanbhag@gmail.com"


def send_alert(image_path, confidence):
    try:
        msg = EmailMessage()
        msg["Subject"] = "🚨 Deepfake Detected"
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL
        msg.set_content(f"""
Deepfake Detection Alert

Prediction : FAKE

Confidence : {confidence:.2f}%

Please check the attached screenshot.
""")

        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                data = f.read()
            msg.add_attachment(
                data,
                maintype="image",
                subtype="jpeg",
                filename=os.path.basename(image_path)
            )

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(SENDER_EMAIL, APP_PASSWORD)
            smtp.send_message(msg)

        print("Alert Email Sent")
    except Exception as e:
        print("Email Error:", e)
        return False

    return True