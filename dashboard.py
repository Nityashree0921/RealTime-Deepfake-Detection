import tkinter as tk
from tkinter import messagebox
import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _start_process(script_path):
    launch_kwargs = {
        "cwd": BASE_DIR,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }

    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        launch_kwargs["startupinfo"] = startupinfo

    return subprocess.Popen(
        [sys.executable, script_path],
        **launch_kwargs,
    )


def logout():
    root.destroy()
    _start_process(os.path.join(BASE_DIR, "login.py"))

from analytics import total_detections, total_real, total_fake

# ----------------------------
# Functions
# ----------------------------
def launch_script(script_name):
    script_path = os.path.join(BASE_DIR, script_name)
    if not os.path.exists(script_path):
        messagebox.showerror("Launch Error", f"Script not found: {script_name}")
        return

    try:
        _start_process(script_path)
    except Exception as e:
        messagebox.showerror("Launch Error", str(e))


def webcam():
    launch_script("app.py")


def image():
    launch_script("image_detector.py")


def video():
    launch_script("video_detector.py")


def audio():
    launch_script("audio_detector.py")


def history():
    launch_script("history.py")


def open_path(path, label):
    if os.path.exists(path):
        try:
            os.startfile(path)
        except OSError:
            try:
                os.startfile(os.path.dirname(path))
            except OSError:
                messagebox.showerror("Error", f"Unable to open {label} automatically.")
    else:
        messagebox.showerror("Error", f"{label} not found.")


def open_screenshots():
    folder = os.path.join(BASE_DIR, "screenshots")
    open_path(folder, "Screenshots folder")

def open_database():
    db = os.path.join(BASE_DIR, "detections.db")
    open_path(db, "Database file")


def get_report_files():
    reports_dir = os.path.join(BASE_DIR, "reports")
    if not os.path.exists(reports_dir):
        return []

    files = [f for f in os.listdir(reports_dir) if f.lower().endswith(".pdf")]
    files.sort(key=lambda name: os.path.getmtime(os.path.join(reports_dir, name)), reverse=True)
    return files


def refresh_reports_label():
    reports_dir = os.path.join(BASE_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    reports = get_report_files()
    if reports:
        latest = reports[0]
        reports_label.config(
            text=f"Latest report: {latest}\nTotal PDF reports: {len(reports)}\nReports folder: {reports_dir}",
            fg="lightgreen"
        )
    else:
        reports_label.config(
            text=f"No PDF reports yet.\nRun a detection to generate one.\nReports folder: {reports_dir}",
            fg="lightyellow"
        )


def open_csv_log():
    csv_path = os.path.join(BASE_DIR, "detections.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            f.write("Date,Time,Prediction,Confidence\n")
    open_path(csv_path, "CSV log file")


def open_reports_folder():
    reports_path = os.path.join(BASE_DIR, "reports")
    os.makedirs(reports_path, exist_ok=True)
    refresh_reports_label()
    open_path(reports_path, "Reports folder")

def about():
    messagebox.showinfo(
        "About Project",
        """
AI Deepfake Detection System

Features

• Webcam Detection
• Image Detection
• Video Detection
• Audio Detection
• Detection History
• Screenshot Capture
• Email Alerts
• SQLite Database
• CSV Logging
• PDF Reports
• Analytics Dashboard

Developed Using

Python
OpenCV
TensorFlow
SQLite
Tkinter
"""
    )

def exit_app():
    root.destroy()


# ----------------------------
# Main Window
# ----------------------------

root = tk.Tk()

root.title("AI Deepfake Detection System")

root.geometry("760x860")

root.configure(bg="#1E1E2E")

root.resizable(False, False)

# ----------------------------
# Header
# ----------------------------

main_frame = tk.Frame(root, bg="#1E1E2E")
main_frame.pack(fill="both", expand=True)

canvas = tk.Canvas(main_frame, bg="#1E1E2E", highlightthickness=0)
scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas, bg="#1E1E2E")

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Header

title = tk.Label(
    scrollable_frame,
    text="AI DEEPFAKE DETECTION SYSTEM",
    font=("Arial", 22, "bold"),
    fg="white",
    bg="#1E1E2E"
)

title.pack(pady=20)

subtitle = tk.Label(
    scrollable_frame,
    text="Multi-Modal Deepfake Detection using Deep Learning",
    font=("Arial", 12),
    fg="lightgray",
    bg="#1E1E2E"
)

subtitle.pack()

user_label = tk.Label(
    scrollable_frame,
    text="Logged in as: admin",
    font=("Arial", 11),
    fg="lightgreen",
    bg="#1E1E2E"
)

user_label.pack()

status = tk.Label(
    scrollable_frame,
    text="● AI STATUS : READY",
    font=("Arial", 11, "bold"),
    fg="lime",
    bg="#1E1E2E"
)

status.pack(pady=10)

# ----------------------------
# Analytics
# ----------------------------

stats = tk.Label(
    scrollable_frame,
    text=f"""
Total Detections : {total_detections()}

REAL : {total_real()}

FAKE : {total_fake()}
""",
    font=("Arial", 13, "bold"),
    fg="white",
    bg="#1E1E2E",
    justify="left"
)

stats.pack(pady=10)

reports_label = tk.Label(
    scrollable_frame,
    text="Loading reports...",
    font=("Arial", 11),
    fg="lightgreen",
    bg="#1E1E2E",
    justify="left"
)

reports_label.pack(pady=(0, 10))
refresh_reports_label()

# ----------------------------
# Button Style
# ----------------------------

btn_style = {
    "font": ("Arial", 13, "bold"),
    "width": 32,
    "height": 2,
    "bd": 0,
    "cursor": "hand2"
}

# ----------------------------
# Buttons
# ----------------------------

tk.Button(
    scrollable_frame,
    text="📷 Webcam Detection",
    bg="#4CAF50",
    fg="white",
    command=webcam,
    **btn_style
).pack(pady=6)

tk.Button(
    scrollable_frame,
    text="🖼 Image Detection",
    bg="#2196F3",
    fg="white",
    command=image,
    **btn_style
).pack(pady=6)

tk.Button(
    scrollable_frame,
    text="🎥 Video Detection",
    bg="#FF9800",
    fg="white",
    command=video,
    **btn_style
).pack(pady=6)

tk.Button(
    scrollable_frame,
    text="🎤 Audio Detection",
    bg="#9C27B0",
    fg="white",
    command=audio,
    **btn_style
).pack(pady=6)

tk.Button(
    scrollable_frame,
    text="📊 Detection History",
    bg="#009688",
    fg="white",
    command=history,
    **btn_style
).pack(pady=6)

tk.Button(
    scrollable_frame,
    text="📁 Open Screenshots Folder",
    bg="#607D8B",
    fg="white",
    command=open_screenshots,
    **btn_style
).pack(pady=6)

tk.Button(
    scrollable_frame,
    text="🗄 Open Detection Database",
    bg="#795548",
    fg="white",
    command=open_database,
    **btn_style
).pack(pady=6)

tk.Button(
    scrollable_frame,
    text="📄 Open CSV Log",
    bg="#673AB7",
    fg="white",
    command=open_csv_log,
    **btn_style
).pack(pady=6)

tk.Button(
    scrollable_frame,
    text="📕 Open PDF Reports",
    bg="#00BCD4",
    fg="white",
    command=open_reports_folder,
    **btn_style
).pack(pady=6)

tk.Button(
    scrollable_frame,
    text="ℹ About Project",
    bg="#3F51B5",
    fg="white",
    command=about,
    **btn_style
).pack(pady=6)

tk.Button(
    scrollable_frame,
    text="🔓 Logout",
    bg="#FF9800",
    fg="white",
    command=logout,
    **btn_style
).pack(pady=6)

tk.Button(
    scrollable_frame,
    text="❌ Exit",
    bg="#F44336",
    fg="white",
    command=exit_app,
    **btn_style
).pack(pady=15)

# ----------------------------
# Footer
# ----------------------------

footer = tk.Label(
    scrollable_frame,
    text="© 2026 AI Deepfake Detection System | Python • TensorFlow • OpenCV",
    font=("Arial", 10),
    fg="gray",
    bg="#1E1E2E"
)

footer.pack(side="bottom", pady=15)

root.mainloop()