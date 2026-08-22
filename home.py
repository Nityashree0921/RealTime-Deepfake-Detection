import tkinter as tk
from tkinter import messagebox
import subprocess
import sys
import os


# =========================================================
# COLORS
# =========================================================

BG = "#080D1A"
CARD = "#111A2E"
CARD2 = "#16223A"
WHITE = "#FFFFFF"
MUTED = "#9AA8C7"
CYAN = "#00D9FF"
BLUE = "#287BFF"
GREEN = "#20D67B"
PURPLE = "#9B5CFF"
ORANGE = "#FF9D42"
RED = "#FF5577"


# =========================================================
# PROJECT PATH
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================================================
# OPEN DETECTION PROGRAM
# =========================================================

def open_program(filename):

    path = os.path.join(BASE_DIR, filename)

    if not os.path.exists(path):
        messagebox.showerror(
            "File Not Found",
            f"{filename} was not found."
        )
        return

    try:

        process = subprocess.Popen(
            [sys.executable, path],
            cwd=BASE_DIR
        )

        # Bring Home page temporarily behind the new program
        root.after(
            300,
            lambda: focus_detection_window(filename)
        )

    except Exception as e:

        messagebox.showerror(
            "Launch Error",
            str(e)
        )
# =========================================================
# BUTTON FUNCTIONS
# =========================================================

def webcam():
    open_program("app.py")


def image_detection():
    open_program("image_detector.py")


def video_detection():
    open_program("video_detector.py")


def audio_detection():
    open_program("audio_detector.py")


def history():
    open_program("history.py")


def dashboard():
    open_program("dashboard.py")


def logout():

    root.destroy()

    login_path = os.path.join(BASE_DIR, "login.py")

    subprocess.Popen(
        [sys.executable, login_path],
        cwd=BASE_DIR
    )


def show_about():

    messagebox.showinfo(
        "About AI Deepfake Detection",
        "AI Deepfake Detection System\n\n"
        "A multi-modal deep learning application "
        "for detecting manipulated images, videos, "
        "audio and live webcam content.\n\n"
        "Technologies:\n"
        "Python • OpenCV • TensorFlow • Tkinter • SQLite"
    )


# =========================================================
# MAIN WINDOW
# =========================================================

root = tk.Tk()

root.title("AI Deepfake Detection System")

root.geometry("1200x800")

root.configure(bg=BG)

root.minsize(900, 650)


# =========================================================
# HEADER
# =========================================================

header = tk.Frame(
    root,
    bg=BG,
    height=70
)

header.pack(
    side="top",
    fill="x"
)

header.pack_propagate(False)


logo = tk.Label(
    header,
    text="◈",
    font=("Arial", 30, "bold"),
    fg=CYAN,
    bg=BG
)

logo.pack(
    side="left",
    padx=(30, 5)
)


brand = tk.Label(
    header,
    text="DEEPFAKE AI",
    font=("Arial", 17, "bold"),
    fg=WHITE,
    bg=BG
)

brand.pack(
    side="left"
)


# Navigation buttons

nav_frame = tk.Frame(
    header,
    bg=BG
)

nav_frame.pack(
    side="right",
    padx=25
)


def nav_button(text, command):

    return tk.Button(
        nav_frame,
        text=text,
        command=command,
        bg=BG,
        fg=MUTED,
        activebackground=BG,
        activeforeground=CYAN,
        font=("Arial", 10, "bold"),
        relief="flat",
        bd=0,
        cursor="hand2"
    )


nav_button("HOME", lambda: scroll_to(0)).pack(
    side="left",
    padx=10
)

nav_button("DETECT", lambda: scroll_to(1)).pack(
    side="left",
    padx=10
)

nav_button("HISTORY", history).pack(
    side="left",
    padx=10
)

nav_button("DASHBOARD", dashboard).pack(
    side="left",
    padx=10
)

nav_button("LOGOUT", logout).pack(
    side="left",
    padx=10
)


# =========================================================
# SCROLLABLE AREA
# =========================================================

container = tk.Frame(
    root,
    bg=BG
)

container.pack(
    fill="both",
    expand=True
)


canvas = tk.Canvas(
    container,
    bg=BG,
    highlightthickness=0
)

scrollbar = tk.Scrollbar(
    container,
    orient="vertical",
    command=canvas.yview
)

canvas.configure(
    yscrollcommand=scrollbar.set
)

scrollbar.pack(
    side="right",
    fill="y"
)

canvas.pack(
    side="left",
    fill="both",
    expand=True
)


content = tk.Frame(
    canvas,
    bg=BG
)


window_id = canvas.create_window(
    (0, 0),
    window=content,
    anchor="nw"
)


def update_scroll_region(event=None):

    canvas.configure(
        scrollregion=canvas.bbox("all")
    )


def resize_content(event):

    canvas.itemconfig(
        window_id,
        width=event.width
    )


content.bind(
    "<Configure>",
    update_scroll_region
)

canvas.bind(
    "<Configure>",
    resize_content
)


# =========================================================
# SMOOTH MOUSE SCROLL
# =========================================================

def mouse_scroll(event):

    if event.delta:

        canvas.yview_scroll(
            int(-event.delta / 120),
            "units"
        )


canvas.bind_all(
    "<MouseWheel>",
    mouse_scroll
)


# =========================================================
# SECTION POSITIONS
# =========================================================

sections = []


def register_section(frame):

    sections.append(frame)


def scroll_to(index):

    if index >= len(sections):
        return

    root.update_idletasks()

    frame = sections[index]

    y = frame.winfo_rooty() - content.winfo_rooty()

    total_height = content.winfo_height()

    if total_height <= 0:
        return

    position = y / total_height

    position = max(0, min(position, 1))

    canvas.yview_moveto(position)

# =========================================================
# HERO SECTION
# =========================================================

hero = tk.Frame(
    content,
    bg=BG,
    height=560
)

hero.pack(
    fill="x"
)

hero.pack_propagate(False)

register_section(hero)


# Left side

hero_left = tk.Frame(
    hero,
    bg=BG
)

hero_left.place(
    relx=0.08,
    rely=0.10,
    relwidth=0.55,
    relheight=0.85
)


small_title = tk.Label(
    hero_left,
    text="◉  NEXT GENERATION AI SECURITY",
    font=("Arial", 11, "bold"),
    fg=CYAN,
    bg=BG
)

small_title.pack(
    anchor="w",
    pady=(25, 15)
)


hero_title = tk.Label(
    hero_left,
    text="DETECT WHAT'S REAL.\nEXPOSE WHAT'S FAKE.",
    font=("Arial", 35, "bold"),
    fg=WHITE,
    bg=BG,
    justify="left"
)

hero_title.pack(
    anchor="w"
)


hero_description = tk.Label(
    hero_left,
    text=(
        "An intelligent multi-modal deepfake detection system "
        "designed to analyze images, videos, audio and live "
        "webcam streams using deep learning."
    ),
    font=("Arial", 13),
    fg=MUTED,
    bg=BG,
    justify="left",
    wraplength=600
)

hero_description.pack(
    anchor="w",
    pady=25
)


start_button = tk.Button(
    hero_left,
    text="▶  START DETECTION",
    command=lambda: scroll_to(1),
    bg=CYAN,
    fg="#001018",
    activebackground="#55E7FF",
    activeforeground="#001018",
    font=("Arial", 13, "bold"),
    relief="flat",
    bd=0,
    cursor="hand2",
    padx=30,
    pady=12
)

start_button.pack(
    anchor="w"
)


# Right visual

visual = tk.Frame(
    hero,
    bg=CARD
)

visual.place(
    relx=0.67,
    rely=0.16,
    relwidth=0.25,
    relheight=0.65
)


visual_icon = tk.Label(
    visual,
    text="◈",
    font=("Arial", 80, "bold"),
    fg=CYAN,
    bg=CARD
)

visual_icon.pack(
    pady=(45, 5)
)


visual_title = tk.Label(
    visual,
    text="AI CORE",
    font=("Arial", 18, "bold"),
    fg=WHITE,
    bg=CARD
)

visual_title.pack()


visual_status = tk.Label(
    visual,
    text="● SYSTEM READY",
    font=("Arial", 10, "bold"),
    fg=GREEN,
    bg=CARD
)

visual_status.pack(
    pady=15
)


visual_text = tk.Label(
    visual,
    text="MULTI-MODAL\nDETECTION ENGINE",
    font=("Arial", 10, "bold"),
    fg=MUTED,
    bg=CARD,
    justify="center"
)

visual_text.pack()


# =========================================================
# STAT BAR
# =========================================================

stats = tk.Frame(
    content,
    bg=CARD,
    height=110
)

stats.pack(
    fill="x",
    padx=50,
    pady=(0, 35)
)

stats.pack_propagate(False)


def stat(parent, number, label, color):

    frame = tk.Frame(
        parent,
        bg=CARD
    )

    frame.pack(
        side="left",
        expand=True,
        fill="both"
    )

    tk.Label(
        frame,
        text=number,
        font=("Arial", 25, "bold"),
        fg=color,
        bg=CARD
    ).pack(
        pady=(18, 0)
    )

    tk.Label(
        frame,
        text=label,
        font=("Arial", 10, "bold"),
        fg=MUTED,
        bg=CARD
    ).pack()


stat(stats, "04", "DETECTION MODES", CYAN)
stat(stats, "AI", "DEEP LEARNING", PURPLE)
stat(stats, "24/7", "READY", GREEN)
stat(stats, "100%", "LOCAL PROCESSING", ORANGE)


# =========================================================
# DETECTION SECTION
# =========================================================

detect_section = tk.Frame(
    content,
    bg=BG
)

detect_section.pack(
    fill="x",
    padx=50,
    pady=30
)

register_section(detect_section)


section_title = tk.Label(
    detect_section,
    text="CHOOSE YOUR DETECTION MODE",
    font=("Arial", 24, "bold"),
    fg=WHITE,
    bg=BG
)

section_title.pack()


section_subtitle = tk.Label(
    detect_section,
    text="Select a modality to analyze your media",
    font=("Arial", 11),
    fg=MUTED,
    bg=BG
)

section_subtitle.pack(
    pady=(5, 25)
)


# Cards row

cards = tk.Frame(
    detect_section,
    bg=BG
)

cards.pack(
    fill="x"
)


def detection_card(
    parent,
    icon,
    title,
    description,
    color,
    command
):

    card = tk.Frame(
        parent,
        bg=CARD,
        height=220
    )

    card.pack(
        side="left",
        expand=True,
        fill="both",
        padx=8
    )

    card.pack_propagate(False)

    tk.Label(
        card,
        text=icon,
        font=("Arial", 35),
        fg=color,
        bg=CARD
    ).pack(
        pady=(20, 5)
    )

    tk.Label(
        card,
        text=title,
        font=("Arial", 14, "bold"),
        fg=WHITE,
        bg=CARD
    ).pack()

    tk.Label(
        card,
        text=description,
        font=("Arial", 9),
        fg=MUTED,
        bg=CARD,
        justify="center",
        wraplength=180
    ).pack(
        pady=10
    )

    button = tk.Button(
        card,
        text="OPEN",
        command=command,
        bg=color,
        fg="#FFFFFF",
        activebackground=color,
        font=("Arial", 9, "bold"),
        relief="flat",
        bd=0,
        cursor="hand2",
        width=12
    )

    button.pack(
        pady=5
    )


detection_card(
    cards,
    "◉",
    "WEBCAM",
    "Real-time face detection through your camera.",
    GREEN,
    webcam
)

detection_card(
    cards,
    "▣",
    "IMAGE",
    "Analyze images and detect manipulated faces.",
    BLUE,
    image_detection
)

detection_card(
    cards,
    "▶",
    "VIDEO",
    "Analyze uploaded videos frame by frame.",
    ORANGE,
    video_detection
)

detection_card(
    cards,
    "♫",
    "AUDIO",
    "Analyze speech and detect spoofed audio.",
    PURPLE,
    audio_detection
)


# =========================================================
# FEATURES SECTION
# =========================================================

features = tk.Frame(
    content,
    bg=CARD2
)

features.pack(
    fill="x",
    padx=50,
    pady=40
)


tk.Label(
    features,
    text="WHY USE OUR SYSTEM?",
    font=("Arial", 23, "bold"),
    fg=WHITE,
    bg=CARD2
).pack(
    pady=(25, 20)
)


feature_text = (
    "✓  Multi-modal deepfake analysis\n\n"
    "✓  Real-time webcam detection\n\n"
    "✓  Automated detection history\n\n"
    "✓  Screenshot capture for fake results\n\n"
    "✓  Email alerts for detected deepfakes\n\n"
    "✓  PDF detection reports\n\n"
    "✓  SQLite database logging"
)


tk.Label(
    features,
    text=feature_text,
    font=("Arial", 12),
    fg=MUTED,
    bg=CARD2,
    justify="left"
).pack(
    pady=(0, 30)
)


# =========================================================
# FOOTER
# =========================================================

footer = tk.Frame(
    content,
    bg="#050914",
    height=150
)

footer.pack(
    fill="x",
    padx=50,
    pady=(30, 0)
)

footer.pack_propagate(False)


tk.Label(
    footer,
    text="◈  AI DEEPFAKE DETECTION SYSTEM",
    font=("Arial", 15, "bold"),
    fg=CYAN,
    bg="#050914"
).pack(
    pady=(25, 5)
)


tk.Label(
    footer,
    text="Multi-Modal Deepfake Detection using Deep Learning",
    font=("Arial", 10),
    fg=MUTED,
    bg="#050914"
).pack()


tk.Label(
    footer,
    text="Python  •  TensorFlow  •  OpenCV  •  SQLite  •  Tkinter",
    font=("Arial", 9),
    fg=MUTED,
    bg="#050914"
).pack(
    pady=5
)


tk.Label(
    footer,
    text="© 2026 AI Deepfake Detection System  |  All Rights Reserved",
    font=("Arial", 9),
    fg="#66708F",
    bg="#050914"
).pack(
    pady=5
)

# =========================================================
# KEYBOARD SCROLL
# =========================================================

def keyboard_scroll(event):

    if event.keysym == "Down":
        canvas.yview_scroll(3, "units")

    elif event.keysym == "Up":
        canvas.yview_scroll(-3, "units")


root.bind(
    "<Down>",
    keyboard_scroll
)

root.bind(
    "<Up>",
    keyboard_scroll
)
# =========================================================
# HOME / END KEY NAVIGATION
# =========================================================

def go_top(event=None):

    canvas.yview_moveto(0)


def go_bottom(event=None):

    canvas.yview_moveto(1)


root.bind(
    "<Home>",
    go_top
)

root.bind(
    "<End>",
    go_bottom
)

# =========================================================
# START
# =========================================================

root.mainloop()