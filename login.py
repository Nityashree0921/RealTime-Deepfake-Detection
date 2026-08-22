import os
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox


# =========================================================
# LOGIN DETAILS
# =========================================================

USERNAME = "admin"
PASSWORD = "admin123"


# =========================================================
# COLORS
# =========================================================

BG = "#0B1020"
CARD = "#151C32"
INPUT = "#202943"
TEXT = "#FFFFFF"
MUTED = "#9CA8C7"
ACCENT = "#00D4FF"
BUTTON = "#00A8CC"
BUTTON_HOVER = "#00C6EE"
ERROR = "#FF5C7A"


# =========================================================
# OPEN HOME PAGE
# =========================================================

def open_home():

    root.destroy()

    home_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "home.py"
        )
    )

    subprocess.Popen(
        [sys.executable, home_path],
        cwd=os.path.dirname(home_path)
    )


# =========================================================
# LOGIN FUNCTION
# =========================================================

def login():

    username = user_entry.get().strip()
    password = pass_entry.get()

    if username == USERNAME and password == PASSWORD:

        open_home()

    else:

        messagebox.showerror(
            "Login Failed",
            "Invalid username or password."
        )


# =========================================================
# SHOW / HIDE PASSWORD
# =========================================================

def toggle_password():

    if pass_entry.cget("show") == "*":

        pass_entry.config(show="")
        eye_button.config(text="Hide")

    else:

        pass_entry.config(show="*")
        eye_button.config(text="Show")


# =========================================================
# MAIN WINDOW
# =========================================================

root = tk.Tk()

root.title("AI Deepfake Detection System")

root.geometry("1000x650")

root.configure(bg=BG)

root.resizable(False, False)


# =========================================================
# LEFT SIDE - PROJECT INFORMATION
# =========================================================

left_frame = tk.Frame(
    root,
    bg=BG,
    width=500,
    height=650
)

left_frame.pack(
    side="left",
    fill="both"
)

left_frame.pack_propagate(False)


# AI Logo

logo = tk.Label(
    left_frame,
    text="◈",
    font=("Arial", 70, "bold"),
    fg=ACCENT,
    bg=BG
)

logo.pack(
    pady=(90, 10)
)


# Project title

title = tk.Label(
    left_frame,
    text="AI DEEPFAKE\nDETECTION",
    font=("Arial", 34, "bold"),
    fg=TEXT,
    bg=BG,
    justify="center"
)

title.pack()


# Subtitle

subtitle = tk.Label(
    left_frame,
    text="Multi-Modal Deepfake Detection\nusing Deep Learning",
    font=("Arial", 14),
    fg=MUTED,
    bg=BG,
    justify="center"
)

subtitle.pack(pady=20)


# Status

status = tk.Label(
    left_frame,
    text="●  AI SYSTEM READY",
    font=("Arial", 12, "bold"),
    fg="#36E27B",
    bg=BG
)

status.pack(pady=15)


# Security text

security = tk.Label(
    left_frame,
    text="Secure • Intelligent • Multi-Modal",
    font=("Arial", 11),
    fg=MUTED,
    bg=BG
)

security.pack(
    pady=20
)


# =========================================================
# RIGHT SIDE - LOGIN CARD
# =========================================================

right_frame = tk.Frame(
    root,
    bg=BG,
    width=500,
    height=650
)

right_frame.pack(
    side="right",
    fill="both"
)

right_frame.pack_propagate(False)


# Card

card = tk.Frame(
    right_frame,
    bg=CARD,
    width=390,
    height=500
)

card.place(
    relx=0.5,
    rely=0.5,
    anchor="center"
)

card.pack_propagate(False)


# Login heading

login_title = tk.Label(
    card,
    text="Welcome Back",
    font=("Arial", 25, "bold"),
    fg=TEXT,
    bg=CARD
)

login_title.pack(
    pady=(45, 5)
)


login_subtitle = tk.Label(
    card,
    text="Sign in to access the detection system",
    font=("Arial", 11),
    fg=MUTED,
    bg=CARD
)

login_subtitle.pack(
    pady=(0, 30)
)


# =========================================================
# USERNAME
# =========================================================

username_label = tk.Label(
    card,
    text="USERNAME",
    font=("Arial", 10, "bold"),
    fg=MUTED,
    bg=CARD
)

username_label.pack(
    anchor="w",
    padx=40
)


user_entry = tk.Entry(
    card,
    font=("Arial", 13),
    bg=INPUT,
    fg=TEXT,
    insertbackground=TEXT,
    relief="flat"
)

user_entry.pack(
    padx=40,
    pady=(8, 20),
    ipady=10,
    fill="x"
)


# =========================================================
# PASSWORD
# =========================================================

password_label = tk.Label(
    card,
    text="PASSWORD",
    font=("Arial", 10, "bold"),
    fg=MUTED,
    bg=CARD
)

password_label.pack(
    anchor="w",
    padx=40
)


password_frame = tk.Frame(
    card,
    bg=INPUT
)

password_frame.pack(
    padx=40,
    pady=(8, 25),
    fill="x"
)


pass_entry = tk.Entry(
    password_frame,
    font=("Arial", 13),
    bg=INPUT,
    fg=TEXT,
    insertbackground=TEXT,
    show="*",
    relief="flat"
)

pass_entry.pack(
    side="left",
    padx=10,
    ipady=9,
    fill="x",
    expand=True
)


eye_button = tk.Button(
    password_frame,
    text="Show",
    command=toggle_password,
    bg=INPUT,
    fg=ACCENT,
    activebackground=INPUT,
    activeforeground=TEXT,
    relief="flat",
    bd=0,
    cursor="hand2"
)

eye_button.pack(
    side="right",
    padx=10
)


# =========================================================
# LOGIN BUTTON
# =========================================================

login_button = tk.Button(
    card,
    text="LOGIN TO SYSTEM",
    command=login,
    bg=BUTTON,
    fg=TEXT,
    activebackground=BUTTON_HOVER,
    activeforeground=TEXT,
    font=("Arial", 12, "bold"),
    relief="flat",
    bd=0,
    cursor="hand2"
)

login_button.pack(
    padx=40,
    fill="x",
    ipady=12
)


# =========================================================
# FOOTER
# =========================================================

footer = tk.Label(
    card,
    text="Authorized access only",
    font=("Arial", 9),
    fg=MUTED,
    bg=CARD
)

footer.pack(
    pady=25
)


# =========================================================
# ENTER KEY
# =========================================================

root.bind(
    "<Return>",
    lambda event: login()
)


# Start cursor in username

user_entry.focus()


# =========================================================
# START APPLICATION
# =========================================================

root.mainloop()