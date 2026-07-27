import os
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox

USERNAME = "admin"
PASSWORD = "admin123"

def login():

    if user_entry.get() == USERNAME and pass_entry.get() == PASSWORD:

        root.destroy()

        dashboard_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "dashboard.py"))
        launch_kwargs = {
            "cwd": os.path.dirname(dashboard_path),
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }

        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            launch_kwargs["startupinfo"] = startupinfo

        subprocess.Popen([sys.executable, dashboard_path], **launch_kwargs)

    else:
        messagebox.showerror("Login Failed", "Invalid Username or Password")


root = tk.Tk()

root.title("AI Deepfake Detection Login")

root.geometry("500x350")

root.configure(bg="#1E1E2E")

title = tk.Label(
    root,
    text="AI DEEPFAKE DETECTION",
    font=("Arial",20,"bold"),
    fg="white",
    bg="#1E1E2E"
)

title.pack(pady=30)

tk.Label(
    root,
    text="Username",
    bg="#1E1E2E",
    fg="white",
    font=("Arial",12)
).pack()

user_entry = tk.Entry(root, font=("Arial",12), width=30)
user_entry.pack(pady=5)

tk.Label(
    root,
    text="Password",
    bg="#1E1E2E",
    fg="white",
    font=("Arial",12)
).pack()

pass_entry = tk.Entry(root, show="*", font=("Arial",12), width=30)
pass_entry.pack(pady=5)

tk.Button(
    root,
    text="Login",
    command=login,
    bg="#4CAF50",
    fg="white",
    font=("Arial",13,"bold"),
    width=20
).pack(pady=25)

root.mainloop()