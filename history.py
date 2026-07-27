import tkinter as tk
from tkinter import ttk, messagebox
import sqlite3

# -----------------------------
# Database
# -----------------------------

conn = sqlite3.connect("detections.db")
cursor = conn.cursor()

# -----------------------------
# Functions
# -----------------------------

def load_data():

    for row in tree.get_children():
        tree.delete(row)

    cursor.execute("""
        SELECT id,date,time,prediction,confidence
        FROM detections
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()

    for row in rows:
        tree.insert("", tk.END, values=row)


def delete_history():

    answer = messagebox.askyesno(
        "Delete",
        "Delete all detection history?"
    )

    if answer:

        cursor.execute("DELETE FROM detections")
        conn.commit()

        load_data()

        messagebox.showinfo(
            "Success",
            "History Deleted Successfully"
        )


def refresh():
    load_data()

# -----------------------------
# Window
# -----------------------------

root = tk.Tk()

root.title("Detection History")

root.geometry("900x550")

root.configure(bg="#1E1E2E")

# -----------------------------
# Heading
# -----------------------------

title = tk.Label(
    root,
    text="Detection History",
    font=("Arial",20,"bold"),
    fg="white",
    bg="#1E1E2E"
)

title.pack(pady=15)

# -----------------------------
# Table
# -----------------------------

columns = (
    "ID",
    "Date",
    "Time",
    "Prediction",
    "Confidence"
)

tree = ttk.Treeview(
    root,
    columns=columns,
    show="headings",
    height=18
)

for col in columns:
    tree.heading(col,text=col)
    tree.column(col,width=150,anchor="center")

tree.pack(fill="both",expand=True,padx=20)

scroll = ttk.Scrollbar(
    root,
    orient="vertical",
    command=tree.yview
)

tree.configure(yscroll=scroll.set)

scroll.pack(side="right",fill="y")

# -----------------------------
# Buttons
# -----------------------------

frame = tk.Frame(root,bg="#1E1E2E")
frame.pack(pady=15)

tk.Button(
    frame,
    text="Refresh",
    bg="#4CAF50",
    fg="white",
    width=15,
    command=refresh
).grid(row=0,column=0,padx=10)

tk.Button(
    frame,
    text="Delete History",
    bg="#F44336",
    fg="white",
    width=15,
    command=delete_history
).grid(row=0,column=1,padx=10)

tk.Button(
    frame,
    text="Close",
    bg="#607D8B",
    fg="white",
    width=15,
    command=root.destroy
).grid(row=0,column=2,padx=10)

# -----------------------------
# Load Data
# -----------------------------

load_data()

root.mainloop()

conn.close()