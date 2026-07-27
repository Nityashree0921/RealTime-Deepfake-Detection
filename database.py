import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
conn = sqlite3.connect(os.path.join(BASE_DIR, "detections.db"), check_same_thread=False)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS detections(

id INTEGER PRIMARY KEY AUTOINCREMENT,

date TEXT,

time TEXT,

prediction TEXT,

confidence REAL

)
""")

conn.commit()

def save_detection(date, time, prediction, confidence):

    cursor.execute(
        "INSERT INTO detections(date,time,prediction,confidence) VALUES(?,?,?,?)",
        (date, time, prediction, confidence)
    )

    conn.commit()


def get_all():

    cursor.execute("SELECT * FROM detections ORDER BY id DESC")

    return cursor.fetchall()