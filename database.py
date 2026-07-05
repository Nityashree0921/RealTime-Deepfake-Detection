import sqlite3

conn = sqlite3.connect("detections.db", check_same_thread=False)

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