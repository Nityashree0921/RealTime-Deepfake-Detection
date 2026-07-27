import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
conn = sqlite3.connect(os.path.join(BASE_DIR, "detections.db"))
cursor = conn.cursor()

def total_detections():
    cursor.execute("SELECT COUNT(*) FROM detections")
    return cursor.fetchone()[0]

def total_real():
    cursor.execute("SELECT COUNT(*) FROM detections WHERE prediction LIKE '%REAL%'")
    return cursor.fetchone()[0]

def total_fake():
    cursor.execute("SELECT COUNT(*) FROM detections WHERE prediction LIKE '%FAKE%'")
    return cursor.fetchone()[0]