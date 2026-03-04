import os
import sqlite3
from typing import Optional

import time  # Add this line

class ProgressDB:
    def __init__(self, path: str = "workouts.sqlite"):
        self.path = path

    def init(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with sqlite3.connect(self.path) as con:
            cur = con.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS workouts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    exercise TEXT NOT NULL,
                    reps INTEGER NOT NULL,
                    avg_accuracy REAL NOT NULL
                )
                """
            )
            con.commit()

    def insert_workout(self, date: str, exercise: str, reps: int, avg_accuracy: float):
        with sqlite3.connect(self.path) as con:
            cur = con.cursor()
            cur.execute(
                "INSERT INTO workouts (date, exercise, reps, avg_accuracy) VALUES (?, ?, ?, ?)",
                (date, exercise, reps, float(avg_accuracy)),
            )
            con.commit()