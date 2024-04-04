import sqlite3
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np  

DB_PATH = Path('manif_database.db')
IMAGES_PATH = Path('images')
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def create_database():
    if not DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
        CREATE TABLE IF NOT EXISTS manif_data (
            id INTEGER PRIMARY KEY,
            manif TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            unique_name TEXT
        )
        ''')
        conn.commit()
        conn.close()
        print("Database&Tables created.")
    else:
        print("Database existing!!")

def save_fig(fig, fig_id, tight_layout=True, fig_extension="png", resolution=300):
    if fig is None:
        print("No figure provided to save.")
        return
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    plt.tight_layout()
    fig.savefig(path, format=fig_extension, dpi=resolution)
    print(f"plot saved in: {path}")

def save_manif(manif, unique_name):
    manif_string = json.dumps(manif.tolist())
    conn = sqlite3.connect('manif_database.db')
    c = conn.cursor()
    c.execute('INSERT INTO manif_data (manif, unique_name) VALUES (?, ?)', (manif_string, unique_name))
    conn.commit()
    conn.close()
    print(f"Matrix {unique_name} saved in db")