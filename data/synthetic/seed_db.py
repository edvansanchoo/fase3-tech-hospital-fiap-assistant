"""Seed SQLite with 5 demo patients PAC-001..PAC-005."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "hospital.db"

PATIENTS = [
    ("PAC-001", "Adulto 45a", "febre, tosse 3 dias", "estável"),
    ("PAC-002", "Adulto 62a", "dor abdominal", "estável"),
    ("PAC-003", "Adulto 52a", "follow-up oncologia mama", "estável"),
    ("PAC-004", "Adulto 70a", "polifarmácia", "estável"),
    ("PAC-005", "Adulto 30a", "consulta geral", "estável"),
]

EXAMS = [
    ("PAC-001", "hemograma", "concluido", "2026-01-10"),
    ("PAC-002", "tomografia_abdominal", "pendente", "2026-01-20"),
    ("PAC-003", "mamografia", "pendente", "2026-01-18"),
    ("PAC-004", "função_renal", "concluido", "2026-01-12"),
    ("PAC-005", "consulta_geral", "concluido", "2026-01-15"),
]

PRESCRIPTIONS = [
    ("PAC-004", "warfarina 5mg", "ativo"),
    ("PAC-004", "amoxicilina 500mg", "ativo"),
]

def seed() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS pacientes (
        patient_id TEXT PRIMARY KEY, perfil TEXT, queixa TEXT, status TEXT
    );
    CREATE TABLE IF NOT EXISTS exames (
        patient_id TEXT, tipo TEXT, status TEXT, data TEXT
    );
    CREATE TABLE IF NOT EXISTS prescricoes (
        patient_id TEXT, medicamento TEXT, status TEXT
    );
  """)
    cur.executemany("INSERT OR REPLACE INTO pacientes VALUES (?,?,?,?)", PATIENTS)
    cur.executemany("INSERT INTO exames VALUES (?,?,?,?)", EXAMS)
    cur.executemany("INSERT INTO prescricoes VALUES (?,?,?)", PRESCRIPTIONS)
    conn.commit()
    conn.close()

if __name__ == "__main__":
    seed()
    print(f"Seeded {DB_PATH}")
