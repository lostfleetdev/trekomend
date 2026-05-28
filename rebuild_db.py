"""
rebuild_db.py — Rebuild the SQLite database from the TMDB CSV.

Imports all columns including poster_path, backdrop_path, homepage, etc.
Creates indexes and FTS5 full-text search.

Usage:  uv run python rebuild_db.py
"""
import csv
import os
import sqlite3
import sys
import time

CSV_PATH = "dataset/TMDB_movie_dataset_v11.csv"
DB_PATH = "new-kaggle-output/tmdb_movies.db"
BACKUP_PATH = "new-kaggle-output/tmdb_movies.db.bak"

# Columns from CSV (in order)
CSV_COLUMNS = [
    "id", "title", "vote_average", "vote_count", "status", "release_date",
    "revenue", "runtime", "adult", "backdrop_path", "budget", "homepage",
    "imdb_id", "original_language", "original_title", "overview", "popularity",
    "poster_path", "tagline", "genres", "production_companies",
    "production_countries", "spoken_languages", "keywords",
]

# Derived columns (computed during import)
DERIVED_COLUMNS = ["primary_genre", "year"]

ALL_COLUMNS = CSV_COLUMNS + DERIVED_COLUMNS

# Column types for SQLite
COLUMN_TYPES = {
    "id": "INTEGER PRIMARY KEY",
    "title": "TEXT",
    "vote_average": "REAL",
    "vote_count": "REAL",
    "status": "TEXT",
    "release_date": "TEXT",
    "revenue": "REAL",
    "runtime": "REAL",
    "adult": "TEXT",
    "backdrop_path": "TEXT",
    "budget": "REAL",
    "homepage": "TEXT",
    "imdb_id": "TEXT",
    "original_language": "TEXT",
    "original_title": "TEXT",
    "overview": "TEXT",
    "popularity": "REAL",
    "poster_path": "TEXT",
    "tagline": "TEXT",
    "genres": "TEXT",
    "production_companies": "TEXT",
    "production_countries": "TEXT",
    "spoken_languages": "TEXT",
    "keywords": "TEXT",
    # Derived columns
    "primary_genre": "TEXT",
    "year": "INTEGER",
}

# Numeric columns to convert
NUMERIC_COLS = {
    "id", "vote_average", "vote_count", "revenue", "runtime", "budget", "popularity",
}


def parse_row(row: dict) -> dict:
    """Convert string values to proper types and compute derived columns."""
    parsed = {}
    for col in CSV_COLUMNS:
        val = row.get(col, "")
        if val == "" or val is None:
            parsed[col] = None
        elif col in NUMERIC_COLS:
            try:
                parsed[col] = float(val) if "." in str(val) else int(val)
            except (ValueError, TypeError):
                parsed[col] = None
        else:
            parsed[col] = val

    # Derived: primary_genre (first genre in the list)
    genres = parsed.get("genres") or ""
    parsed["primary_genre"] = genres.split(",")[0].strip() if genres.strip() else "Unknown"

    # Derived: year from release_date
    rd = parsed.get("release_date") or ""
    if len(rd) >= 4:
        try:
            parsed["year"] = int(rd[:4])
        except ValueError:
            parsed["year"] = None
    else:
        parsed["year"] = None

    return parsed


def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: CSV not found at {CSV_PATH}")
        return 1

    # Backup existing DB
    if os.path.exists(DB_PATH):
        print(f"Backing up existing DB to {BACKUP_PATH}")
        import shutil
        shutil.copy2(DB_PATH, BACKUP_PATH)

    # Remove existing DB
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Removed existing {DB_PATH}")

    print(f"Creating new database at {DB_PATH}")
    db = sqlite3.connect(DB_PATH)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")

    # Create table with all columns
    col_defs = ", ".join(f"{col} {COLUMN_TYPES[col]}" for col in ALL_COLUMNS)
    db.execute(f"CREATE TABLE movies ({col_defs})")
    print("Created movies table")

    # Import CSV
    print(f"Reading {CSV_PATH}...")
    start = time.time()
    seen_ids: dict[int, int] = {}  # id -> index in rows
    rows = []
    skipped = 0
    dupes = 0

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            parsed = parse_row(row)
            # Skip rows with no ID
            if parsed["id"] is None:
                skipped += 1
                continue

            mid = parsed["id"]
            if mid in seen_ids:
                dupes += 1
                # Keep the one with a poster_path, or higher popularity
                existing = rows[seen_ids[mid]]
                existing_has_poster = existing.get("poster_path") not in (None, "")
                new_has_poster = parsed.get("poster_path") not in (None, "")
                if new_has_poster and not existing_has_poster:
                    rows[seen_ids[mid]] = parsed
                elif new_has_poster == existing_has_poster:
                    if (parsed.get("popularity") or 0) > (existing.get("popularity") or 0):
                        rows[seen_ids[mid]] = parsed
            else:
                seen_ids[mid] = len(rows)
                rows.append(parsed)

            if (i + 1) % 200000 == 0:
                print(f"  Read {i + 1:,} rows...")

    elapsed = time.time() - start
    print(f"Read {i + 1:,} rows in {elapsed:.1f}s ({skipped} skipped, {dupes} dupes merged, {len(rows):,} unique)")

    # Insert rows
    print("Inserting rows...")
    start = time.time()
    placeholders = ", ".join(["?"] * len(ALL_COLUMNS))
    insert_sql = f"INSERT INTO movies ({', '.join(ALL_COLUMNS)}) VALUES ({placeholders})"

    batch_size = 50000
    for batch_start in range(0, len(rows), batch_size):
        batch = rows[batch_start:batch_start + batch_size]
        values = [tuple(row[col] for col in ALL_COLUMNS) for row in batch]
        db.executemany(insert_sql, values)
        db.commit()
        print(f"  Inserted {min(batch_start + batch_size, len(rows)):,} / {len(rows):,}")

    elapsed = time.time() - start
    print(f"Inserted {len(rows):,} rows in {elapsed:.1f}s")

    # Create indexes
    print("Creating indexes...")
    db.execute("CREATE INDEX idx_movies_year ON movies (year)")
    db.execute("CREATE INDEX idx_movies_primary_genre ON movies (primary_genre)")
    db.execute("CREATE INDEX idx_movies_vote_average ON movies (vote_average)")
    db.execute("CREATE INDEX idx_movies_popularity ON movies (popularity)")
    db.execute("CREATE INDEX idx_movies_title ON movies (title)")
    db.execute("CREATE INDEX idx_movies_imdb_id ON movies (imdb_id)")
    db.execute("CREATE INDEX idx_movies_status ON movies (status)")
    db.execute("CREATE INDEX idx_movies_release_date ON movies (release_date)")
    print("  Created 8 indexes")

    # Create FTS5 full-text search
    print("Creating FTS5 index...")
    db.execute("""
        CREATE VIRTUAL TABLE movies_fts USING fts5(
            title,
            original_title,
            overview,
            tagline,
            keywords,
            genres,
            content='movies',
            content_rowid='id'
        )
    """)
    db.execute("""
        INSERT INTO movies_fts (rowid, title, original_title, overview, tagline, keywords, genres)
        SELECT id, title, original_title, overview, tagline, keywords, genres FROM movies
    """)
    print("  FTS5 index created")

    # Analyze for query optimizer
    db.execute("ANALYZE")
    db.commit()

    # Verify
    count = db.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
    sample = db.execute("SELECT id, title, poster_path, backdrop_path FROM movies WHERE id = 550").fetchone()
    print(f"\nVerification:")
    print(f"  Total movies: {count:,}")
    print(f"  Fight Club: id={sample[0]}, title={sample[1]}")
    print(f"    poster_path: {sample[2]}")
    print(f"    backdrop_path: {sample[3]}")

    # Check poster_path availability
    poster_count = db.execute("SELECT COUNT(*) FROM movies WHERE poster_path IS NOT NULL AND poster_path != ''").fetchone()[0]
    print(f"  Movies with poster_path: {poster_count:,}")

    db.close()
    print(f"\nDone! Database saved to {DB_PATH}")
    print(f"Backup at {BACKUP_PATH} (if existed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
