#!/usr/bin/env python3
"""
convert_csv.py

Reads raw course‑evaluation CSV exports placed in the ``input/`` directory
and produces three cleaned CSVs in ``output/``:

* **Main.csv**  – one row per course offering with the two key averages (Q 21 & Q 23)
* **Quant.csv** – one row per quantitative question per course offering
* **Qual.csv**  – one row per qualitative comment per course offering, labelled with sentiment

---
Run from the project root:

```bash
python convert_csv.py
```

Dependencies
------------
* `pandas`
* `nltk` (the first run will auto‑download VADER)

Install them with:

```bash
pip install pandas nltk
```
"""

import os
import re
import csv
import sys
from pathlib import Path
from typing import List, Dict

import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

INPUT_DIR  = Path(__file__).parent / "input"
OUTPUT_DIR = Path(__file__).parent / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

# Mapping from full question text → institutional “Question #”.
# Update here if your survey instrument changes.
QUESTION_MAP = {
    1:  "Course organized to help learning",
    2:  "Course developed abilities/skills for subject",
    3:  "Course developed ability to think critically",
    4:  "Material organized around learning outcomes",
    5:  "Course improved problem-solving skills",
    7:  "Satisfaction with effort in course",
    14: "Instructor presented organized content",
    15: "Instructor increased understanding of material",
    16: "Instructor helpful to student individually",
    17: "Instructor provided meaningful feedback",
    18: "Instructor provided timely feedback",
    19: "Instructor encouraged participation",
    20: "Instructor conduct professional",
    21: "Overall effectiveness of instructor's teaching technique",
    22: "Overall demonstration of the significance of subject matter",
    23: "Instructor created an environment conducive to learning",
}

# Reverse lookup for fast mapping
TEXT_TO_QNUM = {v.lower(): k for k, v in QUESTION_MAP.items()}

# Sentiment buckets using VADER compound score
def compound_to_bucket(score: float) -> str:
    if score >= 0.75:
        return "Highly Positive"
    if score >= 0.50:
        return "Positive"
    if score >= 0.10:
        return "Slightly Positive"
    if score > -0.10:
        return "Neutral"
    if score > -0.50:
        return "Slightly Negative"
    if score > -0.75:
        return "Negative"
    return "Highly Negative"

# Ensure VADER data present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

meta_re = re.compile(r"^(Spring|Summer|Fall|Winter)\s+(\d{4}),\s+(.+?)\s+Section", re.I)
possible_resp_re = re.compile(r"There were:\s+(\d+)\s+possible respondents", re.I)

def parse_metadata(lines: List[str]) -> Dict[str, str]:
    """Extract term, year, course title, possible respondent count."""
    term, year, course_title, possible_resp = None, None, None, None
    for l in lines[:4]:
        m_meta = meta_re.search(l)
        if m_meta:
            term, year, course_title = m_meta.group(1).title(), int(m_meta.group(2)), m_meta.group(3).strip()
        m_resp = possible_resp_re.search(l)
        if m_resp:
            possible_resp = int(m_resp.group(1))
    if None in (term, year, course_title, possible_resp):
        raise ValueError("Could not parse metadata in file header.")
    return {
        "Course Title": course_title,
        "Term": term,
        "Year": year,
        "# of Possible Responses": possible_resp,
    }

def clean_cell(cell: str) -> str:
    return (cell or "").strip().replace('\u00a0', ' ')

# --------------------------------------------------------------------------- #
# Core parsing functions
# --------------------------------------------------------------------------- #

def parse_quant_section(rows: List[List[str]], meta: Dict[str, str]) -> List[Dict]:
    quant_rows = []
    for row in rows:
        if not row or row[0].startswith('Text Responses'):
            break
        # Skip formatting rows that have empty Question Text
        if len(row) < 5:
            continue
        q_text = clean_cell(row[1])
        if not q_text:
            continue
        N   = int(float(clean_cell(row[2]))) if clean_cell(row[2]).isdigit() else None
        avg = float(clean_cell(row[3])) if clean_cell(row[3]) else None
        sd  = float(clean_cell(row[4])) if clean_cell(row[4]) else None
        q_num = TEXT_TO_QNUM.get(q_text.lower())
        quant_rows.append({
            **meta,
            "Question #": q_num,
            "Question Text": q_text,
            "N": N,
            "Avg": avg,
            "SD": sd,
        })
    return quant_rows

def parse_qual_section(rows: List[List[str]], meta: Dict[str, str]) -> List[Dict]:
    qual_rows = []
    current_question = None
    for row in rows:
        if not row:
            continue
        cell = clean_cell(row[0])
        if not cell:
            continue
        if cell.lower().startswith('question:'):
            current_question = re.sub(r'^Question:\s*', '', cell, flags=re.I).strip(': ').strip()
            continue
        if current_question is None:
            # Haven't encountered a question yet
            continue
        if cell.lower() in ('n/a', 'na'):
            continue  # ignore placeholder
        sentiment = compound_to_bucket(sia.polarity_scores(cell)['compound'])
        qual_rows.append({
            **{k: meta[k] for k in ('Course Title', 'Term', 'Year')},
            "Question Text": current_question,
            "Answer Text": cell,
            "Sentiment": sentiment,
        })
    return qual_rows

def parse_main_record(quant_rows: List[Dict], meta: Dict[str, str]) -> Dict:
    # Look for the two key averages
    avg_21 = next((r['Avg'] for r in quant_rows if r.get('Question #') == 21), None)
    avg_23 = next((r['Avg'] for r in quant_rows if r.get('Question #') == 23), None)
    return {
        **meta,
        "Avg (21)": avg_21,
        "Avg (23)": avg_23,
    }

# --------------------------------------------------------------------------- #
# Program entry point
# --------------------------------------------------------------------------- #

def process_file(path: Path):
    with path.open('r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    raw_lines = [','.join(r) for r in rows]  # for meta parsing, join back
    meta = parse_metadata(raw_lines)

    # Rows after header (row index 4 onward)
    quant_section = rows[5:]  # skip first five rows incl. quant header
    # find index where Text Responses starts
    text_resp_idx = next((i for i, r in enumerate(quant_section) if r and r[0].startswith('Text Responses')), len(quant_section))
    quant_rows = parse_quant_section(quant_section[:text_resp_idx], meta)
    qual_rows  = parse_qual_section(quant_section[text_resp_idx:], meta)
    main_row   = parse_main_record(quant_rows, meta)

    return main_row, quant_rows, qual_rows

def main():
    main_records: List[Dict]  = []
    quant_records: List[Dict] = []
    qual_records: List[Dict]  = []

    for csv_file in INPUT_DIR.glob('*.csv'):
        try:
            m, q_rows, ql_rows = process_file(csv_file)
        except Exception as exc:
            print(f"Error processing {csv_file.name}: {exc}", file=sys.stderr)
            continue
        main_records.append(m)
        quant_records.extend(q_rows)
        qual_records.extend(ql_rows)

    # Create DataFrames with explicit column order
    main_df  = pd.DataFrame(main_records, columns=["Course Title", "Term", "Year", "# of Possible Responses", "Avg (21)", "Avg (23)"])
    quant_df = pd.DataFrame(quant_records, columns=["Course Title", "Term", "Year", "# of Possible Responses", "Question #", "Question Text", "N", "Avg", "SD"])
    qual_df  = pd.DataFrame(qual_records, columns=["Course Title", "Term", "Year", "Question Text", "Answer Text", "Sentiment"])

    main_df.to_csv(OUTPUT_DIR / 'Main.csv',  index=False)
    quant_df.to_csv(OUTPUT_DIR / 'Quant.csv', index=False)
    qual_df.to_csv(OUTPUT_DIR / 'Qual.csv',  index=False)

    print(f"Wrote Main.csv ({len(main_df)}) Quant.csv ({len(quant_df)}) Qual.csv ({len(qual_df)}) to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
