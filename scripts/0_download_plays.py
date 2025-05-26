# src/preprocessing/download_plays.py
import os
import requests
from bs4 import BeautifulSoup

PLAYS = {
    # Tragedies
    "hamlet": "https://www.gutenberg.org/cache/epub/1524/pg1524.txt",
    "macbeth": "https://www.gutenberg.org/cache/epub/1533/pg1533.txt",
    "romeo_and_juliet": "https://www.gutenberg.org/cache/epub/1513/pg1513.txt",
    "othello": "https://www.gutenberg.org/cache/epub/1531/pg1531.txt",
    "king_lear": "https://www.gutenberg.org/cache/epub/1532/pg1532.txt",
    "julius_caesar": "https://www.gutenberg.org/cache/epub/1522/pg1522.txt",
    "antony_and_cleopatra": "https://www.gutenberg.org/cache/epub/1534/pg1534.txt",
    "titus_andronicus": "https://www.gutenberg.org/cache/epub/1507/pg1507.txt",
    "coriolanus": "https://www.gutenberg.org/cache/epub/1535/pg1535.txt",
    "timon_of_athens": "https://www.gutenberg.org/cache/epub/1536/pg1536.txt",
    
    # Comedies
    "midsummer_nights_dream": "https://www.gutenberg.org/cache/epub/1514/pg1514.txt",
    "much_ado_about_nothing": "https://www.gutenberg.org/cache/epub/1519/pg1519.txt",
    "twelfth_night": "https://www.gutenberg.org/cache/epub/1526/pg1526.txt",
    "merchant_of_venice": "https://www.gutenberg.org/cache/epub/1515/pg1515.txt",
    "as_you_like_it": "https://www.gutenberg.org/cache/epub/1523/pg1523.txt",
    "taming_of_the_shrew": "https://www.gutenberg.org/cache/epub/1508/pg1508.txt",
    "merry_wives_of_windsor": "https://www.gutenberg.org/cache/epub/1517/pg1517.txt",
    "comedy_of_errors": "https://www.gutenberg.org/cache/epub/1504/pg1504.txt",
    "loves_labours_lost": "https://www.gutenberg.org/cache/epub/1510/pg1510.txt",
    "measure_for_measure": "https://www.gutenberg.org/cache/epub/1530/pg1530.txt",
    "alls_well_that_ends_well": "https://www.gutenberg.org/cache/epub/1529/pg1529.txt",
    "the_two_gentlemen_of_verona": "https://www.gutenberg.org/cache/epub/1509/pg1509.txt",
    
    # Histories
    "richard_iii": "https://www.gutenberg.org/cache/epub/1503/pg1503.txt",
    "henry_v": "https://www.gutenberg.org/cache/epub/1521/pg1521.txt",
    "henry_iv_part1": "https://www.gutenberg.org/cache/epub/1516/pg1516.txt",
    "henry_iv_part2": "https://www.gutenberg.org/cache/epub/1518/pg1518.txt",
    "richard_ii": "https://www.gutenberg.org/cache/epub/1512/pg1512.txt",
    "king_john": "https://www.gutenberg.org/cache/epub/1511/pg1511.txt",
    "henry_vi_part1": "https://www.gutenberg.org/cache/epub/1500/pg1500.txt",
    "henry_vi_part2": "https://www.gutenberg.org/cache/epub/1501/pg1501.txt",
    "henry_vi_part3": "https://www.gutenberg.org/cache/epub/1502/pg1502.txt",
    "henry_viii": "https://www.gutenberg.org/cache/epub/1541/pg1541.txt",
    
    # Romances/Late Plays
    "the_tempest": "https://www.gutenberg.org/cache/epub/1540/pg1540.txt",
    "winters_tale": "https://www.gutenberg.org/cache/epub/1539/pg1539.txt",
    "cymbeline": "https://www.gutenberg.org/cache/epub/1538/pg1538.txt",
    "pericles": "https://www.gutenberg.org/cache/epub/1537/pg1537.txt",
    
    # Problem Plays
    "troilus_and_cressida": "https://www.gutenberg.org/cache/epub/1528/pg1528.txt"
}

def download_play(title, url, output_dir="data/raw/dialogue"):
    os.makedirs(output_dir, exist_ok=True)
    response = requests.get(url)
    
    if response.status_code == 200:
        filepath = os.path.join(output_dir, f"{title}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"Downloaded {title} to {filepath}")
    else:
        print(f"Failed to download {title}")

if __name__ == "__main__":
    for title, url in PLAYS.items():
        download_play(title, url)