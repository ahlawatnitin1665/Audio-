"""
ICU Machine Sound Effects Downloader v3
Downloads hospital/ICU machine sounds from free sources (no signup, no attribution).

Usage:
    python download_icu_sounds.py

Sources:
  - SoundDino.com: Hospital ambience, machine sounds, heart/cardiogram/ECG sounds
  - BigSoundBank.com: CC0 medical beeps, ECG, cardiac arrest, doppler sounds
  - Curated direct URLs for specific machine beeps and alarms

Output: ./archive/icu_machine_sounds/
"""

import requests
import re
import os
import time
from urllib.parse import urljoin

# ─── Configuration ───────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "archive", "icu_machine_sounds")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

# ─── SoundDino Pages (scrape data-track attributes from HTML) ────────────────
SOUNDDINO_PAGES = [
    # (page_url, output_subdir, description)
    ("https://sounddino.com/en/effects/hospital/", "sounddino_hospital", "Hospital ambience & machine sounds"),
    ("https://sounddino.com/en/effects/heart-and-cardiogram/", "sounddino_heart", "Heartbeat, cardiogram & ECG sounds"),
]

# (Legacy curated tracks removed — all now covered by full SoundDino page scrape above)

# ─── BigSoundBank (CC0, direct WAV+MP3 downloads) ───────────────────────────
# Each entry: (page_url, output_subdir, description)
BIGSOUNDBANK_SOUNDS = [
    # Cardiac/ECG beeps
    ("https://bigsoundbank.com/electrocardiogram-60-Hz-s0457.html", "bigsoundbank", "ECG 60Hz beep"),
    ("https://bigsoundbank.com/cardiac-arrest-electrocardiogram-1-s0365.html", "bigsoundbank", "Cardiac arrest ECG #1"),
    ("https://bigsoundbank.com/cardiac-arrest-electrocardiogram-2-s0366.html", "bigsoundbank", "Cardiac arrest ECG #2"),
    ("https://bigsoundbank.com/67-hz-electrocardiogram-s0458.html", "bigsoundbank", "ECG 67Hz beep"),
    # Medical devices
    ("https://bigsoundbank.com/blood-pressure-monitor-s0994.html", "bigsoundbank", "Blood pressure monitor"),
    ("https://bigsoundbank.com/portable-vascular-doppler-2-s0493.html", "bigsoundbank", "Vascular doppler #2"),
    ("https://bigsoundbank.com/portable-vascular-doppler-1-s0492.html", "bigsoundbank", "Vascular doppler #1"),
    ("https://bigsoundbank.com/portable-vascular-doppler-4-s0495.html", "bigsoundbank", "Vascular doppler #4"),
    ("https://bigsoundbank.com/portable-vascular-doppler-3-s0494.html", "bigsoundbank", "Vascular doppler #3"),
    # Hospital ambience
    ("https://bigsoundbank.com/motherhood-corridor-s0582.html", "bigsoundbank", "Maternity corridor"),
    ("https://bigsoundbank.com/digital-pager-s0059.html", "bigsoundbank", "Digital pager"),
    ("https://bigsoundbank.com/alphanumeric-pager-s0218.html", "bigsoundbank", "Alphanumeric pager"),
]


def download_file(url, save_path, max_retries=3):
    """Download a file with retries."""
    for attempt in range(max_retries):
        try:
            r = SESSION.get(url, timeout=30)
            r.raise_for_status()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(r.content)
            return True
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"    FAILED: {e}")
                return False
            time.sleep(1)
    return False


def scrape_sounddino_page(page_url, output_subdir):
    """Download all tracks from a SoundDino page by scraping data-track attributes."""
    print(f"    Fetching page...", end=" ", flush=True)
    try:
        r = SESSION.get(page_url, timeout=15)
        tracks = re.findall(r'data-track="([^"]*\.mp3)"', r.text)
        names = re.findall(r'<div class="name">([^<]+)', r.text)

        print(f"{len(tracks)} tracks found")
        count = 0
        for track_path, name in zip(tracks, names):
            filename = os.path.basename(track_path)
            save_path = os.path.join(OUTPUT_DIR, output_subdir, filename)

            if os.path.exists(save_path):
                count += 1
                continue

            dl_url = urljoin("https://sounddino.com", track_path)
            print(f"    Downloading: {name[:50]}...", end=" ", flush=True)
            if download_file(dl_url, save_path):
                print("OK")
                count += 1
            time.sleep(0.3)

        return count
    except Exception as e:
        print(f"    ERROR: {e}")
        return 0


def download_bigsoundbank_sounds():
    """Download CC0 sounds from BigSoundBank."""
    print(f"\n  [BigSoundBank CC0 medical sounds]")
    count = 0
    for page_url, subdir, desc in BIGSOUNDBANK_SOUNDS:
        # Fetch page to find audio download URL
        try:
            r = SESSION.get(page_url, timeout=15)
            # Find MP3 (prefer MP3 over WAV for size)
            audio_match = re.search(r'href="([^"]*\.mp3)"', r.text)
            if not audio_match:
                audio_match = re.search(r'href="([^"]*\.wav)"', r.text)
            if not audio_match:
                print(f"    SKIP (no download link): {desc}")
                continue

            dl_url = audio_match.group(1)
            if dl_url.startswith("/"):
                dl_url = "https://bigsoundbank.com" + dl_url

            filename = dl_url.split("/")[-1].split("?")[0]
            save_path = os.path.join(OUTPUT_DIR, subdir, filename)

            if os.path.exists(save_path):
                count += 1
                continue

            print(f"    {desc}: {filename}...", end=" ", flush=True)
            if download_file(dl_url, save_path):
                print("OK")
                count += 1
        except Exception as e:
            print(f"    FAIL: {desc} - {e}")
        time.sleep(0.3)
    return count


def show_results():
    """Show what was downloaded."""
    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}")

    for root, dirs, files in os.walk(OUTPUT_DIR):
        if not files:
            continue
        level = root.replace(OUTPUT_DIR, "").count(os.sep)
        indent = "  " * level
        total_size_kb = sum(os.path.getsize(os.path.join(root, f)) for f in files) / 1024
        print(f"\n{indent}{os.path.basename(root)}/ ({len(files)} files, {total_size_kb:.0f} KB)")
        for f in sorted(files)[:8]:
            size_kb = os.path.getsize(os.path.join(root, f)) / 1024
            print(f"{indent}  {f} ({size_kb:.0f} KB)")
        if len(files) > 8:
            print(f"{indent}  ... and {len(files)-8} more")


def main():
    print("=" * 60)
    print("ICU Machine Sound Effects Downloader v3")
    print("=" * 60)
    print(f"Output: {OUTPUT_DIR}")

    total = 0

    # 1. SoundDino pages (scrape data-track from HTML)
    for page_url, subdir, desc in SOUNDDINO_PAGES:
        print(f"\n--- SoundDino: {desc} ---")
        total += scrape_sounddino_page(page_url, subdir)

    # 2. BigSoundBank CC0 sounds
    print(f"\n--- BigSoundBank CC0 medical sounds ---")
    total += download_bigsoundbank_sounds()

    print(f"\n{'='*60}")
    print(f"Total files: {total} (new or updated)")

    show_results()


if __name__ == "__main__":
    main()
