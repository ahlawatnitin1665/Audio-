"""Shared patient selection for face/eye and audio monitoring."""

import sys

PATIENTS = {
    "P001": {"name": "Vivaan Gupta", "bed": "Bed 01", "condition": "ARDS"},
    "P002": {"name": "Aarav Sharma", "bed": "Bed 02", "condition": "Sepsis"},
}


def select_patient(source: str = "") -> str:
    print(f"\n{'='*50}")
    title = "PATIENT SELECTION"
    if source:
        title += f" — {source}"
    print(f"  {title}")
    print(f"{'='*50}")
    items = list(PATIENTS.items())
    for i, (pid, info) in enumerate(items, 1):
        print(f"  {i}. {pid}  |  {info['name']:20s} | {info['bed']:7s} | {info['condition']}")
    print(f"{'='*50}")
    try:
        entry = input("Select patient (1/2 or P001/P002, default 1): ").strip().upper()
    except (KeyboardInterrupt, SystemExit):
        sys.exit(0)

    # Map number input → patient ID
    if entry in ("1", "2"):
        pid = items[int(entry) - 1][0]
    elif entry in PATIENTS:
        pid = entry
    else:
        pid = items[0][0]
        print(f"  >>> Using default: {pid} <<<")

    info = PATIENTS[pid]
    print(f"  >>> Monitoring started for: {pid} — {info['name']} ({info['bed']}) <<<\n")
    return pid
