import datetime

LOG_FILE = "system_log.txt"

def log_event(event):
    clean_event = event.encode("ascii", errors="ignore").decode()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now()}] {clean_event}\n")
