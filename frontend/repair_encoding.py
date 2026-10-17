from pathlib import Path
import ftfy

p = Path("index.html")
backup = Path("index.html.encoding-backup")

if not backup.exists():
    raise SystemExit("Backup missing. Aborting to protect the original file.")

text = p.read_text(encoding="utf-8")
fixed = ftfy.fix_text(text)

if fixed == text:
    print("No changes detected. File left unchanged.")
else:
    p.write_text(fixed, encoding="utf-8", newline="")
    print("Encoding repair applied and saved as UTF-8.")
    print("Original backup: index.html.encoding-backup")
