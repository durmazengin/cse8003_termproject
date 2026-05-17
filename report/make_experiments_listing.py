"""Build ASCII-safe experiments log for LaTeX listings."""
from pathlib import Path

REPL = {
    "\u03b1": "a",  # alpha
    "\u03b2": "b",  # beta
    "\u03c6": "p",  # phi
    "\u03b5": "e",  # epsilon
    "\u2014": "-",  # em dash
    "\u2212": "-",  # minus
    "\u2022": "*",  # bullet
    "\u0394": "D",  # Delta
}

src = Path(__file__).with_name("experiments.txt")
dst = Path(__file__).with_name("experiments_listing.txt")
text = src.read_text(encoding="utf-8")
for old, new in REPL.items():
    text = text.replace(old, new)
dst.write_text(text, encoding="ascii", errors="replace")
print(f"Wrote {dst} ({len(text.splitlines())} lines)")
