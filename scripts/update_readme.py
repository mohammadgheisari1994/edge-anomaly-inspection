"""Copy results/<category>/results.md into README.md between the RESULTS markers.

    python scripts/update_readme.py results/pcb1
"""
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
res_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "results" / "pcb1"
table = (res_dir / "results.md").read_text().strip()
table = re.sub(r"^# .*\n+", "", table)  # drop the file's own heading

readme = root / "README.md"
text = readme.read_text()
new = re.sub(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->",
             f"<!-- RESULTS:START -->\n{table}\n<!-- RESULTS:END -->", text, flags=re.S)
readme.write_text(new)
print(f"README updated from {res_dir / 'results.md'}")
