"""Load notebook definitions without downloading data or training full models."""
import json
from pathlib import Path

notebook = json.loads((Path(__file__).resolve().parents[1] / 'experiments.ipynb').read_text())
for cell in notebook['cells']:
    if 'definitions' in cell.get('metadata', {}).get('tags', []):
        exec(compile(''.join(cell['source']), '<notebook>', 'exec'), globals())
