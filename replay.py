import sys
from pathlib import Path

from gym import Model


weights_dir = Path(__file__).parent / "weights"
files = sorted((*weights_dir.glob("*.pt"), *weights_dir.glob("*.pkl")))

if len(sys.argv) == 1:
    for i, path in enumerate(files, 1):
        print(f"{i}: {path.name}")

try:
    selection = sys.argv[1] if len(sys.argv) > 1 else input("Choose weights (number or filename): ")
    choice = int(selection)
    if not 1 <= choice <= len(files):
        raise ValueError
    path = files[choice - 1]
except ValueError:
    path = weights_dir / Path(selection).name
    if not path.is_file():
        raise SystemExit("Invalid choice")

print(f"Replaying {path}")
print(f"Score: {Model.load(path, 'cpu').eval_model('human')}")
