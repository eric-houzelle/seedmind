import sys
from pathlib import Path

# Ensure the repository root is importable so `import seedmind` works in tests.
sys.path.insert(0, str(Path(__file__).resolve().parent))
