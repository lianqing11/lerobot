#!/usr/bin/env python

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from lerobot.value_function.train_value_function import main


if __name__ == "__main__":
    main()
