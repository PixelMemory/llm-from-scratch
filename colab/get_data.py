"""Fetch a real char-level corpus for GPU training.

The bundled data/input.txt is only ~4 KB — fine for a laptop sanity check, but
an A100 would memorize it in seconds. This downloads tiny-shakespeare (~1 MB),
the canonical char-level dataset, and overwrites data/input.txt.

Usage (in Colab or locally):  python colab/get_data.py
"""

import os
import urllib.request

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(HERE, "data", "input.txt")


def main():
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    print(f"downloading tiny-shakespeare -> {DEST}")
    urllib.request.urlretrieve(URL, DEST)
    print(f"done: {os.path.getsize(DEST):,} bytes")


if __name__ == "__main__":
    main()
