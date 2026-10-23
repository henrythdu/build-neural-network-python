import sys
import math


def sigmoid(x):
    # TODO: return 1 / (1 + e^-x), without overflowing for large |x|.
    return 0.0


def main():
    for raw in sys.stdin.read().split("\n"):
        s = raw.strip()
        if not s:
            continue
        print("%.4f" % sigmoid(float(s)))


main()
