import sys
import math


def f4(x):
    return "%.4f" % x


def floats(s):
    return [float(t) for t in s.split(",")]


def fmt(v):
    return ",".join(f4(x) for x in v)


def one_hot(label, k):
    # TODO: a list of k integers, 1 at index `label`, 0 everywhere else.
    oh = [0]*k
    oh[label] = 1
    return oh


def standardize(pixels):
    # TODO: for every pixel p return (p / 255 - 0.1307) / 0.3081. Divide as floating point.
    return [(p/255-0.1307)/0.3081 for p in pixels]


def main():
    k = 0
    for raw in sys.stdin.read().split("\n"):
        s = raw.strip()
        if not s:
            continue
        if s.startswith("K "):
            k = int(s[2:])
            continue
        label_text, image = s.split(None, 1)
        pixels = floats(image.replace("|", ","))
        print(",".join(str(v) for v in one_hot(int(label_text), k)) + ";" + fmt(standardize(pixels)))


main()
