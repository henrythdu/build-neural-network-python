import sys
import math


def f4(x):
    return "%.4f" % x


def floats(s):
    return [float(t) for t in s.split(",")]


def fmt(v):
    return ",".join(f4(x) for x in v)


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def layer(W, b, x, m, n):
    w = [W[i:i+n] for i in range(0,len(W),n)]
    res = []
    for i,vals in enumerate(w):
        sig = sigmoid(sum([math.prod(pair) for pair in zip(vals,x)]) + b[i])
        res.append(sig)

    return res


def main():
    inputs = []
    W = []
    b = []
    for raw in sys.stdin.read().split("\n"):
        s = raw.strip()
        if s.startswith("INPUT "):
            inputs.append(floats(s[6:]))
        elif s.startswith("WEIGHTS "):
            W = floats(s[8:])
        elif s.startswith("BIAS "):
            b = floats(s[5:])
        elif s.startswith("M N "):
            parts = s.split()
            m, n = int(parts[2]), int(parts[3])
            for x in inputs:
                print(fmt(layer(W, b, x, m, n)))


main()
