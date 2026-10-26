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


def grads(x, w, b, y):
    # TODO: forward pass for a, then delta = dL/da * da/dz,
    # then return [delta * x_i for every input] followed by delta (the bias gradient).
    z = sum([xi*wi for xi,wi in zip(x,w)]) + b
    a = sigmoid(z)
    dLda = 2*(a-y)
    dadz = a*(1-a)
    delta = dLda*dadz

    return [delta*xi for xi in x] + [delta]



def main():
    for raw in sys.stdin.read().split("\n"):
        s = raw.strip()
        if not s:
            continue
        xs, ws, bs, ys = s.split(";")
        print(fmt(grads(floats(xs), floats(ws), float(bs), float(ys))))


main()
