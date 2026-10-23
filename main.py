import sys
import math


def sigmoid(x):
    # TODO: return 1 / (1 + e^-x), without overflowing for large |x|.i
    if x >= 0:
        sig = 1 / (1 + math.e**(-x))
    else:
        sig = math.e**(x) / (1 + math.e**x)

    return sig

def neuron(s):
    w,x,b = s.split(";")
    w = [float(i) for i in w.split(",")]
    x = [float(i) for i in x.split(",")]
    b = float(b)
    dot_prod = sum(math.prod(pair) for pair in zip(w,x)) + b
    return sigmoid(dot_prod)


def main():
    for raw in sys.stdin.read().split("\n"):
        s = raw.strip()
        if not s:
            continue
        # print("%.4f" % sigmoid(float(s)))
        print("%.4f" % neuron(s))

main()
