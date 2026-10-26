import sys
import math


def f4(x):
    return "%.4f" % x


def floats(s):
    return [float(t) for t in s.split(",")]


def fmt(v):
    return ",".join(f4(x) for x in v)

EPS = 1e-12


def mse(p, t):
    # TODO: mean of (p - t)^2
    res = sum([(pi-ti)**2 for pi,ti in zip(p,t)])/len(p)
    return res


def mae(p, t):
    # TODO: mean of |p - t|
    res = sum([abs(pi-ti) for pi,ti in zip(p,t)])/len(p)
    return res

def clam(pi):
    return max(EPS,max(pi,min(pi,1-EPS)))

def bce(p, t):
    res = sum([(-(ti*math.log(clam(pi)) + (1-ti)*math.log(1-clam(pi)))) for pi,ti in zip(p,t)])/len(p)
    return res


def main():
    p, t, name = [], [], "mse"
    for raw in sys.stdin.read().split("\n"):
        s = raw.strip()
        if s.startswith("PRED "):
            p = floats(s[5:])
        elif s.startswith("TRUE "):
            t = floats(s[5:])
        elif s.startswith("LOSS "):
            name = s[5:].strip()
    fn = {"mse": mse, "mae": mae, "bce": bce}[name]
    print(f4(fn(p, t)))


main()
