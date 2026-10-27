import sys
import math


def f4(x):
    return "%.4f" % x


def floats(s):
    return [float(t) for t in s.split(",")]


def fmt(v):
    return ",".join(f4(x) for x in v)

def split_batch(list,batch):
    return [list[x:x+batch] for x in range(0,len(list),batch)]

def train(xs, ys, lr, epochs, batch):

    x_batches = split_batch(xs,batch)
    y_batches = split_batch(ys,batch)
    w,b = 0,0
    for _ in range(epochs):
        for x_batch,y_batch in zip(x_batches,y_batches):
            m = len(x_batch)
            dLdw = sum([2*(w*xi + b - yi)*xi for xi,yi in zip(x_batch,y_batch)])/m
            dLdb = sum([2*(w*xi + b - yi) for xi,yi in zip(x_batch,y_batch)])/m
            w -= dLdw*lr
            b -= dLdb*lr


    return w,b


def main():
    lr, epochs, batch = 0.1, 1, 1
    xs, ys = [], []
    for raw in sys.stdin.read().split("\n"):
        s = raw.strip()
        if s.startswith("LR "):
            lr = float(s[3:])
        elif s.startswith("EPOCHS "):
            epochs = int(s[7:])
        elif s.startswith("BATCH "):
            batch = int(s[6:])
        elif s.startswith("DATA "):
            flat = floats(s[5:].replace(";", ","))
            xs, ys = flat[0::2], flat[1::2]
    w, b = train(xs, ys, lr, epochs, batch)
    total = 0.0
    for x, y in zip(xs, ys):
        err = w * x + b - y
        total += err * err
    print(fmt([w, b, total / len(xs)]))


main()
