import sys
import math


def f4(x):
    return "%.4f" % x


def floats(s):
    return [float(t) for t in s.split(",")]


def fmt(v):
    return ",".join(f4(x) for x in v)

EPS = 1e-12

def clip(val):
    return max(EPS,min(val, 1-EPS))

def CE(yi,ai):
    return -(yi*math.log(ai) + (1-yi)*math.log(1-ai))

def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def train(X, y, lr, epochs):
    # TODO: w = k zeros, b = 0. Each epoch: forward over all rows, accumulate the loss,
    # the correct count and the averaged gradients, record loss and accuracy, then update.
    # Return (losses, accs, w, b): per-epoch lists, then the final parameters.
    m,k = len(X),len(X[0])
    w,b = [0]*k,0
    losses= []
    accs = []
    for _ in range(epochs):
        zs = [sum([wj*xi for wj,xi in zip(w,x_row)]) + b for x_row in X]
        a = [sigmoid(z) for z in zs]
        errors = [ai-yi for ai,yi in zip(a,y)]
        dLdw = [sum(err*xj for err,xj in zip(errors,col))/m for col in zip(*X)]
        dLdb = sum(errors)/m
        w = [wj-dLdwj*lr for wj,dLdwj, in zip(w,dLdw)]
        b -= dLdb*lr
        accs.append(sum(1 for ai, yi in zip(a, y) if (ai >= 0.5) == yi) / m)
        loss = sum(CE(yi,ai) for yi,ai in zip(y,a))/m
        losses.append(loss)


    return losses, accs, w, b


def main():
    lr, epochs = 0.1, 1
    X, y = [], []
    for raw in sys.stdin.read().split("\n"):
        s = raw.strip()
        if s.startswith("LR "):
            lr = float(s[3:])
        elif s.startswith("EPOCHS "):
            epochs = int(s[7:])
        elif s.startswith("DATA "):
            rows = [floats(r) for r in s[5:].split(";")]
            X = [r[:-1] for r in rows]
            y = [r[-1] for r in rows]
    losses, accs, w, b = train(X, y, lr, epochs)
    for e in range(epochs):
        print("epoch %d loss=%s acc=%s" % (e + 1, f4(losses[e]), f4(accs[e])))
    print("params " + fmt(w + [b]))


main()
