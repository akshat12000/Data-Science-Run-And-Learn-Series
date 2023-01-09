import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

def relu(x):
    return max(0, x)

def leaky_relu(x):
    return max(0.1 * x, x)

def main():
    print("Sigmoid: ", sigmoid(1))
    print("Tanh: ", tanh(1))
    print("ReLU: ", relu(1))
    print("Leaky ReLU: ", leaky_relu(1))

if __name__ == "__main__":
    main()
