# Python-Neural-Network
a Neural Network made a python
ready to import and use as a library

eg. :
if __name__ == "__main__":
    TABLE = [
    ([0, 0], [0]),   # sum=0, carry=0
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0]),   # sum=0, carry=1
]
    
nn = Neural_Network(2, [2], 1, activation="leaky-relu")
nn.Train(TABLE, 100000, 0.05)

print("--- predictions ---")
for inp, target in TABLE:
    out = nn.Forward(inp)[0]
    print(f"input {inp}  target {target[0]}  output {out:.4f}")
