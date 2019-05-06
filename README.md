# A Neural Network library written in Haskell

```zipWith gerben haskell deeplearning```

An attempt at making a neural network library in Haskell.
This is not something you'd want to use for any project, bugs are probably still present.

What works:
- A simple xor gate example
- NNs can be constructed as a Sequence of Layers (with possiblilty for different structures in the future)
- Batched updates to the network
- Layers, optimizers and loss functions:
  - Dense (Fully Connected) layers
  - ReLU activation
  - Sigmoid activation
  - Mean Squared error loss function
  - SGD optimizer
  

Things to improve:
- No warning when combining layers that do not fit together (This is very dangerous!)
- No multithreading support
