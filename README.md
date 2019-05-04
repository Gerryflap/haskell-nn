# A Neural Network library written in Haskell

An attempt at making a neural network library in Haskell.
This is not something you'd want to use for any project, bugs are probably still present.

What works:
- A simple xor gate example
- NNs can be constructed as a Sequential model with a list of Layers

Things to improve:
- Learning rate is hardcoded in update rules
- No support for arbitrary optimizers
- No support for batch updates, only supports single input vectors
- Code is kind of messy right now
- No multithreading support