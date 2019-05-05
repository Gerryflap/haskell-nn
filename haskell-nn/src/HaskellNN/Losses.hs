module HaskellNN.Losses where
import HaskellNN.Datastructures

-- A loss function takes predictions and true labels for one sample and returns a tuple (loss, gradients)
type LossFunction = (NDArray -> NDArray -> (NDArray, NDArray))

