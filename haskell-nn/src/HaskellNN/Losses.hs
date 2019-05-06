module HaskellNN.Losses where
import HaskellNN.Datastructures

-- A loss function takes predictions and true labels for one sample and returns a tuple (loss, gradients)
type LossFunction = (NDArray -> NDArray -> (NDArray, NDArray))

mse :: LossFunction
mse prd true = (loss, diffs)
    where
        diffs@(Matrix m) = mergend (-) prd true
        sqdiffs = mapNd (**2) diffs
        summedgrads = foldl1 (mergend (+)) $ splitnd diffs
        meangrads = mapNd (/(fromIntegral $ shape 0 diffs)) summedgrads
        summedsq = foldl1 (mergend (+)) $ splitnd sqdiffs
        meansq = mapNd (/(fromIntegral $ shape 0 sqdiffs)) summedsq
        loss = foldl1 (mergend (+)) $ splitnd meansq