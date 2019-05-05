{-# language ExistentialQuantification #-}
-- This library might get phased out later
module Lib where

import System.Random
import Debug.Trace

zipMats :: (Float -> Float -> Float) -> [[Float]] -> [[Float]] -> [[Float]]
zipMats f = \x y -> zipWith (zipWith f) x y

-- Definition of the Layer_ typeclass which defines that all layers can forward and backward propagate
class Layer_ l where
    -- Receives the layer and an input vector and outputs the output vector
    fwd :: l -> [Float] -> [Float]

    -- Receives a layer and a tuple (inputs, error) and outputs a tuple (backpropagated error, updated layer)
    bwd :: l -> ([Float], [Float]) -> ([Float], l)

    -- Initializes weights using a random generator list and passes the new layer and list without the used numbers onwards
    initialize :: [Float] -> l -> ([Float], l)
    initialize list layer = (list, layer)

class Model_ model where
    -- Gets x -> y -> model and returns (loss, trained model)
    train :: [Float] -> [Float] -> model -> (Float, model)

    trainepoch :: [[Float]] -> [[Float]] -> model -> (Float, model)
    trainepoch _ [] model = (0, model)
    trainepoch [] _ model = (0, model)
    trainepoch (x:xs) (y:ys) model = (newloss, newmodel)
        where
            (prevloss, prevmodel) = trainepoch xs ys model
            (loss, newmodel) = train x y prevmodel
            floatlen = fromIntegral $ length xs
            newloss = (loss / (1.0 + floatlen)) + (floatlen * prevloss / (1.0 + floatlen))


-- Definition of the Layer datatype that wraps all Layer_ instances in order to be put in a list or other datatype
data Layer = forall l. Layer_ l => Layer l

instance Layer_ Layer where
    fwd (Layer l) = fwd l
    bwd (Layer l) inp = (output, Layer newlayer)
        where
            (output, newlayer) = bwd l inp
    initialize nums (Layer l) = (newnums, Layer newlayer)
        where
            (newnums, newlayer) = initialize nums l

-- A loss function takes predictions and true labels for one sample and returns a tuple (loss, gradients)
type LossFunction = ([Float] -> [Float] -> (Float, [Float]))

mse :: LossFunction
mse ypred ytrue = (loss, grads)
    where
        loss = (sum $ zipWith (\y1 y2 -> (y1 - y2)^2) ypred ytrue) / (fromIntegral (length ypred))
        grads = zipWith (\y1 y2 -> (y1 - y2) / (fromIntegral (length ypred))) ypred ytrue
-- Sequential Model is a list of layers that can be used as a layer in itself and represents a whole neural network
data SequentialModel = Sequential [Layer] LossFunction

-- Gets all layer inputs as a list of vectors
getinputs :: SequentialModel -> [Float] -> [[Float]]
getinputs (Sequential [] loss) vec = []
getinputs (Sequential (l:ls) loss) vec = vec:inputs
    where
        inputs = getinputs (Sequential ls loss) (fwd l vec)

-- Backprop helper function, variables: reversed list of (inp, layer) -> grads -> (output grads, reversed list of updated layers)
backpropseq_ :: [([Float], Layer)] -> [Float] -> ([Float], [Layer])
backpropseq_ [] grads = (grads, [])
backpropseq_ ((inp, layer):xs) grads = (finalgrads, newlayer:newlayers)
    where
        (newgrads, newlayer) = bwd layer (inp, grads)
        (finalgrads, newlayers) = backpropseq_ xs newgrads


-- Backpropagates and updates the model given: inputs (from getinputs) -> gradients -> Current model -> Returns updated model
backpropseq :: [[Float]] -> [Float] -> SequentialModel ->  ([Float], SequentialModel)
backpropseq ins grads (Sequential layers loss) = (finalgrads, Sequential (reverse reversedlayers) loss)
    where
        zipped = zip ins layers
        (finalgrads, reversedlayers) = backpropseq_ (reverse zipped) grads


instance Layer_ SequentialModel where
    fwd (Sequential [] loss) vec = vec
    fwd (Sequential (l:ls) loss) vec = fwd (Sequential ls loss) (fwd l vec)

    bwd model (inp, grads) = backpropseq (getinputs model inp) grads model

    initialize nums model@(Sequential [] _) = (nums, model)
    initialize nums (Sequential (l:ls) loss) = (newnums, Sequential (newlayer:newlayers) loss)
        where
            (usednums, Sequential newlayers _) = initialize nums (Sequential ls loss)
            (newnums, newlayer) = initialize usednums l

instance Model_ SequentialModel where
    train x y model@(Sequential layers lossf) = (loss, updatedmodel)
        where
            (loss, grads) = lossf (fwd model x) y
            (_, updatedmodel) = bwd model (x, grads)

-- The identity layer that just passes anything through it
data Identity = Identity deriving Show

instance Layer_ Identity where
    fwd _ inp = inp
    bwd l (_, inp) = (inp, l)

-- The ReLU activation function
data ReLU = ReLU

relu :: Float -> Float
relu x  | x > 0 = x
        | otherwise = 0

relu_bwd :: Float -> Float -> Float
relu_bwd activation gradient    | activation > 0 = gradient
                                | otherwise = 0

instance Layer_ ReLU where
    fwd _ inp = map relu inp
    bwd l (inputs, gradients) = (zipWith relu_bwd inputs gradients, l)

data Sigmoid = Sigmoid

sigmoid :: Float -> Float
sigmoid x = 1.0 / (1.0 + exp (-x))

sigmoid_bwd :: Float -> Float -> Float
sigmoid_bwd i x = x * (sigmoid i) * (1.0 - sigmoid i)

instance Layer_ Sigmoid where
    fwd _ inp = map sigmoid inp
    bwd l (inputs, gradients) = (zipWith sigmoid_bwd inputs gradients, l)

-- Dense layer (also called Fully Connected layer)

transpose :: [[Float]] -> [[Float]]
transpose ([]:_) = []
transpose xs = (map head xs):(transpose (map tail xs))

data Dense = Dense [[Float]] [Float]

instance Show Dense where
    show (Dense w b) = "Dense: " ++ show w ++ " " ++ show b

initweights :: [Float] -> [[Float]] -> ([Float], [[Float]])
initweights rands [] = (rands, [])
initweights rands (row:rows) = (newrand, newrow:newweights)
    where
        newrow = take (length row) rands
        passedrand = drop (length row) rands
        (newrand, newweights) = initweights passedrand rows

instance Layer_ Dense where
    fwd (Dense weights biases) inp = zipWith (+) weightsfwd biases
        where
            --weightsfwd = map sum (zipWith (\i w -> map (i*) w) inp $ transpose weights)
            weightsfwd = map (\w -> sum $ zipWith (*) inp w) $ transpose weights
    bwd (Dense weights biases) (inputs, gradients) = (backerror, Dense updatedweights updatedbias)
        where
            backerror = map (\w -> sum $ zipWith (*) gradients w) $ weights
            updatedbias = zipWith (-) biases $ map (0.01*) gradients
            weightgrads = map (\x -> map (x*) gradients) inputs
            updatedweights = zipMats (\w g -> w - 0.01 * g) weights weightgrads
    initialize rands (Dense w b) = (newrands, Dense neww newb)
        where
            (newrands, newb:neww)  = initweights rands (b:w)

denselayer :: Int -> Int -> Dense
denselayer i o = Dense w b
    where
        w = replicate i $ replicate o 0
        b = replicate o 0