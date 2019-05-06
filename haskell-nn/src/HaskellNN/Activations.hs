module HaskellNN.Activations where

-- Input -> Output
type ActivationFwd = (Float -> Float)

-- Input -> Gradient -> Backward Gradient
type ActivationBwd = (Float -> Float -> Float)
type ActivationFn = (ActivationFwd, ActivationBwd)


relu_fwd :: ActivationFwd
relu_fwd x  | x > 0 = x
        | otherwise = 0

relu_bwd :: ActivationBwd
relu_bwd activation gradient    | activation > 0 = gradient
                                | otherwise = 0

relu :: ActivationFn
relu = (relu_fwd, relu_bwd)


sigmoid_fwd :: ActivationFwd
sigmoid_fwd x = 1.0 / (1.0 + exp (-x))

sigmoid_bwd :: ActivationBwd
sigmoid_bwd i x = x * (sigmoid_fwd i) * (1.0 - sigmoid_fwd i)

sigmoid :: ActivationFn
sigmoid = (sigmoid_fwd, sigmoid_bwd)