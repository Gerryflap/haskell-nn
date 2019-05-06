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