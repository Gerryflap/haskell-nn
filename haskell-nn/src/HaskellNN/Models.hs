module HaskellNN.Models where

import HaskellNN.Layers
import HaskellNN.Optimizers
import HaskellNN.Losses
import HaskellNN.Datastructures

data Model = Model Layer LossFunction Optimizer

trainBatch :: Model -> NDArray -> NDArray -> (NDArray, Model)
trainBatch (Model lay lossfn opt) batch@(Matrix _) true@(Matrix _) = (loss, Model newlay lossfn newopt)
    where
        (o, inps) = fwd lay batch
        (loss, backGrads) = lossfn o true
        (_, grads) = bwd lay (inps, backGrads)
        (newopt, finalgrads) = apply opt grads
        newlay = addgrads lay finalgrads

predictBatch :: Model -> NDArray -> NDArray
predictBatch (Model lay lossfn opt) batch@(Matrix _) = fst $ fwd lay batch

initM :: [Float] -> Model -> ([Float], Model)
initM rands (Model lay lossfn opt) = (nrand, Model nlay lossfn opt)
    where
        (nrand, nlay) = initialize rands lay