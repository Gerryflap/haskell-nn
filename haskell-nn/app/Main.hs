module Main where

--import Lib
import System.Random
import Control.Monad
import HaskellNN.Activations
import HaskellNN.Datastructures
import HaskellNN.Layers
import HaskellNN.Models
import HaskellNN.Losses
import HaskellNN.Optimizers

xdata = Matrix [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
ydata = Matrix [[0.0], [1.0], [1.0], [0.0]]

nn = Layer $ Sequential [
        Layer $ Dense (zeros 2 20) (zeros 20 0),
        Layer $ Activation relu,
--        Layer $ Dense (zeros 20 20) (zeros 20 0),
--        Layer $ Activation relu,
        Layer $ Dense (zeros 20 1) (zeros 1 0),
        Layer $ Activation sigmoid
    ]

model = Model nn mse (Optimizer $ SGD 0.1)

train m 0 = "Done.\n" ++ "Final output: " ++ (show $ predictBatch m xdata) ++ "\n Model params:" ++ (show (getparams $ getLayer m))
train m batches = "Loss: " ++ (show loss) ++ "\n" ++ train nm (batches-1)
    where
        (loss, nm) = trainBatch m xdata ydata

main :: IO ()
main = do
    stdgen <- getStdGen
    let rands = randomRs (-3.0, 3.0) stdgen :: [Float]
    let (nrands, imodel) = initM rands model

    putStrLn $ train imodel 1000