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



--xdata = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
--ydata = [[0.0], [1.0], [1.0], [0.0]]
--
--model = Sequential [
--        Layer $ denselayer 2 20,
--        Layer ReLU,
--        Layer $ denselayer 20 20,
--        Layer ReLU,
--        Layer $ denselayer 20 1,
--        Layer Sigmoid
--    ] mse
--
--main :: IO ()
--main = do
--    stdgen <- getStdGen
--    let rands = randomRs (-0.3, 0.3) stdgen :: [Float]
--    let (nrands, imodel) = initialize rands model
--    let (s, trainedmodel) = trainmodel xdata ydata 1 10000 imodel
--    putStrLn s
--    putStrLn $ "Output for data: " ++ (show $ map (fwd trainedmodel) xdata)
--
--
--
--
--trainmodel _ _ _ (-1) m = ("Done. ", m)
--trainmodel x y i iters model    | i `mod` 1000 == 0 = (s ++ nexts, nextm)
--                                | otherwise = (nexts, nextm)
--    where
--        (loss, newmodel) = trainepoch x y model
--        s = "Epoch: " ++ (show (i)) ++ "/" ++ (show (i+iters-1)) ++ "\nLoss: " ++ (show loss) ++ "\n\n"
--        (nexts, nextm) = trainmodel x y (i+1) (iters - 1) newmodel

xdata = Matrix [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
ydata = Matrix [[0.0], [1.0], [1.0], [0.0]]

nn = Layer $ Sequential [
        Layer $ Dense (zeros 1 20) (zeros 20 0),
        Layer $ Activation relu,
        Layer $ Dense (zeros 20 20) (zeros 20 0),
        Layer $ Activation relu,
        Layer $ Dense (zeros 20 1) (zeros 1 0),
        Layer $ Activation sigmoid
    ]

model = Model nn mse (Optimizer $ SGD 0.01)

train m 0 = "Done.\n" ++ "Final output: " ++ (show $ predictBatch m xdata) ++ "\n Model params:" ++ (show (getparams $ getLayer m))
train m batches = "Loss: " ++ (show loss) ++ "\n" ++ train nm (batches-1)
    where
        (loss, nm) = trainBatch m xdata ydata

main :: IO ()
main = do
    stdgen <- getStdGen
    let rands = randomRs (-0.3, 0.3) stdgen :: [Float]
    let (nrands, imodel) = initM rands model

    putStrLn $ train imodel 1000