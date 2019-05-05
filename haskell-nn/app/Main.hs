module Main where

import Lib
import System.Random
import Control.Monad
import HaskellNN
import HaskellNN.Datastructures

xdata = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
ydata = [[0.0], [1.0], [1.0], [0.0]]

model = Sequential [
        Layer $ denselayer 2 20,
        Layer ReLU,
        Layer $ denselayer 20 20,
        Layer ReLU,
        Layer $ denselayer 20 1,
        Layer Sigmoid
    ] mse

main :: IO ()
main = do
    stdgen <- getStdGen
    let rands = randomRs (-0.3, 0.3) stdgen :: [Float]
    let (nrands, imodel) = initialize rands model
    let (s, trainedmodel) = trainmodel xdata ydata 1 10000 imodel
    putStrLn s
    putStrLn $ "Output for data: " ++ (show $ map (fwd trainedmodel) xdata)




trainmodel _ _ _ (-1) m = ("Done. ", m)
trainmodel x y i iters model    | i `mod` 1000 == 0 = (s ++ nexts, nextm)
                                | otherwise = (nexts, nextm)
    where
        (loss, newmodel) = trainepoch x y model
        s = "Epoch: " ++ (show (i)) ++ "/" ++ (show (i+iters-1)) ++ "\nLoss: " ++ (show loss) ++ "\n\n"
        (nexts, nextm) = trainmodel x y (i+1) (iters - 1) newmodel
