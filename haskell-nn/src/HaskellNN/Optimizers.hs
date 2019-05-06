{-# language ExistentialQuantification #-}
module HaskellNN.Optimizers where

import HaskellNN.Datastructures


class Optimizer_ o where
    -- Applies the optimizer to grads and gives (newopt, newgrads)
    apply :: o -> NDTree -> (o, NDTree)

data Optimizer = forall o. Optimizer_ o => Optimizer o

instance Optimizer_ Optimizer where
    apply (Optimizer o) t = (Optimizer no, nt)
        where
            (no, nt) = apply o t

data SGD = SGD Float

instance Optimizer_ SGD where
    apply opt@(SGD lr) grads = (opt, mapNdtNd ((-lr)*) grads)