{-# language ExistentialQuantification #-}
module HaskellNN.Layers where

import HaskellNN.Losses
import HaskellNN.Activations
import HaskellNN.Datastructures


-- Definition of the Layer_ typeclass which defines that all layers can forward and backward propagate
class Layer_ l where
    -- Receives the layer and an input NDArray and outputs (output NDArray, NDTree of all layer outputs)
    fwd :: l -> NDArray -> (NDArray, NDTree)

    -- Receives a layer and a tuple (inputs, error) and outputs a tuple (backpropagated error, weight gradients)
    bwd :: l -> (NDTree, NDArray) -> (NDArray, NDTree)

    -- Updates the layer with new parameters
    setparams :: l -> NDTree -> l
    setparams l _ = l

    getparams :: l -> NDTree
    getparams l = LeafEmpty

    -- Adds the gradients to the parameters
    addgrads :: l -> NDTree -> l
    addgrads l grads = setparams l $ mergendt (mergend (+)) grads $ getparams l

    -- Initializes weights using a random generator list and passes the new layer and list without the used numbers onwards
    initialize :: [Float] -> l -> ([Float], l)
    initialize list layer = (list, layer)


data Layer = forall l. Layer_ l => Layer l

instance Layer_ Layer where
    fwd (Layer l) = fwd l
    bwd (Layer l) inp = (output, grads)
        where
            (output, grads) = bwd l inp
    initialize nums (Layer l) = (newnums, Layer newlayer)
        where
            (newnums, newlayer) = initialize nums l
    setparams (Layer l) t = Layer $ setparams l t
    getparams (Layer l) = getparams l
    addgrads (Layer l) t = Layer $ addgrads l t

-- Models a sequence of layers
data Sequential = Sequential [Layer]

singleFwdOp :: (NDArray, [NDTree]) -> Layer -> (NDArray, [NDTree])
singleFwdOp (inp, inps) l = (outp, inpt:inps)
    where
        (outp, inpt) = fwd l inp

singleBwdOp :: (NDTree, Layer) -> (NDArray, [NDTree]) -> (NDArray, [NDTree])
singleBwdOp (i, l) (e, gl) = (ne, gs:gl)
    where
        (ne, gs) = bwd l (i, e)

singleInitOp :: ([Float], [Layer]) -> Layer -> ([Float], [Layer])
singleInitOp (r, ls) l = (nr, nl:ls)
    where
        (nr, nl) = initialize r l

instance Layer_ Sequential where
    fwd (Sequential layers) input = (outp, Node $ reverse inpRev)
        where
            (outp, inpRev) = foldl singleFwdOp (input, []) layers
    bwd (Sequential layers) (inpt, errors) = (newerror, Node grads)
        where
            (Node inps) = inpt
            il = zip inps layers
            (newerror, grads) = foldr singleBwdOp (errors, []) il
    initialize rands (Sequential layers) = (newrand, Sequential $ reverse newlayers)
        where
            (newrand, newlayers) = foldl singleInitOp (rands, []) layers

    setparams (Sequential layers) (Node inps) = Sequential $ zipWith setparams layers inps

    getparams (Sequential layers) = Node $ map getparams layers

-- Models a Dense (Fully Connected) layer
data Dense = Dense NDArray NDArray

denselayer :: NDArray -> NDArray -> Dense
denselayer w@(Matrix _) b@(Vector _) = Dense w b

instance Layer_ Dense where
    fwd (Dense w b) input = (out, Leaf input)
        where
            out = mergend (+) b (input @@ w)
    bwd (Dense w b) (Leaf inp, errors) = (newErrors, Node [Leaf gradsWeights, Leaf gradsBias])
        where
            newErrors = errors @@ transposeM w
            gradsBias = errors
            summedinp = foldl1 (mergend (+)) $ splitnd inp
            gradsWeights = (expanddims 1 summedinp) @@ (expanddims 0 errors)
    initialize rands (Dense w b) = (newrands, Dense newW newB)
        where
            (newB, rands2) = uniformRandV (shape 0 b) rands
            (newW, newrands) = uniformRandM (shape 0 w) (shape 1 w) rands2
    setparams (Dense w b) (Node [Leaf nw@(Matrix _), Leaf nb@(Vector _)]) = Dense nw nb
    getparams (Dense w b) = Node [Leaf w, Leaf b]

-- Activation Layer
data Activation = Activation ActivationFn

instance Layer_ Activation where
    fwd (Activation (afwd, _)) input = (mapNd afwd input, Leaf input)
    bwd (Activation (_, abwd)) (Leaf inp, errors) = (mergend abwd summedinp errors, LeafEmpty)
        where
            summedinp = foldl1 (mergend (+)) $ splitnd inp