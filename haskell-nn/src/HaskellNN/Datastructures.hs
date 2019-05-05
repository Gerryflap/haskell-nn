module HaskellNN.Datastructures where

data NDArray    = Scalar Float
                | Vector [Float]
                | Matrix [[Float]]
                deriving Show

transpose :: [[Float]] -> [[Float]]
transpose ([]:_) = []
transpose xs = (map head xs):(transpose (map tail xs))

transposeM :: NDArray -> NDArray
transposeM (Matrix m) = Matrix $ transpose m
transposeM _ = error "Cannot transpose something that is not a Matrix"

fromScalar :: NDArray -> Float
fromScalar (Scalar x) = x
fromScalar _ = error "Please supply scalar"

fromVector :: NDArray -> [Float]
fromVector (Vector x) = x
fromVector _ = error "Please supply vector"

fromMatrix :: NDArray -> [[Float]]
fromMatrix (Matrix x) = x
fromMatrix _ = error "Please supply matrix"

splitnd :: NDArray -> [NDArray]
splitnd (Matrix vs) = map (Vector) vs
splitnd (Vector xs) = map (Scalar) xs

zeros :: Int -> Int -> NDArray
zeros 0 0 = Scalar 0
zeros n 0 = Vector $ replicate n 0
zeros 0 n = zeros n 0
zeros n m = Matrix $ replicate n (replicate m 0)

uniformRandM :: Int -> Int -> [Float] -> (NDArray, [Float])
uniformRandM 0 _ rands = (Matrix [], rands)
uniformRandM n m rands = (Matrix ((take m newrands):nextm), drop m newrands)
    where
        (Matrix nextm, newrands) = uniformRandM (n-1) m rands

uniformRandV :: Int -> [Float] -> (NDArray, [Float])
uniformRandV n rands = (Vector $ take n rands, drop n rands)

shape :: Int -> NDArray -> Int
shape 0 (Vector xs) = length xs
shape 0 (Matrix xs) = length xs
shape 1 (Matrix (x:xs)) = length x

-- mergend zips/merges 2 NDArrays with the function f. This can be used for any pointwise function such as + or *
mergend :: (Float -> Float -> Float) -> NDArray -> NDArray -> NDArray
mergend f (Scalar x) (Scalar y) = Scalar $ f x y
mergend f (Vector x) (Vector y) = Vector $ zipWith f x y
mergend f (Matrix x) (Matrix y) = Matrix $ zipWith (zipWith f) x y
mergend f v@(Vector x) m@(Matrix y) | length y == length x =  Matrix $ zipWith (\xv yr -> map (f xv) yr) x y
                                | length (head y) == length x = transposeM $ mergend f v $ transposeM m
                                | otherwise = error "Cannot merge Vector with a Matrix with no equal length on both axes"
mergend f m@(Matrix _) v@(Vector _) = mergend f v m
mergend f (Scalar x) (Vector y) = Vector $ map (x*) y
mergend f v@(Vector _) s@(Scalar _) = mergend f s v
mergend f (Scalar x) (Matrix y) = Matrix $ map (map (x*)) y
mergend f m@(Matrix _) s@(Scalar _) = mergend f s m

mapm :: (Float -> Float) -> NDArray -> NDArray
mapm f (Scalar x) = Scalar $ f x
mapm f (Vector x) = Vector $ f <$> x
mapm f (Matrix x) = Matrix $ (f <$>) <$> x

-- Expands dimensions of the given NDArray. The first integer denotes the axis of expansion
expanddims :: Int -> NDArray -> NDArray
expanddims 0 (Vector x) = Matrix [x]
expanddims 1 (Vector x) = Matrix $ map (\v -> [v]) x
expanddims _ (Scalar x) = Vector [x]
expanddims _ (Matrix _) = error "Expanding Matrices is currently not supported!"
expanddims _ _ = error "Supplied arguments are invalid"

flatten :: NDArray -> NDArray
flatten (Vector x)  | length x == 1 = Scalar $ head x
                    | otherwise = error "Cannot flatten a vector with multiple entries"
flatten (Matrix x)  = Vector $ foldl (\flat arr -> flat ++ arr) [] x

matmul :: NDArray -> NDArray -> NDArray
matmul (Matrix a) (Matrix b) = Matrix [[ sum $ zipWith (*) ar bc | bc <- (transpose b) ] | ar <- a ]
matmul a@(Vector _) b@(Matrix _) = flatten $ expanddims 0 a @@ b
matmul a@(Matrix _) b@(Vector _) = flatten $ a @@ expanddims 1 b

infixl 5 @@
(@@) = matmul

data NDTree = Leaf NDArray
            | Node [NDTree]
            | LeafEmpty
            deriving Show

mergendt :: (NDArray -> NDArray -> NDArray) -> NDTree -> NDTree -> NDTree
mergendt f (Node t1s) (Node t2s) = Node $ zipWith (mergendt f) t1s t2s
mergendt f (Leaf x) (Leaf y) = Leaf $ f x y
mergendt _ LeafEmpty LeafEmpty = LeafEmpty
mergendt _ _ _ = LeafEmpty