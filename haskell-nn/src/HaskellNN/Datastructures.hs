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

mergend :: (Float -> Float -> Float) -> NDArray -> NDArray -> NDArray
mergend f (Scalar x) (Scalar y) = Scalar $ f x y
mergend f (Vector x) (Vector y) = Vector $ zipWith f x y
mergend f (Matrix x) (Matrix y) = Matrix $ zipWith (zipWith f) x y
mergend _ _ _ = error "mergend can only merge 2 NDArrays of the same type"

mapm :: (Float -> Float) -> NDArray -> NDArray
mapm f (Scalar x) = Scalar $ f x
mapm f (Vector x) = Vector $ f <$> x
mapm f (Matrix x) = Matrix $ (f <$>) <$> x


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