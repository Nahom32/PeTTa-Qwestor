(= (_clamp01 $x)
  (if (and (< $x 1) (> $x 0))
     $x
     (if (> $x 1)
      1
      0
      )
   )
)
(= (_clamp11 $x)
  (if (and (< $x 1) (> $x -1))
     $x
     (if (> $x 1)
      1
      -1
      )
   )
)

(= (replaceByIndex $expr $index $new)
    (if (== $expr ())
        $expr
        (let*
            (
                ($first (car-atom $expr))
                ($tail (cdr-atom $expr))
            )
            (if (== $index 0)
                (if (== $tail ())
                    ($new)
                    (cons-atom $new $tail)
                )
                (let $c (replaceByIndex $tail (- $index 1) $new)
                    (cons-atom $first $c)
                )))))
