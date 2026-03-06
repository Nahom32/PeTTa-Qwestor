(= (get-goal $space $goal-name) 
(let* 
   (
    ($goal-val (match $space (state $x $y) (cdr-atom $x)))
    ($val 
    (case $goal-name
        (
        (integerity (index-atom $goal-val 0))
        (energy (index-atom $goal-val 1))
        (competence (index-atom $goal-val 2))
        (affiliation (index-atom $goal-val 3))
        )
   )))
   $val
   

))


(= (update-goal $space $name $value)
    (let*
     (
        ($current-modulator-vec (fetch-modulators $space))
        ($current-goal (cdr-atom (fetch-goals $space)))
        ($update-goal  (case $name 
        ( 
            (integerity (replaceByIndex $current-goal 0 ($name $value)))
            (energy (replaceByIndex $current-goal 1 ($name $value)))
            (competence (replaceByIndex $current-goal 2 ($name $value)))
            (affiliation (replaceByIndex $current-goal 3 ($name $value)))
        )))
        ; here the function goes through the logic of comonadic(Appraisal Comonad Psi) computation using different formulas and logic but it is to show how the space works 
        ($updated-state (state (cons goal-vector $update-goal ) $current-modulator-vec))
        ($_ (remove-state $space))
        
     )
     (add-state $space $updated-state)
      )
)