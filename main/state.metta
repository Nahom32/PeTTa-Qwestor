; !(import! &self "context.py")


; type signiture
; x=GxM where G=goal(goal) vector and M= modulator vector
;  (..................... energy affiliation competence )
(: Context Type)
(: context (-> Atom Number Context))
(: State Type)

; Alpha values (learning rates) control update speed
    (= (alpha) ((urgency_alpha 0.60)         ; How fast urgency adapts
    (resolution_alpha 0.45)      ; How fast resolution adapts  
    (expertise_alpha 0.45)       ; How fast user expertise estimate adapts
    (threshold_alpha 0.40)       ; How fast confidence threshold adapts
    (familiarity_alpha 0.35)     ; How fast topic familiarity adapts
    (failure_alpha 0.55)         ; How fast failure wariness INCREASES
    (failure_decay 0.20)         ; How fast failure wariness DECAYS (opposite of alpha)
    (securing_alpha 0.45)        ; How fast securing/caution adapts
    (approach_alpha 0.40)        ; How fast approach/exploration adapts
    (arousal_alpha 0.40)         ; How fast arousal adapts
    (risk_aversion_alpha 0.45)   ; How fast risk aversion adapts
    (error_tolerance_alpha 0.40) ; How fast error tolerance adapts
    (creativity_alpha 0.40)      ; How fast creativity adapts
    (valence_alpha 0.40)))        ; How fast emotional valence adapts


; a function to add the list of alpha(a hyperparameter)  in to atomespace for modulator-updator (increamental blending)
(= (add-alpha $space $list)
 
 (if (== $list ())
    ()
 (let* (
    ($head (car-atom $list))
    ($tail (cdr-atom $list))
    ($_ (add-atom $space $head))
    )
    (add-alpha $space $tail)
 )
 ))

; get a single hyper-parameter from space to calculate its respective modulator
(= (get-alpha $name)
      (match &self ($name $y) $y)
)

; a function to add llm extracted context(perception) into atomspace 
; (= (add-context $space) 
;   (let* (
;     ($context-value (py-call (context.check)))
;     ($_ (add-atom $space $context-value)))
;     $context-value 
;     )
; )













(= (add-state $space)
    (add-atom $space $state)
)


(= (get-mod $space $modulator-name) 
(let* 
   (
    ($goal-val (match $space (state $x $y) (cdr-atom $y)))
    ($val 
    (case $modulator-name
        (
        (Activation (index-atom $goal-val 0))
        (Resolution (index-atom $goal-val 1))
        (SecuringThreshold (index-atom $goal-val 2))
        (SelectionThreshold (index-atom $goal-val 3))
                    
        )
   )))
   $val
   
))
(= (fetch-modulators $space)(match $space (state $x $y) $y))
(= (fetch-state $space) (match $space (state $x $y) (state $x $y)))
(= (fetch-goals $space)(match $space (state $x $y) $x))

(= (remove-state $space)
 (let $state (fetch-state $space) (remove-atom $space $state))
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



