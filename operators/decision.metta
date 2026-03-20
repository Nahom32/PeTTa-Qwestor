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
            (efficiency (replaceByIndex $current-goal 0 ($name $value)))
            (accuracy (replaceByIndex $current-goal 1 ($name $value)))
            (success_moderate (replaceByIndex $current-goal 2 ($name $value)))
            (knowledge (replaceByIndex $current-goal 3 ($name $value)))
            (novelty (replaceByIndex $current-goal 4 ($name $value)))
            (success_breakthrough (replaceByIndex $current-goal 5 ($name $value)))
            (coherence (replaceByIndex $current-goal 6 ($name $value)))
            (originality (replaceByIndex $current-goal 7 ($name $value)))
            (social (replaceByIndex $current-goal 8 ($name $value)))
            (help_short (replaceByIndex $current-goal 9 ($name $value)))
            (help_long (replaceByIndex $current-goal 10 ($name $value)))
            (over_beneficial (replaceByIndex $current-goal 11 ($name $value)))
            (over_safety (replaceByIndex $current-goal 12 ($name $value)))
            (over_honesty (replaceByIndex $current-goal 13 ($name $value)))
        )))
        ; here the function goes through the logic of comonadic(Appraisal Comonad Psi) computation using different formulas and logic but it is to show how the space works 
        ($updated-state (state (cons goal-vector $update-goal ) $current-modulator-vec))
        ($_ (remove-state $space))
        
     )
     (add-state $space $updated-state)
      )
)

(= (updateGoals $space $goals)
      (let* (
         ($mods (getMods $space))
         ($_ (remove-state $space))
         ($updated (state (cons goal-vector $goals) $mods))
         ($_ (add-state $updated))
         )
         (println! (mods update success 🎉))
   )
)


(= (goalWeight $state_space)
    (let* 
      (
      ($efficiency_base (* (getGoal $state_space efficiency) (- 1 (* 0.3 (getContext $state_space complexity)))))
      ($efficiency  (* $efficiency_base (+ 0.70 (* 0.60 (get$urgency)))))
      ($accuracy_base (* (getGoal $state_space accuracy) (+ 1 (* 0.3 (getContext $state_space  complexity)))))
      ($accuracy  (* $accuracy_base (+ (- 0.70 (* 0.40 $urgency)) (* 0.50 $resolution))))
      ($short_help_base (* (getGoal $state_space helper_short) (+ 0.65 (+ (* 0.55 getContext $state_space urgency) (* 0.25 (- 1.0 (getContext $state_space complexity)))))))
      ($success_moderate_base (- (+ (+ (+ (+ 0.65 (* 0.35 $threshold)) (* 0.35 $securing)) (* 0.30 $low_confidence)) (* 0.20 $complexity)) (* 0.20 $urgency)))
      ($knowledge_base (* $knowledge (- (+ (+ 0.55 (* 0.55 $complexity)) (* 0.30 $resolution))(* 0.20 $urgency))))
      ($breakthrough_base  (* $success_breakthrough (-  (+  (+   (+ 0.45 (* 0.45 $complexity))    (* 0.35 $low_confidence))  (* 0.20 $threshold)) (* 0.15 $urgency))))
      ($novelty_base (* $novelty(- (- (+ (+ (+ 0.45 (* 0.45 $complexity)) (* 0.35 $resolution))(* 0.25 $low_confidence)) (* 0.20 $threshold)) (* 0.15 $urgency))))
      ($coherence_base (* $coherence (+ (+ (+ (+ (+ 0.60 (* 0.30 $resolution)) (* 0.25 (- 1.0 $low_confidence))) (* 0.20 $threshold)) (* 0.10 (- 1.0 $urgency))) (* 0.08 (clamp01 (* (+ $valence 1.0) 0.5))))))
      ($originality_base     (* $originality (- (- (+ (+ (+ 0.45 (* 0.40 $complexity)) (* 0.30 $resolution)) (* 0.20 (clamp01 (* (+ $valence 1.0) 0.5)))) (* 0.15 $urgency)) (* 0.10 $threshold))))
      ($social_base (* $social (+ (+ (+ 0.55 (* 0.35 (- 1.0 $low_confidence))) (* 0.20 $urgency)) (* 0.10 (clamp01 (* (+ $valence 1.0) 0.5))))))
      ($beneficial_base  (* $over_beneficial (+ (+ (+ 0.60 (* 0.45 $threshold)) (* 0.40 $securing)) (* 0.30 $low_confidence))))
      ($help_long (* (getGoal $state_space help_long) (+ (+ 0.55 (* 0.65 $resolution)) (* 0.30 $complexity))))
      ($safety_base (* $over_safety (+ (+ (+ 0.60 (* 0.45 $complexity)) (* 0.55 $threshold)) (* 0.50 $securing))))
      ($honesty_base  (* $over_honesty (+ (+ 0.60 (* 0.60 $low_confidence)) (* 0.30 $threshold))))
      ($goalList ((efficiency $efficiency) ( accuracy$accuracy) ($short_help_base) ($success_moderate_base) ($knowledge_base) ($novelty_base) ($breakthrough_base) ($coherence_base) ($originality_base) ($social_base) ($beneficial_base) ($safety_base) ))
      ))
      (updateGoals $state_space $goalList)
    )



    ; Constants
(= (ACTION-BLOCKED-SCORE) -1000000000.0)
(= (ACTION-ACTIVE-FLOOR) -100000000.0)

; Math helpers
(= (abs $x) (if (< $x 0.0) (* $x -1.0) $x))

; Dictionary & Tuple helpers (designed for 2-element tuples)
(= (get-score (($k1 $v1) ($k2 $v2)) $key)
   (if (== $k1 $key) $v1
       (if (== $k2 $key) $v2 (ACTION-BLOCKED-SCORE))))

(= (update-score (($k1 $v1) ($k2 $v2)) $key $new-val)
   (if (== $k1 $key) (($k1 $new-val) ($k2 $v2))
       (if (== $k2 $key) (($k1 $v1) ($k2 $new-val))
           (($k1 $v1) ($k2 $v2)))))

(= (sort-scores (($k1 $v1) ($k2 $v2)))
   (if (>= $v1 $v2)
       (($k1 $v1) ($k2 $v2))
       (($k2 $v2) ($k1 $v1))))

(= (best-action (($k1 $v1) ($k2 $v2)))
   (if (>= $v1 $v2) $k1 $k2))

; Determine preferred action based on intent and thresholds
(= (get-preferred $intent_type $low_conf $threshold)
   (if (== $intent_type factual) act_search
       (if (== $intent_type reflective) act_think
           (if (or (>= $low_conf 0.40) (>= $threshold 0.45)) act_search act_think))))

; Main action selection logic
(= (selectAction $scores $intent_type $low_conf $threshold $intent_margin)
   (let* (
     ($search_score (get-score $scores act_search))
     ($think_score (get-score $scores act_think))
     ($diff (abs (- $search_score $think_score)))
     ($is_active (and (> $search_score (ACTION-ACTIVE-FLOOR)) 
                      (> $think_score (ACTION-ACTIVE-FLOOR))))
     ($is_tie (and $is_active (<= $diff $intent_margin)))
   )
   (if $is_tie
       (let* (
         ($preferred (get-preferred $intent_type $low_conf $threshold))
         ($new_score (+ (+ (max $search_score $think_score) $intent_margin) 0.0001))
         ($updated_scores (update-score $scores $preferred $new_score))
       ) ((best-action $updated_scores) (sort-scores $updated_scores)))
       ((best-action $scores) (sort-scores $scores))
   )))

; Example Call
; !(selectAction ((act_think 0.7) (act_search 0.72)) factual 0.35 0.44 0.03)

(= (_clamp11 $x)
  (if (and (< $x 1) (> $x -1))
     $x
     (if (> $x 1)
      1
      -1
      )
   )
)

; (= (_score_actions $actionList $weightedGaol $res)
;  (if (== $actionList ())
;     $res
;    (let*  
;    (($h (car-atom $actionList))
;     ($t (cdr-atom $actionList))
;     ($current ( $h)))
;  ($new_res (cons $res $currnt))
; (_score_actions $t $new_res)
;  ))
; )

; (= (score_helper $action $weightedGaol)
; (let* 
;  ($score (map-atom ))

; )

; )