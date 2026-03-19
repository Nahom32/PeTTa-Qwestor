; ─────────────────────────────────────────────
;
; Goal vector accessors and weight computation.
;
; ─────────────────────────────────────────────


; ─────────────────────────────────────────────
; get-goal – read a single goal value by name.
;   match the canonical default-goals ordering.
; ─────────────────────────────────────────────
(= (get-goal $space $goal-name)
(let*
   (
    ($goal-vec  (fetch-goals $space))
    ($goal-list (cdr-atom $goal-vec))
    ($pair
    (case $goal-name
        (
        (efficiency           (index-atom $goal-list 0))
        (accuracy             (index-atom $goal-list 1))
        (success_moderate     (index-atom $goal-list 2))
        (knowledge            (index-atom $goal-list 3))
        (novelty              (index-atom $goal-list 4))
        (success_breakthrough (index-atom $goal-list 5))
        (coherence            (index-atom $goal-list 6))
        (originality          (index-atom $goal-list 7))
        (social               (index-atom $goal-list 8))
        (help_short           (index-atom $goal-list 9))
        (help_long            (index-atom $goal-list 10))
        (over_beneficial      (index-atom $goal-list 11))
        (over_safety          (index-atom $goal-list 12))
        (over_honesty         (index-atom $goal-list 13))
        )
   ))
   )
   (index-atom $pair 1)
))


; ─────────────────────────────────────────────
; getGoal – convenience alias matching the name
;           used inside goalWeight formulas.
; ─────────────────────────────────────────────
(= (getGoal $space $goal-name)
   (get-goal $space $goal-name)
)


; ─────────────────────────────────────────────
; update-goal – replace one goal's value.
; ─────────────────────────────────────────────
(= (update-goal $space $name $value)
    (let*
     (
        ($current-modulator-vec (fetch-modulators $space))
        ($current-goal (cdr-atom (fetch-goals $space)))
        ($update-goal  (case $name
        (
            (efficiency           (replaceByIndex $current-goal 0  ($name $value)))
            (accuracy             (replaceByIndex $current-goal 1  ($name $value)))
            (success_moderate     (replaceByIndex $current-goal 2  ($name $value)))
            (knowledge            (replaceByIndex $current-goal 3  ($name $value)))
            (novelty              (replaceByIndex $current-goal 4  ($name $value)))
            (success_breakthrough (replaceByIndex $current-goal 5  ($name $value)))
            (coherence            (replaceByIndex $current-goal 6  ($name $value)))
            (originality          (replaceByIndex $current-goal 7  ($name $value)))
            (social               (replaceByIndex $current-goal 8  ($name $value)))
            (help_short           (replaceByIndex $current-goal 9  ($name $value)))
            (help_long            (replaceByIndex $current-goal 10 ($name $value)))
            (over_beneficial      (replaceByIndex $current-goal 11 ($name $value)))
            (over_safety          (replaceByIndex $current-goal 12 ($name $value)))
            (over_honesty         (replaceByIndex $current-goal 13 ($name $value)))
        )))
        ($updated-state (state (cons goal-vector $update-goal) $current-modulator-vec))
        ($_ (remove-state $space))
     )
     (add-state $space $updated-state)
      )
)


; ─────────────────────────────────────────────
; updateGoals – replace the entire goal list.
; FIX BUG-1: add-state now passes $space.
; ─────────────────────────────────────────────
(= (updateGoals $space $goals)
      (let* (
         ($mods     (getMods $space))
         ($_ (remove-state $space))
         ($updated  (state (cons goal-vector $goals) $mods))
         ($_ (add-state $space $updated))    ; ← $space was missing before
         )
         (println! (goals update success 🎉))
   )
)


; ─────────────────────────────────────────────
; goalWeight – compute weighted goal values
;             from current context and modulators.
; ─────────────────────────────────────────────
(= (goalWeight $space)
    (let*
      (
        ; ── context signals ──────────────────────────────
        ($complexity       (getContext $space complexity))
        ($failure_signal   (getContext $space failure_signal))
        ($urg_ctx          (getContext $space urgent))

        ; ── modulator values ─────────────────────────────
        ($urgency          (getMod $space m_urgency))
        ($resolution       (getMod $space m_resolution))
        ($threshold        (getMod $space m_threshold))
        ($securing         (getMod $space m_securing))
        ($valence          (getMod $space m_valence))

        ; ── derived signal ───────────────────────────────
        ($low_confidence   (- 1.0 $threshold))

        ; ── current goal values ──────────────────────────
        ($efficiency_goal        (getGoal $space efficiency))
        ($accuracy_goal          (getGoal $space accuracy))
        ($success_moderate_goal  (getGoal $space success_moderate))
        ($knowledge_goal         (getGoal $space knowledge))
        ($novelty_goal           (getGoal $space novelty))
        ($breakthrough_goal      (getGoal $space success_breakthrough))
        ($coherence_goal         (getGoal $space coherence))
        ($originality_goal       (getGoal $space originality))
        ($social_goal            (getGoal $space social))
        ($help_short_goal        (getGoal $space help_short))
        ($help_long_goal         (getGoal $space help_long))
        ($over_beneficial_goal   (getGoal $space over_beneficial))
        ($over_safety_goal       (getGoal $space over_safety))
        ($over_honesty_goal      (getGoal $space over_honesty))

        ; ── valence helper  (map [-1,1] → [0,1]) ─────────
        ($valence_norm     (_clamp01 (* (+ $valence 1.0) 0.5)))

        ; ── weighted goal calculations ─────────────────────
        ($efficiency_base  (* $efficiency_goal (- 1.0 (* 0.30 $complexity))))
        ($efficiency       (* $efficiency_base (+ 0.70 (* 0.60 $urgency))))

        ($accuracy_base    (* $accuracy_goal (+ 1.0 (* 0.30 $complexity))))
        ($accuracy         (* $accuracy_base (+
                               (- 0.70 (* 0.40 $urgency))
                               (* 0.50 $resolution))))

        ($short_help       (* $help_short_goal (+
                               0.65
                               (+ (* 0.55 $urgency)
                                  (* 0.25 (- 1.0 $complexity))))))

        ($success_moderate (-
                               (+ 0.65
                                  (+ (* 0.35 $threshold)
                                     (+ (* 0.35 $securing)
                                        (+ (* 0.30 $low_confidence)
                                           (* 0.20 $complexity)))))
                               (* 0.20 $urgency)))

        ($knowledge_w      (* $knowledge_goal (-
                               (+ 0.55
                                  (+ (* 0.55 $complexity)
                                     (* 0.30 $resolution)))
                               (* 0.20 $urgency))))

        ($breakthrough_w   (* $breakthrough_goal (-
                               (+ 0.45
                                  (+ (* 0.45 $complexity)
                                     (+ (* 0.35 $low_confidence)
                                        (* 0.20 $threshold))))
                               (* 0.15 $urgency))))

        ($novelty_w        (* $novelty_goal (-
                               (+ 0.45
                                  (+ (* 0.45 $complexity)
                                     (+ (* 0.35 $resolution)
                                        (* 0.25 $low_confidence))))
                               (+ (* 0.20 $threshold)
                                  (* 0.15 $urgency)))))

        ($coherence_w      (* $coherence_goal (+
                               0.60
                               (+ (* 0.30 $resolution)
                                  (+ (* 0.25 (- 1.0 $low_confidence))
                                     (+ (* 0.20 $threshold)
                                        (+ (* 0.10 (- 1.0 $urgency))
                                           (* 0.08 $valence_norm))))))))

        ($originality_w    (* $originality_goal (-
                               (+ 0.45
                                  (+ (* 0.40 $complexity)
                                     (+ (* 0.30 $resolution)
                                        (* 0.20 $valence_norm))))
                               (+ (* 0.15 $urgency)
                                  (* 0.10 $threshold)))))

        ($social_w         (* $social_goal (+
                               0.55
                               (+ (* 0.35 (- 1.0 $low_confidence))
                                  (+ (* 0.20 $urgency)
                                     (* 0.10 $valence_norm))))))

        ($beneficial_w     (* $over_beneficial_goal (+
                               0.60
                               (+ (* 0.45 $threshold)
                                  (+ (* 0.40 $securing)
                                     (* 0.30 $low_confidence))))))

        ($help_long_w      (* $help_long_goal (+
                               0.55
                               (+ (* 0.65 $resolution)
                                  (* 0.30 $complexity)))))

        ($safety_w         (* $over_safety_goal (+
                               0.60
                               (+ (* 0.45 $complexity)
                                  (+ (* 0.55 $threshold)
                                     (* 0.50 $securing))))))

        ($honesty_w        (* $over_honesty_goal (+
                               0.60
                               (+ (* 0.60 $low_confidence)
                                  (* 0.30 $threshold)))))

        ; ── assemble ordered goal list ─────────────────────
        ($goalList (
            (efficiency          $efficiency)
            (accuracy            $accuracy)
            (success_moderate    $success_moderate)
            (knowledge           $knowledge_w)
            (novelty             $novelty_w)
            (success_breakthrough $breakthrough_w)
            (coherence           $coherence_w)
            (originality         $originality_w)
            (social              $social_w)
            (help_short          $short_help)
            (help_long           $help_long_w)
            (over_beneficial     $beneficial_w)
            (over_safety         $safety_w)
            (over_honesty        $honesty_w)
        ))
      )
      (updateGoals $space $goalList)
    )
)
