; ─────────────────────────────────────────────
; appraisal.metta
;
; Defines modulators, alpha hyper-parameters,
; goal/context defaults, and the calculate-mod
; update function.
;
; ─────────────────────────────────────────────


; ─────────────────────────────────────────────
; Space-parameterised accessors
; ─────────────────────────────────────────────

(= (getMods $space)  (match $space (state $x $y) $y))
(= (get-goals $space)(match $space (state $x $y) $x))
(= (get_state $space)(match $space (state $x $y) (state $x $y)))

(= (remove-state $space)
 (let $state (get_state $space) (remove-atom $space $state))
)

; Retrieve the value of a single modulator by name.
(= (getMod $space $modulator-name)
    (let*
      (
        ($mod-vec   (match $space (state $x $y) $y))
        ($mod-list  (cdr-atom $mod-vec))
        ($pair      (matchValue $modulator-name $mod-list))
      )
      (index-atom $pair 1)
    )
)

; Map a modulator name to the element at its canonical index.
(= (matchValue $modulator-name $mod-list)
    (case $modulator-name
          (
            (m_urgency           (index-atom $mod-list 0))
            (m_resolution        (index-atom $mod-list 1))
            (m_user_expertise    (index-atom $mod-list 2))
            (m_threshold         (index-atom $mod-list 3))
            (m_topic_familiarity (index-atom $mod-list 4))
            (m_failure_wariness  (index-atom $mod-list 5))
            (m_securing          (index-atom $mod-list 6))
            (m_approach          (index-atom $mod-list 7))
            (m_arousal           (index-atom $mod-list 8))
            (m_risk_aversion     (index-atom $mod-list 9))
            (m_error_tolerance   (index-atom $mod-list 10))
            (m_creativity        (index-atom $mod-list 11))
            (m_valence           (index-atom $mod-list 12))
            ($_ "value not found")
          )
      )
)

; Retrieve a single context signal by name.
(= (getContext $space $name)
 (match $space ($name $val) $val)
)

; Retrieve a single alpha hyper-parameter by name.
(= (getAlpha $space $name)
      (match $space ($name $y) $y)
)


; ─────────────────────────────────────────────
; addState / addItems / addAlpha / add-context
; ─────────────────────────────────────────────

(= (addState $space $state)
    (add-atom $space $state)
)

(= (addItems $baseCase $space $items)
   (if (== $items ())
        $baseCase
        (let* (
               ($h (car-atom $items))
               ($t (cdr-atom $items))
               ($_ (add-atom $space $h))
               )
          (addItems $baseCase $space $t)
          )
     )
)

(= (addAlpha $space $list)    (addItems () $space $list))
(= (add-context $space $list) (addItems () $space $list))


; ─────────────────────────────────────────────
; update-mods – replace the whole modulator vector.
; ─────────────────────────────────────────────
(= (update-mods $space $mods)
      (let* (
         ($goal     (get-goals $space))
         ($_ (remove-state $space))
         ($updated  (state $goal (cons modulator-vector $mods)))
         ($_ (add-state $space $updated))          
         )
         (println! (mods update success 🎉))
   )
)


; ─────────────────────────────────────────────
; calculate-mod
; Computes the new modulator vector via
; incremental blending:  new = (1-α)·old + α·target
; ─────────────────────────────────────────────
(= (calculate-mod $space)
    (let*
      (
        ($_ (println! ("inside calculate-mod")))

        ; ── convenience shorthands ─────────────────────────
        ($cx   (getContext $space complexity))
        ($amb  (getContext $space ambiguity))
        ($thr  (getContext $space threshold))
        ($fs   (getContext $space failure_signal))
        ($exp  (getContext $space expertise))
        ($fam  (getContext $space topic_familiarity))
        ($ri   (getContext $space reflective_intent))
        ($urg  (getContext $space urgent))
        ($val  (getContext $space valence))

        ($target_u (if $urg 1.0 0.0))

        ; ── 1. m_urgency ────────────────────────────────────
        ($m_urgency
          (_clamp01 (+
            (* (- 1.0 (getAlpha $space urgency_alpha))    (getMod $space m_urgency))
            (* (getAlpha $space urgency_alpha)            $target_u))))

        ; ── 2. m_resolution ─────────────────────────────────
        ($m_resolution
          (_clamp01 (+
            (* (- 1.0 (getAlpha $space resolution_alpha)) (getMod $space m_resolution))
            (* (getAlpha $space resolution_alpha)         $cx))))

        ; ── 3. m_user_expertise ─────────────────────────────
        ($m_user_expertise
          (_clamp01 (+
            (* (- 1.0 (getAlpha $space expertise_alpha))  (getMod $space m_user_expertise))
            (* (getAlpha $space expertise_alpha)          $exp))))

        ; ── 4. m_threshold ──────────────────────────────────
        ($m_threshold
          (_clamp01 (+
            (* (- 1.0 (getAlpha $space threshold_alpha))  (getMod $space m_threshold))
            (* (getAlpha $space threshold_alpha)          $thr))))

        ; ── 5. m_topic_familiarity ──────────────────────────
        ($m_topic_familiarity
          (_clamp01 (+
            (* (- 1.0 (getAlpha $space familiarity_alpha)) (getMod $space m_topic_familiarity))
            (* (getAlpha $space familiarity_alpha)         $fam))))

        ; ── 6. m_failure_wariness ───────────────────────────
        ($m_failure_wariness
          (_clamp01 (+
            (* (- 1.0 (getAlpha $space failure_decay))    (getMod $space m_failure_wariness))
            (* (getAlpha $space failure_decay)            $fs))))

        ; ── 7. m_securing ───────────────────────────────────
        ($securing_target
          (_clamp01 (+
            (* 0.50 $thr)
            (+ (* 0.30 $fs)
               (* 0.20 $amb)))))
        ($m_securing
          (_clamp01 (+
            (* (- 1.0 (getAlpha $space securing_alpha))   (getMod $space m_securing))
            (* (getAlpha $space securing_alpha)           $securing_target))))

        ; ── 8. m_approach ───────────────────────────────────
        ($approach_target
          (_clamp01 (+
            (* 0.45 $cx)
            (+ (* 0.25 (- 1.0 $amb))
               (+ (* 0.20 (- 1.0 $thr))
                  (* 0.10 (- 1.0 $fs)))))))
        ($m_approach
          (_clamp01 (+
            (* (- 1.0 (getAlpha $space approach_alpha))   (getMod $space m_approach))
            (* (getAlpha $space approach_alpha)           $approach_target))))

        ; ── 9. m_arousal ────────────────────────────────────
        ($novelty_signal
          (_clamp01 (+
            (* 0.35 $cx)
            (+ (* 0.35 (- 1.0 $fam))
               (+ (* 0.20 $ri)
                  (* 0.10 $amb))))))
        ($arousal_target
          (_clamp01 (+
            0.25
            (+ (* 0.40 $m_urgency)
               (+ (* 0.35 $novelty_signal)
                  (* 0.20 $cx))))))
        ($m_arousal
          (_clamp01 (+
            (* (- 1.0 (getAlpha $space arousal_alpha))    (getMod $space m_arousal))
            (* (getAlpha $space arousal_alpha)            $arousal_target))))

        ; ── 10. m_risk_aversion ─────────────────────────────
        ($risk_aversion_target
          (_clamp01 (+
            (* 0.40 $thr)
            (+ (* 0.30 $fs)
               (+ (* 0.20 $amb)
                  (* 0.10 (if (== $urg 1) 1 0)))))))
        ($m_risk_aversion
          (_clamp01 (+
            (* (- 1.0 (getAlpha $space risk_aversion_alpha)) (getMod $space m_risk_aversion))
            (* (getAlpha $space risk_aversion_alpha)         $risk_aversion_target))))

        ; ── 11. m_error_tolerance ───────────────────────────
        ($error_tolerance_target
          (_clamp01 (+
            0.45
            (+ (* 0.25 (- 1.0 $thr))
               (+ (* 0.20 $fam)
                  (+ (* 0.15 (- 1.0 $fs))
                     (- (* -0.20 $amb)
                        (* 0.10 (if (== $urg 1) 1 0)))))))))
        ($m_error_tolerance
          (_clamp01 (+
            (* (- 1.0 (getAlpha $space error_tolerance_alpha)) (getMod $space m_error_tolerance))
            (* (getAlpha $space error_tolerance_alpha)         $error_tolerance_target))))

        ; ── 12. m_creativity ────────────────────────────────
        ($creativity_target
          (_clamp01 (+
            0.30
            (+ (* 0.30 (- 1.0 $fam))
               (+ (* 0.20 $cx)
                  (+ (* 0.15 (- 1.0 $thr))
                     (- (* 0.20 $ri)
                        (* 0.10 $fs))))))))
        ($m_creativity
          (_clamp01 (+
            (* (- 1.0 (getAlpha $space creativity_alpha))     (getMod $space m_creativity))
            (* (getAlpha $space creativity_alpha)             $creativity_target))))

        ; ── 13. m_valence ───────────────────────────────────
        ($valence_target
          (_clamp11 (- $val (* 0.10 $fs))))
        ($m_valence
          (_clamp11 (+
            (* (- 1.0 (getAlpha $space valence_alpha))        (getMod $space m_valence))
            (* (getAlpha $space valence_alpha)                $valence_target))))
      )

      ; ── return the full updated modulator-vector ──────────
      (modulator-vector
        (m_urgency           $m_urgency)
        (m_resolution        $m_resolution)
        (m_user_expertise    $m_user_expertise)
        (m_threshold         $m_threshold)
        (m_topic_familiarity $m_topic_familiarity)
        (m_failure_wariness  $m_failure_wariness)
        (m_securing          $m_securing)
        (m_approach          $m_approach)
        (m_arousal           $m_arousal)
        (m_risk_aversion     $m_risk_aversion)
        (m_error_tolerance   $m_error_tolerance)
        (m_creativity        $m_creativity)
        (m_valence           $m_valence))
    )
)


; ─────────────────────────────────────────────
; Alpha hyper-parameters (learning rates)
; ─────────────────────────────────────────────
(= (alpha) (
    (urgency_alpha        0.60)
    (resolution_alpha     0.45)
    (expertise_alpha      0.45)
    (threshold_alpha      0.40)
    (familiarity_alpha    0.35)
    (failure_alpha        0.55)
    (failure_decay        0.20)
    (securing_alpha       0.45)
    (approach_alpha       0.40)
    (arousal_alpha        0.40)
    (risk_aversion_alpha  0.45)
    (error_tolerance_alpha 0.40)
    (creativity_alpha     0.40)
    (valence_alpha        0.40)))


; ─────────────────────────────────────────────
; Default modulator vector (13 modulators)
; ─────────────────────────────────────────────
(= (default-mods) (
         modulator-vector
        (m_urgency           0.20)
        (m_resolution        0.40)
        (m_user_expertise    0.50)
        (m_threshold         0.30)
        (m_topic_familiarity 0.50)
        (m_failure_wariness  0.10)
        (m_securing          0.10)
        (m_approach          0.40)
        (m_arousal           0.40)
        (m_risk_aversion     0.40)
        (m_error_tolerance   0.45)
        (m_creativity        0.45)
        (m_valence           0.00)
))


; ─────────────────────────────────────────────
; Default context values
; ─────────────────────────────────────────────
(= (context)(
   ; core signals
   (complexity              0.30)
   (threshold               0.30)
   (ambiguity               0.00)
   (urgent                  0.00)
   (intent_type             mixed)
   ; user and topic
   (expertise               0.50)
   (topic_familiarity       0.50)   
   (failure_signal          0.30)
   (verify_request          False)  
   ; advanced signals
   (reflective_intent       0.50)
   (needs_external_evidence 0.30)
   (needs_task_plan         0.20)
   (needs_multi_source_integration 0.30)
   (valence                 0.00)
))


; ─────────────────────────────────────────────
; Default goal vector
; ─────────────────────────────────────────────
(= (default-goals) (goal-vector
 (efficiency           0.60)
 (accuracy             0.70)
 (success_moderate     0.62)
 (knowledge            0.52)
 (novelty              0.46)           
 (success_breakthrough  0.44)          
 (coherence            0.58)
 (originality          0.48)
 (social               0.58)
 (help_short           0.55)
 (help_long            0.45)
 (over_beneficial      0.60)
 (over_safety          0.65)
 (over_honesty         0.65)
))
