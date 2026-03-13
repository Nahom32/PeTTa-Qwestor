
!(import! &self ../utils/general-helper)

!(bind! &sp (new-space))
; a function to normaliz by truncating a number that deviates from its range

(= (calculate-mod )
    (let* (
      ($_ (println! ("inside calculate-mod")))
      ; 1. urgency (m_urgency)
      ; ($m_urgency (_clamp01 (+ (* (- 1.0 (getAlpha urgency_alpha)) (getMod m_urgency)) (* (getAlpha resolution_alpha) (getContext complexity)))))

      ; 2. RESOLUTION (m_resolution) - depth of processing
      ($m_resolution (_clamp01 (+ (* (- 1.0 (getAlpha resolution_alpha)) (getMod m_resolution)) (* (getAlpha resolution_alpha) (getContext complexity)))))

      ; 3. USER EXPERTISE (m_user_expertise)
      ($m_user_expertise (_clamp01 (+ (* (- 1.0 (getAlpha expertise_alpha)) (getMod m_user_expertise)) (* (getAlpha expertise_alpha) (getContext expertise)))))

      ; 4. THRESHOLD (m_threshold) - confidence needed for assertions
      ($m_threshold (_clamp01 (+ (* (- 1.0 (getAlpha threshold_alpha)) (getMod m_threshold)) (* (getAlpha threshold_alpha) (getContext threshold)))))
      
      ; 5. TOPIC FAMILIARITY (m_topic_familiarity)
      ($m_topic_familiarity (_clamp01 (+ (* (- 1.0 (getAlpha familiarity_alpha)) (getMod m_topic_familiarity)) (* (getAlpha familiarity_alpha) (getContext topic_familarity)))))

      ; 6. FAILURE WARINESS (m_failure_wariness) - KEY DESIGN PATTERN
      ($m_failure_wariness (_clamp01 (+ (* (- 1.0 (getAlpha failure_decay)) (getMod m_failure_wariness)) (* (getAlpha failure_decay) (getContext failure_signal)))))

      ; 7. SECURING (m_securing) - general caution
      ($securing_target (_clamp01 (+ (* 0.5 (getContext threshold)) (+ (* 0.30 (getContext failure_signal)) (* 0.20 (getContext ambiguity))))))
      ($m_securing (_clamp01 (+ (* (- 1.0 (getAlpha securing_alpha)) (getMod m_securing)) (* (getAlpha securing_alpha) $securing_target))))

      ; 8. m_approach
      ($approach_target (_clamp01 (+ (* 0.45 (getContext complexity)) (+ (* 0.25  (- 1 (getContext ambiguity))) (+ (* 0.2 (- 1 (getContext threshold))) (* 0.10 (- 1.0 (getContext failure_signal))))))))
      ($m_approach (_clamp01 (+ (* (- 1.0 (getAlpha approach_alpha)) (getMod m_approach)) (* (getAlpha approach_alpha) $approach_target))))

      ; 9. m_arousal

      ; 10. m_risk_aversion
      ($risk_aversion_target (_clamp01 (+ (* 0.4 (getContext threshold)) (+ (* 0.3 (getContext failure_signal)) (+ (* 0.2 (getContext ambiguity)) (* 0.1 (if (== (getContext urgent) 1) 1 0)))))))
      ($m_risk_aversion (_clamp01 (+ (* (- 1 (getAlpha risk_aversion_alpha)) (getMod m_risk_aversion)) (* (getAlpha risk_aversion_alpha) $risk_aversion_target))))

      ; ; 11. m_error_tolerance
      ; ($m_error_tolerance )

      ; ; 12. m_creativity
      ; ($m_creativity )

      ; 13. m_valence
      ; ($m_valence )

      ($result (modulator-vector (m_resoultion $m_resolution) (m_user_expertise $m_user_expertise) (m_threshold $m_threshold) (m_topic_familiarity $m_topic_familiarity) (m_failure_wariness $m_failure_wariness) (m_securing $m_securing) (m_approach $m_approach) (m_risk_aversion $m_risk_aversion)))

      )
      $result
      )
)

(= (addState $space $state)
    (add-atom $space $state)
)



(= (update-mods $space $mods)
      (let* (
         ($goal (get-goals $space))
         ($_ (remove-state $space))
         ($updated (state $goal (cons modulator-vector $mods)))
         ($_ (add-state $updated))
         )
         (println! (mods update success 🎉))
   )
)

(= (getMod $modulator-name) 
(let* 
   (
    ($goal-val (match &space (state $x $y) (cdr-atom $y)))
    ($val 
    (case $modulator-name
        (
        (m_urgency (index-atom $goal-val 0))
        (m_resolution (index-atom $goal-val 1))
        (m_user_expertise (index-atom $goal-val 2))
        (m_threshold (index-atom $goal-val 3))
        (m_topic_familiarity (index-atom $goal-val 4))
        (m_failure_wariness (index-atom $goal-val 5))
        (m_securing (index-atom $goal-val 6))
        (m_approach (index-atom $goal-val 7))
        (m_arousal (index-atom $goal-val 8))
        (m_risk_aversion (index-atom $goal-val 9))
        (m_error_tolerance (index-atom $goal-val 10))
        (m_creativity (index-atom $goal-val 11))
        (m_valence (index-atom $goal-val 12))            
        )
   )))
   (index-atom $val 1)
   
))

(= (getContext $name)
 (match &space ($name $val) $val)
)

(= (getAlpha $name)
      (match &space ($name $y) $y)
)

(= (getMods $space)(match $space (state $x $y) $y))
(= (get_state $space) (match $space (state $x $y) (state $x $y)))
(= (get-goals $space)(match $space (state $x $y) $x))
(= (remove-state $space)
 (let $state (get_state $space) (remove-atom $space $state))
)


(= (addAlpha $space $list)
(if (== $list ())
    ()
 (let* (
    ($head (car-atom $list))
    ($tail (cdr-atom $list))
    ($_ (add-atom $space $head))
    )
    (addAlpha $space $tail)
 )
 ))

(= (add-context $space $list)
 
 (if (== $list ())
    ()
 (let* (
    ($head (car-atom $list))
    ($tail (cdr-atom $list))
    ($_ (add-atom $space $head))
    )
    (add-context $space $tail)
 )
 ))



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
    (valence_alpha 0.40)))       ; How fast emotional valence adapts

; defaul modulator-vector 
(= (default-mods) (
         modulator-vector 
        (m_urgency 0.2)
        (m_resolution 0.4)
        (m_user_expertise 0.5)
        (m_threshold 0.3)
        (m_topic_familiarity 0.5)
        (m_failure_wariness 0.1)
        (m_securing 0.1)
        (m_approach 0.4)
        (m_arousal 0.4)
        (m_risk_aversion 0.40)
        (m_error_tolerance 0.45)
        (m_creativity 0.45)
        (m_valence 0)
))


(= (context)(
   ; core signals 
   (complexity 0.3)
   (threshold 0.3)
   (ambiguity 0.0)
   (urgent 0.0)
   (intent_type mixed)
   ; user and topic
   (expertise 0.5)
   (topic_familarity 0.5)
   (failure_signal 0.3)
   (verfiy_request 0.3)
   ; advanced signals
   (reflective_intent None)
   (needs_external_evidence 0.3)
   (needs_task_plan 0.2)
   (needs_multi_source_integration 0.3)
   (valence 0.0)
))

(= (default-goals) (goal-vector
 (efficiency 0.60)
 (accuracy 0.70)
 (success_moderate 0.62)
 (knowledge 0.52)
 (novelty 0.46)
 (novelty 0.46)
 (success_breakthrough 0.44)
 (coherence 0.58)
 (originality 0.48)
 (social 0.58)
 (help_short 0.55)
 (help_long 0.45)
 (over_beneficial 0.60)
 (over_safety 0.65)
 (over_honesty 0.65)
))
