; !(import! &self ../main/state)
!(import! &self ../utils/general-helper)




!(bind! &sp (new-space))
; a function to normaliz by truncating a number that deviates from its range

; 1. extract context
; 2. calculate a modulator using alpha
; 3. update the modulator part in the state
(= (calculate-mod)
    (let* (
      ; 2. RESOLUTION (m_resolution) - depth of processing
      ($m_resolution (_clamp01 (+ (* (- 1.0 (get-alpha resolution_alpha)) (get-mod m_resolution)) (* (get-alpha resolution_alpha) (get-context complexity)))))
      ; 3. USER EXPERTISE (m_user_expertise)
      ($m_user_expertise (_clamp01 (+ (* (- 1.0 (get-alpha expertise_alpha)) (get-mod m_user_expertise)) (* (get-alpha expertise_alpha) (get-context expertise)))))
      ; 4. THRESHOLD (m_threshold) - confidence needed for assertions
      ($m_threshold (_clamp01 (+ (* (- 1.0 (get-alpha threshold_alpha)) (get-mod m_threshold)) (* (get-alpha threshold_alpha) (get-context threshold)))))
      ; ; 5. TOPIC FAMILIARITY (m_topic_familiarity)
      ($m_topic_familiarity (_clamp01 (+ (* (- 1.0 (get-alpha familiarity_alpha)) (get-mod m_topic_familiarity)) (* (get-alpha familiarity_alpha) (get-context topic_familarity)))))
      ; ; 6. FAILURE WARINESS (m_failure_wariness) - KEY DESIGN PATTERN
      ($m_failure_wariness (_clamp01 (+ (* (- 1.0 (get-alpha failure_decay)) (get-mod m_failure_wariness)) (* (get-alpha failure_decay) (get-context failure_signal)))))
      ; ; 7. SECURING (m_securing) - general caution
      ; ($securing_target (_clamp01 (+ (* 0.5 (get-context threshold) (+ (* 0.30 (get-context failure_signal)) (* 0.20 (get-context ambiguity)))))))
      ; ($m_securing (_clamp01 (_clamp01 (+ (* (- 1.0 (get-alpha securing_alpha)) (get-mod m_securing)) (* (get-alpha securing_alpha) $securing_target)))))
      ($result (modulator-vector (m_resoultion $m_resolution) (m_user_expertise $m_user_expertise) (m_threshold $m_threshold) (m_topic_familiarity $m_topic_familiarity) (m_failure_wariness $m_failure_wariness)))

      )
      $result
      )
)

(= (add-state $space $state)
    (add-atom $space $state)
)

(= (remove-state $space)
 (let $state (fetch-state $space) (remove-atom $space $state))
)

(= (update-modulator $space $name $value)
    (let*
     (
        ($current-goal-vec (get-goals $space))
        ($current-modulator (cdr-atom (get-mods $space)))
        ($update-modulator  (case $name 
        ( 
        (m_urgency (replaceByIndex $current-modulator 0 ($name $value)))
        (m_resolution (replaceByIndex $current-modulator 1 ($name $value)))
        (m_user_expertise (replaceByIndex $current-modulator 2 ($name $value)))
        (m_threshold (replaceByIndex $current-modulator 3 ($name $value)))
        (m_topic_familiarity (replaceByIndex $current-modulator 4 ($name $value)))
        (m_failure_wariness (replaceByIndex $current-modulator 5 ($name $value)))
        (m_securing (replaceByIndex $current-modulator 6 ($name $value)))
        (m_approach (replaceByIndex $current-modulator 7 ($name $value)))
        (m_arousal (replaceByIndex $current-modulator 8 ($name $value)))
        (m_risk_aversion (replaceByIndex $current-modulator 9 ($name $value)))
        (m_error_tolerance (replaceByIndex $current-modulator 10 ($name $value)))
        (m_creativity (replaceByIndex $current-modulator 11 ($name $value)))
        (m_valence (replaceByIndex $current-modulator 12 ($name $value)))            
        
        )))
        ; here the function goes through the logic of monadic(D) computation using different formulas and logic but it is to show how the space works 
        ($updated-state (state $current-goal-vec (cons modulator-vector $update-modulator)))
        ($_ (remove-state $space))
        ($_ (add-state $space $updated-state))
        
     )
      (println! (update success 🎉))
      )
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

(= (get-mod $modulator-name) 
(let* 
   (
    ($goal-val (match &sp (state $x $y) (cdr-atom $y)))
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

(= (get-context $name)
 (match &sp ($name $val) $val)
)

(= (get-alpha $name)
      (match &sp ($name $y) $y)
)

(= (get-mods $space)(match $space (state $x $y) $y))
(= (get_state $space) (match $space (state $x $y) (state $x $y)))
(= (get-goals $space)(match $space (state $x $y) $x))
(= (remove-state $space)
 (let $state (get_state $space) (remove-atom $space $state))
)


(= (add-alpha $list)
 
 (if (== $list ())
    (println! ("alpha added"))
 (let* (
    ($head (car-atom $list))
    ($tail (cdr-atom $list))
    ($_ (add-atom &sp $head))
    )
    (add-alpha $tail)
 )
 ))

(= (add-context  $list)
 
 (if (== $list ())
     (println! ("context added"))
 (let* (
    ($head (car-atom $list))
    ($tail (cdr-atom $list))
    ($_ (add-atom &sp $head))
    )
    (add-context $tail)
 )
 ))



; context
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
(= (default-mods) (modulator-vector 
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
   (integerity 0.2) 
   (energy 0.4) 
   (competence 0.5) 
   (affiliation 0.7) 
))


!(add-alpha (alpha))
!(add-state &sp (state (default-goals) (default-mods)))


!(add-context (context)) 
; !(+ (* (- 1.0 (get-alpha resolution_alpha)) (get-mod m_resolution)) (* (get-alpha resolution_alpha) (get-context complexity)))
; !(_clamp01 (+ (* (- 1.0 (get-alpha resolution_alpha)) (get-mod m_resolution)) (* (get-alpha resolution_alpha) (get-context complexity))))
; !(get-mods &sp)
; !(update-modulator &sp m_resolution 0.9)
; !(update-goal &sp energy 0.8)
; !(get_state &sp)
; !(update-mods &sp ((m_urgency 0.0) (m_resolution 0.9) (m_user_expertise 0.5) (m_threshold 0.3) (m_topic_familiarity 0.5) (m_failure_wariness 0.1) (m_securing 0.1) (m_approach 0.4) (m_arousal 0.4) (m_risk_aversion 0.4) (m_error_tolerance 0.45) (m_creativity 0.45) (m_valence 0)))
; !(get_state &sp)

; !(get-context valence)
!(calculate-mod)
