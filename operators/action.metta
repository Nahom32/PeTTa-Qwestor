!(import! &self appraisal)
(= (Actions $space)
 ((AG act_respond (
        (efficiency 1.00)
        (accuracy (eval (_clamp01 (- 1.00 (* 1.10 (getContext $space complexity))))))
        (success_moderate  (eval (_clamp01 (- 0.80 (* 0.20 (getContext $space complexity))))))
        (knowledge  (eval (_clamp01 (+ 0.28 (* 0.18 (getContext $space complexity))))))
        (novelty (eval (_clamp01 (+ 0.18 (* 0.10 (getContext $space complexity))))))
        (success_breakthrough (eval (_clamp01 (+ 0.18  (* 0.12  (getContext $space complexity))))))
        (coherence (eval (_clamp01 (- 0.72 (* 0.10 (getContext $space complexity))))))
        (originality (eval (_clamp01 (+ 0.16 (* 0.10 (getContext $space complexity))))))
        (social (eval (_clamp01 (+ 0.74 (* 0.08 (- 1.0 (getContext $space complexity)))))))
        (help_short (eval (_clamp01 (- 0.95 (* 0.20 (getContext $space complexity))))))
        (help_long (eval (_clamp01 (+ 0.25 (* 0.20 (getContext $space complexity))))))
        (over_beneficial (eval (_clamp01 (- 0.45 (* 0.15 (getContext $space complexity))))))
        (over_safety (eval (_clamp01 (- 0.45 (* 0.20 (getContext $space complexity))))))
        (over_honesty 0.60)

 ))
 (AG act_clarify (
       (efficiency 0.65)
       (accuracy (eval (_clamp01 (+ 0.55 (* 0.25 (getContext $space complexity))))))
       (success_moderate 0.72)
       (knowledge (eval (_clamp01 (+ 0.45 (* 0.20 (getContext $space complexity))))))
       (novelty (eval (_clamp01 (+ 0.28 (* 0.12 (getContext $space complexity))))))
       (success_breakthrough (eval (_clamp01 (+ 0.28 (* 0.12 (getContext $space complexity))))))
       (coherence 0.82)
       (originality (eval (_clamp01 (+ 0.18 (* 0.10 (getContext $space complexity))))))
       (social  0.95)
       (help_short (eval (_clamp01 (+ 0.55 (* 0.10 (- 1.0 (getContext $space complexity)))))))
       (help_long (eval (_clamp01 (+ 0.40 (* 0.20 (getContext $space complexity))))))
       (over_beneficial 0.85)
       (over_safety 0.90)
       (over_honesty 0.95)
 ))

 (AG act_search (
    (efficiency 0.25)
    (accuracy (eval (_clamp01 (+ 0.30 (* 0.90 (getContext $space complexity))))))
    (success_moderate (eval (_clamp01 (+ 0.55 (* 0.20 (getContext $space complexity))))))
    (knowledge (eval (_clamp01 (+ 0.68 (* 0.22 (getContext $space complexity))))))
    (novelty (eval (_clamp01 (+ 0.58 (* 0.18 (getContext $space complexity))))))
    (success_breakthrough (eval (_clamp01 (+ 0.45 (* 0.18 (getContext $space complexity))))))
    (coherence 0.58)
    (originality (eval (_clamp01 (+ 0.48 (* 0.16 (getContext $space complexity))))))
    (social (eval (_clamp01 (+ 0.42 (* 0.12 (- 1.0 (getContext $space complexity)))))))
    (help_short (eval (_clamp01 (+ 0.35 (* 0.10 (- 1.0 (getContext $space complexity)))))))
    (help_long (eval (_clamp01 (+ 0.55 (* 0.35 (getContext $space complexity))))))
    (over_beneficial 0.72)
    (over_safety 0.78)
    (over_honesty 0.82)

 ))
(AG act_verify (
    (efficiency 0.35)
    (accuracy (eval (_clamp01 (+ 0.75 (* 0.20 (getContext $space complexity))))))
    (success_moderate 0.90)
    (knowledge (eval (_clamp01 (+ 0.62 (* 0.15 (getContext $space complexity))))))
    (novelty (eval (_clamp01 (+ 0.25 (* 0.08 (getContext $space complexity))))))
    (success_breakthrough (eval (_clamp01 (+ 0.38 (* 0.10 (getContext $space complexity))))))
    (coherence 0.86)
    (originality (eval (_clamp01 (+ 0.24 (* 0.08 (getContext $space complexity))))))
    (social 0.88)
    (help_short (eval (_clamp01 (+ 0.40 (* 0.10 (- 1.0 (getContext $space complexity)))))))
    (help_long (eval (_clamp01 (+ 0.50 (* 0.20 (getContext $space complexity))))))
    (over_beneficial 0.96)
    (over_safety 0.97)
    (over_honesty 0.97)
))
(AG act_decompose (
    (efficiency 0.45)
    (accuracy (eval (_clamp01 (+ 0.55 (* 0.35 (getContext $space complexity))))))
    (success_moderate (eval (_clamp01 (+ 0.65 (* 0.15 (getContext $space complexity))))))
    (knowledge (eval (_clamp01 (+ 0.72 (* 0.20 (getContext $space complexity))))))
    (novelty (eval (_clamp01 (+ 0.62 (* 0.18 (getContext $space complexity))))))
    (success_breakthrough (eval (_clamp01 (+ 0.62 (* 0.22 (getContext $space complexity))))))
    (coherence 0.80)
    (originality (eval (_clamp01 (+ 0.66 (* 0.18 (getContext $space complexity))))))
    (social 0.72)
    (help_short (eval (_clamp01 (+ 0.30 (* 0.05 (- 1.0 (getContext $space complexity)))))))
    (help_long (eval (_clamp01 (+ 0.70 (* 0.25 (getContext $space complexity))))))
    (over_beneficial 0.70)
    (over_safety 0.76)
    (over_honesty 0.80)
))

(AG act_think (
    (efficiency 0.40)
    (accuracy (eval (_clamp01 (+ 0.60 (* 0.25 (getContext $space complexity))))))
    (success_moderate (eval (_clamp01 (+ 0.45 (* 0.20 (getContext $space complexity))))))
    (knowledge (eval (_clamp01 (+ 0.66 (* 0.18 (getContext $space complexity))))))
    (novelty (eval (_clamp01 (+ 0.70 (* 0.18 (getContext $space complexity))))))
    (success_breakthrough (eval (_clamp01 (+ 0.68 (* 0.20 (getContext $space complexity))))))
    (coherence 0.74)
    (originality (eval (_clamp01 (+ 0.74 (* 0.16 (getContext $space complexity))))))
    (social 0.58)
    (help_short (eval (_clamp01 (+ 0.35 (* 0.10 (- 1.0 (getContext $space complexity)))))))
    (help_long (eval (_clamp01 (+ 0.60 (* 0.25 (getContext $space complexity))))))
    (over_beneficial 0.78)
    (over_safety 0.84)
    (over_honesty 0.90)
))

(AG act_synthesize (
    (efficiency 0.30)
    (accuracy (eval (_clamp01 (+ 0.74 (* 0.12 (getContext $space complexity))))))
    (success_moderate 0.82)
    (knowledge (eval (_clamp01 (+ 0.78 (* 0.16 (getContext $space complexity))))))
    (novelty (eval (_clamp01 (+ 0.56 (* 0.12 (getContext $space complexity))))))
    (success_breakthrough (eval (_clamp01 (+ 0.54 (* 0.14 (getContext $space complexity))))))
    (coherence 0.84)
    (originality (eval (_clamp01 (+ 0.82 (* 0.12 (getContext $space complexity))))))
    (social 0.68)
    (help_short (eval (_clamp01 (+ 0.42 (* 0.08 (- 1.0 (getContext $space complexity)))))))
    (help_long (eval (_clamp01 (+ 0.72 (* 0.18 (getContext $space complexity))))))
    (over_beneficial 0.90)
    (over_safety 0.92)
    (over_honesty 0.95)
 )
 )
 ))


 (= (ActionList $space)
    (collapse (match $space (AG $x $y) $x))
 )
 (= (getAction $space $aciton)
    (match $space (AG $aciton $effects) $effects)
 )
 (= (bindAG $space) 
  (add-atom $space (Actions))
 )

(= (addAction $space $action)
   (trace! "action added into space" (add-atom $space $action))
)
(= (addActions $space $aglist)
(if (== $aglist ())
  (completed)
    (let* (($h (car-atom $aglist))
          ($t (cdr-atom $aglist))
          ($_ (addAction $space $h))
          )
        (addActions $space $t)
            
    )
))



(= (_hallucination_penalty $action $cx $ambiguity)
    (let*
      (($action_list ((act_respond 0.90)
                     (act_search 0.30)
                     (act_verify 0.12)
                     (act_clarify 0.15)
                     (act_decompose 0.40)
                     (act_think 0.22)
                     (act_synthesize 0.20)))
        ($base  (if (== (find_by_key $action $action_list) ())
                                0.50
                                (find_by_key $action $action_list)))

         ($result (case $action (
             (act_respond (+ $base (+ (* 0.25 $cx) (* 0.20 $ambiguity))))
             (act_search  (+ $base (* 0.10 $ambiguity)))
             (act_decompose  (+ $base (* 0.10 $cx)))
         )))
        )
        _clamp01($result)
      
    ))


(= (_redundancy_penalty $action $cx $familiarity $urgency)
     (let*
      (($action_list (
                     (act_search 0.42)
                     (act_respond 0.35)
                     (act_verify 0.30)
                     (act_clarify 0.18)
                     (act_decompose 0.72)
                     (act_think 0.82)
                     (act_synthesize 0.26)
                     
                     ))
      ($base  (if (== (find_by_key $action $action_list) ())
                                0.35
                        (find_by_key $action $action_list))) 
         )
            (if (== $action act_respond) 

                    (_clamp01 (+ 0.45 (* 0.25 (- 1.0 $cx)) (* 0.15 $familiarity) (* 0.10 (- 1.0 $urgency))))

                    $base
                  ))
         )



(= (_premature_penalty $action $cx $ambiguity $threshold)
     (let*
      (($action_list (
                     (act_search 0.20)
                     (act_verify 0.08)
                     (act_clarify 0.12)
                     (act_decompose 0.10)
                     (act_think 0.15)
                     (act_synthesize 0.06)
                     ))
      ($base  (if (== (find_by_key $action $action_list) ())
                                0.20
                        (find_by_key $action $action_list))))
                        
         (if (== $action act_respond) 

                    (_clamp01(+ (+ 0.40 (* 0.35 $cx))  (+ (* 0.25 $ambiguity) (* 0.20  $threshold))))
                    $base
             )
         ))



(= (action_score_wraper $space $goalWeight $action_list $scored_action)
 (let* (
   ($turn_count (getState turn_count))
   ($appraisal_list (update-mod $space))
   ($cx (find_by_key cx $appraisal_list))
   ($ambiguity (find_by_key ambiguity $appraisal_list))
   ($threshold_signal (find_by_key threshold_signal $appraisal_list))
   ($familiarity_signal (find_by_key familiarity_signal $appraisal_list))
   ($failure_signal (find_by_key failure_signal $appraisal_list))
   ($urgent_flag (find_by_key urgent_flag $appraisal_list))
   ($intent_type (find_by_key intent_type $appraisal_list))
   ($verify_request (find_by_key verify_request $appraisal_list))
   ($reflective_intent (find_by_key reflective_intent $appraisal_list))
   ($needs_external_evidence (find_by_key needs_external_evidence $appraisal_list))
   ($needs_task_plan (find_by_key needs_task_plan $appraisal_list))
   ($needs_multi_source_integration (find_by_key needs_task_plan $appraisal_list))
   ($u (find_by_key u $appraisal_list))
   ($res (find_by_key res $appraisal_list))
   ($ux (find_by_key ux $appraisal_list))
   ($threshold (find_by_key m_threshold $appraisal_list))
   ($familiarity (find_by_key familiarity $appraisal_list))
   ($securing (find_by_key securing $appraisal_list))
   ($approach (find_by_key approach $appraisal_list))
   ($arousal (find_by_key arousal $appraisal_list))
   ($risk_aversion (find_by_key risk_aversion $appraisal_list))
   ($error_tolerance (find_by_key error_tolerance $appraisal_list))
   ($creativity (find_by_key creativity $appraisal_list))
   ($confidence  (_clamp01((+ (* 0.55  $familiarity)  (+ (* 0.25  (- 1.0  $ambiguity)) (* 0.20  (- 1.0  $cx)))))))
   ($low_confidence (_clamp01(- 1.0  $confidence)))
   ($answerability (_clamp01((* (- 1.0 $ambiguity) (* (- 1.0 $threshold_signal)  $familiarity_signal)))))
   
 )
 (action_score_wraper 
 $space 
 $goalWeight
 $action_list
 $scored_action
 $turn_count
   $cx
   $ambiguity
   $threshold_signal
   $familiarity_signal
   $failure_signal
   $urgent_flag
   $intent_type
   $verify_request
   $reflective_intent
   $needs_external_evidence
   $needs_task_plan
   $needs_multi_source_integration
   $u
   $res
   $ux
   $threshold
   $familiarity
   $securing
   $approach
   $arousal
   $risk_aversion
   $error_tolerance
   $creativity
   $confidence
   $low_confidence
   $answerability
 )

))

(= (action_score $space $goalWeight $action_list $scored_action)
   (if (= $action_list ())
     $scored_action
     (let*
        ($score 0.0)
        (($h $t)  (decons-atom $action_list))
        (case $h
        (act_clarify
  (+ $score
     (+ (* 0.90  $ambiguity)
        (+ (* -0.35 $ux)
           (+ (* -0.15 $u)
              (+ (* 0.20 $threshold)
                 (+ (* 0.20 $securing)
                    (+ (* 0.10 $coherence)
                       (+ (* -0.08 $valence)
                          (+ (* 0.22 $social)
                             (+ (* -0.06 $originality)
                                (+ (* 0.08 (- 1.0 $error_tolerance))
                                   (+ (* -0.55 $answerability)
                                      (+ (* -0.20 $help_short)
                                         (+ (* -0.15 $anti_redundant)
                                            (if (and (> $ambiguity 0.75)
                                                     (or (> $threshold_signal 0.55)
                                                         (> $low_confidence 0.45)))
                                                0.18
                                                0.0)
)))))))))))))))
   (act_respond
  (+ $score
     (+ (* 0.35 $u)
        (+ (* 0.25 (- 1.0 $ambiguity))
           (+ (* 0.15 $ux)
              (+ (* -0.20 $cx)
                 (+ (* 0.20 $familiarity)
                    (+ (* -0.35 $threshold)
                       (+ (* -0.30 $failure_wariness)
                          (+ (* -0.35 $securing)
                             (+ (* -0.20 $low_confidence)
                                (+ (* 0.10 (- 1.0 $arousal))
                                   (+ (* 0.12 $coherence)
                                      (+ (* 0.10 $valence)
                                         (+ (* 0.14 $social)
                                            (+ (* -0.06 $originality)
                                               (+ (* -0.18 $risk_aversion)
                                                  (+ (* 0.30 $help_short)
                                                     (+ (* -0.15 $help_long)
                                                        (+ (* 0.45 $answerability)
                                                           (+ (* 0.22 $error_tolerance)
                                                              (+ (* 0.16 $help_short)
                                                                 (+ (* 0.12 $anti_redundant)
                                                                    (if (>= $cx 0.50)
                                                                        (+ (* -0.08 $knowledge)
                                                                           (* -0.10 $success_breakthrough))
                                                                        0.0)
))))))))))))))))))))
         
        
    
     )
   )
   
   ) 

(act_search
  (+ $score
     (+ (* 0.35 $cx)
        (+ (* 0.20 $res)
           (+ (* -0.15 $u)
              (+ (* 0.35 $threshold)
                 (+ (* 0.35 (- 1.0 $familiarity))
                    (+ (* 0.30 $failure_wariness)
                       (+ (* 0.15 $securing)
                          (+ (* 0.08 $arousal)
                             (+ (* 0.06 $coherence)
                                (+ (* 0.02 $valence)
                                   (+ (* 0.10 $originality)
                                      (+ (* 0.06 $social)
                                         (+ (* 0.08 (- 1.0 $risk_aversion))
                                            (+ (* 0.10 (- 1.0 $error_tolerance))
                                               (+ (* 0.10 $creativity)
                                                  (+ (* 0.06 $help_long)
                                                     (+ (* -0.08 $help_short)
                                                        (+ (* 0.14 $knowledge)
                                                           (+ (* 0.12 $novelty)
                                                              (+ (* 0.08 $success_breakthrough)
                                                                 (+ (* 0.50 $needs_external_evidence)
                                                                    (+ (* 0.12 $needs_multi_source_integration)
                                                                      (+ (* -0.08 $needs_task_plan) (* reflective_search_penalty reflective_intent))
))))))))))))))))))))))))
(act_verify
  (+ $score
     (+ (* 0.65 $threshold)
        (+ (* 0.75 $low_confidence)
           (+ (* 0.35 $failure_wariness)
              (+ (* 0.15 $cx)
                 (+ (* -0.20 $u)
                    (+ (* -0.10 $ambiguity)
                       (+ (* 0.30 $securing)
                          (+ (* 0.14 $coherence)
                             (+ (* -0.14 $valence)
                                (+ (* 0.10 $social)
                                   (+ (* -0.08 $originality)
                                      (+ (* 0.25 $risk_aversion)
                                         (+ (* -0.08 $arousal)
                                            (+ (* 0.55 (- 1.0 $error_tolerance))
                                               (+ (* 0.08 (- 1.0 $creativity))
                                                  (+ (* 0.08 $help_long)
                                                     (+ (* -0.10 $help_short)
                                                        (+ (* 0.32 (if $verify_request 1.0 0.0))
                                                           (* 0.05 $knowledge)
))))))))))))))))))))

(act_decompose
  (+ $score
     (+ (* 0.30 $cx)
        (+ (* 0.30 $res)
           (+ (* 0.10 (- 1.0 $ambiguity))
              (+ (* -0.12 $u)
                 (+ (* -0.28 $ambiguity)
                    (+ (if (and (>= $cx 0.60)
                                (<= $ambiguity 0.60))
                           0.10
                           0.0)
                       (+ (if (< $cx 0.35)
                              -0.35
                              0.0)
                          (+ (* 0.10 $approach)
                             (+ (* 0.10 $arousal)
                                (+ (* 0.10 $coherence)
                                   (+ (* 0.04 $valence)
                                      (+ (* 0.12 $originality)
                                         (+ (* 0.08 $social)
                                            (+ (* 0.08 $creativity)
                                               (+ (* -0.08 (- 1.0 $error_tolerance))
                                                  (+ (* 0.12 $help_long)
                                                     (+ (* -0.12 $help_short)
                                                        (+ (* 0.08 $knowledge)
                                                           (+ (* 0.06 $novelty)
                                                              (+ (* 0.10 $success_breakthrough)
                                                                 (+ (* 0.24 $needs_task_plan)
                                                                    (+ (* -0.12 $needs_external_evidence)
                                                                       (* 0.02 $needs_multi_source_integration)
))))))))))))))))))))))))

(act_think
  (+ $score
     (+ (* 0.35 $cx)
        (+ (* 0.25 $ambiguity)
           (+ (* 0.35 $approach)
              (+ (* 0.10 $low_confidence)
                 (+ (* 0.10 (- 1.0 $u))
                    (+ (* -0.10 $threshold)
                       (+ (* 0.20 $arousal)
                          (+ (* 0.08 $coherence)
                             (+ (* 0.02 $valence)
                                (+ (* 0.14 $originality)
                                   (+ (* 0.04 $social)
                                      (+ (* 0.10 (- 1.0 $risk_aversion))
                                         (+ (* 0.26 $creativity)
                                            (+ (* -0.14 (- 1.0 $error_tolerance))
                                               (+ (* 0.10 $help_long)
                                                  (+ (* -0.08 $help_short)
                                                     (+ (* 0.10 $knowledge)
                                                        (+ (* 0.12 $novelty)
                                                           (+ (* 0.16 $success_breakthrough)
                                                              (+ (* $reflective_think_bonus $reflective_intent)
                                                                 (+ (* -0.30 (* $anti_redundant
                                                                                (+ 0.70 (* 0.30 $familiarity))))
                                                                    (+ (* -0.16 $answerability)
                                                                       (if (and (>= $cx 0.70)
                                                                                (>= $approach 0.62)
                                                                                (or (>= $ambiguity 0.25)
                                                                                    (>= $low_confidence 0.30)))
                                                                           0.07
                                                                           (if (and (>= $cx 0.65)
                                                                                    (>= $approach 0.58)
                                                                                    (or (>= $ambiguity 0.22)
                                                                                        (>= $low_confidence 0.28)))
                                                                               0.03
                                                                               0.0))
))))))))))))))))))))))))

(act_synthesize
  (+ $score
     (+ (* 0.24 $cx)
        (+ (* 0.12 $res)
           (+ (* -0.10 $u)
              (+ (* 0.16 (- 1.0 $ambiguity))
                 (+ (* 0.14 (- 1.0 $familiarity))
                    (+ (* 0.12 $approach)
                       (+ (* 0.08 $arousal)
                          (+ (* 0.16 $creativity)
                             (+ (* 0.16 $coherence)
                                (+ (* 0.08 $valence)
                                   (+ (* 0.22 $originality)
                                      (+ (* 0.10 $social)
                                         (+ (* 0.06 (- 1.0 $low_confidence))
                                            (+ (* 0.12 $knowledge)
                                               (+ (* 0.08 $novelty)
                                                  (+ (* 0.10 $success_breakthrough)
                                                     (+ (* 0.14 $help_long)
                                                        (+ (* -0.10 $help_short)
                                                           (+ (* -0.12 $risk_aversion)
                                                              (+ (* -0.18 $threshold)
                                                                 (+ (* -0.16 $failure_wariness)
                                                                    (+ (* 0.55 $needs_multi_source_integration)
                                                                       (+ (* -0.12 $needs_external_evidence)
                                                                          (+ (* -0.18 $needs_task_plan)
                                                                             (+ (if (and (>= $cx 0.55)
                                                                                         (<= $ambiguity 0.60))
                                                                                    0.16
                                                                                    0.0)
                                                                                (+ (if (>= $ambiguity 0.80)
                                                                                       -0.28
                                                                                       0.0)
                                                                                   (if $verify_request
                                                                                       -0.25
                                                                                       0.0)
                                                                                )
                                                                             )))))))))))))))))))))))))))



   
   
   ))
   ))