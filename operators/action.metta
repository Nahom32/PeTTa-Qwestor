! (register-module! "../../PeTTa-Qwestor")
! (import! &self PeTTa-Qwestor:utils:general-helper)
!(import! &self appraisal)
(= (Actions $space)
    ((AG act_respond (
                (efficiency 1.00)
                (accuracy (eval (clampValues (- 1.00 (* 1.10 (getContext $space complexity))))))
                (success_moderate  (eval (clampValues (- 0.80 (* 0.20 (getContext $space complexity))))))
                (knowledge  (eval (clampValues (+ 0.28 (* 0.18 (getContext $space complexity))))))
                (novelty (eval (clampValues (+ 0.18 (* 0.10 (getContext $space complexity))))))
                (success_breakthrough (eval (clampValues (+ 0.18  (* 0.12  (getContext $space complexity))))))
                (coherence (eval (clampValues (- 0.72 (* 0.10 (getContext $space complexity))))))
                (originality (eval (clampValues (+ 0.16 (* 0.10 (getContext $space complexity))))))
                (social (eval (clampValues (+ 0.74 (* 0.08 (- 1.0 (getContext $space complexity)))))))
                (help_short (eval (clampValues (- 0.95 (* 0.20 (getContext $space complexity))))))
                (help_long (eval (clampValues (+ 0.25 (* 0.20 (getContext $space complexity))))))
                (over_beneficial (eval (clampValues (- 0.45 (* 0.15 (getContext $space complexity))))))
                (over_safety (eval (clampValues (- 0.45 (* 0.20 (getContext $space complexity))))))
                (over_honesty 0.60)

        ))
(AG act_clarify (
        (efficiency 0.65)
        (accuracy (eval (clampValues (+ 0.55 (* 0.25 (getContext $space complexity))))))
        (success_moderate 0.72)
        (knowledge (eval (clampValues (+ 0.45 (* 0.20 (getContext $space complexity))))))
        (novelty (eval (clampValues (+ 0.28 (* 0.12 (getContext $space complexity))))))
        (success_breakthrough (eval (clampValues (+ 0.28 (* 0.12 (getContext $space complexity))))))
        (coherence 0.82)
        (originality (eval (clampValues (+ 0.18 (* 0.10 (getContext $space complexity))))))
        (social  0.95)
        (help_short (eval (clampValues (+ 0.55 (* 0.10 (- 1.0 (getContext $space complexity)))))))
        (help_long (eval (clampValues (+ 0.40 (* 0.20 (getContext $space complexity))))))
        (over_beneficial 0.85)
        (over_safety 0.90)
        (over_honesty 0.95)
))

(AG act_search (
        (efficiency 0.25)
        (accuracy (eval (clampValues (+ 0.30 (* 0.90 (getContext $space complexity))))))
        (success_moderate (eval (clampValues (+ 0.55 (* 0.20 (getContext $space complexity))))))
        (knowledge (eval (clampValues (+ 0.68 (* 0.22 (getContext $space complexity))))))
        (novelty (eval (clampValues (+ 0.58 (* 0.18 (getContext $space complexity))))))
        (success_breakthrough (eval (clampValues (+ 0.45 (* 0.18 (getContext $space complexity))))))
        (coherence 0.58)
        (originality (eval (clampValues (+ 0.48 (* 0.16 (getContext $space complexity))))))
        (social (eval (clampValues (+ 0.42 (* 0.12 (- 1.0 (getContext $space complexity)))))))
        (help_short (eval (clampValues (+ 0.35 (* 0.10 (- 1.0 (getContext $space complexity)))))))
        (help_long (eval (clampValues (+ 0.55 (* 0.35 (getContext $space complexity))))))
        (over_beneficial 0.72)
        (over_safety 0.78)
        (over_honesty 0.82)

))
(AG act_verify (
        (efficiency 0.35)
        (accuracy (eval (clampValues (+ 0.75 (* 0.20 (getContext $space complexity))))))
        (success_moderate 0.90)
        (knowledge (eval (clampValues (+ 0.62 (* 0.15 (getContext $space complexity))))))
        (novelty (eval (clampValues (+ 0.25 (* 0.08 (getContext $space complexity))))))
        (success_breakthrough (eval (clampValues (+ 0.38 (* 0.10 (getContext $space complexity))))))
        (coherence 0.86)
        (originality (eval (clampValues (+ 0.24 (* 0.08 (getContext $space complexity))))))
        (social 0.88)
        (help_short (eval (clampValues (+ 0.40 (* 0.10 (- 1.0 (getContext $space complexity)))))))
        (help_long (eval (clampValues (+ 0.50 (* 0.20 (getContext $space complexity))))))
        (over_beneficial 0.96)
        (over_safety 0.97)
        (over_honesty 0.97)
))
(AG act_decompose (
        (efficiency 0.45)
        (accuracy (eval (clampValues (+ 0.55 (* 0.35 (getContext $space complexity))))))
        (success_moderate (eval (clampValues (+ 0.65 (* 0.15 (getContext $space complexity))))))
        (knowledge (eval (clampValues (+ 0.72 (* 0.20 (getContext $space complexity))))))
        (novelty (eval (clampValues (+ 0.62 (* 0.18 (getContext $space complexity))))))
        (success_breakthrough (eval (clampValues (+ 0.62 (* 0.22 (getContext $space complexity))))))
        (coherence 0.80)
        (originality (eval (clampValues (+ 0.66 (* 0.18 (getContext $space complexity))))))
        (social 0.72)
        (help_short (eval (clampValues (+ 0.30 (* 0.05 (- 1.0 (getContext $space complexity)))))))
        (help_long (eval (clampValues (+ 0.70 (* 0.25 (getContext $space complexity))))))
        (over_beneficial 0.70)
        (over_safety 0.76)
        (over_honesty 0.80)
))

(AG act_think (
        (efficiency 0.40)
        (accuracy (eval (clampValues (+ 0.60 (* 0.25 (getContext $space complexity))))))
        (success_moderate (eval (clampValues (+ 0.45 (* 0.20 (getContext $space complexity))))))
        (knowledge (eval (clampValues (+ 0.66 (* 0.18 (getContext $space complexity))))))
        (novelty (eval (clampValues (+ 0.70 (* 0.18 (getContext $space complexity))))))
        (success_breakthrough (eval (clampValues (+ 0.68 (* 0.20 (getContext $space complexity))))))
        (coherence 0.74)
        (originality (eval (clampValues (+ 0.74 (* 0.16 (getContext $space complexity))))))
        (social 0.58)
        (help_short (eval (clampValues (+ 0.35 (* 0.10 (- 1.0 (getContext $space complexity)))))))
        (help_long (eval (clampValues (+ 0.60 (* 0.25 (getContext $space complexity))))))
        (over_beneficial 0.78)
        (over_safety 0.84)
        (over_honesty 0.90)
))

(AG act_synthesize (
        (efficiency 0.30)
        (accuracy (eval (clampValues (+ 0.74 (* 0.12 (getContext $space complexity))))))
        (success_moderate 0.82)
        (knowledge (eval (clampValues (+ 0.78 (* 0.16 (getContext $space complexity))))))
        (novelty (eval (clampValues (+ 0.56 (* 0.12 (getContext $space complexity))))))
        (success_breakthrough (eval (clampValues (+ 0.54 (* 0.14 (getContext $space complexity))))))
        (coherence 0.84)
        (originality (eval (clampValues (+ 0.82 (* 0.12 (getContext $space complexity))))))
        (social 0.68)
        (help_short (eval (clampValues (+ 0.42 (* 0.08 (- 1.0 (getContext $space complexity)))))))
        (help_long (eval (clampValues (+ 0.72 (* 0.18 (getContext $space complexity))))))
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
; (= (bindAG $space)
;     (add-atom $space (Actions))
; )

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
    (let* (
            ($action_list ((act_respond 0.90)
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
(clampValues $result)
)
)


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

    (clampValues (+ (+ 0.45 (+ (* 0.25 (- 1.0 $cx)) (* 0.15 $familiarity))) (* 0.10 (- 1.0 $urgency))))

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

    (clampValues (+ (+ 0.40 (* 0.35 $cx))  (+ (* 0.25 $ambiguity) (* 0.20  $threshold))))
    $base
)
))


;;==============================================================
;; actionScoreWrapper — thin entry point.
;; Fetches the appraisal context once and delegates to action_score.
;;==============================================================
(= (actionScoreWrapper $space $goalWeight $action_list $scored_action)
    (let* (($appraisal_list (calculateMod $space)))
        (action_score $space $goalWeight $action_list $scored_action $appraisal_list)
    )
)


;;==============================================================
;; weightedSum — sum a flat list of (* weight value) terms.
;; Each element must be a 2-tuple: (weight value).
;; Eliminates deeply-nested (+ a (+ b (+ c ...))) chains.
;;==============================================================
(= (weightedSum $pairs)
    (if (== $pairs ())
        0.0
        (let* (
                ($head  (car-atom $pairs))
                ($tail  (cdr-atom $pairs))
                ($w     (car-atom $head))
                ($v     (car-atom (cdr-atom $head)))
            )
        (+ (* $w $v) (weightedSum $tail))
    )
)
)


;;==============================================================
;; action_score — core scoring loop.
;;
;; Arguments:
;;   $space          — the agent's knowledge space
;;   $goalWeight     — weighted goal list  (from goalWeight)
;;   $action_list    — remaining actions to score
;;   $scored_action  — accumulator of (action score) pairs
;;   $appraisal_list — flat modulatorVector from calculateMod
;;
;; All context variables are extracted here from $appraisal_list
;; or from the space directly, so no free variables exist in the
;; case branches.
;;==============================================================
(= (action_score $space $goalWeight $action_list $scored_action $appraisal_list)
    (if (= $action_list ())
        $scored_action
        (let* (
                ;; ── Appraisal / modulator variables ────────────────────────
                ($cx                (find_by_key m_resolution        $appraisal_list))
                ($ambiguity         (getContext $space ambiguity))
                ($threshold_signal  (getContext $space threshold))
                ($familiarity_sig   (getContext $space topic_familiarity))
                ($verify_request    (getContext $space verify_request))
                ($reflective_intent (getContext $space reflective_intent))
                ($needs_ext_ev      (getContext $space needs_external_evidence))
                ($needs_task_plan   (getContext $space needs_task_plan))
                ($needs_multi_src   (getContext $space needs_multi_source_integration))
                ($u                 (find_by_key m_urgency           $appraisal_list))
                ($res               (find_by_key m_resolution        $appraisal_list))
                ($ux                (find_by_key m_user_expertise     $appraisal_list))
                ($threshold         (find_by_key m_threshold         $appraisal_list))
                ($familiarity       (find_by_key m_topic_familiarity $appraisal_list))
                ($failure_wariness  (find_by_key m_failure_wariness  $appraisal_list))
                ($securing          (find_by_key m_securing          $appraisal_list))
                ($approach          (find_by_key m_approach          $appraisal_list))
                ($arousal           (find_by_key m_arousal           $appraisal_list))
                ($risk_aversion     (find_by_key m_risk_aversion     $appraisal_list))
                ($error_tolerance   (find_by_key m_error_tolerance   $appraisal_list))
                ($creativity        (find_by_key m_creativity        $appraisal_list))
                ($valence           (find_by_key m_valence           $appraisal_list))
                ;; ── Alpha / tuning params (read once per loop from space) ──
                ($reflective_think_bonus    (getAlpha $space reflective_think_bonus))
                ($reflective_search_penalty (getAlpha $space reflective_search_penalty))
                ($anti_redundant            (getContext $space anti_redundant))
                ;; ── Derived variables ──────────────────────────────────────
                ($confidence    (clampValues (+ (* 0.55 $familiarity)
                            (+ (* 0.25 (- 1.0 $ambiguity))
                                (* 0.20 (- 1.0 $cx))))))
                ($low_confidence  (clampValues (- 1.0 $confidence)))
                ($answerability   (clampValues (* (- 1.0 $ambiguity)
                            (* (- 1.0 $threshold_signal) $familiarity_sig))))
                ;; ── Current action + its effects from the AG atom ─────────
                (($h $t)       (decons-atom $action_list))
                ($effects      (getAction $space $h))
                ($coherence    (find_by_key coherence           $effects))
                ($social       (find_by_key social              $effects))
                ($originality  (find_by_key originality         $effects))
                ($knowledge    (find_by_key knowledge           $effects))
                ($novelty      (find_by_key novelty             $effects))
                ($success_bk   (find_by_key success_breakthrough $effects))
                ($help_short   (find_by_key help_short          $effects))
                ($help_long    (find_by_key help_long           $effects))

                ;; ── Score for the current action ──────────────────────────
                ($score
                    (clampValues
                        (case $h (

                                (act_clarify
                                    (weightedSum (
                                            (0.90  $ambiguity)
                                            (-0.35 $ux)
                                            (-0.15 $u)
                                            (0.20  $threshold)
                                            (0.20  $securing)
                                            (0.10  $coherence)
                                            (-0.08 $valence)
                                            (0.22  $social)
                                            (-0.06 $originality)
                                            (0.08  (- 1.0 $error_tolerance))
                                            (-0.55 $answerability)
                                            (-0.20 $help_short)
                                            (-0.15 $anti_redundant)
                                            (1.0   (if (and (> $ambiguity 0.75)
                                                        (or (> $threshold_signal 0.55)
                                                            (> $low_confidence 0.45)))
                                                    0.18
                                                    0.0))
                                    ))
                        )

                    (act_respond
                        (weightedSum (
                                (0.35  $u)
                                (0.25  (- 1.0 $ambiguity))
                                (0.15  $ux)
                                (-0.20 $cx)
                                (0.20  $familiarity)
                                (-0.35 $threshold)
                                (-0.30 $failure_wariness)
                                (-0.35 $securing)
                                (-0.20 $low_confidence)
                                (0.10  (- 1.0 $arousal))
                                (0.12  $coherence)
                                (0.10  $valence)
                                (0.14  $social)
                                (-0.06 $originality)
                                (-0.18 $risk_aversion)
                                (0.46  $help_short)
                                (-0.15 $help_long)
                                (0.45  $answerability)
                                (0.22  $error_tolerance)
                                (0.12  $anti_redundant)
                                (1.0   (if (>= $cx 0.50)
                                        (+ (* -0.08 $knowledge)
                                            (* -0.10 $success_bk))
                                        0.0))
                        ))
            )

        (act_search
            (weightedSum (
                    (0.35  $cx)
                    (0.20  $res)
                    (-0.15 $u)
                    (0.35  $threshold)
                    (0.35  (- 1.0 $familiarity))
                    (0.30  $failure_wariness)
                    (0.15  $securing)
                    (0.08  $arousal)
                    (0.06  $coherence)
                    (0.02  $valence)
                    (0.10  $originality)
                    (0.06  $social)
                    (0.08  (- 1.0 $risk_aversion))
                    (0.10  (- 1.0 $error_tolerance))
                    (0.10  $creativity)
                    (0.06  $help_long)
                    (-0.08 $help_short)
                    (0.14  $knowledge)
                    (0.12  $novelty)
                    (0.08  $success_bk)
                    (0.50  $needs_ext_ev)
                    (0.12  $needs_multi_src)
                    (-0.08 $needs_task_plan)
                    ($reflective_search_penalty $reflective_intent)
            ))
)

(act_verify
    (weightedSum (
            (0.65  $threshold)
            (0.75  $low_confidence)
            (0.35  $failure_wariness)
            (0.15  $cx)
            (-0.20 $u)
            (-0.10 $ambiguity)
            (0.30  $securing)
            (0.14  $coherence)
            (-0.14 $valence)
            (0.10  $social)
            (-0.08 $originality)
            (0.25  $risk_aversion)
            (-0.08 $arousal)
            (0.55  (- 1.0 $error_tolerance))
            (0.08  (- 1.0 $creativity))
            (0.08  $help_long)
            (-0.10 $help_short)
            (0.32  (if $verify_request 1.0 0.0))
            (0.05  $knowledge)
    ))
)

(act_decompose
    (weightedSum (
            (0.30  $cx)
            (0.30  $res)
            (0.10  (- 1.0 $ambiguity))
            (-0.12 $u)
            (-0.28 $ambiguity)
            (1.0   (if (and (>= $cx 0.60) (<= $ambiguity 0.60)) 0.10 0.0))
            (1.0   (if (< $cx 0.35) -0.35 0.0))
            (0.10  $approach)
            (0.10  $arousal)
            (0.10  $coherence)
            (0.04  $valence)
            (0.12  $originality)
            (0.08  $social)
            (0.08  $creativity)
            (-0.08 (- 1.0 $error_tolerance))
            (0.12  $help_long)
            (-0.12 $help_short)
            (0.08  $knowledge)
            (0.06  $novelty)
            (0.10  $success_bk)
            (0.24  $needs_task_plan)
            (-0.12 $needs_ext_ev)
            (0.02  $needs_multi_src)
    ))
)

(act_think
    (weightedSum (
            (0.35  $cx)
            (0.25  $ambiguity)
            (0.35  $approach)
            (0.10  $low_confidence)
            (0.10  (- 1.0 $u))
            (-0.10 $threshold)
            (0.20  $arousal)
            (0.08  $coherence)
            (0.02  $valence)
            (0.14  $originality)
            (0.04  $social)
            (0.10  (- 1.0 $risk_aversion))
            (0.26  $creativity)
            (-0.14 (- 1.0 $error_tolerance))
            (0.10  $help_long)
            (-0.08 $help_short)
            (0.10  $knowledge)
            (0.12  $novelty)
            (0.16  $success_bk)
            ($reflective_think_bonus $reflective_intent)
            (-0.30 (* $anti_redundant (+ 0.70 (* 0.30 $familiarity))))
            (-0.16 $answerability)
            (1.0   (if (and (>= $cx 0.70)
                        (>= $approach 0.62)
                        (or (>= $ambiguity 0.25)
                            (>= $low_confidence 0.30)))
                    0.07
                    (if (and (>= $cx 0.65)
                            (>= $approach 0.58)
                            (or (>= $ambiguity 0.22)
                                (>= $low_confidence 0.28)))
                        0.03
                        0.0)))
    ))
)

(act_synthesize
    (weightedSum (
            (0.24  $cx)
            (0.12  $res)
            (-0.10 $u)
            (0.16  (- 1.0 $ambiguity))
            (0.14  (- 1.0 $familiarity))
            (0.12  $approach)
            (0.08  $arousal)
            (0.16  $creativity)
            (0.16  $coherence)
            (0.08  $valence)
            (0.22  $originality)
            (0.10  $social)
            (0.06  (- 1.0 $low_confidence))
            (0.12  $knowledge)
            (0.08  $novelty)
            (0.10  $success_bk)
            (0.14  $help_long)
            (-0.10 $help_short)
            (-0.12 $risk_aversion)
            (-0.18 $threshold)
            (-0.16 $failure_wariness)
            (0.55  $needs_multi_src)
            (-0.12 $needs_ext_ev)
            (-0.18 $needs_task_plan)
            (1.0   (if (and (>= $cx 0.55) (<= $ambiguity 0.60)) 0.16 0.0))
            (1.0   (if (>= $ambiguity 0.80) -0.28 0.0))
            (1.0   (if $verify_request -0.25 0.0))
    ))
)

))
)
)
)
;; ── Recurse with the scored action appended ─────────────────
(action_score
    $space
    $goalWeight
    $t
    (cons-atom ($h $score) $scored_action)
    $appraisal_list
)
)
))