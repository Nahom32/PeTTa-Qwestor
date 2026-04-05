; ─────────────────────────────────────────────────────────────
; action-score-test.metta
;
; Test suite for action_score and its helpers.
;
; Structure:
;   1. Space bootstrap  — identical to appraisal-test.metta
;   2. Unit tests       — weightedSum helper
;   3. Scenario tests   — action_score with specific context overrides
;      [A] High-ambiguity / low-familiarity  → expect act_clarify to win
;      [B] High-complexity / reflective      → expect act_think to win
;      [C] High-failure / external evidence  → expect act_search to win
;      [D] Low-complexity / high-familiarity → expect act_respond to win
;      [E] Verify requested explicitly       → expect act_verify to win
;   4. Helper function: assert-score-positive
; ─────────────────────────────────────────────────────────────
!(import! &self ../action)
!(import! &self ../decision)
!(import! &self ../appraisal)
!(import! &self ../../main/main-loop)
!(import! &self ../../utils/general-helper)

; ─────────────────────────────────────────────────────────────
; Bootstrap: create and populate the shared test space
; ─────────────────────────────────────────────────────────────
!(bind! &tspace (new-space))
!(addAlpha   &tspace (alpha))
!(addContext &tspace (context))
!(addState   &tspace (state (default-goals) (default-mods)))
!(Actions &tspace)
!(addActions &tspace (Actions &tspace))
!(collapse (get-atoms &tspace))
!(weightedSum ((0.3 0.8) (0.2 0.9) (0.5 0.1)))

; ─────────────────────────────────────────────────────────────
; Helper assertions
; ─────────────────────────────────────────────────────────────

; assert-true: expect a boolean expression to be True
; (= (assert-true $label $expr)
;     (if $expr
;         (println! (PASS $label))
;         (println! (FAIL $label))
;     )
; )

; ; assert-in-range: expect $val in [$lo, $hi]
; (= (assert-in-range $label $val $lo $hi)
;     (assert-true $label
;         (and (>= $val $lo) (<= $val $hi)))
; )

; ; assert-greater: expect $a > $b
; (= (assert-greater $label $a $b)
;     (assert-true $label (> $a $b))
; )


; ; ─────────────────────────────────────────────────────────────
; ; Section 1 — Unit tests: weightedSum
; ; ─────────────────────────────────────────────────────────────

; ; Test 1-A: empty list → 0.0
; !(test
;     (weightedSum ())
;     0.0)

; ; Test 1-B: single term — (1.0 * 0.5) = 0.5
; !(test
;     (weightedSum ((1.0 0.5)))
;     0.5)

; ; Test 1-C: multiple terms — (0.5*0.4) + (0.3*1.0) + (-0.2*0.5)
; ;           = 0.20 + 0.30 - 0.10 = 0.40
; !(test
;     (weightedSum ((0.5 0.4) (0.3 1.0) (-0.2 0.5)))
;     0.4)

; ; Test 1-D: clampValues used downstream — weightedSum itself is unbounded,
; ;           verify a large positive result is > 1 (clamping done externally)
; !(assert-true
;     "weightedSum-unbounded"
;     (> (weightedSum ((1.0 0.8) (1.0 0.8) (1.0 0.8))) 1.0)
; )


; ; ─────────────────────────────────────────────────────────────
; ; Section 2 — Scenario setups
; ;
; ; Each scenario creates an isolated space so context overrides
; ; don't bleed across tests.
; ; ─────────────────────────────────────────────────────────────

; ; ── Scenario A: High ambiguity, low familiarity, no external evidence
; ;    Expected winner: act_clarify (high ambiguity drives clarify score)
; ; ─────────────────────────────────────────────────────────────
; !(bind! &sp-A (new-space))
; !(addAlpha   &sp-A (alpha))
; !(addContext &sp-A (context))
; !(addState   &sp-A (state (default-goals) (default-mods)))
; !(addActions &sp-A (Actions &sp-A))
; ; Override context signals for scenario A
; !(add-atom &sp-A (ambiguity               0.90))
; !(add-atom &sp-A (topic_familiarity       0.10))
; !(add-atom &sp-A (threshold               0.60))
; !(add-atom &sp-A (complexity              0.20))
; !(add-atom &sp-A (needs_external_evidence 0.10))
; !(add-atom &sp-A (anti_redundant          0.10))

; !(let* (
;         ($appraisal-A   (calculateMod &sp-A))
;         ($goalW-A       (goalWeight &sp-A))
;         ($all_actions   (ActionList &sp-A))
;         ($scores-A      (action_score &sp-A $goalW-A $all_actions () $appraisal-A))
;         ($clarify-score (find_by_key act_clarify $scores-A))
;         ($respond-score (find_by_key act_respond $scores-A))
;     )
; (let* (
;         ($_ (println! (Scenario-A scores: $scores-A)))
;         ($_ (assert-in-range "A: clarify score in [0,1]" $clarify-score 0.0 1.0))
;         ($_ (assert-greater  "A: clarify > respond"       $clarify-score $respond-score))
;     )
; (println! (Scenario-A done))
; ))


; ; ── Scenario B: High complexity, reflective intent → act_think expected
; ; ─────────────────────────────────────────────────────────────
; !(bind! &sp-B (new-space))
; !(addAlpha   &sp-B (alpha))
; !(addContext &sp-B (context))
; !(addState   &sp-B (state (default-goals) (default-mods)))
; !(addActions &sp-B (Actions &sp-B))
; !(add-atom &sp-B (complexity              0.80))
; !(add-atom &sp-B (ambiguity               0.35))
; !(add-atom &sp-B (reflective_intent       0.90))
; !(add-atom &sp-B (topic_familiarity       0.40))
; !(add-atom &sp-B (threshold               0.25))
; !(add-atom &sp-B (needs_external_evidence 0.10))
; !(add-atom &sp-B (anti_redundant          0.05))

; !(let* (
;         ($appraisal-B  (calculateMod &sp-B))
;         ($goalW-B      (goalWeight &sp-B))
;         ($all_actions  (ActionList &sp-B))
;         ($scores-B     (action_score &sp-B $goalW-B $all_actions () $appraisal-B))
;         ($think-score  (find_by_key act_think  $scores-B))
;         ($search-score (find_by_key act_search $scores-B))
;     )
; (let* (
;         ($_ (println! (Scenario-B scores: $scores-B)))
;         ($_ (assert-in-range "B: think score in [0,1]" $think-score 0.0 1.0))
;         ($_ (assert-greater  "B: think > search"        $think-score $search-score))
;     )
; (println! (Scenario-B done))
; ))


; ; ── Scenario C: High failure signal, external evidence needed → act_search
; ; ─────────────────────────────────────────────────────────────
; !(bind! &sp-C (new-space))
; !(addAlpha   &sp-C (alpha))
; !(addContext &sp-C (context))
; !(addState   &sp-C (state (default-goals) (default-mods)))
; !(addActions &sp-C (Actions &sp-C))
; !(add-atom &sp-C (complexity              0.55))
; !(add-atom &sp-C (ambiguity               0.40))
; !(add-atom &sp-C (failure_signal          0.80))
; !(add-atom &sp-C (needs_external_evidence 0.90))
; !(add-atom &sp-C (topic_familiarity       0.20))
; !(add-atom &sp-C (threshold               0.50))
; !(add-atom &sp-C (anti_redundant          0.10))

; !(let* (
;         ($appraisal-C  (calculateMod &sp-C))
;         ($goalW-C      (goalWeight &sp-C))
;         ($all_actions  (ActionList &sp-C))
;         ($scores-C     (action_score &sp-C $goalW-C $all_actions () $appraisal-C))
;         ($search-score (find_by_key act_search  $scores-C))
;         ($respond-score (find_by_key act_respond $scores-C))
;     )
; (let* (
;         ($_ (println! (Scenario-C scores: $scores-C)))
;         ($_ (assert-in-range "C: search score in [0,1]" $search-score 0.0 1.0))
;         ($_ (assert-greater  "C: search > respond"       $search-score $respond-score))
;     )
; (println! (Scenario-C done))
; ))


; ; ── Scenario D: Low complexity, high familiarity, simple question → act_respond
; ; ─────────────────────────────────────────────────────────────
; !(bind! &sp-D (new-space))
; !(addAlpha   &sp-D (alpha))
; !(addContext &sp-D (context))
; !(addState   &sp-D (state (default-goals) (default-mods)))
; !(addActions &sp-D (Actions &sp-D))
; !(add-atom &sp-D (complexity              0.10))
; !(add-atom &sp-D (ambiguity               0.05))
; !(add-atom &sp-D (topic_familiarity       0.90))
; !(add-atom &sp-D (threshold               0.10))
; !(add-atom &sp-D (failure_signal          0.05))
; !(add-atom &sp-D (needs_external_evidence 0.05))
; !(add-atom &sp-D (anti_redundant          0.05))

; !(let* (
;         ($appraisal-D   (calculateMod &sp-D))
;         ($goalW-D       (goalWeight &sp-D))
;         ($all_actions   (ActionList &sp-D))
;         ($scores-D      (action_score &sp-D $goalW-D $all_actions () $appraisal-D))
;         ($respond-score (find_by_key act_respond $scores-D))
;         ($clarify-score (find_by_key act_clarify $scores-D))
;     )
; (let* (
;         ($_ (println! (Scenario-D scores: $scores-D)))
;         ($_ (assert-in-range "D: respond score in [0,1]" $respond-score 0.0 1.0))
;         ($_ (assert-greater  "D: respond > clarify"       $respond-score $clarify-score))
;     )
; (println! (Scenario-D done))
; ))


; ; ── Scenario E: verify_request=True, high uncertainty → act_verify
; ; ─────────────────────────────────────────────────────────────
; !(bind! &sp-E (new-space))
; !(addAlpha   &sp-E (alpha))
; !(addContext &sp-E (context))
; !(addState   &sp-E (state (default-goals) (default-mods)))
; !(addActions &sp-E (Actions &sp-E))
; !(add-atom &sp-E (verify_request          True))
; !(add-atom &sp-E (threshold               0.70))
; !(add-atom &sp-E (failure_signal          0.60))
; !(add-atom &sp-E (complexity              0.40))
; !(add-atom &sp-E (ambiguity               0.30))
; !(add-atom &sp-E (topic_familiarity       0.30))
; !(add-atom &sp-E (anti_redundant          0.10))

; !(let* (
;         ($appraisal-E  (calculateMod &sp-E))
;         ($goalW-E      (goalWeight &sp-E))
;         ($all_actions  (ActionList &sp-E))
;         ($scores-E     (action_score &sp-E $goalW-E $all_actions () $appraisal-E))
;         ($verify-score (find_by_key act_verify  $scores-E))
;         ($search-score (find_by_key act_search  $scores-E))
;     )
; (let* (
;         ($_ (println! (Scenario-E scores: $scores-E)))
;         ($_ (assert-in-range "E: verify score in [0,1]" $verify-score 0.0 1.0))
;         ($_ (assert-greater  "E: verify > search"        $verify-score $search-score))
;     )
; (println! (Scenario-E done))
; ))


; ; ─────────────────────────────────────────────────────────────
; ; Section 3 — Invariants that must hold across ALL scenarios
; ;
; ; For every action in every scored list, the output must be
; ; a numeric value clamped to [0.0, 1.0].
; ; ─────────────────────────────────────────────────────────────
; !(let* (
;         ($appraisal-all (calculateMod &tspace))
;         ($goalW-all     (goalWeight &tspace))
;         ($all_actions   (ActionList &tspace))
;         ($scores-all    (action_score &tspace $goalW-all $all_actions () $appraisal-all))
;         ($_ (println! (Default-scenario all-scores: $scores-all)))
;         ; Every individual score must be in [0,1]
;         ($_ (assert-in-range "default: act_clarify in [0,1]"   (find_by_key act_clarify   $scores-all) 0.0 1.0))
;         ($_ (assert-in-range "default: act_respond in [0,1]"   (find_by_key act_respond   $scores-all) 0.0 1.0))
;         ($_ (assert-in-range "default: act_search in [0,1]"    (find_by_key act_search    $scores-all) 0.0 1.0))
;         ($_ (assert-in-range "default: act_verify in [0,1]"    (find_by_key act_verify    $scores-all) 0.0 1.0))
;         ($_ (assert-in-range "default: act_decompose in [0,1]" (find_by_key act_decompose $scores-all) 0.0 1.0))
;         ($_ (assert-in-range "default: act_think in [0,1]"     (find_by_key act_think     $scores-all) 0.0 1.0))
;         ($_ (assert-in-range "default: act_synthesize in [0,1]"(find_by_key act_synthesize $scores-all) 0.0 1.0))
;     )
; (println! (All-invariant-checks done))
; )
