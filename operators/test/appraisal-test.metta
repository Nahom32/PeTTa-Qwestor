!(import! &self ../space)
!(import! &self ../appraisal)
!(import! &self ../action)
!(import! &self ../decision)
!(import! &self ../../main/main-loop)
!(import! &self ../../utils/general-helper)

; --------------- adding state and parameters to the space ---------------
!(bind! &space (new-space))
!(addAlpha &space (alpha))
!(addContexts &space (context))
!(addState &space (state (default-mods) (default-goals)))
!(match  &space (state $x $y) (state $x $y))

; --------------- adding actions to the space -----------------------------

!(addActions &space (Actions &space))
; !(update-goal &space kcnowledge 0.7)
; ── example usage ───────────────────────────────────────────
; !(selectAction ((act_think 0.7) (act_search 0.72)) factual 0.35 0.44 0.03)
; !(getGoal &space over_honesty)
; !(let $x (goalWeight &space) $x)
!(ActionList &space)
