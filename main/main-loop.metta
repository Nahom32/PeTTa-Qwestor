!(import! &self state)
!(import! &self ../operators/appraisal)
!(import! &self ../operators/decision)
!(import! &self "context.py")
!(import! &self space)


; ─────────────────────────────────────────────
; Main loop overview
; ─────────────────────────────────────────────
; 1. Parse user query → context signals via LLM (context.py).
; 2. Store context signals in the space.
; 3. Compute updated modulators with calculateMod.
; 4. Update the state's modulator vector.
; 5. [Planned] Decision module: update goal weights, choose action.
; 6. [Planned] Generate LLM response, feed user reply back into context.


; ─────────────────────────────────────────────
; addContexts
; If no context was parsed (empty list), inject
; the built-in defaults instead.
; ─────────────────────────────────────────────
(= (addContexts $space $context)
    (if (== $context ())
        (addContext $space (context))
        (addContext $space $context)
    )
)


; ─────────────────────────────────────────────
; setUpSpace
; Populate a fresh space with alpha hyper-params
; and an initial state (default goals + default mods).
; ─────────────────────────────────────────────
(= (setUpSpace $space)
    (let*
        (
            ($_ (addAlpha $space (alpha)))
            ($_ (addState $space (state (default-goals) (default-mods) (default-anti-goals))))
        )
    "atoms added"
)
)


(= (mainLoop $space $query)
    (let* (
            ($context       (py-call (context.wrap_parser $query)))
            ($_ (addContexts $space $context))
            ($calculated-mods (calculateMod $space))
            ($_ (update-mods $space $calculated-mods))
        )
    $calculated-mods
)
)

; ─────────────────────────────────────────────
; Entry point
; ─────────────────────────────────────────────
; !(setUpSpace &space)
; !(mainLoop &space "Can you verify if the quantum computing breakthrough is real? I need detailed analysis comparing expert opinions.")
(..................)