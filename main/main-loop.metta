!(import! &self state)
!(import! &self ../operators/appraisal)
!(import! &self "context.py")
!(import! &self space)


; 1. getting context from space space/generating the repspective context from llm by giving some query.
; 2. getting state from the space
; 3. calculating mods by using calculate-mods function in appraisal module
; 4. update only mods by using update-mods  function in appraisal module.
; 5. decision module (this part hasn't been began.) yet we know that it uses the modulators and stuffs to update the goal vectos and then decides what to do next.
; 6. after that the state will be given to LLM then LLM generates all content then the user will also respond to it after  that that response will be update the context-->modulator,and stuffs
;  that is what i  understood so far.
; if there is anything that need further study i will explore  and implement it.
(= (mainLoop $space)
(let* (
    ;    ($context (py-call (context.wrap_parser "Can you verify if the quantum computing breakthrough is real? I need detailed analysis comparing expert opinions.")))
       ($context ())
       ($_ (let $temp (if (== $context ())
                        (add-context $space (context))
                        (add-context $space $context)
                 )))
    
       ($_ (addAlpha $space (alpha)))
       ($_ (addState $space (state (default-goals) (default-mods))))
       ($calculated-mods  (calculate-mod))
   )
    $calculated-mods
))


; !(getAlpha expertise_alpha)
; !(calculate-mod)
; !(addState &space (state (default-goals) (default-mods)))
; !(add-context &space (context))
; !(addAlpha &space (alpha))
; !(getMod m_resolution)
; !(get_state &space)
; !(calculate-mod)
!(mainLoop &space)
