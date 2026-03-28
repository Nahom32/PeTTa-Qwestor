!(import! &self ../state)
; !(bind! &contextSpace (new-space))
; !(add-context &contextSpace)

!(bind! &sp (new-space))
!(add-state &sp (state (goalVector
   (integerity 0.2) 
   (energy 0.4) 
   (competence 0.5) 
   (affiliation 0.7) 
) (modulatorVector 
(m_urgency 0.2) 
(m_risk_aversion 0.3) 
(m_failure_wariness 0.1)
(m_error_tolerance 0.5)
(m_user_expertise 0.5)
(m_topic_familiarity 0.5)
(m_creativity 0.5)
)))





; !(remove-state &sp)
; !(get-state &sp)
!(getGoal &sp competence)
; !(get-modulator &sp Resolution)
; !(update-modulator &sp Resolution 0.45)
; !(update-demand &sp energy 0.8)
; !(get-state &sp)
; !(get-state &sp)
