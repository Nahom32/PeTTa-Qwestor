import math
def _parse_with_openai(query: str, api_key: str, model: str) -> dict[str, Any] | None:
    try:
        messages_mod = importlib.import_module("langchain_core.messages")
        openai_mod = importlib.import_module("langchain_openai")

        HumanMessage = getattr(messages_mod, "HumanMessage")
        SystemMessage = getattr(messages_mod, "SystemMessage")
        ChatOpenAI = getattr(openai_mod, "ChatOpenAI")

        llm = ChatOpenAI(model=model, temperature=0, api_key=api_key)

        system = (
            "Return JSON only (no markdown). "
            'Schema: {"urgent": boolean, "complexity": number, "ambiguity": number, "expertise": number, "threshold": number, "topic_familiarity": number, "failure_signal": number, "intent_type": string, "reflective_intent": number, "verify_request": boolean, "needs_external_evidence": number, "needs_task_plan": number, "needs_multi_source_integration": number, "valence": number}. '
            "Rules: complexity, ambiguity, expertise, threshold, topic_familiarity, failure_signal are each 0..1. "
            "Rules: valence is in [-1,1], where -1 is strongly negative/frustrated tone, +1 is strongly positive/satisfied tone, and 0 is neutral. "
            "Rules: intent_type must be one of reflective|factual|mixed. "
            "Rules: reflective_intent is 0..1 and measures how much deliberate internal reasoning is likely beneficial before final answer. "
            "Rules: verify_request is true only if user explicitly asks to verify/check/fact-check a claim before answering. "
            "Rules: needs_external_evidence, needs_task_plan, needs_multi_source_integration are each 0..1. "
            "Interpretation: needs_external_evidence is high when answering likely requires fresh/source-backed evidence gathering beyond internal memory. "
            "Interpretation: needs_task_plan is high when the user asks for an ordered plan, breakdown, roadmap, or stepwise execution structure. "
            "Interpretation: needs_multi_source_integration is high when the user asks to synthesize/compare/conflict-resolve across multiple viewpoints or sources. "
            "Interpretation: expertise 0 means novice user language, 1 means expert-level user language."
            "Interpretation: threshold is risk/safety sensitivity (higher means more caution needed). "
            "Interpretation: topic_familiarity is how likely the assistant is to already know this topic well (higher means more familiar). "
            "Interpretation: failure_signal is high when the user indicates previous answer/correction problems."
        )

        out = llm.invoke([SystemMessage(content=system), HumanMessage(content=query)])
        raw = out.content if hasattr(out, "content") else str(out)
        payload = _extract_json(_to_text(raw))

        urgent_raw = payload.get("urgent", None)
        complexity_raw = payload.get("complexity", None)
        ambiguity_raw = payload.get("ambiguity", None)
        expertise_raw = payload.get("expertise", None)
        threshold_raw = payload.get("threshold", 0.3)
        topic_familiarity_raw = payload.get("topic_familiarity", 0.5)
        failure_signal_raw = payload.get("failure_signal", 0.0)
        intent_type_raw = str(payload.get("intent_type", "mixed")).strip().lower()
        reflective_intent_raw = payload.get("reflective_intent", 0.5)
        verify_request_raw = payload.get("verify_request", False)
        needs_external_evidence_raw = payload.get("needs_external_evidence", 0.3)
        needs_task_plan_raw = payload.get("needs_task_plan", 0.2)
        needs_multi_source_integration_raw = payload.get(
            "needs_multi_source_integration", 0.3
        )
        valence_raw = payload.get("valence", 0.0)

        urgent = _coerce_bool(urgent_raw)
        if urgent is None:
            return None
        verify_request = _coerce_bool(verify_request_raw)
        if verify_request is None:
            verify_request = False

        try:
            complexity = _clamp01(float(complexity_raw))
            ambiguity = _clamp01(float(ambiguity_raw))
            expertise = _clamp01(float(expertise_raw))
            threshold = _clamp01(float(threshold_raw))
            topic_familiarity = _clamp01(float(topic_familiarity_raw))
            failure_signal = _clamp01(float(failure_signal_raw))
            reflective_intent = _clamp01(float(reflective_intent_raw))
            needs_external_evidence = _clamp01(float(needs_external_evidence_raw))
            needs_task_plan = _clamp01(float(needs_task_plan_raw))
            needs_multi_source_integration = _clamp01(
                float(needs_multi_source_integration_raw)
            )
            valence = _clamp11(float(valence_raw))
        except Exception:
            return None

        if intent_type_raw not in {"reflective", "factual", "mixed"}:
            intent_type_raw = "mixed"
        (
            needs_external_evidence,
            needs_task_plan,
            needs_multi_source_integration,
        ) = _calibrate_action_signals(
            needs_external_evidence=needs_external_evidence,
            needs_task_plan=needs_task_plan,
            needs_multi_source_integration=needs_multi_source_integration,
            ambiguity=ambiguity,
            intent_type=intent_type_raw,
            reflective_intent=reflective_intent,
        )

        return {
            "urgent": urgent,
            "complexity": complexity,
            "ambiguity": ambiguity,
            "expertise": expertise,
            "threshold": threshold,
            "topic_familiarity": topic_familiarity,
            "failure_signal": failure_signal,
            "intent_type": intent_type_raw,
            "reflective_intent": reflective_intent,
            "verify_request": verify_request,
            "needs_external_evidence": needs_external_evidence,
            "needs_task_plan": needs_task_plan,
            "needs_multi_source_integration": needs_multi_source_integration,
            "valence": valence,
        }

    except Exception:
        return None

def check():
    return [
        "context",
        ["urgent", 0.1],
        ["complexity", 0.2],
        ["ambiguity", 0.3],
        ["expertise", 0.4],
        ["threshold", 0.4],
    ]

def trunc(number):
    return math.trunc(number * 100) / 100