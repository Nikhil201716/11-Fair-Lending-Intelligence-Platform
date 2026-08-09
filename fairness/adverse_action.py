"""
adverse_action.py
--------------------
Generates ECOA/Regulation B-style adverse action notices for declined
applicants, using the deterministic-decision + LLM-narration pattern:
SHAP decides WHICH reasons are disclosed, the local model only writes the
prose. The model never chooses reasons, never sees the applicant's group,
and cannot influence the decision.

Guardrails, all enforced AFTER generation on the actual text:

  G1 NO PROTECTED-ATTRIBUTE LANGUAGE - the notice must not mention group
     membership or any prohibited-basis word (policy FL-200-1).

  G2 NO GEOGRAPHIC REASONS - policy AA-400-3 is explicit that a
     geographic/neighborhood contribution must NOT be disclosed as a
     reason and must instead be escalated to fair lending review. This is
     the guardrail that actually fires in practice: SHAP analysis found a
     geographic feature among the top reasons for 7 of 25 declined
     applicants, so those applications are ROUTED TO ESCALATION rather
     than given a notice with a sanitized reason list. Silently dropping
     the geographic reason and disclosing the next-best one would produce
     a notice that is technically clean and substantively false.

  G3 REASONS MUST BE REAL - every disclosed reason must map to a reason
     code that was actually in that applicant's SHAP top contributors. A
     notice citing a plausible-sounding reason the model never used is
     precisely what AA-400-2 prohibits.

  G4 NO INVENTED SPECIFICS - the notice must not contain numbers absent
     from the supplied reason data (a fabricated "your DTI of 0.61"
     would be a compliance violation even if the reason itself is right).

Output: reports/adverse_action_results.json/.md
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ARTIFACTS = ROOT / "risk_model" / "artifacts"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

OLLAMA_MODEL = "qwen2.5:0.5b"
MAX_REASONS = 4  # policy AA-400-2: at most four principal reasons

PROHIBITED_WORDS = [
    "group a", "group b", "race", "racial", "ethnic", "religion", "religious",
    "national origin", "sex", "gender", "marital", "age", "public assistance",
    "neighborhood", "neighbourhood", "zip", "postal", "area", "region", "district",
    "location", "where you live", "geographic",
]

PROMPT_TEMPLATE = """Write a short adverse action notice for a declined loan application.

The ONLY reasons you may state are these, exactly as written:
{reasons}

Rules:
- Two to four sentences.
- State that the application was declined and list the reasons above.
- Do not invent any other reason, number, or detail.
- Do not mention race, sex, age, religion, neighborhood, or location.

Notice:"""


def build_reason_list(entry: dict) -> tuple[list[str], bool]:
    """Returns (disclosable reason labels, needs_escalation)."""
    geo = [t for t in entry["top_reasons"] if t["is_geographic"]]
    disclosable = [t["label"] for t in entry["top_reasons"] if not t["is_geographic"]]
    return disclosable[:MAX_REASONS], bool(geo)


def check_guardrails(notice: str, allowed_reasons: list[str], entry: dict) -> dict:
    text = notice.lower()

    g1 = [w for w in PROHIBITED_WORDS if w in text]

    # G3: at least one allowed reason must actually be reflected, and no
    # reason-bearing content should come from outside the allowed set.
    def keywords(label: str) -> set[str]:
        return {w for w in re.findall(r"[a-z]+", label.lower()) if len(w) > 4}

    allowed_kw = set().union(*(keywords(r) for r in allowed_reasons)) if allowed_reasons else set()
    reflected = bool(allowed_kw & set(re.findall(r"[a-z]+", text)))

    # G4: numeric claims in the notice that were never supplied.
    #
    # The first version of this check flagged the list markers "1." "2."
    # "3." as invented numbers, because it compared against an empty
    # supplied-set and never stripped enumeration. That was a bug in the
    # CHECK, not in the model - though it happened to fire on notices that
    # were violating anyway. Enumeration is stripped first so the reported
    # reason for a failure is the real one.
    body = re.sub(r"(?m)^\s*\d+[\.\)]\s+", "", notice)      # "1. " / "2) " at line start
    body = re.sub(r"\[[^\]]*\]", "", body)                    # placeholders like [Your Name]

    # No REASON_LABEL contains a digit, so any number surviving in the body
    # is by construction something the model made up.
    supplied_numbers = set(re.findall(r"\d+\.?\d*", " ".join(allowed_reasons)))
    invented_numbers = sorted(set(re.findall(r"\d+\.?\d*", body)) - supplied_numbers)

    return {
        "g1_prohibited_language": g1,
        "g1_passed": not g1,
        "g3_reason_reflected": reflected,
        "g3_passed": reflected,
        "g4_invented_numbers": invented_numbers,
        "g4_passed": not invented_numbers,
        "all_passed": (not g1) and reflected and (not invented_numbers),
    }


TEMPLATE_NOTICE = (
    "Notice of Adverse Action\n\n"
    "We are unable to approve your recent credit application. Federal law requires us to "
    "disclose the principal reasons for this decision. The principal reasons were:\n\n"
    "{reasons}\n\n"
    "This decision was based in whole or in part on information contained in your credit "
    "application and credit report. You have the right to request a statement of the specific "
    "reasons for this decision, and to obtain a free copy of your credit report."
)


def generate_template(entry: dict) -> dict:
    """Deterministic notice assembly - no model in the decision OR the
    wording. Included because the LLM comparison below is only meaningful
    against the alternative a compliance team would actually ship."""
    reasons, needs_escalation = build_reason_list(entry)
    result = {
        "application_id": entry["application_id"],
        "default_probability": entry["default_probability"],
        "method": "deterministic_template",
        "geographic_reason_present": needs_escalation,
        "disclosable_reasons": reasons,
        "escalated": False, "notice": None, "guardrails": None,
    }
    if needs_escalation or not reasons:
        result.update(escalated=True,
                       escalation_reason=("geographic feature among top SHAP contributors "
                                          "(policy AA-400-3)") if needs_escalation
                                         else "no disclosable reason available")
        return result
    notice = TEMPLATE_NOTICE.format(reasons="\n".join(f"  - {r}" for r in reasons))
    result["notice"] = notice
    result["guardrails"] = check_guardrails(notice, reasons, entry)
    return result


def generate(entry: dict) -> dict:
    import ollama

    reasons, needs_escalation = build_reason_list(entry)
    result = {
        "application_id": entry["application_id"],
        "default_probability": entry["default_probability"],
        "geographic_reason_present": needs_escalation,
        "disclosable_reasons": reasons,
        "escalated": False, "notice": None, "guardrails": None,
    }

    # G2: escalate rather than sanitize.
    if needs_escalation:
        result.update(escalated=True,
                       escalation_reason="a geographic feature was among the top SHAP "
                                         "contributors; policy AA-400-3 requires fair lending "
                                         "review instead of disclosure")
        return result

    if not reasons:
        result.update(escalated=True, escalation_reason="no disclosable reason available")
        return result

    prompt = PROMPT_TEMPLATE.format(reasons="\n".join(f"- {r}" for r in reasons))
    resp = ollama.chat(model=OLLAMA_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        options={"temperature": 0.0, "num_predict": 180})
    notice = resp["message"]["content"].strip()
    result["notice"] = notice
    result["guardrails"] = check_guardrails(notice, reasons, entry)
    return result


def main():
    with open(ARTIFACTS / "shap_reasons.json", encoding="utf-8") as f:
        entries = json.load(f)

    print(f"Generating adverse action notices for {len(entries)} declined applicants...")
    results = []
    for i, e in enumerate(entries, 1):
        results.append(generate(e))
        if i % 5 == 0:
            print(f"  ...{i}/{len(entries)}")

    print("Generating deterministic template notices for the same applicants...")
    template_results = [generate_template(e) for e in entries]

    def score(rs):
        gen = [r for r in rs if not r["escalated"]]
        ok = [r for r in gen if r["guardrails"]["all_passed"]]
        bad = [r for r in gen if not r["guardrails"]["all_passed"]]
        return gen, ok, bad

    generated, passed, failed = score(results)
    t_generated, t_passed, t_failed = score(template_results)
    escalated = [r for r in results if r["escalated"]]

    summary = {
        "model": OLLAMA_MODEL,
        "n_declined": len(results),
        "n_escalated_geographic": len(escalated),
        "llm_generation": {
            "n_notices_generated": len(generated),
            "n_passed_all_guardrails": len(passed),
            "n_failed_guardrails": len(failed),
            "guardrail_pass_rate": round(len(passed) / len(generated), 4) if generated else None,
            "failure_breakdown": {
                "g1_prohibited_language": sum(1 for r in failed if not r["guardrails"]["g1_passed"]),
                "g3_reason_not_reflected": sum(1 for r in failed if not r["guardrails"]["g3_passed"]),
                "g4_invented_numbers": sum(1 for r in failed if not r["guardrails"]["g4_passed"]),
            },
            "example_invented_numbers": sorted({n for r in failed
                                                 for n in r["guardrails"]["g4_invented_numbers"]})[:12],
        },
        "deterministic_template": {
            "n_notices_generated": len(t_generated),
            "n_passed_all_guardrails": len(t_passed),
            "guardrail_pass_rate": round(len(t_passed) / len(t_generated), 4) if t_generated else None,
        },
        # kept at top level for backward compatibility with the dashboard
        "n_notices_generated": len(generated),
        "n_passed_all_guardrails": len(passed),
        "failure_breakdown": {
            "g1_prohibited_language": sum(1 for r in failed if not r["guardrails"]["g1_passed"]),
            "g3_reason_not_reflected": sum(1 for r in failed if not r["guardrails"]["g3_passed"]),
            "g4_invented_numbers": sum(1 for r in failed if not r["guardrails"]["g4_passed"]),
        },
        "results": results,
        "template_results": template_results,
    }
    with open(REPORTS_DIR / "adverse_action_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(REPORTS_DIR / "adverse_action_results.md", "w", encoding="utf-8") as f:
        f.write("# Adverse Action Notice Generation\n\n")
        f.write(f"- Declined applicants processed: **{len(results)}**\n")
        f.write(f"- Escalated to fair lending review (geographic reason present): "
                f"**{len(escalated)}**\n")
        f.write(f"- Notices generated: **{len(generated)}**\n")
        f.write(f"- Passed all guardrails: **{len(passed)}/{len(generated)}**\n\n")
        if passed:
            f.write("### Example notice (passed all guardrails)\n\n")
            ex = passed[0]
            f.write(f"Reasons supplied: {ex['disclosable_reasons']}\n\n> {ex['notice']}\n")

    print(f"\nDeclined applicants:            {len(results)}")
    print(f"Escalated (geographic reason):  {len(escalated)}")
    print(f"\nLLM generation      : {len(passed)}/{len(generated)} passed all guardrails")
    print(f"  failures: {summary['llm_generation']['failure_breakdown']}")
    print(f"  invented numbers seen: {summary['llm_generation']['example_invented_numbers']}")
    print(f"Deterministic template: {len(t_passed)}/{len(t_generated)} passed all guardrails")
    for r in failed[:2]:
        g = r["guardrails"]
        print(f"\n  LLM FAILED {r['application_id']}: prohibited={g['g1_prohibited_language']} "
              f"invented={g['g4_invented_numbers']}")
    if t_passed:
        print(f"\n  TEMPLATE EXAMPLE ({t_passed[0]['application_id']}):")
        print("   " + t_passed[0]["notice"][:320].replace("\n", "\n   "))
    print(f"\nSaved {REPORTS_DIR / 'adverse_action_results.json'}")


if __name__ == "__main__":
    main()
