"""
generate_policy_corpus.py
----------------------------
Builds the document corpus the RAG assistant retrieves over: a fictional
lender's internal credit policy manual plus plain-language summaries of
the fair-lending regulatory concepts the rest of this project implements
(adverse action notices, disparate impact, the 80% rule, protected
attributes, proxy variables).

IMPORTANT - these documents are SYNTHETIC and deliberately paraphrased.
They describe real regulatory CONCEPTS (ECOA/Regulation B adverse-action
requirements, the four-fifths rule, disparate impact vs. disparate
treatment) in this fictional lender's own words. They are NOT reproductions
of statutory text, and nothing here should be read as legal advice or as a
quotation of any real regulation. The point is a realistic retrieval
corpus with known, checkable content - not a legal reference.

Each document is chunked with a KNOWN structure so the evaluation harness
(rag/evaluate_retrieval.py) can score retrieval against a labeled
question -> correct-chunk mapping rather than eyeballing whether answers
"look right."

Output: data/policy_docs/*.md, data/policy_corpus.json
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "policy_docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Each doc: id, title, and sections. Section ids are stable so the eval
# harness can reference them as ground-truth answers.
DOCUMENTS = [
    {
        "doc_id": "CP-100",
        "title": "Meridian Lending Credit Policy — Underwriting Standards",
        "sections": [
            {
                "section_id": "CP-100-1",
                "heading": "Debt-to-Income Thresholds",
                "text": (
                    "Meridian Lending evaluates every application against a debt-to-income (DTI) "
                    "ratio computed as total monthly debt obligations divided by gross monthly "
                    "income. Applications with a DTI at or below 0.36 are eligible for standard "
                    "pricing. Applications with a DTI above 0.36 and at or below 0.45 require "
                    "secondary review by a senior underwriter. Applications with a DTI above 0.45 "
                    "are declined under standard policy unless a documented compensating factor "
                    "applies, such as verified reserves covering six months of payments."
                ),
            },
            {
                "section_id": "CP-100-2",
                "heading": "Credit History Length",
                "text": (
                    "A minimum credit history of 24 months is required for unsecured personal "
                    "loans. Applicants with 24 to 47 months of history are limited to a maximum "
                    "principal of 15,000. Applicants with 48 months or more of history may be "
                    "considered for the full product range. Thin-file applicants may substitute "
                    "12 months of verified rental payment history for up to 12 months of the "
                    "credit history requirement."
                ),
            },
            {
                "section_id": "CP-100-3",
                "heading": "Delinquency Look-Back",
                "text": (
                    "The delinquency look-back window is 24 months. A single 30-day delinquency "
                    "within the window does not by itself trigger decline. Two or more 30-day "
                    "delinquencies, or any single delinquency of 60 days or greater, require "
                    "secondary review. Any charge-off, repossession, or bankruptcy discharged "
                    "within 24 months results in decline under standard policy."
                ),
            },
        ],
    },
    {
        "doc_id": "FL-200",
        "title": "Fair Lending Standards — Protected Attributes and Prohibited Factors",
        "sections": [
            {
                "section_id": "FL-200-1",
                "heading": "Prohibited Basis Factors",
                "text": (
                    "Credit decisions at Meridian Lending must never be based on a prohibited "
                    "basis factor. Under the Equal Credit Opportunity Act as implemented by "
                    "Regulation B, prohibited bases include race, color, religion, national "
                    "origin, sex, marital status, age (provided the applicant has capacity to "
                    "contract), receipt of public assistance income, and the good-faith exercise "
                    "of rights under consumer credit protection law. No model feature, scorecard "
                    "input, or underwriter judgment may use these attributes as a decision input."
                ),
            },
            {
                "section_id": "FL-200-2",
                "heading": "Proxy Variables and Redlining Risk",
                "text": (
                    "A feature that does not name a protected attribute may still function as a "
                    "proxy for one. Geographic features carry particular risk because residential "
                    "patterns are correlated with protected class membership. Any feature derived "
                    "from an applicant's address, neighborhood, census tract, or ZIP code must be "
                    "reviewed for proxy behavior before entering a model. A geographic feature is "
                    "presumed to be a proxy if its correlation with a protected class indicator "
                    "exceeds 0.30 and it does not demonstrate independent predictive value after "
                    "controlling for verified financial capacity."
                ),
            },
            {
                "section_id": "FL-200-3",
                "heading": "Disparate Treatment versus Disparate Impact",
                "text": (
                    "Disparate treatment occurs when an applicant is treated differently because "
                    "of a protected attribute, whether or not the difference was intentional. "
                    "Disparate impact occurs when a facially neutral policy applied uniformly "
                    "produces a materially worse outcome for a protected class and the policy is "
                    "not justified by business necessity, or a less discriminatory alternative "
                    "achieving the same business objective is available. A model may produce "
                    "disparate impact even when no protected attribute appears among its inputs."
                ),
            },
        ],
    },
    {
        "doc_id": "FL-300",
        "title": "Fair Lending Standards — Testing and the Four-Fifths Rule",
        "sections": [
            {
                "section_id": "FL-300-1",
                "heading": "The Four-Fifths (80%) Rule",
                "text": (
                    "The four-fifths rule is the primary screening test for adverse impact. "
                    "Compute the selection rate for each group, then divide the selection rate of "
                    "the lower group by that of the higher group to obtain the disparate impact "
                    "ratio. A ratio below 0.80 is treated as evidence of adverse impact requiring "
                    "investigation. The four-fifths rule is a screening threshold and not a safe "
                    "harbor: a ratio at or above 0.80 does not by itself establish that a model "
                    "is fair, and statistical significance testing should accompany it."
                ),
            },
            {
                "section_id": "FL-300-2",
                "heading": "Demographic Parity and Equalized Odds",
                "text": (
                    "Demographic parity difference measures the gap in approval rates between "
                    "groups without conditioning on outcome. Equalized odds compares true positive "
                    "rates and false positive rates across groups, conditioning on the actual "
                    "outcome. These metrics can conflict: a model satisfying demographic parity "
                    "may violate equalized odds when the underlying outcome base rates genuinely "
                    "differ between groups. Model reviews must state which fairness definition "
                    "was chosen and why, because no single model can generally satisfy all of "
                    "them simultaneously."
                ),
            },
            {
                "section_id": "FL-300-3",
                "heading": "Remediation and the Fairness-Accuracy Tradeoff",
                "text": (
                    "When testing identifies adverse impact, remediation options include removing "
                    "or replacing the offending feature, reweighting training data, and applying "
                    "group-specific decision thresholds. Every remediation must be measured for "
                    "its effect on predictive accuracy, and the tradeoff must be documented. A "
                    "remediation that eliminates disparity while destroying model performance is "
                    "not automatically preferable, and the review committee must record the "
                    "rationale for the option selected."
                ),
            },
        ],
    },
    {
        "doc_id": "AA-400",
        "title": "Adverse Action Notice Requirements",
        "sections": [
            {
                "section_id": "AA-400-1",
                "heading": "Timing and Delivery",
                "text": (
                    "When an application is declined, the applicant must receive an adverse action "
                    "notice within 30 days of receipt of a completed application. The notice must "
                    "be delivered in writing or, where the applicant has consented, electronically. "
                    "A notice is also required when an existing account is terminated or when "
                    "credit terms are changed unfavorably for a specific account holder."
                ),
            },
            {
                "section_id": "AA-400-2",
                "heading": "Statement of Specific Reasons",
                "text": (
                    "An adverse action notice must disclose the specific principal reasons for the "
                    "decision. Generic statements such as 'did not meet our credit standards' or "
                    "'internal scoring model' are insufficient. Where a credit scoring model "
                    "produced the decision, the reasons disclosed must be the factors that most "
                    "affected that individual applicant's score. Meridian Lending discloses at "
                    "most four principal reasons per notice, ordered by their contribution to the "
                    "decision."
                ),
            },
            {
                "section_id": "AA-400-3",
                "heading": "Prohibited Content in Notices",
                "text": (
                    "An adverse action notice must never state or imply that a protected attribute "
                    "influenced the decision, and must never cite a reason that was not an actual "
                    "input to the decision. Reasons must be traceable to model features or "
                    "documented underwriter findings. Where a geographic or neighborhood feature "
                    "contributed to a score, it must not be disclosed as a reason and must instead "
                    "be escalated to fair lending review, because reliance on it may itself "
                    "constitute a violation."
                ),
            },
        ],
    },
    {
        "doc_id": "MG-500",
        "title": "Model Governance — Documentation and Review",
        "sections": [
            {
                "section_id": "MG-500-1",
                "heading": "Model Cards",
                "text": (
                    "Every model used in a credit decision must have a model card recording its "
                    "intended use, training data period, feature list, performance metrics overall "
                    "and by group, fairness test results, known limitations, and the date of last "
                    "review. A model without a current model card may not be deployed to "
                    "production decisioning."
                ),
            },
            {
                "section_id": "MG-500-2",
                "heading": "Explainability Requirements",
                "text": (
                    "Any model producing credit decisions must support per-applicant reason "
                    "generation. Global feature importance alone is insufficient because adverse "
                    "action notices require the reasons specific to the individual applicant. "
                    "Feature attribution methods used for this purpose must be documented, and "
                    "the review must confirm that attributions are consistent with the model's "
                    "actual behavior rather than assumed from the model family."
                ),
            },
            {
                "section_id": "MG-500-3",
                "heading": "Review Cadence",
                "text": (
                    "Credit models undergo full fair lending review annually and after any change "
                    "to the feature set, training population, or decision threshold. Monitoring "
                    "for population drift runs monthly. A material drift finding triggers an "
                    "off-cycle review regardless of when the last annual review occurred."
                ),
            },
        ],
    },
]


def main():
    corpus = []
    for doc in DOCUMENTS:
        lines = [f"# {doc['title']}", "", f"_Document ID: {doc['doc_id']}_", ""]
        for sec in doc["sections"]:
            lines += [f"## {sec['heading']}", "", sec["text"], ""]
            corpus.append({
                "chunk_id": sec["section_id"],
                "doc_id": doc["doc_id"],
                "doc_title": doc["title"],
                "heading": sec["heading"],
                "text": sec["text"],
            })
        (DOCS_DIR / f"{doc['doc_id']}.md").write_text("\n".join(lines), encoding="utf-8")

    with open(DATA_DIR / "policy_corpus.json", "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2)

    print(f"Wrote {len(DOCUMENTS)} policy documents ({len(corpus)} sections) to {DOCS_DIR}")
    for doc in DOCUMENTS:
        print(f"  {doc['doc_id']}: {doc['title']} ({len(doc['sections'])} sections)")


if __name__ == "__main__":
    main()
