"""
eval_questions.py
--------------------
The labeled question set the retrieval harness is scored against.

Each question is tagged with a TYPE, and the mix is deliberate:

  lexical    - the question reuses the document's own terminology
               ("four-fifths rule", "disparate impact"). BM25 should do
               well here; if dense retrieval beat BM25 even on these, the
               question set would be suspect.
  paraphrase - the question describes the concept in a normal person's
               words, sharing little or no vocabulary with the source
               section ("if someone gets turned down, how fast..."). This
               is where dense retrieval should earn its keep.
  hard       - the question plausibly relates to several sections, and
               the correct one is decided by a specific detail.

RESULT NOTE (added after actually running the evaluation): the "hard"
bucket did NOT turn out to be hard for retrieval - every method scored a
perfect 1.000 MRR on it, same as the lexical bucket. Those questions are
conceptually hard for a human to ANSWER, but they still contain
lexically distinctive terms ("disparate impact ratio", "debt-to-income
ratio"), so finding the right section was easy. The bucket is kept, and
this note kept with it, because quietly deleting a bucket that failed to
discriminate would misrepresent how carefully the test set was designed.
The paraphrase bucket is the only one that actually separates the
methods, and every headline difference in the report comes from it.

Without the paraphrase bucket this comparison would be meaningless -
all three methods score 1.000 on everything else.
"""

# (question, correct_section_id, type)
EVAL_QUESTIONS = [
    # --- lexical: uses the corpus's own terms ---
    ("What is the four-fifths rule and what ratio triggers investigation?", "FL-300-1", "lexical"),
    ("What is the difference between disparate treatment and disparate impact?", "FL-200-3", "lexical"),
    ("What are the prohibited basis factors under Regulation B?", "FL-200-1", "lexical"),
    ("What must a model card record?", "MG-500-1", "lexical"),
    ("What is the delinquency look-back window?", "CP-100-3", "lexical"),
    ("What does demographic parity difference measure versus equalized odds?", "FL-300-2", "lexical"),
    ("What is the debt-to-income threshold for standard pricing?", "CP-100-1", "lexical"),

    # --- paraphrase: same meaning, different words ---
    ("If someone gets turned down, how quickly must the company tell them?", "AA-400-1", "paraphrase"),
    ("Can we just say 'you didn't meet our standards' when we reject somebody?", "AA-400-2", "paraphrase"),
    ("Is it a problem if we use where a person lives to decide their score?", "FL-200-2", "paraphrase"),
    ("How much borrowing history does somebody need before we can lend to them?", "CP-100-2", "paraphrase"),
    ("What do we do if our testing shows one group is being treated worse?", "FL-300-3", "paraphrase"),
    ("How often do we have to check our scoring systems again?", "MG-500-3", "paraphrase"),
    ("Are we allowed to tell the customer their neighborhood hurt their application?", "AA-400-3", "paraphrase"),
    ("Do we need to explain decisions for one specific person, or is overall importance enough?",
     "MG-500-2", "paraphrase"),

    # --- hard: several sections are plausible; a detail decides it ---
    ("A model has no protected attribute among its inputs. Can it still be discriminatory?",
     "FL-200-3", "hard"),
    ("Our disparate impact ratio is 0.83. Does that mean we are in the clear?", "FL-300-1", "hard"),
    ("A feature built from ZIP code correlates 0.45 with a protected class. What is the rule?",
     "FL-200-2", "hard"),
    ("We fixed the bias but the model got noticeably worse. Is that automatically the right call?",
     "FL-300-3", "hard"),
    ("An applicant has a 0.42 debt-to-income ratio. What happens to their application?",
     "CP-100-1", "hard"),
]
