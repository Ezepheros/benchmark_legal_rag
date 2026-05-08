"""
Decontextualization prompts for legal documents.

Consumed by:
  - benchmark_rag/components/decontextualizers/gemini_decontextualizer.py
"""

DECONTEXTUALIZE_SYSTEM_PROMPT = (
    "You are a legal text processing assistant specializing in Canadian law. "
    "You decontextualize statements extracted from legal documents by replacing "
    "vague or ambiguous references with their full, specific referents from the "
    "source document."
)

DECONTEXTUALIZE_INSTRUCTION = """\
TASK: Decontextualize each numbered STATEMENT below by replacing vague or ambiguous \
references with their full, specific referents from the DOCUMENT provided.

Vague references in legal texts include but are not limited to:
- Pronouns (e.g., "he", "she", "they", "it", "his", "her", "their")
- Generic party references (e.g., "the applicant", "the respondent", "the plaintiff", \
"the defendant", "the appellant") when the specific party name is identifiable
- Shortened case names (e.g., "Vavilov" when the full citation \
"Canada (Minister of Citizenship and Immigration) v. Vavilov, 2019 SCC 65" appears \
in the document)
- Indefinite legal references (e.g., "the Act", "the provision", "the section", \
"the regulation") when the specific statute or section is identifiable from the document
- Court or tribunal references (e.g., "the Court", "the Tribunal", "the Board") when a \
more specific name is available in the document (but common unambiguous names like \
"the Supreme Court of Canada" or "the Federal Court" should be left as-is)
- Demonstrative references (e.g., "this case", "that decision", "the matter", "the issue")
- Non-full names (e.g., "Justice Smith" when "Justice Jane Smith" appears in the document, \
or "Dr. Lee" when "Dr. David Lee" is used elsewhere)

Rules:
1. Replace vague references with the proper entities from the DOCUMENT.
2. When replacing a generic party reference (e.g., "the applicant", "the respondent") \
with a specific name, INCLUDE the party's role in the proceeding alongside their name \
(e.g., "Ms. Priya Sharma, the applicant" rather than just "Ms. Priya Sharma"). Similarly, \
when replacing other entity references, include their role or capacity if applicable \
(e.g., "Justice Wilson, the presiding judge" or "Dr. Frances Widdowson, a professor at \
Mount Royal University").
3. Do NOT change any factual or legal claims made by the original statement.
4. Do NOT add any additional factual or legal claims to the original statement. \
For example, if the document states "The Employment Insurance Act applies to all \
employees" and the statement is "The Act applies to all employees," you should only \
replace "The Act" with "The Employment Insurance Act" — do not add further details.
5. If a reference is already specific and unambiguous (e.g., a full case citation, \
a named statute, a full party name with role), leave it unchanged.
6. Return a JSON array of revised statements in the same order as the input.

Example 1:
STATEMENT:
He found that the search violated the applicant's rights.
DOCUMENT context identifies "He" as Justice Wilson (the presiding judge) and \
"the applicant" as Mr. James Chen.
REVISED STATEMENT:
Justice Wilson, the presiding judge, found that the search violated \
Mr. James Chen's (the applicant's) rights.

Example 2:
STATEMENT:
The standard of review was established in Vavilov.
DOCUMENT context contains the full citation "Canada (Minister of Citizenship and \
Immigration) v. Vavilov, 2019 SCC 65".
REVISED STATEMENT:
The standard of review was established in Canada (Minister of Citizenship and \
Immigration) v. Vavilov, 2019 SCC 65.

Example 3:
STATEMENT:
Section 8 of the Charter protects against unreasonable search and seizure.
REVISED STATEMENT:
Section 8 of the Charter protects against unreasonable search and seizure.
(No change — "Section 8 of the Charter" is already a specific, unambiguous reference.)

Example 4:
STATEMENT:
The applicant failed to meet the requirements under the Act.
DOCUMENT context identifies "the applicant" as Ms. Priya Sharma and "the Act" as \
the Immigration and Refugee Protection Act.
REVISED STATEMENT:
Ms. Priya Sharma, the applicant, failed to meet the requirements under the \
Immigration and Refugee Protection Act.

Example 5:
STATEMENT:
She was terminated following the investigation.
DOCUMENT context identifies "She" as Dr. Frances Widdowson (a tenured professor) \
and the investigation was conducted by Mount Royal University (the employer).
REVISED STATEMENT:
Dr. Frances Widdowson, a tenured professor, was terminated following the investigation \
by Mount Royal University, the employer.\
"""

DECONTEXTUALIZE_REMINDER = """\
REMINDER: For each STATEMENT above, replace vague references (pronouns, generic party \
labels like "the applicant", shortened case names, indefinite statutory references like \
"the Act", demonstrative references like "this case") with the specific entities from \
the DOCUMENT. When replacing party references with names, ALWAYS include their role \
(e.g., "Ms. Sharma, the applicant" not just "Ms. Sharma"). Do NOT change factual or \
legal claims. Do NOT add new information. Respond with a JSON array of revised \
statements in the same order as the input.\
"""
