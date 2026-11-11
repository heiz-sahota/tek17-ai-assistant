# Judge rubric, version 2

The grading guide for the LLM judge, and for you when you grade answers by hand
to calibrate it. The judge and you must use exactly the same rubric.

**What the judge receives for each answer**

| Input | Example |
|---|---|
| QUESTION | What is the Zone D multiplier now? |
| SOURCES | The chunks your system retrieved, each labelled `[reference]` |
| ANSWER | The assistant's answer and its citations |
| REFERENCE | The golden-set answer, and the question `type` (`fact`, `number`, `condition`, `out_of_scope` ...) |

**How to grade**

- Grade each criterion **separately**. One answer can pass one criterion and fail another.
- Write the **reasoning first**, then the verdict: `pass`, `fail` or `n/a`.
- Judge only against the SOURCES and the REFERENCE, never against outside knowledge.
- When unsure between pass and fail, choose **fail** and say why. A strict judge is easier to calibrate than a lenient one.

---

## 1. Grounded

**Question:** Is every claim in the answer supported by the SOURCES?

| Verdict | When |
|---|---|
| pass | Every fact, number and condition in the answer appears in the sources, possibly reworded |
| fail | Any claim is missing from the sources, contradicts them, or is a number that doesn't match |
| n/a | The answer is a refusal with no factual claims |

**Edge cases**
- Rewording is fine: "12.5%" for "12.5 percent".
- A claim that is true in real life but not in the sources **fails**. The assistant may only use its sources.
- A calculation passes if each input comes from the sources and the arithmetic is right.

**Pass example**
- Question: What is the fuel surcharge?
- Sources: `[01_Pricing_Model_Overview / 2]` "Fuel surcharge: 12.5 percent of the subtotal, from 1 January 2026."
- Answer: "The fuel surcharge is 12.5 percent of the subtotal, from 1 January 2026."
- Reasoning: Both the rate and the base (subtotal) and the date appear in the source. → **pass**

**Fail example**
- Question: What is the fuel surcharge?
- Same source.
- Answer: "The fuel surcharge is 12.5 percent of the subtotal and is reviewed every quarter."
- Reasoning: Nothing in the sources says it is reviewed quarterly. → **fail**

---

## 2. Correct

**Question:** Does the answer match the REFERENCE answer in substance?

| Verdict | When |
|---|---|
| pass | Same facts and values as the reference, in any wording; extra correct detail is fine |
| fail | Contradicts the reference, gives an outdated value, or answers a different question |
| n/a | The question is `out_of_scope` |

**Edge cases**
- An old value that was correct before a change **fails**, even if it appears in a change log.
- Units and rounding must match: "503 NOK" fails if the reference is "503.50 NOK".

**Pass example**
- Question: What is the Zone D multiplier now?
- Reference: 1.60, since 1 July 2025.
- Answer: "Zone D uses a multiplier of 1.60, raised from 1.50 on 1 July 2025."
- Reasoning: The current value 1.60 matches; mentioning the old value as history is correct extra detail. → **pass**

**Fail example**
- Same question and reference.
- Sources include `[06_Pricing_Change_Log / Version 2025.2]` "Zone D multiplier raised from 1.50 to 1.60."
- Answer: "The Zone D multiplier is 1.50."
- Reasoning: 1.50 is the value before the change; the question asks for the current one. → **fail**

---

## 3. Complete

**Question:** Does the answer include every essential part of the reference that the question needs?

| Verdict | When |
|---|---|
| pass | All parts needed to answer the question are there: values, conditions, exceptions |
| fail | A part the question needs is missing, so the answer is misleading or unusable on its own |
| n/a | The question is `out_of_scope` |

**Edge cases**
- Missing background is fine; missing a **condition** that changes the answer is not.
- For "when and how much" questions, both parts are essential.
- Judge completeness against what the QUESTION asks. Details in the reference answer that the question doesn't ask for are not required.

**Pass example**
- Question: When does the oversize surcharge apply and how much is it?
- Reference: 150 NOK, when the longest side is over 120 cm.
- Answer: "An oversize surcharge of 150 NOK applies when the parcel's longest side is over 120 cm."
- Reasoning: Both the amount and the condition are present. → **pass**

**Fail example**
- Same question and reference.
- Answer: "The oversize surcharge is 150 NOK."
- Reasoning: The question also asks when it applies; the condition (longest side over 120 cm) is missing. → **fail**

---

## 4. Cited

**Question:** Does each citation point to a source that actually supports the claim it is used for?

| Verdict | When |
|---|---|
| pass | Every citation contains the information it is attached to, and every key claim has a citation |
| fail | A citation doesn't support its claim, a key claim has no citation, or citations are padded with irrelevant sources |
| n/a | The answer is a refusal with no citations |

**Edge cases**
- A citation that was really retrieved but is **irrelevant** fails. The code check only proves a citation exists; this criterion checks that it supports the claim.
- Citing two sources for one claim is fine if both support it.

**Pass example**
- Question: Which zone is postal code 6995?
- Answer: "Postal code 6995 is an island exception and is always priced as Zone D." Citations: `03_Postal_Zone_Rules / Rule Z-2`.
- Reasoning: Rule Z-2 lists 6995 among the island exceptions priced as Zone D. → **pass**

**Fail example**
- Same question.
- Answer: same text. Citations: `03_Postal_Zone_Rules / Rule Z-2`, `02_Weight_and_Size_Price_Table / 4`, `05_Master_Data_Admin_Guide / 1`.
- Reasoning: Rule Z-2 supports the answer, but the fixed-fees table and the admin guide say nothing about postal code 6995. Padding with irrelevant sources. → **fail**

---

## 5. Refusal correct

**Question:** Did the assistant answer when it should, and decline when it should?

| Verdict | When |
|---|---|
| pass | An in-scope question is answered, or an out-of-scope question is declined or explained as out of scope |
| fail | An out-of-scope question gets an answer or a guess, or an in-scope question is refused |

**Edge cases**
- Explaining from the sources that something is **not covered** counts as a correct refusal.
- A refusal of an in-scope question fails even if it is polite: it means retrieval or the prompt let the user down.
- Off-task requests (poems, code, general chat) must be declined.

**Pass example (out of scope)**
- Question: What is the price to send a parcel to Sweden? Type: `out_of_scope`.
- Answer: "The pricing model doesn't cover parcels sent outside Norway, including Sweden." Citation: `01_Pricing_Model_Overview / 7`.
- Reasoning: No price was invented, and the answer explains from the sources that it is out of scope. → **pass**

**Fail example (out of scope)**
- Same question.
- Answer: "Parcels to Sweden cost about 300 NOK."
- Reasoning: A price was invented for something the sources don't cover. → **fail**

**Fail example (in scope, refused)**
- Question: How are final prices rounded? Type: `fact`.
- Answer: "I couldn't find this in the sources I have access to."
- Reasoning: The rounding rule is in the pricing documents (nearest 0.50 NOK, halfway rounded up); the question is in scope. → **fail**

---

## A worked example across all five criteria

- Question: Does the zone multiplier apply to the home delivery fee?
- Reference: No. The zone multiplier is applied in step 3, and fixed fees such as home delivery are added after it, in step 4. Type: `condition`.
- Sources: `[01_Pricing_Model_Overview / 3]` the calculation order.
- Answer: "Yes, the home delivery fee of 49 NOK is multiplied by the zone multiplier." Citations: `01_Pricing_Model_Overview / 3`.

| Criterion | Reasoning | Verdict |
|---|---|---|
| Grounded | The cited calculation order adds fixed fees after the zone multiplier, so the claim contradicts the source | fail |
| Correct | The reference says no; the answer says yes | fail |
| Complete | It gives the fee amount but reaches the wrong conclusion; completeness can't pass when the core is wrong | fail |
| Cited | The citation is the right section, but it does not support the claim made | fail |
| Refusal correct | In-scope question, and it answered | pass |

The citation check in code would **pass** this answer, because `01_Pricing_Model_Overview / 3` was really retrieved. This is exactly what the judge is for.

---

## Output format for the judge

One JSON object per answer, reasoning before each verdict:

```json
{
  "rubric_version": "1",
  "grounded":        {"reasoning": "...", "verdict": "pass|fail|n/a"},
  "correct":         {"reasoning": "...", "verdict": "pass|fail|n/a"},
  "complete":        {"reasoning": "...", "verdict": "pass|fail|n/a"},
  "cited":           {"reasoning": "...", "verdict": "pass|fail|n/a"},
  "refusal_correct": {"reasoning": "...", "verdict": "pass|fail"}
}
```

**Judge settings:** temperature 0, JSON output, and if possible a different and stronger deployment than the one that answers.

**Reported metrics:** pass rate per criterion over the questions where it applies (`n/a` excluded), plus how many answers passed all applicable criteria.

**Changing the rubric:** raise the version number, and re-run both the judge and the calibration. Scores from different rubric versions are not comparable.