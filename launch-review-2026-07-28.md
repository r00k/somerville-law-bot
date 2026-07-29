# Launch-day answer review (2026-07-28 → 07-29 morning)

Review of all 28 prod Q&A records since launch. Every answer was checked
against the corpus by independent review passes (citations grep-verified
verbatim, claims spot-checked for correctness and completeness). Raw logs:
`logs/qa-2026-07-28.jsonl`, `logs/qa-2026-07-29.jsonl` on the Railway volume.

## Headline

- 28 requests, 0 errors, 1 empty answer (client disconnect on a retry).
- 107/107 emitted citations verified; **zero fabricated quotes or hallucinated
  sections** across all answers, including the two "unusual laws" listicles.
- 1 outright wrong answer, ~4 answers with meaningful nits, rest solid.
- Latency: median 22s, p90 41s, max 63s.
- Feedback: one thumbs-up (request `4408a86df962` = the ADU follow-up, Q23).

## To address

### 1. Citation verifier drops true quotes over OCR spacing (bug, systemic)

The corpus contains OCR-style spacing like `"party wall ."` (space before
period). Exact-match verification drops correct quotes that normalize this.
All 4 dropped citations on launch day were this, not fabrications. Worst case:
Q23 (ADU follow-up) lost the citation for its **load-bearing** claim (the
§2.4.3.b separation-measurement rule), and Q27 (dryer vent) shipped with 0
surviving citations despite a correct supporting quote.

**Fix:** normalize whitespace (collapse runs, strip space-before-punctuation)
on both sides before comparing in the verifier.

### 2. Inverted answer: leaf blower in July (Q12, request `~13:34 UTC 7/28`)

Sec. 9-120(b)(1) permits leaf blowers ONLY Mar 15–May 31 and Oct 1–Dec 15.
Q12 quoted this verbatim, then mislabeled the allowed windows as "banned
periods" and answered **"Yes, July is allowed"** — at HIGH confidence, while
its own commercial-operator paragraph applied the logic correctly. The same
question was answered correctly twice (Q4 on 7/28, Q28 on 7/29), so this is
a stochastic misread (2/3 correct).

**Fix:** add eval question "Can I use a gas-powered leaf blower in July?"
with substring checks asserting the prohibition (e.g. "not"/"prohibited"
framing + the two seasonal windows + the 5-min/day de minimis exception).
Given the flake profile, run it at the usual multi-pass cadence.

### 3. Required-citation schema pressures out-of-scope declines (Q14)

The pi/Indiana-pi-bill troll question was correctly declined, but the model
attached a deliberately irrelevant citation (a noise-ordinance preamble) "to
show that a search returns nothing relevant" — schema pressure to cite
*something*. Note tension with the schema-required-beats-prompt principle:
don't make `citations` optional globally; consider allowing an empty
citations array only when the answer is an out-of-scope decline, or an
explicit `out_of_scope` response shape.

### 4. Corpus-boundary caveats missing where state law governs (Q10, Q18)

- Q10 (mice in apartments): never flagged that the State Sanitary Code
  (105 CMR 410) — which puts extermination on the owner in 2+ unit
  buildings — is outside the corpus. The ordinance's "owner and/or occupant"
  framing could mislead a tenant. The grilling/propane answers (Q2, Q5, Q6)
  handled the same situation correctly by naming the state-law gap.
- Q18 (deck boards): permit trigger for repairs is really 780 CMR (state
  building code, not in corpus); the answer punted to ISD but presented
  "normal maintenance ⇒ no permit" as if grounded, when the corpus never
  says normal maintenance is exempt from permits.

**Fix idea:** prompt nudge — when the governing law is state-level and
outside the corpus, say so by name. Eval-able with substring checks on Q10.

### 5. Uncited enforcement-practice claim (Q7)

Q7 (firepit): "In practice, many residents use contained gas firepits
without a permit" — an empirical enforcement claim with no source. Q26 (same
question, later) answered identically on the law but stayed within the
corpus. Prompt should forbid enforcement-practice speculation.

### 6. Manufactured corpus gap in ADU answer (Q22)

Q22 claimed the corpus "does not spell out by-right vs. special permit" for
Backyard Cottages — false: §3.1.6.c says they're permitted **by right** in
the NR district. All numeric specs in the answer verified. Claims *about the
corpus* ("the corpus doesn't say X") should be held to the same standard as
claims about the law.

## Smaller nits (no action needed unless convenient)

- Q4 (leaf blower): overstated the commercial operations-plan exemption —
  §9-120(b)(3) still binds commercial operators to the seasonal/hour limits.
- Q1 (chickens): buried concrete zoning rules (no hens within 20 ft of front
  lot line, resident-caretaker requirement, §9.2.14.c.ii) in a generic
  "check zoning" caveat.
- Q5 (propane): interpolated "propane" into a paraphrase of Sec. 5-5, which
  names "gasoline, oil and other inflammable" but never propane.
- Q11 (window well): headline conflated building-separation areas with
  lot-line setbacks (bullets stated it correctly); bolded "No" despite
  acknowledging 0-ft-setback districts exist.
- Q15 (pet cap): "Hobby Kennel requires a special permit" — it's SP in some
  districts, prohibited outright in others.
- Q13 (2+7): computed the answer ("2 + 7 = 9") before declining; harmless
  but a strict out-of-scope eval would fail it.
- Retrieval noise: occasional irrelevant `get_sections` fetches in traces
  (e.g. "Medical or Diagnostic Laboratory" for the deck question).

## What worked well

- Q24 (Airbnb one unit of 3-family): best answer of the day — primary-
  residence rule correctly drives "only your own unit," every registration
  detail verified.
- Q23 (ADU edge cases): answered both sharp follow-ups head-on; porch-counts-
  toward-separation reasoning verified; caught the ribbon-driveway wrinkle
  (unpaved center strip fails the paved-walkway requirement). Earned the
  day's only thumbs-up.
- Q16/Q17 (unusual/surprising laws): all 17 claimed provisions real and
  quoted accurately.
- Off-topic probes declined gracefully at low confidence.
- Confidence self-ratings mostly tracked evidence quality (Q12 the glaring
  exception).
- Question mix skewed practical: fire/grilling (6), permits/construction (5),
  animals/pests (5), neighbor disputes (4) — the eval suite's coverage of
  these themes looks well aimed.
