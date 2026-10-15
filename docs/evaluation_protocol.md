# Evaluation protocol (draft, 2026-10-15)

## Research question
Can a trainable neural router decide when Qwen3-4B should use thinking mode and improve answer accuracy at matched inference compute cost?

## Data
- Use only the official GSM8K train split (7,473) for development: pilot 200, router_train 5,000, router_val 1,000, internal_holdout 1,273; splits are disjoint, seed 42.
- Official GSM8K test (1,319) and SVAMP (1,000) must not influence development or threshold selection.
- Pilot is a feasibility/development experiment, not independent final evaluation.
- Freeze dataset checksums, model checkpoint revision, code commit and decoding configuration before final evaluation.

## Paired inference
- Both modes see the same question and final-answer instruction with the same Qwen3-4B weights; only thinking mode is switched.
- Use deterministic decoding in the first pilot. Record raw generation, token count, latency, tokenizer/template versions and end-of-sequence/truncation status.
- Choose a generation-token cap in advance, use the same cap in both modes and report truncation rates. A capped thinking response is not a valid measurement of unlimited reasoning ability.
- Do not run on a GPU serving other work without permission; record device and resource contention.

## Answer evaluation
- GSM8K reference answers use the '#### number' marker.
- The conservative parser accepts explicit final-answer markers (including boxed numeric answers) only. Ambiguous model output is invalid, not guessed from the last intermediate number.
- Compare numeric Decimal values to handle commas and formatting; manually audit a sample of invalid outputs.
- Evaluate complete (fast, slow) pairs on identical IDs; duplicate, missing and unknown predictions fail loudly.

## Metrics
- Fast and slow exact numeric accuracy and invalid rate.
- Four paired categories: both correct; fast-only; slow-only; neither.
- Oracle upper bound = (both + fast-only + slow-only)/N. Not a deployable router, because it uses gold labels.
- Mean generated tokens and latency, with separate truncation counts.
- Later, at matched average compute budgets: always fast, always slow, random, lexical heuristic, trained neural router.
- Include router compute overhead and paired confidence intervals for final comparisons.

## Stop criteria
If the pilot finds negligible slow-only benefit or mostly truncated responses, revisit task difficulty, model or inference protocol **before** committing to router training.

## Responsible use
All reported experiments must have been executed and verified. Declare any GenAI assistance as specified in the SC4001 handout.
