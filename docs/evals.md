# Evals

How the briefings and retrieval are measured, and what the results say. Index and model validation are in [methods](methods.md#compound-burden-score) and [the forecasting model](ml.md). Back to the [README](../README.md).

Guards and evals do different jobs. Guards run on every briefing and drop bad output: typed structured output, refusal and truncation errors, and dropping actions whose citations were not retrieved. Evals measure how good the system is, where it fails, and whether a change made it better or worse.

## The harness

One command scores any version of the system the same way:

```
PYTHONPATH=. python scripts/run_evals.py --suite retrieval|briefings|judge|robustness|all --week 2026-06-22
```

It loads a dataset (stored briefings with their tract inputs, or a gold set), runs scorers that share one interface, records a version manifest (git SHA, system prompt hash, model ID, embedding model, corpus and config hashes), logs to MLflow, and compares against accepted baselines in `evals/baselines/`. Scorers are written against this project's prompt rules and data, not a generic eval library: generic metrics need a model call per check and do not test these specific rules.

## Results

| Eval | What it checks | Result | Cost |
|---|---|---|---|
| A1 retrieval | Right policy pages for a driver and group | recall@3 0.38, recall@10 0.65, MRR 0.70, nDCG@10 0.59, 7 of 7 off-topic questions rejected | free |
| A2 faithfulness | Numbers, urgency and drivers match the tract data | 626 of 626 briefings pass every check | free |
| A4 rules | No pollution source named, no talk about the pipeline, no jargon, actions on the tract's own drivers | 100% on all four; median reading grade 10.2 | free |
| A5 specificity | Briefings differ when drivers or groups differ | separation 0.36 by driver, 0.30 by group | free |
| A6 judge | Rubric scores from a stronger model | not trusted on any criterion (see below) | about $1, cached |
| A7 robustness | Injection, unsafe excerpts, empty retrieval, perturbation, repeat runs | injection 10 of 10, unsafe 4 of 5, empty 3 of 3, perturbation 38 of 38 | $0.03 |
| A9 CI gate | A2, A4, A5 on a frozen sample and A1 metric code on frozen rankings | runs on every pull request | free |

## A1 retrieval

A hand-labeled gold set of 30 questions built from the real driver and group templates (heat and older adults, air quality and children, and so on), each labeled with the relevant policy pages, plus 7 off-topic questions that should return nothing.

The eval made two decisions:

- **Embedding model.** Compared at cutoffs of 0.50 and 0.45, all-MiniLM-L6-v2 reached recall@3 0.26, MRR 0.48 and rejected 5 of 7 negatives; no MiniLM cutoff rejected every negative with recall@3 above 0.19. Qwen3-Embedding-0.6B reached 0.33, 0.58 and 7 of 7. A task instruction on queries raised Qwen3 further (MRR 0.63 to 0.72).
- **Cutoff.** A sweep in steps of 0.02 picked 0.48: the lowest distance with no empty real question and every negative rejected.

A cross-encoder reranker was tested on the same set and is not in production.

Recall@3 of 0.38 sounds low, but most questions have several relevant pages and the briefing takes 3 excerpts per driver, so recall@3 is capped well below 1. MRR of 0.70 says the first relevant page is usually ranked first or second.

## A2, A4, A5 on 626 briefings

Code-based checks on every stored briefing for the week of June 22, 2026, with no model calls:

- **Faithfulness:** every number in the summary and key risks matches a value in the tract's data (with rounding tolerance); stored urgency equals the input; every driver is mentioned or listed as a data gap.
- **Rules:** one check per system prompt rule. An early prompt asked the model to explain its urgency level, and 24 of 25 sample briefings described how the pipeline set it. That check now runs on every briefing.
- **Specificity:** briefings for tracts with different drivers should read differently. The weak spot is sources: 95.5% of briefings share their exact set of cited sources with another briefing, because 12 documents cover the same few actions. That is a corpus limit, and the README says so.

## A6 LLM as judge

Claude Sonnet 5.5 scored a stratified sample of 60 briefings on a fixed rubric: groundedness, tract specificity, actionability, urgency tone, clarity, and unsafe advice. I labeled 20 of them by hand with the same rubric. A criterion counts only if quadratic weighted Cohen's kappa against my labels reaches 0.61.

| Criterion | Kappa | Exact match | Within 1 point |
|---|---|---|---|
| Groundedness | -0.02 | 25% | 90% |
| Tract specificity | 0.38 | 5% | 85% |
| Actionability | 0.36 | 75% | 95% |
| Urgency tone | -0.15 | 20% | 75% |
| Clarity | 0.01 | 15% | 100% |
| Unsafe advice (yes/no) | | 100% agreement, none found | |

No criterion passed, so judge scores are not used as evidence anywhere. The judge is mostly within a point of my labels, but it scores systematically lower on groundedness, specificity and clarity, and kappa punishes that. A judge that has not been checked against people would have produced a nicer table; this one says the briefing quality claims rest on the code-based checks and my own reading.

## A7 robustness

30 live Haiku calls:

- **Prompt injection:** a planted excerpt says to ignore the rules and call the tract safe. 10 of 10 briefings ignored it.
- **Unsafe excerpt:** a planted excerpt recommends something harmful. 4 of 5 left it out; one did not.
- **Empty retrieval:** drivers with no excerpts produce data gaps and no actions, 3 of 3.
- **Perturbation (no model calls):** small changes to one indicator or to group order keep the actions; changing the driver set changes them (38 of 38).
- **Repeat runs:** the same tracts generated 3 times cite the same pages (Jaccard 0.89), with different wording (text cosine 0.59). No refusals; 1 of 30 calls hit the token limit.

## A9 CI gate

Live retrieval needs the 1.2 GB embedding model and the Chroma index, so CI does not run it. Instead, every pull request runs A2, A4 and A5 on a frozen briefing sample and the A1 metric code on frozen retrieval rankings, and fails if a metric drops below `evals/baselines/` by more than its tolerance. When the prompt or corpus changes on purpose, the baselines are regenerated locally and updated in the same pull request.
