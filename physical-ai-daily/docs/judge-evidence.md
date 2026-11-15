# LLM-as-a-judge: what the evidence supports (as of 2026-11-15)

This file covers only practices backed by measured evidence or by explicit vendor eval docs.
Labels used throughout:

- **[P]** a paper or technical report with numbers
- **[V]** explicit guidance in vendor docs (Anthropic, OpenAI, Google), usually with no data attached
- **[U]** unproven: a practitioner or vendor blog, or a claim with no public data behind it

Caveat that applies to almost everything below: most studies judge *model responses* on chat and
summarization benchmarks (MT-Bench, SummEval, Topical-Chat, BiGGen). Very little measures *ranking
news stories by importance*, which is our task. Q9 covers the little that does. Effect sizes are
often small and depend on the judge model, and older results (GPT-3.5, Claude-v1) can overstate
biases that frontier models have since reduced.

---

## Q1. Holistic Likert score vs decomposed criteria vs checklist (yes/no)

| Source | Setup | Finding | Effect size |
|---|---|---|---|
| [P] CheckEval, Lee et al., v3 Oct 2025 — https://arxiv.org/abs/2403.18771 | SummEval and Topical-Chat. 12 judges (Llama-3.1, Mistral, Qwen2.5, Gemma2, GPT-4o/4-Turbo/4o-mini). Yes/no checklist per sub-dimension; score = fraction of "yes", equally weighted. Compared with G-Eval (Likert) and SEEval. | **Inter-judge agreement rises a lot. Agreement with humans rises only a little.** A learned weighting of the questions gave no reliable gain. | Inter-evaluator Krippendorff α, all models: G-Eval 0.09 → CheckEval 0.48 (SummEval) and 0.06 → 0.45 (Topical-Chat). Large models: 0.67. Human correlation, best judge: SummEval ρ 0.52 → 0.55, Topical-Chat ρ 0.64 → 0.65, i.e. +0.01 to +0.03. |
| [P] TICK, Cook et al. 2024 — https://arxiv.org/abs/2410.03608 | 612 instructions. GPT-4o judge with CoT. Checklist generated per instruction, compared with direct 1–5 scoring and preference judgments, against triply-annotated human labels. | Checklists help moderately. They also make human raters more consistent with each other. | Exact agreement with humans: direct scoring 46.4% → TICK 52.2% (+5.8 pts). Human inter-annotator α: 0.194 → 0.256 when given checklists. |
| [P] "Ask, Then Grade", Nandwani & Gakhar, NeurIPS 2026 workshop "Can We Trust the Judge?" — https://openreview.net/forum?id=yhBUZYSlDZ (listing: https://neurips.cc/virtual/2026/161241) | SummEval, QAGS, USR Topical-Chat. 6 judges. Compares single binary, rubric-enhanced binary, a 6-question BinEval-style binary decomposition, plain 5-point ratings, and superlative-anchored scales. | **Plain 5-point ratings had the highest mean Spearman with humans for every judge.** Binary decomposition is "useful for atomic verification and question-level diagnosis, but not a generally superior default." | 5-point beat BinEval in 31 of 36 planned contrasts after conservative correction. Workshop paper; I read only the abstract and listing. |
| [P] LLM-Rubric, Hashemi et al. 2025 — https://arxiv.org/abs/2501.00274 | 9 rubric questions on dialogue; target is human overall satisfaction (1–4). The LLM's answer distributions per question feed a small network trained on human labels. | Combining decomposed answers with a mapping **learned from human labels** beats using the LLM's answers uncalibrated. | RMSE < 0.5, about 2× better than the uncalibrated baseline. |
| [V] Google Vertex AI eval docs — https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/determine-eval | n/a | Recommends starting from adaptive, auto-generated per-prompt rubrics (`GENERAL_QUALITY`). Static rubrics use 1–5 with a descriptor per level. | No data. |

**Bottom line.** Checklists reliably make judges agree with *each other*: large effect on α. They make
judges agree with *humans* only slightly (+0.01 to +0.06), and a 2026 study finds plain graded
5-point ratings agree with humans *better* than binary decomposition on holistic constructs. The
evidence is mixed. Checklists win on consistency and on atomic, verifiable facts. Graded ratings win
or tie on human validity for holistic qualities.

## Q2. Pointwise scoring vs pairwise comparison

| Source | Setup | Finding | Effect size |
|---|---|---|---|
| [P] Zheng et al. 2023, MT-Bench — https://arxiv.org/abs/2306.05685 (tables in v4 HTML) | MT-Bench, about 3K expert votes; GPT-4 judge; pairwise vs single-answer grading. | Pointwise and pairwise agree with humans about equally, at roughly the human–human level. | Agreement excluding ties (S2): pairwise 85%, single 85%, human–human 81%. Including ties (S1): 66% vs 60% vs 63%. |
| Same paper, Table 2 | 80 pairs, each judged in both orders. | **Pairwise judges are order-sensitive.** | Consistent across a swap: GPT-4 65%, GPT-3.5 46%, Claude-v1 24%. Claude-v1 favored the first position 75% of the time. |
| Same paper, mitigation | Count a win only if it holds in both orders; otherwise call it a tie. | Recommended. The paper gives no measured effect for it. Few-shot examples raised consistency. | Few-shot consistency: GPT-4 65% → 77.5%, Claude-v1 23.8% → 63.7%. |
| [P] Wang et al. 2023, "LLMs are not Fair Evaluators" — https://arxiv.org/abs/2305.17926 | Vicuna benchmark, 80 queries, human win/tie/lose labels; GPT-4 and ChatGPT judges. | Swapping order flips many verdicts. Averaging over both orders (BPC) plus evidence-first reasoning (MEC) moves results closer to humans. | Conflict rate: GPT-4 46% (close pair), 5% (clear gap); ChatGPT 82.5%. GPT-4 accuracy vs humans: 52.7% vanilla → 62.5% with MEC + BPC; κ 0.24 → 0.37. |
| [P] Tripathi et al. 2025, "Pairwise or Pointwise?" — https://arxiv.org/abs/2504.14716 | IFEval-TweakSet and MT-Bench. Llama-3.x, Qwen2.5, o3-mini, o4-mini judges. A distractor (assertiveness, prolixity, sycophancy) is added to the weaker response. | **Pairwise is much easier to sway with surface features than absolute scoring.** | Preference flips: pairwise about 35% vs absolute about 9% (open models). For o-series models, absolute is 17–26%, still below pairwise at 29–44%. |
| [P] Shi et al. 2024, position-bias study — https://arxiv.org/abs/2406.07791 | 15 judges, 22 tasks, 150K+ instances. | Position bias depends on the judge and task, and is **strongly driven by the quality gap**: close pairs are the most order-sensitive. | Effect is qualitative in the abstract. |
| [P] Liu et al. 2024, PairS — https://arxiv.org/abs/2403.16950 | Long-form generation evaluation. | Pairwise ranking with debiased, both-order comparisons beats direct scoring. | "Significant" gains; I did not extract numbers. |
| [V] OpenAI eval best practices — https://developers.openai.com/api/docs/guides/evaluation-best-practices | n/a | "Use pairwise comparison or pass/fail for more reliability." | No data. |

**Bottom line.** Agreement with humans is similar for both formats. Pairwise is order-biased, and
the bias is worst exactly where pairwise is used, on close calls. Running both orders and only
counting consistent wins is the standard fix. Pairwise is also more vulnerable to distractor
features, so the neutrality of the inputs matters more there.

## Q3. A panel of different models vs one strong judge

| Source | Setup | Finding | Effect size |
|---|---|---|---|
| [P] Verga et al. 2024, PoLL — https://arxiv.org/abs/2404.18796 | KILT QA (NQ, TriviaQA, HotpotQA) with binary correctness, plus Chatbot Arena ranking. Panel of Command-R, Claude-3-Haiku and GPT-3.5 (**three different families**) vs GPT-4. Max vote for binary labels, mean for 1–5 scores. | The panel beats the single large judge and shows less intra-model bias. | κ vs humans: NQ 0.763 vs 0.627; TQA 0.906 vs 0.841; HPQA 0.867 vs 0.830. Arena Kendall τ: 0.778 vs 0.667. Spread of accuracy deltas (SD): PoLL 2.2 vs up to 6.1 for single judges. **7–8× cheaper.** GPT-4's κ swung from 0.52 to 0.73 across small prompt edits. |
| [P] Goel et al. 2025, "Great Models Think Alike" — https://arxiv.org/abs/2502.04313 | CAPA, an error-overlap similarity metric, across many models. | Judges favor models similar to themselves, and **errors grow more correlated as models get more capable**. A same-family panel buys less independence than its size suggests. | No numbers in the abstract. Direction is clear. |
| [P] Panickssery et al. 2024 — https://arxiv.org/abs/2404.13076 | GPT-4 and Llama-2 judging their own vs others' text. | Self-preference grows linearly with self-recognition ability. | Linear correlation; the abstract gives no coefficient. |
| [P] Zheng et al. 2023 | MT-Bench. | Self-enhancement: GPT-4 favored itself by about 10% and Claude-v1 by about 25% above human win rates. Authors call the data limited. | about +10% / +25% |

**Bottom line.** A multi-family panel beats one strong judge on agreement, bias and cost. The
measured gain comes from *disjoint model families*. Panels drawn from one family, or from models
that resemble each other, are expected to share blind spots.

## Q4. Reasoning before the score (CoT)

| Source | Setup | Finding | Effect size |
|---|---|---|---|
| [P] Wang et al. 2023 — https://arxiv.org/abs/2305.17926 | Same setup as Q2. Explanation first (EC) vs score first. | Explanation first helps. | GPT-4 accuracy 52.7% → 56.5%. ChatGPT 44.4% → 52.6%, κ 0.06 → 0.23. |
| [P] Chiang & Lee 2023, "A Closer Look…" — https://arxiv.org/abs/2310.05657 | SummEval and Topical-Chat; ChatGPT; N=20 samples averaged. | Asking for a rationale (rate-explain or analyze-rate) beats a bare score. Auto-generated CoT steps are not always useful. | SummEval coherence τ: 0.248 (score only) → 0.44–0.48 (with rationale). Gains are smaller or mixed on other attributes. |
| [P] Liu et al. 2023, G-Eval — https://arxiv.org/abs/2303.16634 | SummEval; GPT-4. | Auto-CoT gives a tiny gain. | ρ 0.500 → 0.514. |
| [P] Zheng et al. 2023 | Math questions. | CoT, and a reference answer even more, cut grading failures. | Failures: 70% default → 30% CoT → 15% reference-guided. |
| [P] Yamauchi et al. 2025, "Empirical study of LLM-as-a-judge design" — https://arxiv.org/abs/2506.13639 | BiGGen-Bench; GPT-4o and Llama-3.1-70B. | **When clear criteria and a reference are given, CoT adds almost nothing.** It helps when criteria are missing. | GPT-4o r: 0.664 → 0.666 with criteria. Without criteria and reference: 0.422 → 0.487. |
| [P] Wang et al. 2025, "Judgment distribution" — https://arxiv.org/abs/2503.03064 | RewardBench and MT-Bench, pointwise and pairwise; GPT-4o, Llama-3.1-8B, Mistral-7B, Prometheus-2. | CoT collapses the score distribution. With mean-of-distribution scoring, no-CoT beat CoT in 14 of 16 cases. **GPT-4o was nearly unaffected.** | Small models: +3 to +11 pts accuracy without CoT. GPT-4o: −0.6 to +0.4. |
| [V] Anthropic, "Develop tests" — https://platform.claude.com/docs/en/test-and-evaluate/develop-tests | n/a | "Encourage reasoning: Use a grader model with thinking on, so that it reasons before it produces an evaluation score. This increases evaluation performance, particularly for tasks requiring complex judgment." | No data. |
| [V] OpenAI eval best practices | n/a | "Add reasoning and chain-of-thought as reasoning before scoring improves eval performance." | No data. |

**Bottom line.** Evidence is mixed. For frontier judges with clear criteria, the effect is about
zero to small and positive. It helps more with weak criteria or complex judgment. It can hurt small
models when you read scores from token probabilities. Both vendors recommend reasoning before the
score. A short rationale is low-risk.

## Q5. Scale granularity, level anchors, references and few-shot examples

| Source | Setup | Finding | Effect size |
|---|---|---|---|
| [P] "Grading Scale Impact…", 2026 — https://arxiv.org/abs/2601.03444 | 6 benchmarks, 150 items, 12 human raters, 6 judges (GPT-4o, Gemini-2.5-flash, Llama-3.3-70B, Qwen3-32B, DeepSeek-v3.2, Mistral-7B). Scales 0–5, 0–10, 0–100. Measured by ICC. | 0–5 gives the best human–LLM agreement pooled. 0–10 is weakest pooled, but best on MT-Bench. | Pooled ICC: 0–5 0.853, 0–10 0.805, 0–100 0.840 (Δ ≈ 0.05). Best scale on 4 of 6 benchmarks. No significance tests. |
| [P] "Ask, Then Grade", 2026 (Q1) | n/a | Superlative endpoint anchors reduced use of the scale boundaries and did not improve agreement. | Null effect. |
| [P] Prometheus, Kim et al. 2023 — https://arxiv.org/abs/2310.08491 | Fine-tuned 7B judge; correlation with GPT-4 scores. Ablations remove the score rubric (per-level descriptions), the reference answer, or the feedback. | Every component helps. **The reference answer helps most.** The rubric matters most on unseen criteria. | Removing the rubric: −0.02 (seen), −0.10 (unseen). Removing the reference: −0.22 (seen), −0.11 (Vicuna). Removing feedback: −0.17 to −0.19. |
| [P] Prometheus 2, Kim et al. 2024 — https://arxiv.org/abs/2405.01535 | FLASK, human correlation. | Removing the reference answer hurts even GPT-4. | GPT-4 r 0.679 → 0.616. |
| [P] Yamauchi et al. 2025 (Q4) | BiGGen-Bench. | Criteria matter more than the reference. Removing both is very damaging. | GPT-4o r: 0.666 default, 0.591 without criteria, 0.638 without reference, 0.487 without both. |
| [P] Zheng et al. 2023 | Pairwise order consistency. | Few-shot examples raise consistency. | GPT-4 65% → 77.5%. |
| [P] Verga et al. 2024 | GPT-4, NQ. | Few-shot beats zero-shot, but the prompt wording mattered as much as the examples. | κ 0.518 → 0.627. A "don't overthink" prompt reached 0.725. |
| [V] Anthropic docs | n/a | "Have detailed, clear rubrics"; "Be empirical or specific… judge from a scale of 1–5". | No data. |
| [U] FutureAGI glossary (in search results) | n/a | Claims κ rises from about 0.4 to about 0.78 with anchored 1–5 levels. | Vendor claim, no paper. Treat as unproven. |

**Bottom line.** Explicit criteria and descriptions for each level reliably help (−0.07 to −0.10 r
when removed). A reference or exemplar helps even more where one exists. Coarser scales (0–5) beat
0–10 by a small, inconsistent margin of about 0.05 ICC. Fancier anchor wording does not help.

## Q6. Known biases and what measurably reduces them

| Bias | Source | Setup | Size of bias | What reduced it |
|---|---|---|---|---|
| Position (pairwise) | [P] Zheng 2023; [P] Wang 2023 | See Q2 | GPT-4 order-consistent 65%; conflict rate 46% on close pairs | Both orders and count only consistent wins (Zheng). Averaging both orders: +~10 pts accuracy with MEC (Wang). Few-shot: +12.5 pts consistency. |
| Position (listwise / many candidates) | [P] CALM, Ye et al. 2024 — https://arxiv.org/abs/2410.02736 | 12 bias types; GPT-4o, Claude-3.5 and others | Pairwise robustness GPT-4o 0.78, Claude-3.5 0.83. **Most models fall below 0.5 with 3–4 candidates.** | Rotate or shuffle the order; avoid many candidates at once (paper's advice). |
| Position (listwise) | [P] Tang et al. 2023, permutation self-consistency — https://arxiv.org/abs/2310.07712 | 5 list-ranking datasets; GPT-3.5, Llama-2-70B | Order changes the output ranking | Rank several shuffled orders and aggregate to the central ranking: **+7–18% (GPT-3.5), +8–16% (Llama-2-70B)**. |
| Verbosity / length | [P] Zheng 2023 | "Repetitive list" attack | Fooled GPT-3.5 and Claude-v1 91%, **GPT-4 only 8.7%** | Stronger judge |
| Verbosity / length | [P] Dubois et al. 2024, length-controlled AlpacaEval — https://arxiv.org/abs/2404.04475 | AlpacaEval vs Chatbot Arena | Raw win rate is length-biased | Regression-based length control: Spearman with Arena 0.94 → 0.98 |
| Verbosity / length | [P] CALM | n/a | Robustness: GPT-4o 0.98, Claude-3.5 0.95. **Small for frontier judges.** | n/a |
| Authority (fake citations) | [P] CALM | Fake book, quote or URL inserted | Robustness 0.75–0.89. Quotes and books sway judges more than URLs. | Protective instructions (suggested, not measured) |
| Bandwagon ("X% of people think…") | [P] CALM | Claimed majority opinion added | **Claude-3.5 robustness 0.61**; GPT-4o 0.79 | n/a |
| Sentiment / emotional tone | [P] CALM | Negative tone added to the better answer | Robustness 0.24–0.36, the worst bias measured | Neutral input text |
| Distractor features (assertive, verbose, sycophantic) | [P] Tripathi 2025 | See Q2 | Flips: pairwise ~35%, absolute ~9% | Absolute scoring |
| Self-preference | [P] Panickssery 2024; [P] CALM; [P] Zheng 2023 | n/a | Claude-3.5 ≈7.5% score error, GPT-4o ≈1.7%; GPT-4 ≈+10% | Judge with a different model from the generator. Anthropic's docs say the same: "Generally best practice to evaluate with a different model from the one used to generate the evaluated output." [V] |
| Leniency (saying "correct" when unsure) | [P] Thakur et al. 2024, "Judging the Judges" — https://arxiv.org/abs/2406.12624 | TriviaQA; 13 judges | P(correct when unsure) well above 0.5 for most; 0.99 for Llama-2-70B | Stronger judges. Report κ or π, not just % agreement. |
| LLM-generated text preferred over human text | [P] G-Eval | SummEval-style data | G-Eval-4 preferred GPT-3.5 summaries even when humans did not | Preliminary |
| Ranking fails on weaker items | [P] Thakur 2024 | n/a | Judges separate the top well but the weaker candidates poorly | n/a |

## Q7. Multiple samples, temperature, averaging

| Source | Setup | Finding | Effect size |
|---|---|---|---|
| [P] Yamauchi et al. 2025 — https://arxiv.org/abs/2506.13639 | BiGGen-Bench; GPT-4o with CoT; 5 samples at T=1 vs greedy. | The mean of samples beats greedy. Mean beats median and majority vote. | GPT-4o r 0.635 → 0.666 (+0.03). Llama-70B 0.593 → 0.641 (+0.05). |
| [P] Wang et al. 2025 — https://arxiv.org/abs/2503.03064 | Token-probability mean vs mode. | The mean beats the mode everywhere. | GPT-4o +2.7 to +4.0 pts. Small models +7 to +10 pts. |
| [P] Wang et al. 2023 | MEC with k evidence samples at T=1. | More samples help, with diminishing returns. k=3 is the best cost trade-off. T=0.6–1.0 is better than 0.2 or 1.4. | GPT-4: k=1 56.5%, k=3 58.7%, k=6 60.9%. |
| [P] Schroeder & Wood-Doughty 2024 — https://arxiv.org/abs/2412.12509 | BBH, SQuAD, MT-Bench; 7–8B judges; 100 replications. | Single-shot judgments are unreliable. Temperature 0 only *hides* the variance. | McDonald's ω 0.42–0.80 at T>0 (many rated "questionable"). Small models only. |
| [P] Chiang & Lee 2023 | ChatGPT. | Rationale-style prompts are more robust to temperature. | Qualitative. |
| [P] 2601.03444 (Q5) | T = 0.1–1.0. | Agreement barely moved with temperature. | ±0.02–0.05 ICC |

**Bottom line.** Averaging a few samples (3–5) gives a small, consistent gain: +0.03 to +0.05 r for
frontier judges, more for small ones. Averaging across a panel gives some of the same benefit.

## Q8. Calibrating against human labels

| Source | Setup | Finding | Effect size |
|---|---|---|---|
| [P] Thakur et al. 2024 — https://arxiv.org/abs/2406.12624 | TriviaQA; 400 human-annotated items. | **Percent agreement misleads.** Judges above 90% agreement can still be off by more than 10 points in score. Use chance-corrected metrics (Scott's π or Cohen's κ). | The best judge was 8 π points below the human–human level. |
| [P] LLM-Rubric 2025 | Learned mapping from rubric answers to human overall scores. | Calibrating on human labels roughly halves error. | about 2× RMSE improvement |
| [P] Wang et al. 2023, human-in-the-loop calibration | Send the 20% most order-unstable items to a human. | Large gain from a small human budget. | GPT-4 accuracy 62.5% → 73.8%, κ 0.37 → 0.56, which is human level (71.7%, 0.54). |
| [P] Shankar et al. 2024, EvalGen — https://arxiv.org/abs/2404.12272 | User study. | **Criteria drift**: people refine their criteria while grading outputs, so the rubric should be revised after looking at real outputs. | Qualitative |
| [P] "How many human reviews do you need?", 2026 — https://arxiv.org/abs/2605.16354 | Two-stage sampling: LLM labels everything, humans label a subsample; doubly robust estimator. | Provides sample-size formulas. Put more human labels where the LLM is less predictable. | Formulas only; no single number. |
| [U] Practitioner heuristic (dev.to blog in search results) | n/a | About 200 paired labels for a κ≈0.6 estimate with ±0.05 CI; about 400 if κ≈0.4. | Unproven heuristic. It follows from standard κ standard-error math. |
| [V] OpenAI | n/a | "Validate agreement against your human labels before optimizing for cost or latency"; "Use human feedback to calibrate automated scoring." | No data |

## Q9. Judging news or story importance and editorial selection

| Source | Setup | Finding | Effect size |
|---|---|---|---|
| [P] Public Service Algorithm, 2025 — https://arxiv.org/abs/2506.22270 | 30 news articles; 27 human editors; 8 LLMs. Four editorial criteria, each scored 0–5 with **one prompt per criterion** and incremental guidance. Articles ranked on the averaged scores. | LLMs match editors on some criteria and miss on others. LLMs underrate "diversity" and "forward-looking". **Model choice matters enormously.** | Human panel ICC(2,k) 0.75–0.94. Best LLM top-5 overlap 75% (Qwen-72B, Mistral-NeMo). NDCG@5 ranged from 0.51 to 0.95 across models. |
| [P] Hagar et al. 2025, LLM-assisted news discovery — https://arxiv.org/abs/2509.25491 | 63 trade-press articles; newsworthiness 1–5 using true/new/interesting/important guide questions; GPT-4o, GPT-4.1, GPT-5, o3, o4-mini vs 3 annotators. | Coarse agreement is OK; exact scores are poor. **LLMs over-rate vague items, favor technical novelty over real-world impact, and inflate scores.** Humans also disagree with each other. | Human–human κ 0.03–0.29. LLM exact accuracy 14–46%; ±1 accuracy 72–92%; r 0.23–0.52. Live deployment (o3): exact 27%, ±1 47%. |

**Bottom line.** News importance is a low-agreement construct even among humans (κ as low as 0.03).
Judge models differ hugely. LLMs agree with editors on coarse top-k selection, roughly 75% top-5
overlap in the best case, and are known to over-value novelty and vagueness.

---

## Our design against the evidence

| Element | Verdict | Evidence and comment |
|---|---|---|
| Cards built only from stored sources, verbatim quotes, plain neutral labels | **Supported** | CALM: sentiment (robustness 0.24–0.36), authority (0.75–0.89) and bandwagon (0.61 for Claude-3.5) are real biases. Neutral, attributed, source-grounded inputs address them. Prometheus and Yamauchi show grounding material improves agreement. |
| Raw engagement counts on the card | **Partly. Evidence warns.** | CALM's bandwagon test shows Claude-3.5 swayed by claimed majority opinion (robustness 0.61). Engagement counts are legitimate evidence of importance, but they may get overweighted. Test it: score a few cards with and without counts. |
| Three judges on Claude Opus, Sonnet and Fable | **Partly supported. Evidence says diversify.** | PoLL's gains come from *three disjoint families*. Ours are all Anthropic models. Goel et al. 2025: errors correlate more as capability rises; judges favor similar models. Our high agreement (14/19 overall scores within 1 point) may partly be shared bias, not accuracy. Cards contain mostly verbatim, non-model text, so self-preference is a smaller concern than correlated blind spots. |
| Each judge sees cards in its own random order | **Supported, but thin** | Permutation self-consistency (+7–18%) and CALM (multi-candidate position robustness < 0.5) both say order matters in lists. Three permutations, one per judge, mixes order effects with model differences. Stronger: each judge scores ≥ 2 orders, or scores each card in a separate call so there is no list context. |
| Criteria shown in a random order per judge | **No direct evidence** | No study found on criterion-order effects. Harmless. |
| Five criteria scored 1–10, anchors at 1/3/5/8/10 | **Supported for anchors; the scale is slightly worse than 0–5** | Level descriptions help (Prometheus: −0.10 r without them on unseen criteria). 0–5 beat 0–10 pooled by about 0.05 ICC (2601.03444), but 0–10 was best on MT-Bench. Small and inconsistent: a low-priority change. Unanchored levels (2, 4, 6, 7, 9) are a mild consistency risk. |
| 2–4 sentences of reasoning before the numbers | **Supported, low stakes** | Both Anthropic and OpenAI recommend it. For frontier judges with clear criteria, the measured effect is about 0 to small positive (Yamauchi: +0.002; Wang 2023: +3.8 pts). Keep it; it also produces an audit trail. |
| Holistic overall 1–10 (not an average), following a stated goal and priorities | **Mixed or unknown** | No direct test of holistic-overall vs average-of-criteria found. "Ask, Then Grade" finds graded holistic 5-point ratings agree with humans best. LLM-Rubric finds a combination *learned from human labels* beats an uncalibrated one. A fixed hand-set combination has no evidence either way. |
| Top-5 and proposed lineup per judge, Borda tie-break | **Reasonable, no direct evidence** | Rank aggregation across orders and judges is in line with permutation self-consistency. Borda is only used to break ties, so it matters little. |
| Aggregate = mean of overall scores | **Supported** | Mean-of-samples beats majority and median (Yamauchi), and PoLL uses mean pooling for graded scores. Note: three same-family judges ≠ three independent samples. |
| Flag judge spread ≥ 3 | **Supported in spirit** | Wang 2023: sending the 20% most unstable items to a human took κ from 0.37 to 0.56, human level. The threshold is ours and untested. Make sure the flag actually goes to a human. |
| Cut line (ranks 3–7) settled by a fresh judge doing every pair in both orders; a win counts only if consistent both ways | **Strongly supported** | This is the exact Zheng 2023 mitigation. Position bias is worst on close pairs (Shi 2024; Wang: 46% conflict on close pairs vs 5% on clear gaps), which is exactly the cut line. Two caveats: (1) pairwise is more swayed by surface features (Tripathi: 35% vs 9% flips), so card neutrality matters most here; (2) if most pairs come out as ties, the round decides nothing, so fall back to the pointwise means. Consider making the fresh judge a non-Anthropic model. |
| Hard editorial rules applied after scoring | **Supported** | Deterministic checks for verifiable facts are what both vendors' docs prefer where possible ("automate when possible", code-graded). |
| **No human-label calibration** | **Evidence says do something different** | This is our biggest gap. Agreement *between* judges is not validity (Thakur: > 90% agreement can hide > 10-point errors). News importance has low human agreement (Hagar κ 0.03–0.29) and judges vary hugely by model (PSA NDCG@5 0.51–0.95). OpenAI's guidance is explicit: validate against human labels first. |
| Single sample per judge | **Small improvement available** | 3–5 samples averaged at T≈1: +0.03 to +0.05 r (Yamauchi). The cost is only worth it at the cut line or on flagged stories. |

### The proposed change: per-criterion yes/no checklists, fixed mapping, fixed overall, ±1 adjustment

| Part | Verdict | Evidence |
|---|---|---|
| 4–5 observable yes/no questions per criterion, answered by quoting the card | **Mixed** | CheckEval: large gain in *inter-judge* agreement (α +0.39) and a tiny gain in *human* agreement (+0.01 to +0.03 ρ). TICK: +5.8 pts exact agreement. "Ask, Then Grade" (2026): plain 5-point ratings beat binary decomposition on human correlation in 31 of 36 contrasts for holistic constructs. **Our judges already agree closely** (criterion spreads 0.5–0.9, overall within 1 point for 14/19). The checklist's main proven benefit, consistency, is the thing we least need. Importance is a holistic construct, where graded ratings did better. |
| Fixed mapping from answers to a criterion score; overall = fixed combination | **Not supported as designed** | CheckEval's equal-weight fraction works, but a learned weighting gave no gain there. LLM-Rubric shows mappings help when *learned from human labels* (about 2× RMSE). A hand-set mapping with no human labels has no evidence, and it would remove the holistic, goal-driven judgment that matches the editorial task. |
| Bounded ±1 adjustment with a reason quoting the card | **No evidence either way** | Plausible as a safety valve. Untested. |
| Quoting the card for every answer | **Weakly supported** | Grounding and evidence-first reasoning help (Wang 2023 EC; Prometheus reference). Quoting also makes judgments auditable. |

**Verdict.** Use checklists only for *atomic, verifiable facts* such as "primary source is a paper
or official post", "has a demo or video", "independent coverage exists", "claim is quantified".
That is the use "Ask, Then Grade" endorses, and many of these could be hard rules or card fields
computed in code. Keep graded, anchored criterion scores and a holistic overall for the judgment
itself. Revisit a fixed mapping only once we have human labels to fit it against.

---

## What to change, ranked by strength of evidence

1. **Add human calibration (strong evidence; OpenAI [V], Thakur [P], Wang HITLC [P], Hagar and PSA [P]).**
   Each week, have the editor (the user) rank or label the shortlist blind before seeing the judges'
   output. Track top-5 overlap and NDCG@5 (the PSA paper's metrics) plus Spearman on overall scores,
   accumulated across weeks. Treat the judge setup as unvalidated until about 50–100 labeled stories
   exist; a target CI for κ needs about 200 pairs (unproven heuristic). Route spread ≥ 3 and the cut
   line to that human review.
2. **Put at least one non-Anthropic model on the panel, and preferably on the pairwise cut-line judge (strong-to-moderate: PoLL [P], Goel 2025 [P], Panickssery [P], Anthropic docs [V] on different-model grading).**
   PoLL's κ and τ gains and its lower bias come from disjoint families. Keep one Claude model, and
   compare run-1-style agreement before and after. If agreement falls, the old agreement was partly
   shared bias.
3. **Do not adopt the full checklist-to-fixed-score change. Use checklists only for atomic facts (moderate, mixed: "Ask, Then Grade" [P], CheckEval [P], TICK [P]).**
   The proven gain is consistency, which we already have. Graded ratings matched humans better on
   holistic constructs.
4. **Reduce list-order effects (moderate: permutation self-consistency [P], CALM [P]).** Score each
   card in its own call, or have each judge score ≥ 2 shuffled orders and average.
5. **Keep the both-orders pairwise cut line (strong: Zheng [P], Wang [P]).** Already right. Add a fallback when
   most pairs tie, and keep cards length-matched for this step, since pairwise is distractor-sensitive (Tripathi [P]).
6. **Test the engagement-count bandwagon effect (moderate: CALM [P]).** Score about 5 cards with and
   without counts. If scores move more than intended, show counts as bucketed or relative values,
   or move them into a dedicated criterion.
7. **Watch for novelty over-rating (single study: Hagar [P]).** LLMs favored technical novelty over
   real-world impact and over-rated vague items. If impact is a priority, give it an explicit
   anchored criterion.
8. **Average 3 samples per judge on cut-line and flagged stories (small but consistent: Yamauchi [P], Wang 2025 [P]).** +0.03–0.05 r.
9. **Optionally move criteria to 0–5 or 1–5 with every level anchored (weak; about 0.05 ICC, inconsistent: 2601.03444 [P]; Anthropic [V] example uses 1–5).** Do it only alongside the calibration in item 1, so the change can be measured.
10. **Keep reasoning before the score (vendor guidance [V]; measured effect about 0 to small for frontier judges [P]).** No change.
