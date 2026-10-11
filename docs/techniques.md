# The prompting techniques

Every experiment asks the model the same thing: here is a document split into
numbered sentences; return the numbers of the sentences that belong in the
summary. A **technique** is the instruction that tells the model _how_ to
decide. This document says what each one asks for.

The prompts themselves are in
`extsum/src/extsum/prompts/sets/<set>/techniques/`. There are three wordings
of the same techniques ("prompt sets"): `aspect` for ACLSum, `document` for
news, `bill` for BillSum.

## What every prompt has in common

Each prompt is assembled from the same parts, in this order:

| Part | Content |
| --- | --- |
| Header | "You are an expert in extractive summarization" and the task: select the sentences for one aspect (ACLSum), or a concise summary of the whole document (Part C) |
| Preamble | only for techniques that add information, such as TF-IDF scores |
| Example | only in one-shot variants |
| Input | the sentences, as `Sentence 1: ...`, `Sentence 2: ...` |
| Instructions | **the technique**: the only part that differs between techniques |
| Rules | 1-based indices, JSON only, empty list if nothing matches, and the cap in capped variants |
| Return format | `{"selected_sentences": [...]}` |

In every technique the reasoning is asked for **internally**: the model is
told not to write it out, only the final JSON. The techniques change what the
model is told to think about, not what it outputs.

## The four variants of a technique

Each technique is run in up to four variants:

| | Uncapped | Capped |
| --- | --- | --- |
| **Zero-shot** | instructions only | instructions + "select at most K sentences" |
| **One-shot** | one worked example first | one worked example + the cap |

- **The cap** is a rule in the prompt, and the answer is also cut to the first
  K selected sentences if the model returns more. K is per aspect on ACLSum
  (median gold count) and per dataset in Part C (3 for CNN/DailyMail, 6 for
  BillSum).
- **The example** is a document from a split that is not evaluated, with its
  reference answer. It is the same example for every model.

## The techniques

### Baseline

| Technique | What the model is told |
| --- | --- |
| `vanilla` | Select the sentences that express the aspect (ACLSum), or that together form a concise, informative summary (Part C). No guidance on how. |

### Reasoning techniques

These tell the model to go through a procedure in its head before answering.

| Technique | The procedure |
| --- | --- |
| `chain_of_thought` | Step by step for each sentence: identify what it claims, decide which aspect it belongs to, conclude yes or no. In Part C: identify the central topic, judge what each sentence contributes, drop redundant or minor ones, check coverage. |
| `least_to_most` | Start from the easy, general question and work down. First consider the document's overall purpose; then select. In Part C: split the content into its main units (facts, causes, actions, outcomes) and pick the sentence that best expresses each. |
| `self_ask` | For each sentence, ask a question and answer it: "Does this sentence primarily express the aspect?" In Part C, several questions: is it essential, would removing it lose something, is it already said elsewhere, is it only background. Keep the sentences answered yes. |
| `explanation_based` | For each sentence you would select, form a one-sentence justification of why it belongs. Keep only the sentences with a clear justification. |
| `scoring_based` | Give each sentence a score from 1 to 5 for relevance, and select those scoring 4 or 5. |
| `salience_inference` | First state the document's most salient theme in one sentence. Then score each sentence 0, 1 or 2 for how strongly it expresses that theme. Select the 2s, or the 1s if there are none. |
| `contrastive_joint` | Classify every sentence into exactly one of challenge, approach, outcome or none, and return those labelled with the target aspect. The point is to decide between aspects instead of judging one in isolation. |
| `self_critique` | Draft a list of candidates, then critique it: remove any sentence that only shares keywords with the target. Return the pruned list. In the one-shot variant the example shows a draft being pruned. |

### Techniques that add information

| Technique | What is added |
| --- | --- |
| `tool_augmented` | A **TF-IDF centrality score** is printed beside each sentence, for example `Sentence 2: ... [centrality: 0.87]`, and the model is told to use it together with the content. The score is computed before the prompt is built, without any model: each sentence's TF-IDF similarity to every other sentence in the document, summed and rescaled to 0–1. A high score means the sentence shares vocabulary with many others. |
| `simulated_tool_augmented` | No scores are given. The model is told to imagine a tool, `check_aspect(sentence, aspect)`, that returns "match" or "no_match", and to select the matches. It tests whether the *framing* of using a tool helps without real tool output. |
| `negative_aware` | One-shot only. The example shows not just the selected sentences but also the **near-misses**: sentences that look relevant and were not selected. On BillSum the near-misses are the unselected sentences with the highest overlap with the reference summary; in the other sets they are the unselected sentences of the example. |

### Trace variants

For four reasoning techniques there is a second one-shot version whose example
also shows the **reasoning**, written once by a large reference model
(Qwen3.5-397B) and reused for every model:

`chain_of_thought_trace`, `self_ask_trace`, `scoring_based_trace`,
`salience_inference_trace`.

They exist to test whether showing how to reason helps more than showing only
the answer. How the traces were produced is described in the
[Part A analysis](analysis_part_a_aclsum.md#how-the-traces-were-produced).

### Ablations (Part C only)

An ablation is a technique with one ingredient removed, to see whether that
ingredient is what matters.

| Ablation | What is removed | What it tests |
| --- | --- | --- |
| `tool_no_meta` | the TF-IDF scores; the rest of the tool prompt is kept | whether the gain of `tool_augmented` comes from the scores |
| `tool_no_roleplay` | the "you are an expert" framing and the system prompt; the scores are kept | whether the gain comes from the role-play |
| `scoring_no_length` | the instruction to keep the selection concise | whether `scoring_based` needs to be told to be brief |
| `scoring_no_redundancy` | the instruction to avoid redundant sentences | whether it needs to be told to avoid repeats |

## Which techniques run where

| | ACLSum (Part A) | CNN/DailyMail and BillSum (Part C) |
| --- | --- | --- |
| Techniques | 12 + 4 trace variants = 16 | 12 + 4 trace + 4 ablations = 20 |
| Variants per model | 54 | 52 |
| `tool_augmented` | all four variants | zero-shot, capped only |
| `scoring_based` and its ablations | all four variants | zero-shot, uncapped only |
| `negative_aware`, trace variants | one-shot only | one-shot only |
| Ablations | not run | one variant each |

The two large models (Qwen3.5-27B, Gemma4-31B) run only each dataset's five
best techniques:

| Dataset | Five best techniques |
| --- | --- |
| ACLSum | `self_ask`, `salience_inference`, `self_critique`, `contrastive_joint`, `negative_aware` |
| CNN/DailyMail | `tool_no_roleplay`, `tool_augmented`, `tool_no_meta`, `vanilla`, `salience_inference` |
| BillSum | `tool_no_roleplay`, `tool_augmented`, `least_to_most`, `negative_aware`, `explanation_based` |

## How the wording differs between the three sets

The technique is the same idea in each set; the wording is adapted to the
task.

- **`aspect`** asks for sentences that "primarily express" one aspect, and
  gives short definitions of challenge, approach and outcome.
- **`document`** asks for "a concise and informative summary of the entire
  document", and its instructions are longer: they spell out what to drop
  (redundancy, minor details, quotations, background).
- **`bill`** is the `document` set with legislative wording: purpose,
  provisions, requirements, affected parties, funding, deadlines, exceptions.
  It also tells the model that a TF-IDF score is "supporting evidence only"
  and "not a direct measure" of importance, where the `aspect` set says higher
  scores mean "greater importance".

Because the Part C instructions are more detailed than the Part A ones, a
technique's result in Part A and in Part C is not a comparison of the same
text on different data. It is the same idea, worded for each task.

## Two optional mechanisms, not used in the reported results

Both are implemented and switched off in every experiment config:

- **Self-consistency**: run the same prompt several times with sampling and
  keep the sentences chosen by a majority.
- **Dynamic capper**: instead of a fixed K, a second call asks the model to
  prune its own selection to the essential sentences.
