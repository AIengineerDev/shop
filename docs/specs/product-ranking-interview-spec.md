# Product ranking for marketplace search

Given a user, a search query, request context, and a fixed set of candidate products, return the same products ordered by estimated relevance to that user and query. The model assigns one score per candidate. Higher scores rank first. Retrieval and eligibility checks happen upstream. If scoring fails, return the incoming candidate order.

## 1. Input and data collection

The supplied [ranking_items.csv](../../data/ranking_items.csv) has ten impressions, two query groups, two sessions, and five candidates per group. One row represents a displayed product. The file supports schema checks, baseline scoring, and metric tests. A comparison of learned models requires a larger export of search requests from the same task. The sibling Parquet files come from a simulated home feed and lack the query and retrieval scores needed for this comparison.

| Contract | CSV fields | Use |
|---|---|---|
| Identifiers | `session_id`, `qid` (query_id), `user_id`, `item_id` | Group candidates by the individual query request, even if the same text occurs again. Keep sessions intact in splits. IDs serve as keys. |
| Query and context | `query_text`, `device` (device_type), `user_aov_tier` | The tier is Low, Mid, or High average-order-value affinity as known at request time. |
| Relevance and product features | `bm25` (bm25_score), `semantic_sim`, `price`, `hist_ctr_7d` (historical_ctr_7d), `rating` (review_rating) | The prompt defines `semantic_sim` as cosine **distance**. Lower values mean closer matches under that definition. Verify the producing function on a known query–item pair before applying a sign change or monotonic constraint. |
| Exposure | `displayed_pos` (displayed_position) | One-based rank under the logging policy. Use for reconstructing the incumbent order and analyzing exposure bias. Exclude it from serving features. |
| Outcomes | `clicked`, `added_to_cart`, `purchased`, `relevance_grade` | Three binary pointwise targets and one supplied ordinal target, respectively. |

For training data, log the request timestamp, candidate IDs and retrieval scores before reranking, and the displayed order afterward. Record which items entered the viewport, the serving policy version, and any randomized assignment probabilities. Join clicks, dwell duration, wishlist adds, cart adds, and order lines to a stable impression ID with event timestamps. Log feature values as served or retain point-in-time snapshots. These records distinguish an eligible candidate, a displayed item, and an examined item. An unshown candidate has an unknown behavioral label.

## 2. Labels

Use the supplied grade for the fixture. For future logs, assign the highest qualifying event to each attributed impression.

| Grade | Meaning |
|---|---|
| 0 | No click and no stronger attributed event |
| 1 | Low-dwell click |
| 2 | High-dwell click or wishlist add |
| 3 | Add to cart |
| 4 | Purchase |

Keep all three binary outcomes independently. A purchase sets grade 4 even if the click was missed. A clicked row with no cart or purchase can have grade 1 or 2. The CSV has no dwell duration or wishlist flag, so its binaries alone cannot reconstruct this distinction. Confirm the dwell threshold and handling of missing dwell before generating new grades. Keep returns as a separate outcome for evaluation.

Prefer a direct impression-to-order link. For unmatched orders, agree on the attribution window, repeat-exposure tie rule, and cross-session policy before assigning credit. Count each order line once. Record which labels have had enough time to mature. Report visibility and missing-event rates so instrumentation gaps are distinguishable from weak engagement.

## 3. Features and models

Start with BM25, cosine distance, log price, historical CTR, and review rating. Add query length, device, AOV tier, price rank within the candidate set, and a price-by-tier interaction. Compute group features over the same candidate set at training and serving. Calculate historical CTR from clicks and impressions strictly before request time, with smoothing for low counts. Fit categorical encodings and priors on training data. Handle unseen categories and missing history explicitly.

| Candidate | What it tests |
|---|---|
| Logged order and descending BM25 | Establish the incumbent and lexical baselines on identical requests. |
| Pointwise gradient-boosted trees | Fit separate click, cart, and purchase targets. Compare their orderings to see which proxies agree with graded relevance. Evaluate calibration separately if probabilities are needed. |
| Grouped LambdaMART | Fit the 0–4 grades using request groups and the gain table used by the evaluator. This is the first learned ranking candidate for the tabular features. |
| Query–product text reranker | Consider if error analysis finds lexical or semantic mismatches and product text, larger training data, and serving budget are available. |

## 4. Training and bias

Split representative search traffic chronologically into training, validation, and final test periods. Keep whole sessions and query groups together. Exclude boundary-crossing sessions or move the cutoff, and use only training labels observable by that cutoff. Fit features on training data, tune on validation, then score the frozen configuration on the test set once. The fixture has no timestamps and cannot support this experiment.

For LambdaMART, keep each query's rows contiguous and pass the corresponding group sizes. Tune tree size, learning rate, and early stopping on validation NDCG@10. Groups with equal grades have no preference pairs. Keep them in evaluation coverage and retain unengaged sessions for business metrics. Version the data window, splits, feature schema, label rules, and gain table with each run.

Logged behavior favors items users had an opportunity to see. Start with an uncorrected model and inspect outcomes by original position. If exploration logs exist, verify whether the recorded probability describes item selection, position assignment, or examination before using it in a correction. A shuffle of positions 1–10 can inform comparisons among those ten items. It cannot establish how item 11 would perform at position 1. Use independent relevance judgments or an online experiment for comparisons outside logged exposure.

## 5. Offline evaluation

Use mean **NDCG@10** per request as the primary offline metric. As a starting assumption, map grades 0–4 to gains **0, 1, 3, 7, 15**. Discount gain at one-based rank `r` by dividing by `log2(r + 1)`, then normalize by the ideal ordering of the same candidates. Skip zero-ideal-gain requests in the NDCG mean and report their count and share. Use a fixed item-ID tie break for model scores.

Also report NDCG@3, recall of purchased items in the top three, and reciprocal rank of the first purchased item on requests with a purchase. Publish the corresponding denominators. Compare with both baselines using paired differences on the same requests. Where users recur, bootstrap users with their requests intact. Inspect results by device, AOV tier, query frequency, and new versus returning users. A judged query–product sample can test semantic relevance independently of logged behavior.

**Fixture acceptance test.** With the gains above, the displayed order yields NDCG@10 of **0.99006** for wireless headphones and **1.00000** for mechanical keyboard, averaging **0.99503**. Descending BM25 averages **0.97312**. Both purchases are already at position 1. These values test grouping, ordering, and the evaluator. Add cases covering tied scores, multiple purchases, and zero ideal gain before evaluating a larger export.

## 6. Online evaluation

Randomize by user between incumbent and challenger, keeping retrieval and eligibility fixed. A proposed online primary metric is the **fraction of eligible search sessions with at least one attributed grade 2–4 outcome**. It captures high dwell, wishlisting, cart adds, and purchases. Verify with product that these events represent successful search, particularly for queries where long dwell can indicate confusion.

Track purchases per search session, revenue per search session, reformulation, and zero-click sessions. Monitor latency, fallback rate, and delayed returns. Before rollout, measure the control baseline and agree on the minimum worthwhile effect, guardrail tolerances, sample size, and observation window. Check assignment and event logging during a small ramp. Use user-clustered intervals and allow attributed outcomes to mature before making the launch decision.

## 7. Questions for the interviewer

1. Is relevance judged primarily by query–product fit, by the supplied behavior grades, or by a specific business outcome when those disagree?
2. How is high dwell defined, and what grade should a click receive when dwell is missing? Can a wishlist or purchase receive credit without a recorded click?
3. Which impression gets credit for a later order when the same product appeared in multiple requests or sessions?
4. Are all retrieved candidates available, or only displayed results? Is there a larger timestamped search export, judged relevance set, or randomized exposure log?
5. Does the producer calculate cosine distance as specified? What candidate count, latency budget, and deliverable should the live exercise target?

For the live exercise, the first deliverable is a loader, the two deterministic baselines, and an evaluator that reproduces the fixture values above. The next milestone, once representative logs are available, is a chronological baseline-versus-LambdaMART comparison. An offline improvement qualifies the model for an online test.
