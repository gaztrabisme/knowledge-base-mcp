export const meta = {
  name: 'gate-audit',
  description: 'Audit the retrieval golden gate: LLM-judge label quality + query realism, then synthesize a report card',
  phases: [
    { title: 'Judge', detail: 'parallel relevance-judging of a stratified sample for label completeness + correctness' },
    { title: 'Realism', detail: 'characterize synthetic-query vs real-workload gap' },
    { title: 'Synthesize', detail: 'combine judgments + deterministic stats into wiki/gate-audit.md report card' },
  ],
}

// args = { coverage_stats: {...}, significance_floor: "text", n_batches: 6, sample_queries: [...] }
const A = args || {}
const nBatches = A.n_batches || 6

const JUDGE_SCHEMA = {
  type: 'object',
  required: ['verdicts'],
  properties: {
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        required: ['query', 'gold_uniquely_answers', 'query_is_vague', 'paraphrase_drift',
                   'unlabeled_relevant_count', 'better_than_gold_present'],
        properties: {
          query: { type: 'string' },
          gold_uniquely_answers: { type: 'string', enum: ['yes', 'partial', 'no'] },
          query_is_vague: { type: 'boolean' },
          paraphrase_drift: { type: 'boolean' },
          unlabeled_relevant_count: { type: 'integer' },
          better_than_gold_present: { type: 'boolean' },
          note: { type: 'string' },
        },
      },
    },
  },
}

const judgePrompt = (i) => `You are auditing a retrieval benchmark's LABEL QUALITY. Read the file
\`data/golden/audit/sample_batch_${i}.jsonl\` (one JSON object per line). Each row has:
- query : the (hard, paraphrased) developer question being evaluated
- easy_query : the original, pre-paraphrase question (compare to detect paraphrase drift)
- gold_source_text : the text of the ONE chunk labeled as the correct answer
- gold_parent_ids : the gold parent id(s)
- retrieved_candidates : the top-10 the live system returned, each with {rank, is_labeled_positive, heading_path, book, snippet}

For EACH row, judge — strictly, like a skeptical IR researcher:
1. gold_uniquely_answers: does gold_source_text actually and specifically answer \`query\`?
   "yes" = fully and uniquely; "partial" = related but incomplete/tangential; "no" = does not answer it.
2. query_is_vague: true if the query is so generic that many unrelated chunks could equally answer it
   (a bad benchmark query), false if it targets specific content.
3. paraphrase_drift: true if the hard \`query\` was paraphrased so far from \`easy_query\` that the gold
   chunk no longer cleanly answers it (the hardening broke the label).
4. unlabeled_relevant_count: among retrieved_candidates with is_labeled_positive=false, how many ALSO
   substantively answer \`query\` (i.e. would be acceptable answers a human would mark relevant)?
   This measures LABEL INCOMPLETENESS — the single-positive label missing other correct chunks.
5. better_than_gold_present: true if any unlabeled retrieved candidate answers \`query\` BETTER than gold.
Judge from the snippets; they are truncated but sufficient. Be calibrated, not generous: only count a
candidate as relevant if it genuinely answers the specific question. Return one verdict per row.`

phase('Judge')
const judgeResults = await parallel(
  Array.from({ length: nBatches }, (_, i) => () =>
    agent(judgePrompt(i), { label: `judge:batch${i}`, phase: 'Judge', schema: JUDGE_SCHEMA })
  )
)
const verdicts = judgeResults.filter(Boolean).flatMap(r => r.verdicts || [])
log(`Judged ${verdicts.length} queries across ${nBatches} batches`)

// aggregate the judgments deterministically (no LLM) so the synthesizer gets clean numbers
const n = verdicts.length || 1
const agg = {
  n: verdicts.length,
  gold_not_unique: verdicts.filter(v => v.gold_uniquely_answers !== 'yes').length,
  gold_no: verdicts.filter(v => v.gold_uniquely_answers === 'no').length,
  vague: verdicts.filter(v => v.query_is_vague).length,
  drift: verdicts.filter(v => v.paraphrase_drift).length,
  has_unlabeled_relevant: verdicts.filter(v => (v.unlabeled_relevant_count || 0) > 0).length,
  better_than_gold: verdicts.filter(v => v.better_than_gold_present).length,
  mean_unlabeled_relevant: (verdicts.reduce((s, v) => s + (v.unlabeled_relevant_count || 0), 0) / n).toFixed(3),
}
agg.pct = {
  gold_not_unique: (100 * agg.gold_not_unique / n).toFixed(1),
  vague: (100 * agg.vague / n).toFixed(1),
  drift: (100 * agg.drift / n).toFixed(1),
  has_unlabeled_relevant: (100 * agg.has_unlabeled_relevant / n).toFixed(1),
  better_than_gold: (100 * agg.better_than_gold / n).toFixed(1),
}
log(`Label incompleteness: ${agg.pct.has_unlabeled_relevant}% of queries have >=1 unlabeled-but-relevant top-10 hit; ` +
    `${agg.pct.better_than_gold}% have a better-than-gold unlabeled hit`)

phase('Realism')
const REALISM_SCHEMA = {
  type: 'object',
  required: ['realism_grade', 'findings', 'workload_gap'],
  properties: {
    realism_grade: { type: 'string', enum: ['high', 'medium', 'low'] },
    findings: { type: 'array', items: { type: 'string' } },
    workload_gap: { type: 'string' },
  },
}
const realism = await agent(
  `You are assessing whether a retrieval benchmark's QUERIES match how a real developer (and an AI coding
agent like Claude Code) actually searches a technical knowledge base. These queries were SYNTHETICALLY
generated by an LLM that was shown the answer chunk and asked to write a question it answers, then
paraphrased to remove the chunk's vocabulary.

Computed stats: ${JSON.stringify(A.coverage_stats || {})}

Read ~40 of the actual queries to assess: run
\`cut -c1-200 data/golden/golden_queries_hard.jsonl | head -40\` or Read the file
\`data/golden/golden_queries_hard.jsonl\` and look at the "query" fields (also "original_query"
for the pre-paraphrase form).

Assess: (a) do these read like real dev searches or like generated-from-the-answer questions?
(b) real devs/agents often search with short keyword/jargon phrases or error strings — how far are these
full-sentence natural-language questions from that? (c) what workload distribution is MISSING from this
gate (e.g. short keyword queries, exact-error lookups, multi-hop, navigational "where is X defined")?
Grade realism high/medium/low and list concrete findings.`,
  { label: 'realism', phase: 'Realism', schema: REALISM_SCHEMA }
)

phase('Synthesize')
const SYNTH_SCHEMA = {
  type: 'object',
  required: ['markdown', 'overall_grade', 'top_remediations'],
  properties: {
    markdown: { type: 'string' },
    overall_grade: { type: 'string' },
    top_remediations: { type: 'array', items: { type: 'string' } },
  },
}
const synth = await agent(
  `Write a rigorous AUDIT REPORT CARD for a retrieval-evaluation "golden gate" as a markdown document.
You are a skeptical IR-evaluation expert. The gate decides every retrieval change in this project, so its
validity is load-bearing. Grade it against established test-collection criteria (à la TREC): construct
validity, label completeness, label correctness, query realism/external validity, coverage/stratification,
and statistical power / discrimination.

EVIDENCE TO USE (cite the numbers):

1. Deterministic coverage/power stats:
${JSON.stringify(A.coverage_stats || {}, null, 2)}

2. Statistical discrimination floor (bootstrap CI of each metric's mean over n=200; the pipeline is
deterministic so this sampling CI IS the minimum detectable effect):
${A.significance_floor || '(not available)'}

NUANCE you must state correctly: the CI above is MARGINAL (uncertainty in the absolute metric). A
PAIRED bootstrap on two correlated systems can resolve a smaller systematic gap than this marginal
width, because per-query variance cancels. So do NOT claim "0.012 is noise" purely from the marginal
CI; claim instead: (a) 0.012 is ~3.6x below the marginal NDCG CI half-width (0.0434), (b) the v6.3
rejection was never significance-tested at all, and (c) the rigorous paired test requires re-indexing
the Qwen3 candidate (currently remote-gated). Even under high cross-system correlation the paired CI
would be ~half the marginal (~0.02), so a 0.012 gap is very likely not significant — but frame it as
"not established as significant / pending paired test," not as a proven null.

3. Label-quality judgments over a stratified ${agg.n}-query sample:
${JSON.stringify(agg, null, 2)}

4. Query-realism assessment:
${JSON.stringify(realism || {}, null, 2)}

KEY CONTEXT the report must reckon with:
- Every query has a SINGLE positive label; the corpus is 48k chunks. If label incompleteness is high,
  NDCG@10 is understated and the R@50 "ceiling" (0.96) and the "8 never-retrieved misses" may be LABEL
  artifacts, not corpus-coverage gaps.
- v6.3 REJECTED an embedding upgrade on a 0.802-vs-0.790 aggregate NDCG gap (n=200). Use the discrimination
  floor to state plainly whether a gap that size is distinguishable from sampling noise. This retroactively
  judges whether that decision was statistically sound.
- Queries are synthetic (generated from the answer) and paraphrased; hard_negative_child_ids is empty.

Produce: a graded report card (A–F per criterion with one-line justification + the cited number), an
overall grade, an explicit verdict on whether the gate can currently distinguish the ~0.01-NDCG deltas it
has been used to adjudicate, and a prioritized remediation list (highest impact first, each with the
specific fix). Be concrete and honest — negative findings about the gate are the whole point.
Return the full markdown in 'markdown', plus overall_grade and top_remediations[].`,
  { label: 'synthesize', phase: 'Synthesize', schema: SYNTH_SCHEMA }
)

return {
  label_quality: agg,
  realism,
  overall_grade: synth.overall_grade,
  top_remediations: synth.top_remediations,
  markdown: synth.markdown,
}
