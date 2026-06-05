export const meta = {
  name: 'judge-validation',
  description: 'Blind Claude re-grade of a sample of deepseek relevance judgments; returns per-item grades',
  phases: [{ title: 'Grade', detail: 'parallel blind relevance grading of validation batches' }],
}

const nBatches = (args && args.n_batches) || 6

const SCHEMA = {
  type: 'object',
  required: ['verdicts'],
  properties: {
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        required: ['idx', 'grade'],
        properties: {
          idx: { type: 'integer' },
          grade: { type: 'integer', enum: [0, 1, 2] },
        },
      },
    },
  },
}

const prompt = (i) => `You are independently grading retrieval relevance to validate another judge.
Read \`data/golden/audit/jval_batch_${i}.jsonl\` (one JSON object per line: idx, query, candidate_text).

For EACH row, grade how well candidate_text answers THE QUERY — STRICTLY:
  2 = directly and specifically answers the query (a developer would accept this as the answer)
  1 = answers a genuine PART of the specific question, or restates the same fact (NOT just "same topic")
  0 = same topic but does NOT answer this specific question, or only mentions it in passing, or irrelevant
These candidates were retrieved for the query, so they are ALL topically related — topical relatedness
is NOT relevance. When in doubt between 0 and 1, choose 0. Judge only from the text shown.

Return one verdict per row with its idx and your grade.`

phase('Grade')
const results = await parallel(
  Array.from({ length: nBatches }, (_, i) => () =>
    agent(prompt(i), { label: `grade:batch${i}`, phase: 'Grade', schema: SCHEMA })
  )
)
const verdicts = results.filter(Boolean).flatMap(r => r.verdicts || [])
log(`Claude graded ${verdicts.length} items`)
return { claude_grades: verdicts }
