// Runs each recorded compile through the real library's kv-stable picker and prints one JSON line
// per session: the plan for every turn. test_fold.py diffs it.
//
//   node oracle_fold.mjs sessions.json
//
// CONTEXT_MANAGER overrides the library entry point (default: the harness's node_modules).
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const entry = process.env.CONTEXT_MANAGER ?? fileURLToPath(new URL(
  '../../../../../node_modules/.pnpm/@animalabs+context-manager@0.10.1/node_modules/@animalabs/context-manager/dist/src/index.js',
  import.meta.url,
))
const { AutobiographicalStrategy } = await import(entry)
const { KvStableStrategy, Picker } = await import(new URL('adaptive/index.js', `file://${entry}`).href)   // not re-exported at the root
const pricing = new AutobiographicalStrategy({})

for (const turns of JSON.parse(readFileSync(process.argv[2], 'utf8'))) {
  console.log(JSON.stringify(turns.map(({ chunks, head, tail, headTokens, tailTokens, summaries, budget, options }) => {
    const solver = new KvStableStrategy(options)
    const result = new Picker(solver).run({
      chunks,
      summaries: new Map(summaries.map((s) => [s.id, s])),
      recallPairTokens: new Map(summaries.map((s) => [s.id, s.recall])),
      headTokens,
      tailTokens,
      headChunkIds: new Set(head),
      tailChunkIds: new Set(tail),
    }, budget)
    const plan = solver.lastPlan()
    return {
      levels: Object.fromEntries(chunks.filter((c) => !c.pinned).map((c) => [c.id, result.finalResolutions.get(c.id)])),
      tokens: result.finalTokens,
      perturbation: plan.perturbation,
      override: plan.override ?? null,
      escalated: plan.escalated,
      exhausted: result.exhausted,
      produced: result.produced.map((p) => [p.range.firstChunkId, p.range.lastChunkId]),
      recall: summaries.map((s) => pricing.recallPairCost(s)),
    }
  })))
}
