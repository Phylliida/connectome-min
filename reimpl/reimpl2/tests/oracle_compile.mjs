// Replays sessions through the real library's AutobiographicalStrategy.select (rebuild, kv-stable
// planner, emission and its passes, commit) plus ContextManager.compile's tool-message split, and
// prints one JSON line per session: the state after every event. test_compile.py diffs it.
//
//   node oracle_compile.mjs sessions.json
//
// CONTEXT_MANAGER overrides the library entry point (default: the harness's node_modules). Summaries
// and merges are injected as the library would record them; no model is called. The strategy runs
// with the harness's carrierPolicy, 'live-strip'.
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const entry = process.env.CONTEXT_MANAGER ?? fileURLToPath(new URL(
  '../../../../../node_modules/.pnpm/@animalabs+context-manager@0.10.1/node_modules/@animalabs/context-manager/dist/src/index.js',
  import.meta.url,
))
const { MessageStore, AutobiographicalStrategy, splitMixedToolMessages } = await import(entry)

// The message store reaches its Chronicle store only to persist; estimation never does.
const chronicle = new Proxy({}, {
  get: (_, key) => key === 'then' ? undefined : key === 'currentBranch' ? () => ({ name: 'main' }) : () => null,
})

for (const { config, events } of JSON.parse(readFileSync(process.argv[2], 'utf8'))) {
  const store = new MessageStore(chronicle)
  const messages = []
  const view = {
    getAll: () => messages,
    get: (id) => messages.find((m) => m.id === id),
    estimateTokens: (message) => store.estimateTokens(message),
    setTokenCalibration: (factor) => store.setTokenCalibration(factor),
    getTokenCalibration: () => store.getTokenCalibration(),
  }
  const strategy = new AutobiographicalStrategy({ ...config, adaptiveResolution: true, foldingStrategy: 'kv-stable', carrierPolicy: 'live-strip' })
  // Branch guards need a ContextManager-opened store; nothing here switches branches.
  strategy.requireBranchMutation = strategy.requireLoadedBranch = () => {}
  strategy._storeView = view
  const byId = (id) => strategy.summaries.find((s) => s.id === id)
  const states = []
  for (const event of events) {
    let compiled = null
    switch (event.kind) {
      case 'append':
        messages.push(event.message)
        strategy.rebuildChunks(view)        // onNewMessage
        break
      case 'usage':
        strategy.reportRealInputTokens(event.tokens)
        break
      case 'summarize':                    // what completed ticks leave behind
        for (const memory of event.memories) {
          if (strategy.compressionQueue.length === 0) break
          const chunk = strategy.chunks[strategy.compressionQueue.shift()]
          const id = `L1-${chunk.index}`
          const ids = chunk.messages.map((m) => m.id)
          chunk.compressed = true
          strategy.markChunkRecordCompressed(chunk.recordId, id)
          strategy.summaries.push({ id, level: 1, content: memory.text, tokens: memory.tokens, sourceLevel: 0, sourceIds: ids,
            sourceRange: { first: ids[0], last: ids.at(-1) }, created: 0, ...(memory.blocks ? { responseContent: memory.blocks } : {}) })
        }
        break
      case 'merge': {                      // what a completed merge leaves behind
        // The port picks the children; one this store lacks means the two already disagree, which
        // the next compared state shows.
        const children = event.children.map(byId)
        if (children.length > 0 && children.every(Boolean)) {
          strategy.summaries.push({ id: event.id, level: children[0].level + 1, content: event.text, tokens: event.tokens,
            sourceLevel: children[0].level, sourceIds: event.children,
            sourceRange: { first: children[0].sourceRange.first, last: children.at(-1).sourceRange.last }, created: 0,
            ...(event.blocks ? { responseContent: event.blocks } : {}) })
          for (const child of children) child.parentId = event.id
        }
        break
      }
      case 'compile':
        try {
          const entries = strategy.select(view, null, { maxTokens: event.budget, reserveForResponse: 0 })
          compiled = { messages: entries.flatMap((e) => {
            const parts = splitMixedToolMessages([{ participant: e.participant, content: e.content }])
            return parts.map((part, i) => ({ ...part, ...(e.cacheMarker && i === parts.length - 1 ? { cacheBreakpoint: true } : {}) }))
          }) }
        } catch (error) {
          if (error.name !== 'OverBudgetError') throw error
          compiled = { refusal: [error.message.split(' but ')[0], error.actual, error.budget] }
        }
        break
      default:
        throw new Error(`unknown event ${event.kind}`)
    }
    states.push({
      compiled,
      levels: Object.fromEntries([...strategy.resolutions].filter(([, level]) => level !== 0)),
      queue: [...strategy.compressionQueue],
      multiplier: strategy._calibration,
      armed: strategy._calibrationArmed ? strategy._lastCompileEstimate : 0,
    })
  }
  console.log(JSON.stringify(states))
}
