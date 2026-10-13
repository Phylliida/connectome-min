// Replays scenarios through the real @animalabs/context-manager and prints, per scenario, one
// JSON line holding the strategy's state after every event. test_faithful.py diffs it.
//
//   node oracle.mjs scenarios.json
//
// CONTEXT_MANAGER overrides the library entry point (default: the harness's node_modules).
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const entry = process.env.CONTEXT_MANAGER ?? fileURLToPath(new URL(
  '../../../node_modules/.pnpm/@animalabs+context-manager@0.10.1/node_modules/@animalabs/context-manager/dist/src/index.js',
  import.meta.url,
))
const { MessageStore, AutobiographicalStrategy } = await import(entry)

// The message store reaches its Chronicle store only to persist; estimation never does.
// selectAdaptive's PickerInputs for a budget, captured where it hands them to the picker; null
// when it refuses (OverBudgetError) before getting there.
const STOP = new Error('captured')
function plannerInputs(strategy, view, maxTokens) {
  let captured = null
  strategy.buildPicker = (inputs) => { captured = inputs; throw STOP }
  try {
    strategy.selectAdaptive(view, { maxTokens, reserveForResponse: 0 })
  } catch (error) {
    if (error !== STOP && error.name !== 'OverBudgetError') throw error
  }
  if (captured === null) return null
  const middle = captured.chunks.filter((c) => !c.pinned)
  return {
    leaves: middle.map((c) => [c.id, c.sequence, c.rawTokens, c.currentResolution, c.l1Id ?? null, c.salience]),
    headTokens: captured.headTokens,
    tailTokens: captured.tailTokens,
    zone: captured.headChunkIds.size + captured.tailChunkIds.size,
    now: Math.max(0, ...captured.chunks.map((c) => c.sequence)),
  }
}

const chronicle = new Proxy({}, {
  get: (_, key) => key === 'then' ? undefined : key === 'currentBranch' ? () => ({ name: 'main' }) : () => null,
})

for (const { config, events } of JSON.parse(readFileSync(process.argv[2], 'utf8'))) {
  const store = new MessageStore(chronicle)
  const messages = []
  const view = {
    getAll: () => messages,
    estimateTokens: (message) => store.estimateTokens(message),
    setTokenCalibration: (factor) => store.setTokenCalibration(factor),
    getTokenCalibration: () => store.getTokenCalibration(),
  }
  const strategy = new AutobiographicalStrategy(config)
  // Branch guards need a ContextManager-opened store; nothing here switches branches.
  strategy.requireBranchMutation = strategy.requireLoadedBranch = () => {}
  strategy._storeView = view
  const states = []
  for (const event of events) {
    let planner = null
    switch (event.kind) {
      case 'append':
        messages.push(event.message)
        strategy.rebuildChunks(view)        // onNewMessage
        break
      case 'compile':
        strategy.rebuildChunks(view)        // select, before the picker runs
        planner = plannerInputs(strategy, view, event.budget)
        strategy._lastCompileEstimate = event.window
        strategy._calibrationArmed = true
        break
      case 'usage':
        strategy.reportRealInputTokens(event.tokens)
        break
      case 'demand':
        strategy.enqueueL1ForRange(event.first, event.last)
        break
      case 'summarize':                    // what a completed tick leaves behind
        for (let n = 0; n < event.count && strategy.compressionQueue.length > 0; n++) {
          const chunk = strategy.chunks[strategy.compressionQueue.shift()]
          chunk.compressed = true
          strategy.markChunkRecordCompressed(chunk.recordId, `L1-${chunk.index}`)
        }
        break
      default:
        throw new Error(`unknown event ${event.kind}`)
    }
    states.push({
      chunks: strategy.chunks.map((chunk) => chunk.messages.map((message) => message.id)),
      queue: [...strategy.compressionQueue],
      multiplier: strategy._calibration,
      head: strategy.getHeadWindowEnd(view),
      tail: strategy.getRecentWindowStart(view),
      postStrip: strategy.postStripEstimates(view),
      store: messages.map((message) => store.estimateTokens(message)),
      textOnly: messages.map((message) => strategy.estimateTextOnlyTokens(message)),
      salience: messages.map((message) => AutobiographicalStrategy.computeStaticSalience(message)),
      planner,
    })
  }
  console.log(JSON.stringify(states))
}
