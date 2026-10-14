// Replays sessions through the real library's AutobiographicalStrategy against a scripted model, and
// prints one JSON line per session: the state after every event. test_l1.py diffs its ticks (L1s and
// merges); test_drive.py its passes, which run the harness's own sequence (compaction-autobiographical
// index.ts: foldPass, compileFolds, driveTick, releaseRetryableQuarantine), transcribed below.
//
//   node oracle_l1.mjs sessions.json
//
// The strategy is initialized on an in-memory state store with Chronicle's slot semantics, with the
// harness's carrierPolicy. The model records each request as JSON.stringify sends it and answers from
// the event's script, then a default; it counts the calls it starts and notes a retryable failure, as
// the harness's bridge hooks do. CONTEXT_MANAGER overrides the library entry point (default: the
// harness's node_modules).
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const entry = process.env.CONTEXT_MANAGER ?? fileURLToPath(new URL(
  '../../../../../node_modules/.pnpm/@animalabs+context-manager@0.10.1/node_modules/@animalabs/context-manager/dist/src/index.js',
  import.meta.url,
))
const { MessageStore, AutobiographicalStrategy, splitMixedToolMessages } = await import(entry)
const DEFAULT_REPLY = { stopReason: 'end_turn', content: [{ type: 'text', text: 'Nothing else stood out.' }] }

// The message store reaches its Chronicle store only to persist; estimation never does.
const chronicle = new Proxy({}, {
  get: (_, key) => key === 'then' ? undefined : key === 'currentBranch' ? () => ({ name: 'main' }) : () => null,
})

const copy = (value) => JSON.parse(JSON.stringify(value))
const BRANCH = { name: 'main' }

/** Chronicle's state slots: snapshots, and append logs read back live. */
class StateStore {
  arrays = new Map()
  scalars = new Map()
  registrations = []
  seq = 0
  registerState(registration) {
    this.registrations.push(registration)
    if (registration.strategy === 'snapshot') this.scalars.set(registration.id, null)
    else this.arrays.set(registration.id, [])
  }
  updateStateStrategy() {}
  listStates() { return this.registrations.map(({ id, strategy }) => ({ id, strategy })) }
  currentBranch() { return BRANCH }
  currentSequence() { return this.seq - 1 }
  getStateJson(id) { return this.scalars.has(id) ? this.scalars.get(id) : this.arrays.get(id) ?? null }
  setStateJson(id, value) {
    if (this.arrays.has(id)) this.arrays.set(id, copy(value))
    else this.scalars.set(id, copy(value))
    return this.record()
  }
  appendToStateJsonWithIdentity(id, item, idField, sequenceField) {
    const record = this.record()
    this.array(id).push({ ...copy(item), [idField]: record.id, [sequenceField]: record.sequence })
    return record
  }
  appendToStateJson(id, item) {
    this.array(id).push(copy(item))
    return this.record()
  }
  editStateItem(id, index, payload) {
    this.array(id)[index] = JSON.parse(payload.toString('utf8'))
    return this.record()
  }
  getStateLen(id) { return this.arrays.get(id)?.length ?? null }
  redactStateItems(id, start, end) {
    this.array(id).splice(start, end - start)
    return this.record()
  }
  compactState() { return null }
  array(id) {
    if (!this.arrays.has(id)) this.arrays.set(id, [])
    return this.arrays.get(id)
  }
  record() {
    const sequence = this.seq++
    return { id: `r${sequence}`, sequence }
  }
}

for (const { config, events } of JSON.parse(readFileSync(process.argv[2], 'utf8'))) {
  const messageStore = new MessageStore(chronicle)
  const messages = []
  const view = {
    getAll: () => messages,
    get: (id) => messages.find((m) => m.id === id),
    estimateTokens: (message) => messageStore.estimateTokens(message),
    setTokenCalibration: (factor) => messageStore.setTokenCalibration(factor),
    getTokenCalibration: () => messageStore.getTokenCalibration(),
  }
  let requests = []
  let replies = []
  let started = 0
  let transientFailed = false
  const membrane = {
    async complete(request) {
      started++
      requests.push(JSON.stringify(request))
      const reply = replies.shift() ?? DEFAULT_REPLY
      if (reply.throw) {
        const { message, status, type, retryable } = reply.throw
        if (retryable === true) transientFailed = true
        throw Object.assign(new Error(message), { httpStatus: status, type, retryable, scripted: true })
      }
      return copy(reply)
    },
  }
  const store = new StateStore()
  const ctx = { messageStore: view, membrane, store, namespace: 'cm', currentSequence: 0, tools: undefined, systemPrompt: undefined }
  const strategy = new AutobiographicalStrategy({ ...config, adaptiveResolution: true, foldingStrategy: 'kv-stable', carrierPolicy: 'live-strip' })
  const compile = (maxTokens, reserveForResponse) => {
    try {
      const entries = strategy.select(view, null, { maxTokens, reserveForResponse })
      return { messages: entries.flatMap((e) => {
        const parts = splitMixedToolMessages([{ participant: e.participant, content: e.content }])
        return parts.map((part, i) => ({ ...part, ...(e.cacheMarker && i === parts.length - 1 ? { cacheBreakpoint: true } : {}) }))
      }) }
    } catch (error) {
      if (error.name !== 'OverBudgetError') throw error
      return { refusal: [error.message.split(' but ')[0], error.actual, error.budget] }
    }
  }
  const queued = () => {
    const { l1QueueLength, mergeQueueLength } = strategy.getProgressSnapshot()
    return l1QueueLength + mergeQueueLength
  }
  await strategy.initialize(ctx)
  const states = []
  for (const event of events) {
    let error = false
    let compiled = null
    requests = []
    switch (event.kind) {
      case 'append':
        messages.push(event.message)
        strategy.rebuildChunks(view)        // onNewMessage
        break
      case 'calibrate':
        messageStore.setTokenCalibration(event.multiplier)
        break
      case 'host':
        ctx.tools = event.tools ?? undefined
        ctx.systemPrompt = event.system ?? undefined
        break
      case 'tick':
        replies = [...event.replies]
        try {
          await strategy.tick(ctx)
        } catch (thrown) {
          error = thrown?.scripted ? thrown.message : true   // the provider's own error, or another
        }
        break
      case 'pass': {                       // foldPass
        for (const message of event.messages) {
          messages.push(message)
          strategy.rebuildChunks(view)
        }
        if (event.host) {
          ctx.tools = event.host.tools ?? undefined
          ctx.systemPrompt = event.host.system ?? undefined
        }
        if (event.usage != null) strategy.reportRealInputTokens(event.usage)
        // compileFolds
        compiled = compile(event.budget, event.reserve)
        if (compiled.refusal) {
          const [, actual, budget] = compiled.refusal
          const affordable = actual + event.reserve
          if (!(actual <= budget || affordable > event.window)) compiled = compile(affordable, event.reserve)
        }
        // kickTick: driveTick, then releaseRetryableQuarantine
        replies = [...event.replies]
        const quarantined = new Set(strategy.getCompressionQuarantineStatus().keys)
        transientFailed = false
        try {
          for (;;) {
            const calls = started
            const before = queued()
            await strategy.tick(ctx)
            if (started !== calls || queued() >= before) break
          }
        } catch (thrown) {
          error = thrown?.scripted ? thrown.message : true   // the provider's own error, or another
        }
        if (transientFailed) {
          for (const key of strategy.getCompressionQuarantineStatus().keys) {
            if (!quarantined.has(key)) await strategy.clearCompressionRefusalQuarantine(key)
          }
        }
        break
      }
      default:
        throw new Error(`unknown event ${event.kind}`)
    }
    states.push({
      requests,
      error,
      summaries: strategy.summaries.map((s) => ({
        id: s.id, level: s.level, content: s.content, tokens: s.tokens, sourceIds: s.sourceIds,
        sourceRange: s.sourceRange, mergedInto: s.mergedInto ?? null, responseContent: s.responseContent ?? null,
        provenance: s.provenance ?? null,
      })),
      chunks: strategy.chunks.map((c) => c.messages.map((m) => m.id)),
      compressed: strategy.chunks.filter((c) => c.compressed).map((c) => c.index),
      queue: [...strategy.compressionQueue],
      quarantine: [...strategy.readCompressionQuarantineProjection().values()]
        .map(({ record }) => [record.chunkSourceHash, record.canonicalRequestHash]).sort(),
      counter: strategy.summaryIdCounter,
      merges: strategy.mergeQueue.map((m) => ({
        level: m.level, sourceIds: m.sourceIds, attempts: m.attempts ?? 0, lastStopReason: m.lastStopReason ?? null,
        lastOutcome: m.lastOutcome ?? null, hadRefusal: m.hadRefusal === true, serverErrorStreak: m.serverErrorStreak ?? 0,
      })),
      mergeQuarantine: [...strategy.mergeQuarantine.values()].map((r) => JSON.stringify(r.sourceIds)).sort(),
      compiled,
      levels: Object.fromEntries([...strategy.resolutions].filter(([, level]) => level !== 0)),
      multiplier: strategy._calibration,
      armed: strategy._calibrationArmed ? strategy._lastCompileEstimate : 0,
    })
  }
  console.log(JSON.stringify(states))
}
