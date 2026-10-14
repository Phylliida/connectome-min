// Replays sessions through the real library's AutobiographicalStrategy.tick (L1s and merges) against a
// scripted model, and prints one JSON line per session: the state after every event. test_l1.py diffs it.
//
//   node oracle_l1.mjs sessions.json
//
// The strategy is initialized on an in-memory state store with Chronicle's slot semantics. The model
// records each request as JSON.stringify sends it and answers from the tick's script, then a default.
// CONTEXT_MANAGER overrides the library entry point (default: the harness's node_modules).
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const entry = process.env.CONTEXT_MANAGER ?? fileURLToPath(new URL(
  '../../../../../node_modules/.pnpm/@animalabs+context-manager@0.10.1/node_modules/@animalabs/context-manager/dist/src/index.js',
  import.meta.url,
))
const { MessageStore, AutobiographicalStrategy } = await import(entry)
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
  const membrane = {
    async complete(request) {
      requests.push(JSON.stringify(request))
      const reply = replies.shift() ?? DEFAULT_REPLY
      if (reply.throw) {
        const { message, status, type, retryable } = reply.throw
        throw Object.assign(new Error(message), { httpStatus: status, type, retryable })
      }
      return copy(reply)
    },
  }
  const store = new StateStore()
  const ctx = { messageStore: view, membrane, store, namespace: 'cm', currentSequence: 0, tools: undefined, systemPrompt: undefined }
  const strategy = new AutobiographicalStrategy({ ...config, adaptiveResolution: true, foldingStrategy: 'kv-stable' })
  await strategy.initialize(ctx)
  const states = []
  for (const event of events) {
    let error = false
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
        } catch {
          error = true
        }
        break
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
    })
  }
  console.log(JSON.stringify(states))
}
