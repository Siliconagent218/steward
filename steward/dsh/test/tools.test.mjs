import assert from 'node:assert/strict'
import { test } from 'node:test'
import { egoTools } from '@steward/dsh-tool-ego'
import { worldTools } from '@steward/dsh-tool-world'
import { receiptDenial, reviewExecution } from '@steward/dsh-robot-safety'

function fakeEgo(receipts = {}) {
  const calls = []
  const job = (state) => ({ id: 'job-1', skill: 'pick', receipt: 'rcpt-1', state, message: '', log: [] })
  return {
    calls,
    summary: async () => ({ objects: [] }),
    query: async (q) => { calls.push(['query', q]); return [] },
    observe: async () => ({ visible: [] }),
    explore: async () => job('running'),
    predict: async (skill, params) => {
      calls.push(['predict', skill, params])
      return { id: 'rcpt-1', skill, params, risk: 0.1, feasible: true, withinRiskLimit: true, maxRisk: 0.35, issues: [],
        expiresAt: 60, plan: { target: params.target, path: [[0, 0], [1, 1]], reach: 0.3 } }
    },
    receipt: async (id) => {
      if (!(id in receipts)) throw new Error(`unknown receipt '${id}'`)
      return receipts[id]
    },
    run: async (id) => { calls.push(['run', id]); return job('running') },
    job: async () => job('succeeded'),
    wait: async (id, timeoutS) => { calls.push(['wait', id, timeoutS]); return job('succeeded') },
    stop: async () => ({ stopped: null }),
    reportProblem: async (report) => { calls.push(['report', report]); return { id: 1, ...report } },
  }
}

const exec = (name, args) => ({ name, arguments: args, signal: new AbortController().signal, callId: 'c1' })

function tool(tools, name) {
  const found = tools.find(t => t.name === name)
  assert.ok(found, `tool ${name} is registered`)
  return found
}

test('ego tools expose the robot vocabulary without World State concepts', () => {
  const names = egoTools(fakeEgo()).map(t => t.name)
  assert.deepEqual(names, ['ego_state', 'ego_query', 'ego_observe', 'ego_explore', 'ego_predict', 'skill_run',
    'skill_status', 'body_stop', 'report_problem'])
  for (const t of egoTools(fakeEgo())) assert.doesNotMatch(t.description, /ground truth|world state/i)
})

test('ego_predict forwards only the given parameters and hides the raw path', async () => {
  const ego = fakeEgo()
  const result = await tool(egoTools(ego), 'ego_predict').execute({ skill: 'pick', target: 'e:4' }, exec('ego_predict'))
  assert.deepEqual(ego.calls[0], ['predict', 'pick', { target: 'e:4' }])
  assert.equal(result.receipt, 'rcpt-1')
  assert.equal(result.plan.path, undefined)
  assert.equal(result.plan.reach, 0.3)
})

test('skill_run waits for completion by default', async () => {
  const ego = fakeEgo()
  const result = await tool(egoTools(ego), 'skill_run').execute({ receipt: 'rcpt-1' }, exec('skill_run'))
  assert.deepEqual(ego.calls.map(c => c[0]), ['run', 'wait'])
  assert.equal(result.state, 'succeeded')
})

test('world_edit maps floor and support placements to World Control API edits', async () => {
  const edits = []
  const world = { edit: async (request) => { edits.push(request); return { ok: true, ...request } } }
  const edit = tool(worldTools(world), 'world_edit')
  await edit.execute({ action: 'move', entity: 'w:cup-1', on: 'w:sideboard', offset_x: 0.1 }, exec('world_edit'))
  await edit.execute({ action: 'spawn', entity: 'w:bowl-2', asset: 'bowl', x: 1, y: -0.5, yaw: 0.3 }, exec('world_edit'))
  await edit.execute({ action: 'remove', entity: 'w:plate-1' }, exec('world_edit'))
  assert.deepEqual(edits, [
    { action: 'move', entity: 'w:cup-1', on: 'w:sideboard', offset: [0.1, 0] },
    { action: 'spawn', entity: 'w:bowl-2', asset: 'bowl', at: { x: 1, y: -0.5 }, yaw: 0.3 },
    { action: 'remove', entity: 'w:plate-1' },
  ])
  await assert.rejects(edit.execute({ action: 'move', entity: 'w:cup-1', x: 1 }, exec('world_edit')), /both x and y/)
  await assert.rejects(edit.execute({ action: 'spawn', entity: 'w:x', asset: 'cup' }, exec('world_edit')), /needs a placement/)
})

test('report_problem forwards the goal, step, diagnosis and message to the operator alert', async () => {
  const ego = fakeEgo()
  const report = { goal: '把红碗放到操作台', step: '抓取 e:4', diagnosis: '抓取两次都没对准：e:4 位置不确定度 4 cm', message: '请检查碗的位置' }
  const result = await tool(egoTools(ego), 'report_problem').execute({ ...report, extra: 'dropped' }, exec('report_problem'))
  assert.deepEqual(ego.calls.at(-1), ['report', report])
  assert.equal(result.diagnosis, report.diagnosis)
})

test('safety policy denies skill_run without a receipt, with an unknown one, or above the risk limit', async () => {
  const receipts = {
    ok: { id: 'ok', risk: 0.1, feasible: true, issues: [] },
    risky: { id: 'risky', risk: 0.6, feasible: true, issues: ['target position uncertainty 4.0 cm'] },
    used: { id: 'used', risk: 0.1, feasible: true, issues: [], used: true },
  }
  const ego = fakeEgo(receipts)
  assert.equal((await reviewExecution(ego, exec('skill_run', {}), 0.35)).kind, 'deny')
  assert.match((await reviewExecution(ego, exec('skill_run', { receipt: 'nope' }), 0.35)).reason, /could not be verified/)
  assert.match((await reviewExecution(ego, exec('skill_run', { receipt: 'risky' }), 0.35)).reason, /above the limit/)
  assert.match((await reviewExecution(ego, exec('skill_run', { receipt: 'used' }), 0.35)).reason, /already executed/)
  assert.equal(await reviewExecution(ego, exec('skill_run', { receipt: 'ok' }), 0.35), undefined)
  assert.equal(await reviewExecution(ego, exec('ego_state', {}), 0.35), undefined)
  assert.equal(receiptDenial(receipts.ok, 0.05), 'receipt ok has risk 0.1 above the limit 0.05: ')
})
