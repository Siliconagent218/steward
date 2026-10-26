// Cross-language check of the Ego Harness layer without a language model: the Python Steward
// host and Ego Runtime run as a subprocess; the TypeScript Ego API provider, robot tools and
// safety policy drive the Phase 0 mission over HTTP.
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { test } from 'node:test'
import { Context } from '@deepseek-ai/cordis'
import { EgoRuntimeClient } from '@steward/dsh-ego-runtime-client'
import { reviewExecution } from '@steward/dsh-robot-safety'
import { egoTools } from '@steward/dsh-tool-ego'

const STEWARD = resolve(import.meta.dirname, '../..')
const PYTHON = join(STEWARD, '.venv/bin/python')

function startSteward(root) {
  const child = spawn(PYTHON, ['-m', 'steward.cli', '--root', root, 'up', '--scenario', 'phase0-kitchen', '--speed', '6', '--wca-port', '0',
    '--robot-port', '0', '--api-port', '0'], { cwd: STEWARD, stdio: ['ignore', 'pipe', 'inherit'] })
  return new Promise((resolvePromise, reject) => {
    let buffer = ''
    child.stdout.on('data', (chunk) => {
      buffer += chunk
      const line = buffer.split('\n').find(l => l.startsWith('{'))
      if (line) resolvePromise({ child, info: JSON.parse(line) })
    })
    child.once('exit', code => reject(new Error(`steward up exited with ${code}`)))
  })
}

function buildKitchen(root) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(PYTHON, ['-m', 'steward.cli', '--root', root, 'build-kitchen'], { cwd: STEWARD, stdio: 'ignore' })
    child.once('exit', code => code === 0 ? resolvePromise() : reject(new Error(`build-kitchen exited with ${code}`)))
  })
}

test('robot tools complete the cup-to-table mission through the Ego API provider', { timeout: 240_000 }, async () => {
  const root = mkdtempSync(join(tmpdir(), 'steward-dsh-'))
  await buildKitchen(root)
  const { child, info } = await startSteward(root)
  try {
    const ego = new EgoRuntimeClient(new Context(), { endpoint: info.egoApi, timeoutMs: 15_000 })
    const tools = Object.fromEntries(egoTools(ego).map(t => [t.name, t]))
    const run = (name, args = {}) => tools[name].execute(args, { name, arguments: args, signal: AbortSignal.timeout(200_000) })

    assert.equal((await run('ego_explore')).state, 'succeeded')
    let state = await run('ego_state')
    const cup = state.objects.find(o => o.category === 'cup')
    const table = state.objects.find(o => o.category === 'table')
    assert.ok(cup && table, 'cup and table are perceived after exploring')

    let receipt = await run('ego_predict', { skill: 'pick', target: cup.id })
    if (!receipt.withinRiskLimit) {
      const denied = await reviewExecution(ego, { name: 'skill_run', arguments: { receipt: receipt.receipt } }, 0.35)
      assert.equal(denied.kind, 'deny', 'the safety policy refuses an over-limit receipt')
      const [tx, ty] = cup.position
      const nav = await run('ego_predict', { skill: 'navigate', x: tx - 0.4, y: ty - 1.0, yaw: Math.atan2(1.0, 0.4) })
      assert.ok(nav.withinRiskLimit, JSON.stringify(nav.issues))
      assert.equal((await run('skill_run', { receipt: nav.receipt })).state, 'succeeded')
      await new Promise(r => setTimeout(r, 1500))
      receipt = await run('ego_predict', { skill: 'pick', target: cup.id })
    }
    assert.ok(receipt.withinRiskLimit, JSON.stringify(receipt.issues))
    assert.equal(await reviewExecution(ego, { name: 'skill_run', arguments: { receipt: receipt.receipt } }, 0.35), undefined)
    const picked = await run('skill_run', { receipt: receipt.receipt })
    assert.equal(picked.state, 'succeeded', picked.message)

    const place = await run('ego_predict', { skill: 'place', on: table.id })
    assert.ok(place.withinRiskLimit, JSON.stringify(place.issues))
    assert.equal((await run('skill_run', { receipt: place.receipt })).state, 'succeeded')

    state = await run('ego_state')
    const cupNow = state.objects.find(o => o.id === cup.id)
    assert.equal(cupNow.on, table.id, 'the robot believes the cup now rests on the table')

    const truth = await fetch(`${info.wca}/wca/scene`).then(r => r.json())
    assert.ok(truth.relations.some(r => r.type === 'supported-by' && r.a.entity === 'w:cup-1' && r.b.entity === 'w:table-1'),
      'World State confirms the cup is on the table')
  } finally {
    child.kill('SIGTERM')
  }
})
