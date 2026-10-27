/**
 * Consumer of `ctx.world`: the World Harness tools for building the simulated world.
 *
 * Authoring flow: create a draft asset (from a generator or an authored entity template), read
 * its validation report, render previews and inspect them with the file read tool, publish it,
 * compose a scene from published assets, save a scenario, then load it as an episode.
 * @module @steward/dsh-tool-world
 */

import type { Context } from '@deepseek-ai/cordis'
import { defineTool, type InferValue, type ToolDefinition } from '@deepseek-ai/dsh-tools'
import type { JsonObject, WorldEditRequest, WorldService } from '@steward/dsh-world'

export const name = 'steward-tool-world'
export const inject = ['tools', 'world']

const JSON_OUTPUT = { type: 'json' } as const
type Json = InferValue<typeof JSON_OUTPUT>

function json(value: unknown): Json {
  return JSON.parse(JSON.stringify(value)) as Json
}

function text(value: unknown): [{ type: 'text'; text: string }] {
  return [{ type: 'text', text: JSON.stringify(value, null, 1) }]
}

function asObject(value: unknown, what: string): JsonObject {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) throw new Error(`${what} must be a JSON object`)
  return value as JsonObject
}

const TEMPLATE_GUIDE = 'An entity template is a World State Entity with id "w:template" and an identity pose: '
  + '{id, role: environment|object|robot|actor, category: {value}, pose: {value: {p: [0,0,0], q: [1,0,0,0]}}, '
  + 'body: {kind: "rigid", root: {value: fixed|planar|free}, links: [{id, name, geometry: {collision: [{kind: '
  + '"primitive", shape: box|sphere|cylinder|capsule, size: full extents in metres (box [x,y,z], cylinder [r, length]), '
  + 'pose: {p, q}, rgba}], visual: [...]}, inertial: {value: {mass}}, material: {value: {friction, fragility}}, '
  + 'affordances: [{value: "graspable"}]}], joints: [{id, parent, child, type: {value: fixed|revolute|prismatic}, '
  + 'origin: {value: {p, q}}, axis: {value}, limits: {value: {lo, hi}}, q: {value: [0]}, qd: {value: [0]}}]}}. '
  + 'The origin sits on the floor at the footprint centre. Read an existing asset with asset_get for a full example.'

/**
 * Build the World Harness tool definitions over a World service.
 * @param world - the `ctx.world` service.
 * @returns tool definitions in registration order.
 */
export function worldTools(world: WorldService): ToolDefinition[] {
  return [
    defineTool({
      name: 'world_status',
      description: 'Read the loaded scenario, simulation time and speed, entities, robots (what they hold) and recent '
        + 'world events such as grasps.',
      parameters: {},
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(_args, exec) {
        return json(await world.status(exec.signal))
      },
    }),
    defineTool({
      name: 'world_scene',
      description: 'Read the true current World State: every entity with its exact pose and the relations between '
        + 'entities (what rests on what, what is held). This is ground truth that robots cannot see.',
      parameters: {
        detail: { type: 'boolean', description: 'Return the full SceneState instead of a compact listing (default false).' },
      },
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(args, exec) {
        const scene = await world.scene(exec.signal) as { entities: JsonObject[]; relations: JsonObject[]; stamp: number }
        if (args.detail) return json(scene)
        const entities = scene.entities.map((e) => {
          const p = (e.pose as { value: { p: number[] } }).value.p
          return { id: e.id, category: (e.category as { value: string }).value, role: e.role,
            position: p.map(v => Math.round(v * 1000) / 1000) }
        })
        const relations = scene.relations.map(r => `${(r.a as JsonObject).entity} ${r.type} ${(r.b as JsonObject).entity}`)
        return json({ t: scene.stamp, entities, relations })
      },
    }),
    defineTool({
      name: 'asset_search',
      description: 'List published assets, optionally filtered by a name/category substring or an exact category.',
      parameters: {
        query: { type: 'string', description: 'Substring of the asset name or category.' },
        category: { type: 'string', description: 'Exact category, for example "cup".' },
      },
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(args, exec) {
        return json({ assets: await world.searchAssets(args.query, args.category, exec.signal),
          generators: await world.generators(exec.signal) })
      },
    }),
    defineTool({
      name: 'asset_get',
      description: 'Read one published asset, including its entity template and validation report.',
      parameters: { name: { type: 'string', required: true, description: 'Asset name.' } },
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(args, exec) {
        return json(await world.getAsset(args.name, exec.signal))
      },
    }),
    defineTool({
      name: 'asset_create',
      description: 'Create a draft asset and validate it physically (compiles, has mass, rests without tipping or '
        + 'sinking when dropped). Give either a generator with params (generators: building {width, depth, height}, '
        + 'counter {width, depth, height}, dining_table {width, depth, height, mass}, cabinet_with_drawer {width, depth, '
        + 'height, drawer_height}, cup {radius, height, mass, fragile}, plate, bowl, steward_bot) or an authored '
        + `template. ${TEMPLATE_GUIDE} Nothing is published until asset_publish.`,
      parameters: {
        generator: { type: 'string', description: 'Generator name.' },
        params: { type: 'json', description: 'Generator parameters as a JSON object.' },
        template: { type: 'json', description: 'Authored entity template as a JSON object.' },
      },
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(args, exec) {
        return json(await world.createAsset({
          generator: args.generator,
          params: args.params === undefined ? undefined : asObject(args.params, 'params'),
          template: args.template === undefined ? undefined : asObject(args.template, 'template'),
        }, exec.signal))
      },
    }),
    defineTool({
      name: 'asset_preview',
      description: 'Render front and top preview images of a draft template and return the PNG file paths. Open the '
        + 'images with the file read tool to check the shape before publishing.',
      parameters: {
        template: { type: 'json', required: true, description: 'Entity template as a JSON object.' },
        stem: { type: 'string', description: 'File name stem for the images.' },
      },
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(args, exec) {
        return json({ images: await world.previewAsset(asObject(args.template, 'template'), args.stem ?? 'draft', exec.signal) })
      },
    }),
    defineTool({
      name: 'asset_publish',
      description: 'Validate and publish a template under a name. Publishing an identical template again keeps the '
        + 'same version.',
      parameters: {
        name: { type: 'string', required: true, description: 'Asset name, for example "tall-glass".' },
        template: { type: 'json', required: true, description: 'Entity template as a JSON object.' },
        meta: { type: 'json', description: 'Free-form metadata object, for example the description it came from.' },
      },
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(args, exec) {
        return json(await world.publishAsset(args.name, asObject(args.template, 'template'),
          args.meta === undefined ? undefined : asObject(args.meta, 'meta'), exec.signal))
      },
    }),
    defineTool({
      name: 'scene_compose',
      description: 'Compose a scene from published assets, check that nothing interpenetrates and that it settles, '
        + 'and save it. Spec: {name, entities: [{id: "w:<unique>", asset, at: {x, y, yaw}} or {id, asset, on: '
        + '"w:<support id placed earlier>", offset: [dx, dy], yaw}], regions: [{id, label, polygon: [[x, y], ...]}]}. '
        + 'Place robots with at; place objects on furniture with on.',
      parameters: { spec: { type: 'json', required: true, description: 'Scene specification object.' } },
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(args, exec) {
        return json(await world.composeScene(asObject(args.spec, 'spec'), exec.signal))
      },
    }),
    defineTool({
      name: 'scene_render',
      description: 'Render overview images of a saved scene and return the PNG paths for the file read tool.',
      parameters: { name: { type: 'string', required: true, description: 'Scene name.' } },
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(args, exec) {
        return json({ images: await world.renderScene(args.name, exec.signal) })
      },
    }),
    defineTool({
      name: 'scenario_save',
      description: 'Save a scenario: {name, scene, robots: ["w:<robot id>"], task: "<instruction for the robots>"}. '
        + 'The task must not reveal ground truth the robots should discover themselves.',
      parameters: { scenario: { type: 'json', required: true, description: 'Scenario object.' } },
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(args, exec) {
        return json(await world.saveScenario(asObject(args.scenario, 'scenario'), exec.signal))
      },
    }),
    defineTool({
      name: 'world_catalog',
      description: 'List saved scenes and scenarios.',
      parameters: {},
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(_args, exec) {
        return json(await world.catalog(exec.signal))
      },
    }),
    defineTool({
      name: 'episode_control',
      description: 'Load a saved scenario as the running episode (robots get fresh bodies and ports), or reset the '
        + 'current episode to its initial state.',
      parameters: {
        action: { type: 'string', enum: ['load', 'reset'], required: true, description: 'load or reset.' },
        scenario: { type: 'string', description: 'Scenario name for load.' },
      },
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(args, exec) {
        return json(await world.episode(args.action, args.scenario, exec.signal))
      },
    }),
    defineTool({
      name: 'world_clock',
      description: 'Pause or resume the simulation, or set its speed factor.',
      parameters: {
        action: { type: 'string', enum: ['pause', 'resume', 'speed'], required: true, description: 'Clock action.' },
        value: { type: 'number', description: 'Speed factor for speed, for example 2.' },
      },
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(args, exec) {
        return json(await world.clock(args.action, args.value, exec.signal))
      },
    }),
    defineTool({
      name: 'world_edit',
      description: 'Change the running World State without stopping the episode (robots keep working and must '
        + 'notice the change themselves). move: relocate an object or piece of furniture; objects resting on moved '
        + 'furniture move with it. spawn: add a new entity from a published asset (see asset_search). remove: delete '
        + 'an entity; objects resting on it fall. Place with x/y (floor position in metres, world frame) or with on '
        + '(support entity id, e.g. "w:table") plus offset_x/offset_y in the support frame. yaw in radians; move keeps '
        + 'the current yaw when omitted. Robots, the building and objects a robot is holding cannot be edited; an edit '
        + 'that would interpenetrate another entity is rejected and leaves the world unchanged. Read world_scene first '
        + 'for ids and positions.',
      parameters: {
        action: { type: 'string', enum: ['move', 'spawn', 'remove'], required: true, description: 'Edit kind.' },
        entity: { type: 'string', required: true, description: 'Entity id, e.g. "w:cup-1"; for spawn a new unique id.' },
        asset: { type: 'string', description: 'Published asset name for spawn, e.g. "cup".' },
        x: { type: 'number', description: 'Floor x position in metres.' },
        y: { type: 'number', description: 'Floor y position in metres.' },
        on: { type: 'string', description: 'Support entity id to place on, e.g. "w:sideboard".' },
        offset_x: { type: 'number', description: 'x offset on the support, in the support frame (default 0).' },
        offset_y: { type: 'number', description: 'y offset on the support, in the support frame (default 0).' },
        yaw: { type: 'number', description: 'Heading in radians.' },
      },
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(args, exec) {
        const request: WorldEditRequest = { action: args.action, entity: args.entity }
        if (args.asset !== undefined) request.asset = args.asset
        if (args.on !== undefined) {
          request.on = args.on
          request.offset = [args.offset_x ?? 0, args.offset_y ?? 0]
        } else if (args.x !== undefined || args.y !== undefined) {
          if (args.x === undefined || args.y === undefined) throw new Error('give both x and y, or on')
          request.at = { x: args.x, y: args.y }
        } else if (args.action !== 'remove') {
          throw new Error(`${args.action} needs a placement: x and y, or on`)
        }
        if (args.yaw !== undefined) request.yaw = args.yaw
        return json(await world.edit(request, exec.signal))
      },
    }),
    defineTool({
      name: 'observer_report',
      description: 'Read how accurate each robot\'s own world model is compared with the true World State: object '
        + 'recall, false detections, position error, uncertainty calibration (NEES) and self-localisation error.',
      parameters: {},
      output: { schema: JSON_OUTPUT, render: (_a, v) => text(v) },
      async execute(_args, exec) {
        return json(await world.observerReport(exec.signal))
      },
    }),
  ]
}

export function apply(ctx: Context): void {
  for (const tool of worldTools(ctx.world)) ctx.tools.register(tool)
}
