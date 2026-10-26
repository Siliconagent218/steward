/**
 * Service Definition of the World capability, mounted as `ctx.world` in the World Harness.
 *
 * The World capability reads and controls the Steward plane: World State (ground truth),
 * episodes and the simulation clock, Observer reports, and asset, scene and scenario authoring.
 * It must never be mounted in an Ego Harness.
 * @module @steward/dsh-world
 */

import { Service, type Context } from '@deepseek-ai/cordis'

declare module '@deepseek-ai/cordis' {
  interface Context {
    world: WorldService
  }
}

/** JSON object as returned by the Steward host. */
export interface JsonObject {
  [key: string]: unknown
}

/** Failure reported by a World provider, carrying the host status when one exists. */
export class WorldError extends Error {
  constructor(message: string, readonly status?: number) {
    super(message)
    this.name = 'WorldError'
  }
}

/** Asset creation input: exactly one of a generator (with params) or an authored entity template. */
export interface AssetCreateRequest {
  generator?: string
  params?: JsonObject
  template?: JsonObject
}

/** The `world` service. Every method accepts the caller's cancellation signal. */
export abstract class WorldService extends Service {
  constructor(ctx: Context) {
    super(ctx, 'world')
  }

  /**
   * Loaded scenario, simulation time, entities, robots and recent events.
   * @param signal - caller cancellation.
   * @returns the status record.
   */
  abstract status(signal?: AbortSignal): Promise<JsonObject>

  /**
   * The current World State as a SceneState.
   * @param signal - caller cancellation.
   * @returns the World State.
   */
  abstract scene(signal?: AbortSignal): Promise<JsonObject>

  /**
   * Load a scenario or reset the current episode.
   * @param action - `load` or `reset`.
   * @param scenario - scenario name for `load`.
   * @param signal - caller cancellation.
   * @returns the episode result, including robot ports for `load`.
   */
  abstract episode(action: 'load' | 'reset', scenario: string | undefined, signal?: AbortSignal): Promise<JsonObject>

  /**
   * Pause, resume or change the simulation speed.
   * @param action - `pause`, `resume` or `speed`.
   * @param value - speed factor for `speed`.
   * @param signal - caller cancellation.
   * @returns the clock state.
   */
  abstract clock(action: 'pause' | 'resume' | 'speed', value: number | undefined, signal?: AbortSignal): Promise<JsonObject>

  /**
   * Observer evaluation of every robot's Ego-State against the World State.
   * @param signal - caller cancellation.
   * @returns per-robot report.
   */
  abstract observerReport(signal?: AbortSignal): Promise<JsonObject>

  /**
   * Names of the procedural asset generators.
   * @param signal - caller cancellation.
   * @returns generator names.
   */
  abstract generators(signal?: AbortSignal): Promise<string[]>

  /**
   * Search the asset library.
   * @param query - substring of the asset name or category.
   * @param category - exact category.
   * @param signal - caller cancellation.
   * @returns matching assets with their latest versions.
   */
  abstract searchAssets(query: string | undefined, category: string | undefined, signal?: AbortSignal): Promise<JsonObject[]>

  /**
   * Read one published asset record, including its template.
   * @param assetName - asset name.
   * @param signal - caller cancellation.
   * @returns the asset record.
   */
  abstract getAsset(assetName: string, signal?: AbortSignal): Promise<JsonObject>

  /**
   * Build and validate a draft template without publishing it.
   * @param request - generator with params, or an authored template.
   * @param signal - caller cancellation.
   * @returns the template and its validation report.
   */
  abstract createAsset(request: AssetCreateRequest, signal?: AbortSignal): Promise<JsonObject>

  /**
   * Render preview images of a draft template.
   * @param template - entity template.
   * @param stem - file name stem.
   * @param signal - caller cancellation.
   * @returns image file paths.
   */
  abstract previewAsset(template: JsonObject, stem: string, signal?: AbortSignal): Promise<string[]>

  /**
   * Validate and publish a template to the asset library.
   * @param assetName - asset name.
   * @param template - entity template.
   * @param meta - free-form metadata stored with the version.
   * @param signal - caller cancellation.
   * @returns the published version, validation report and preview paths.
   */
  abstract publishAsset(assetName: string, template: JsonObject, meta: JsonObject | undefined, signal?: AbortSignal): Promise<JsonObject>

  /**
   * Compose, validate and save a scene from published assets.
   * @param spec - scene specification (name, entities with placements, regions).
   * @param signal - caller cancellation.
   * @returns the saved scene and its physical check report.
   */
  abstract composeScene(spec: JsonObject, signal?: AbortSignal): Promise<JsonObject>

  /**
   * Render overview images of a saved scene.
   * @param sceneName - scene name.
   * @param signal - caller cancellation.
   * @returns image file paths.
   */
  abstract renderScene(sceneName: string, signal?: AbortSignal): Promise<string[]>

  /**
   * Save a scenario (scene, robots, task, perturbation policy).
   * @param scenario - scenario record.
   * @param signal - caller cancellation.
   * @returns the saved scenario path.
   */
  abstract saveScenario(scenario: JsonObject, signal?: AbortSignal): Promise<JsonObject>

  /**
   * Saved scene and scenario names.
   * @param signal - caller cancellation.
   * @returns both name lists.
   */
  abstract catalog(signal?: AbortSignal): Promise<{ scenes: string[]; scenarios: string[] }>
}
