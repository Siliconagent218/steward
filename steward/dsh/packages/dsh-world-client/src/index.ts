/**
 * Service Provider of `ctx.world` that calls the Steward World Control API.
 * @module @steward/dsh-world-client
 */

import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import { WorldError, WorldService } from '@steward/dsh-world'
import type { AssetCreateRequest, JsonObject } from '@steward/dsh-world'
import { baseUrl, HttpJsonError, requestJson } from '@steward/http-json'

/** Validated plugin configuration. */
export interface Config {
  /** Base URL of the World Control API. */
  endpoint: string
  /** Deadline for ordinary requests in milliseconds. */
  timeoutMs: number
  /** Deadline for validation, composition and rendering requests in milliseconds. */
  slowTimeoutMs: number
}

/** `ctx.world` provider backed by HTTP calls to the World Control API. */
export class WorldApiClient extends WorldService {
  static Config = z.object({
    endpoint: z.string().default('http://127.0.0.1:7400'),
    timeoutMs: z.number().min(100).default(15_000),
    slowTimeoutMs: z.number().min(1000).default(120_000),
  })

  private readonly base: string

  constructor(ctx: Context, readonly config: Config) {
    super(ctx)
    this.base = baseUrl(config.endpoint)
  }

  private async call<T>(method: 'GET' | 'POST', path: string, body: unknown, signal: AbortSignal | undefined,
    slow = false): Promise<T> {
    try {
      const timeoutMs = slow ? this.config.slowTimeoutMs : this.config.timeoutMs
      return await requestJson<T>(this.base, method, path, body, { signal, timeoutMs })
    } catch (error) {
      if (error instanceof HttpJsonError) throw new WorldError(error.message, error.status)
      throw error
    }
  }

  status(signal?: AbortSignal): Promise<JsonObject> {
    return this.call('GET', '/wca/status', undefined, signal)
  }

  scene(signal?: AbortSignal): Promise<JsonObject> {
    return this.call('GET', '/wca/scene', undefined, signal)
  }

  episode(action: 'load' | 'reset', scenario: string | undefined, signal?: AbortSignal): Promise<JsonObject> {
    return this.call('POST', '/wca/episode', { action, scenario }, signal, true)
  }

  clock(action: 'pause' | 'resume' | 'speed', value: number | undefined, signal?: AbortSignal): Promise<JsonObject> {
    return this.call('POST', '/wca/clock', { action, value }, signal)
  }

  observerReport(signal?: AbortSignal): Promise<JsonObject> {
    return this.call('GET', '/observer/report', undefined, signal)
  }

  async generators(signal?: AbortSignal): Promise<string[]> {
    return (await this.call<{ generators: string[] }>('GET', '/assets/generators', undefined, signal)).generators
  }

  async searchAssets(query: string | undefined, category: string | undefined, signal?: AbortSignal): Promise<JsonObject[]> {
    const params = new URLSearchParams()
    if (query) params.set('query', query)
    if (category) params.set('category', category)
    return (await this.call<{ assets: JsonObject[] }>('GET', `/assets?${params}`, undefined, signal)).assets
  }

  getAsset(assetName: string, signal?: AbortSignal): Promise<JsonObject> {
    return this.call('GET', `/assets/get?name=${encodeURIComponent(assetName)}`, undefined, signal)
  }

  createAsset(request: AssetCreateRequest, signal?: AbortSignal): Promise<JsonObject> {
    return this.call('POST', '/assets/create', request, signal, true)
  }

  async previewAsset(template: JsonObject, stem: string, signal?: AbortSignal): Promise<string[]> {
    return (await this.call<{ images: string[] }>('POST', '/assets/preview', { template, stem }, signal, true)).images
  }

  publishAsset(assetName: string, template: JsonObject, meta: JsonObject | undefined, signal?: AbortSignal): Promise<JsonObject> {
    return this.call('POST', '/assets/publish', { name: assetName, template, meta }, signal, true)
  }

  composeScene(spec: JsonObject, signal?: AbortSignal): Promise<JsonObject> {
    return this.call('POST', '/scenes/compose', spec, signal, true)
  }

  async renderScene(sceneName: string, signal?: AbortSignal): Promise<string[]> {
    return (await this.call<{ images: string[] }>('POST', '/scenes/render', { name: sceneName }, signal, true)).images
  }

  saveScenario(scenario: JsonObject, signal?: AbortSignal): Promise<JsonObject> {
    return this.call('POST', '/scenarios/save', scenario, signal)
  }

  async catalog(signal?: AbortSignal): Promise<{ scenes: string[]; scenarios: string[] }> {
    const [scenes, scenarios] = await Promise.all([
      this.call<{ scenes: string[] }>('GET', '/scenes', undefined, signal),
      this.call<{ scenarios: string[] }>('GET', '/scenarios', undefined, signal),
    ])
    return { scenes: scenes.scenes, scenarios: scenarios.scenarios }
  }
}

export default WorldApiClient
