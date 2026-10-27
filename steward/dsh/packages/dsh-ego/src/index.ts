/**
 * Service Definition of the robot Ego capability, mounted as `ctx.ego`.
 *
 * The Ego capability is the Ego Harness view of one robot: the Ego-State (the robot's own,
 * partial and uncertain reconstruction of the world), ESim predictions that return Prediction
 * Receipts, and execution of exactly the plan a receipt predicted. Providers talk to the
 * robot's Ego Runtime; no provider may expose World State.
 * @module @steward/dsh-ego
 */

import { Service, type Context } from '@deepseek-ai/cordis'

declare module '@deepseek-ai/cordis' {
  interface Context {
    ego: EgoService
  }
}

/** JSON object as returned by the Ego Runtime. */
export interface JsonObject {
  [key: string]: unknown
}

/** Skills the Ego Runtime can predict and run. */
export type EgoSkill = 'navigate' | 'pick' | 'place' | 'open' | 'close'

/** Filter for Ego-State object queries. */
export interface EgoQuery {
  /** Category to keep, for example `cup`. */
  category?: string
  /** Ego entity id whose neighbourhood to search. */
  near?: string
  /** Neighbourhood radius in metres; the runtime default applies when absent. */
  radius?: number
}

/** ESim result that authorises running one predicted plan once before it expires. */
export interface PredictionReceipt {
  id: string
  skill: EgoSkill
  params: JsonObject
  /** Risk score in [0, 1] derived from Ego-State uncertainty and geometry. */
  risk: number
  feasible: boolean
  /** True when feasible and the risk does not exceed the runtime limit. */
  withinRiskLimit: boolean
  maxRisk: number
  issues: string[]
  /** Robot clock time after which the receipt is rejected. */
  expiresAt: number
  plan: JsonObject
  used?: boolean
}

/** One skill execution. */
export interface SkillJob {
  id: string
  skill: string
  receipt: string
  state: 'running' | 'succeeded' | 'failed' | 'stopped'
  message: string
  log: string[]
}

/** Failure reported by an Ego provider, carrying the runtime status when one exists. */
export class EgoError extends Error {
  constructor(message: string, readonly status?: number) {
    super(message)
    this.name = 'EgoError'
  }
}

/**
 * The `ego` service. Every method accepts the caller's cancellation signal; providers must
 * stop waiting when it aborts.
 */
export abstract class EgoService extends Service {
  constructor(ctx: Context) {
    super(ctx, 'ego')
  }

  /**
   * Robot pose, held object, current skill and the objects in the Ego-State.
   * @param signal - caller cancellation.
   * @returns the compact Ego-State summary.
   */
  abstract summary(signal?: AbortSignal): Promise<JsonObject>

  /**
   * Objects in the Ego-State matching a filter.
   * @param query - category and neighbourhood filter.
   * @param signal - caller cancellation.
   * @returns matching objects with position, uncertainty and support.
   */
  abstract query(query: EgoQuery, signal?: AbortSignal): Promise<JsonObject[]>

  /**
   * What the robot camera detects right now.
   * @param signal - caller cancellation.
   * @returns visible categories and ranges, plus a saved camera image path when available.
   */
  abstract observe(signal?: AbortSignal): Promise<JsonObject>

  /**
   * Start looking around in place to extend the Ego-State.
   * @param signal - caller cancellation.
   * @returns the started job.
   */
  abstract explore(signal?: AbortSignal): Promise<SkillJob>

  /**
   * Ask ESim to predict a skill and obtain a receipt.
   * @param skill - skill to predict.
   * @param params - skill parameters (target, support, goal pose).
   * @param signal - caller cancellation.
   * @returns the Prediction Receipt.
   */
  abstract predict(skill: EgoSkill, params: JsonObject, signal?: AbortSignal): Promise<PredictionReceipt>

  /**
   * Read an issued receipt.
   * @param id - receipt id.
   * @param signal - caller cancellation.
   * @returns the receipt as the runtime stores it.
   */
  abstract receipt(id: string, signal?: AbortSignal): Promise<PredictionReceipt>

  /**
   * Execute the plan of a receipt. The runtime re-checks validity, expiry and risk.
   * @param receiptId - receipt to execute.
   * @param signal - caller cancellation.
   * @returns the started job.
   */
  abstract run(receiptId: string, signal?: AbortSignal): Promise<SkillJob>

  /**
   * Read a job.
   * @param jobId - job id.
   * @param signal - caller cancellation.
   * @returns the job record.
   */
  abstract job(jobId: string, signal?: AbortSignal): Promise<SkillJob>

  /**
   * Wait until a job leaves the running state or the timeout passes.
   * @param jobId - job id.
   * @param timeoutS - maximum wait in seconds.
   * @param signal - caller cancellation.
   * @returns the job record at return time.
   */
  abstract wait(jobId: string, timeoutS: number, signal?: AbortSignal): Promise<SkillJob>

  /**
   * Stop all motion and hold the current pose.
   * @param signal - caller cancellation.
   * @returns the id of the interrupted job, if any.
   */
  abstract stop(signal?: AbortSignal): Promise<JsonObject>
}
