/**
 * JSON requests against Steward HTTP services.
 * @module @steward/http-json
 */

/** Non-2xx response, unreachable service or expired deadline. */
export class HttpJsonError extends Error {
  constructor(message: string, readonly status?: number) {
    super(message)
    this.name = 'HttpJsonError'
  }
}

/** Per-request options. */
export interface RequestOptions {
  /** Caller cancellation; aborting rejects the request. */
  signal?: AbortSignal
  /** Deadline in milliseconds for the whole request. */
  timeoutMs: number
}

/**
 * Validate and normalise a service base URL.
 * @param endpoint - configured base URL.
 * @returns the URL without a trailing slash.
 * @throws HttpJsonError when the value is not an absolute http(s) URL.
 */
export function baseUrl(endpoint: string): string {
  let url: URL
  try {
    url = new URL(endpoint)
  } catch {
    throw new HttpJsonError(`endpoint ${JSON.stringify(endpoint)} is not an absolute URL`)
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new HttpJsonError(`endpoint ${JSON.stringify(endpoint)} must use http or https`)
  }
  return url.toString().replace(/\/$/, '')
}

/**
 * Send one JSON request and parse the JSON response.
 * @param base - normalised base URL from {@link baseUrl}.
 * @param method - HTTP method.
 * @param path - path with optional query string.
 * @param body - JSON body for POST requests.
 * @param options - cancellation and deadline.
 * @returns the parsed response body.
 * @throws HttpJsonError carrying the service's error message and status.
 */
export async function requestJson<T>(base: string, method: 'GET' | 'POST', path: string, body: unknown,
  options: RequestOptions): Promise<T> {
  const deadline = AbortSignal.timeout(options.timeoutMs)
  const signal = options.signal ? AbortSignal.any([options.signal, deadline]) : deadline
  let response: Response
  try {
    response = await fetch(`${base}${path}`, {
      method,
      headers: { 'content-type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    })
  } catch (error) {
    if (options.signal?.aborted) throw error
    const reason = deadline.aborted ? `timed out after ${options.timeoutMs} ms` : (error as Error).message
    throw new HttpJsonError(`${method} ${base}${path} failed: ${reason}`)
  }
  const text = await response.text()
  let parsed: unknown = undefined
  if (text) {
    try {
      parsed = JSON.parse(text)
    } catch {
      throw new HttpJsonError(`${method} ${path} returned non-JSON content`, response.status)
    }
  }
  if (!response.ok) {
    const message = typeof parsed === 'object' && parsed !== null && 'error' in parsed
      ? String((parsed as { error: unknown }).error)
      : `HTTP ${response.status}`
    throw new HttpJsonError(message, response.status)
  }
  return parsed as T
}
