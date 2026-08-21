const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8001'
const STREAM_TIMEOUT_MS = 30000 // 30s -- generous for a real Gemini call, but never infinite

/**
 * Consumes a real Server-Sent Events stream from the backend, calling
 * onChunk(text) for every text chunk, onNewItem() whenever the backend
 * signals a visually-distinct new block (e.g. a fresh riddle after a
 * correct guess), and resolving once the stream sends {done: true}.
 *
 * Includes a hard 30s timeout as a last-resort safety net so the UI can
 * never get stuck indefinitely, even in the event of an unexpected
 * network hang.
 */
async function consumeSSEStream(response, { onChunk, onNewItem, onDone }) {
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let timedOut = false

  const timeoutId = setTimeout(() => {
    timedOut = true
    reader.cancel().catch(() => {})
  }, STREAM_TIMEOUT_MS)

  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) {
        if (timedOut) {
          onChunk?.(' (This is taking longer than expected -- please try again.)')
          onDone?.({ done: true, timedOut: true })
        }
        break
      }
      buffer += decoder.decode(value, { stream: true })

      const lines = buffer.split('\n\n')
      buffer = lines.pop()

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const payload = JSON.parse(line.slice('data: '.length))

        if (payload.new_item) {
          onNewItem?.()
        } else if (payload.done) {
          onDone?.(payload)
        } else if (typeof payload.chunk === 'string') {
          onChunk?.(payload.chunk)
        }
      }
    }
  } finally {
    clearTimeout(timeoutId)
  }
}

export async function startActivity(activity, callbacks) {
  const response = await fetch(`${API_BASE}/api/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ activity }),
  })
  if (!response.ok) {
    throw new Error(`Failed to start activity: ${response.status}`)
  }
  const sessionId = response.headers.get('X-Session-Id')
  await consumeSSEStream(response, callbacks)
  return sessionId
}

export async function sendChatTurn({ sessionId, activity, message, action }, callbacks) {
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, activity, message, action }),
  })
  if (!response.ok) {
    throw new Error(`Chat request failed: ${response.status}`)
  }
  await consumeSSEStream(response, callbacks)
}

export async function endSession(sessionId) {
  if (!sessionId) return
  try {
    await fetch(`${API_BASE}/api/session/${sessionId}`, { method: 'DELETE' })
  } catch {
    // best-effort cleanup -- the server's own 60s inactivity sweep is
    // the real backstop if this call fails (e.g. tab closed abruptly)
  }
}

export async function checkHealth() {
  const response = await fetch(`${API_BASE}/api/health`)
  return response.json()
}
