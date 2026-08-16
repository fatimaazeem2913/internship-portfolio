/**
 * capstoneApi.js
 * -----------------
 * Wraps every backend call behind clean functions, isolating fetch() and
 * SSE-parsing details from the components.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function consumeSSE(response, { onChunk, onNewItem, onDone, onError }) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const rawEvent = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);

      const eventTypeMatch = rawEvent.match(/^event: (.+)$/m);
      const dataMatch = rawEvent.match(/^data: (.+)$/m);
      if (!eventTypeMatch || !dataMatch) continue;

      const eventType = eventTypeMatch[1];
      const data = JSON.parse(dataMatch[1]);

      if (eventType === "chunk") onChunk?.(data.text);
      else if (eventType === "new_item") onNewItem?.();
      else if (eventType === "done") onDone?.(data);
      else if (eventType === "error") onError?.(data.message);
    }
  }
}

export async function startSession(activity, callbacks) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}/api/session/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ activity }),
    });
  } catch (e) {
    throw new ApiError("Could not reach the server.", 0);
  }

  if (!response.ok) {
    throw new ApiError(`Failed to start session (status ${response.status})`, response.status);
  }

  const sessionId = response.headers.get("x-session-id");
  await consumeSSE(response, callbacks);
  return sessionId;
}

export async function sendTurn(sessionId, message, action, callbacks) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message, action }),
    });
  } catch (e) {
    throw new ApiError("Could not reach the server.", 0);
  }

  if (!response.ok) {
    if (response.status === 404) {
      throw new ApiError("This session has ended. Returning to the home screen.", 404);
    }
    throw new ApiError(`Request failed (status ${response.status})`, response.status);
  }

  await consumeSSE(response, callbacks);
}

export async function endSession(sessionId) {
  if (!sessionId) return;
  try {
    await fetch(`${API_BASE_URL}/api/session/${sessionId}`, { method: "DELETE" });
  } catch (e) {
    console.warn("Failed to explicitly end session:", e);
  }
}
