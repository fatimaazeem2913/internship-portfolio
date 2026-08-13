/**
 * sessionStorage.js
 * ---------------------
 * Persists ALL sessions (their messages, titles, and metadata) to the
 * browser's localStorage, so a full page refresh (or closing and
 * reopening the browser) doesn't lose any conversation -- Day 13's
 * "chats survive a full page refresh" requirement.
 *
 * WHY THIS IS A SEPARATE MODULE, NOT INLINE IN App.jsx:
 * Same separation-of-concerns principle as chatApi.js -- isolating
 * localStorage's specific, slightly awkward API (it only stores strings,
 * so every read/write needs JSON.stringify/parse) behind clean functions
 * means App.jsx's logic never has to think about serialization at all,
 * and this module can be tested/reasoned about independently.
 *
 * WHAT localStorage ACTUALLY IS, AND ITS REAL LIMITS:
 * A simple browser-provided key-value store, scoped to one specific
 * website's origin, that persists across page reloads and browser
 * restarts (unlike React's in-memory useState, which resets on every
 * refresh). It has a real, hard size limit (typically ~5-10MB depending
 * on the browser) -- fine for text conversations, but NOT a substitute
 * for a real backend database for large-scale or multi-device use. This
 * is the frontend's own version of the same honest limitation documented
 * for the backend's in-memory session store in Day 11/13: convenient for
 * a single browser on a single device, not a real persistence layer.
 */

const STORAGE_KEY = "multiSessionChat.sessions";

/**
 * Loads all persisted sessions from localStorage.
 * @returns {Array<{id: string, title: string|null, messages: Array, createdAt: string}>}
 */
export function loadSessions() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch (e) {
    console.warn("Failed to load sessions from localStorage -- starting fresh.", e);
    return [];
  }
}

/**
 * Saves the full sessions array to localStorage, overwriting whatever
 * was there before. Called after every state change that should persist
 * (new message, new session, title update, etc.).
 */
export function saveSessions(sessions) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
  } catch (e) {
    console.warn("Failed to save sessions to localStorage (quota exceeded?).", e);
  }
}

export function clearAllSessions() {
  localStorage.removeItem(STORAGE_KEY);
}
