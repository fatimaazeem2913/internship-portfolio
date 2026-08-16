/**
 * useInactivityTimer.js
 * -------------------------
 * Requirement #2: "remaining inactive for more than 60 seconds shall
 * terminate the current session, clear its conversation history, and
 * redirect the user to the home screen."
 *
 * This is the FRONTEND half of the 60-second expiry -- it fires
 * onTimeout after 60 real seconds with no user interaction. The BACKEND
 * independently enforces the same rule via session_store.py's sweep
 * (see main.py) -- don't rely on the client alone to enforce something
 * that matters, but a responsive client-side timer is what actually
 * gives the USER a redirect experience rather than a silently-broken session.
 */

import { useEffect, useRef, useCallback } from "react";

const INACTIVITY_TIMEOUT_MS = 60_000;

export function useInactivityTimer(onTimeout, isActive) {
  const timerRef = useRef(null);

  const resetTimer = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    if (isActive) {
      timerRef.current = setTimeout(onTimeout, INACTIVITY_TIMEOUT_MS);
    }
  }, [onTimeout, isActive]);

  useEffect(() => {
    if (!isActive) {
      if (timerRef.current) clearTimeout(timerRef.current);
      return;
    }

    resetTimer();

    const events = ["mousedown", "keydown", "touchstart", "scroll"];
    events.forEach((event) => window.addEventListener(event, resetTimer));

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      events.forEach((event) => window.removeEventListener(event, resetTimer));
    };
  }, [isActive, resetTimer]);
}
