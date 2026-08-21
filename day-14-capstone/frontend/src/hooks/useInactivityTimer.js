import { useEffect, useRef, useCallback } from 'react'

/**
 * Client-side inactivity timer, matching the backend's 60-second
 * server-side session expiry (requirement #2). This is a UX convenience
 * only -- the real enforcement lives server-side (a closed tab can't run
 * this timer, so the server never relies on it alone). Calls onWarning()
 * a few seconds before expiry, and onTimeout() when the client-side
 * timer runs out.
 */
export function useInactivityTimer({ timeoutSeconds = 60, warningSeconds = 10, onWarning, onTimeout }) {
  const timeoutRef = useRef(null)
  const warningRef = useRef(null)

  const reset = useCallback(() => {
    clearTimeout(timeoutRef.current)
    clearTimeout(warningRef.current)

    warningRef.current = setTimeout(() => {
      onWarning?.()
    }, (timeoutSeconds - warningSeconds) * 1000)

    timeoutRef.current = setTimeout(() => {
      onTimeout?.()
    }, timeoutSeconds * 1000)
  }, [timeoutSeconds, warningSeconds, onWarning, onTimeout])

  useEffect(() => {
    reset()
    return () => {
      clearTimeout(timeoutRef.current)
      clearTimeout(warningRef.current)
    }
  }, [reset])

  return { reset }
}
