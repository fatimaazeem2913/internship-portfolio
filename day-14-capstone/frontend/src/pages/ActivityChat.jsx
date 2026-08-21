import { useState, useEffect, useRef, useCallback } from 'react'
import { getActivityById } from '../activities.js'
import { startActivity, sendChatTurn, endSession } from '../api.js'
import { useInactivityTimer } from '../hooks/useInactivityTimer.js'

const ACTION_LABELS = {
  hint: '💡 Hint',
  give_up: '🏳️ Give Up',
}

export default function ActivityChat({ activityId, onBack }) {
  const activity = getActivityById(activityId)
  const [messages, setMessages] = useState([]) // [{ role: 'model'|'user', text }]
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const [showTimeoutWarning, setShowTimeoutWarning] = useState(false)
  const bottomRef = useRef(null)
  const startedRef = useRef(false)

  const { reset: resetInactivityTimer } = useInactivityTimer({
    timeoutSeconds: 60,
    warningSeconds: 10,
    onWarning: () => setShowTimeoutWarning(true),
    onTimeout: () => {
      // Requirement #2: 60s inactivity terminates the session, clears
      // history, and redirects to home.
      endSession(sessionId)
      onBack()
    },
  })

  const appendToLastMessage = useCallback((chunk) => {
    setMessages((prev) => {
      const updated = [...prev]
      updated[updated.length - 1] = {
        ...updated[updated.length - 1],
        text: updated[updated.length - 1].text + chunk,
      }
      return updated
    })
  }, [])

  // Trims the visible chat by whole (AI message, user reply) EXCHANGES,
  // exactly mirroring the backend's 6-exchange history cap -- never a
  // naive slice(-N), which can cut through the middle of an exchange and
  // leave an orphaned bubble on screen. Messages always start with a
  // model bubble (the opening riddle/question/greeting), then alternate
  // model, user, model, user, ... with at most one trailing unpaired
  // model bubble (the current pending message, awaiting a reply).
  const MAX_VISIBLE_EXCHANGES = 6

  const trimToExchanges = useCallback((allMessages, maxExchanges = MAX_VISIBLE_EXCHANGES) => {
    const pairs = []
    let i = 0
    while (i + 1 < allMessages.length) {
      pairs.push([allMessages[i], allMessages[i + 1]])
      i += 2
    }
    const trailing = allMessages.length % 2 === 1 ? [allMessages[allMessages.length - 1]] : []
    return pairs.slice(-maxExchanges).flat().concat(trailing)
  }, [])

  const pushMessage = useCallback((msg) => {
    setMessages((prev) => trimToExchanges([...prev, msg]))
  }, [trimToExchanges])

  const startNewModelMessage = useCallback(() => {
    pushMessage({ role: 'model', text: '' })
  }, [pushMessage])

  // Feedback + new riddle/question stay in ONE bubble, joined by a
  // paragraph break, instead of rendering as two separate bubbles.
  const insertParagraphBreak = useCallback(() => {
    appendToLastMessage('\n\n')
  }, [appendToLastMessage])

  useEffect(() => {
    if (startedRef.current) return
    startedRef.current = true

    setIsStreaming(true)
    startNewModelMessage()
    startActivity(activityId, {
      onChunk: appendToLastMessage,
      onDone: () => setIsStreaming(false),
    })
      .then(setSessionId)
      .catch((err) => {
        console.error(err)
        appendToLastMessage('Sorry, something went wrong starting this activity.')
        setIsStreaming(false)
      })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    return () => {
      endSession(sessionId)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const sendTurn = useCallback(
    async ({ message, action }) => {
      if (!sessionId || isStreaming) return
      resetInactivityTimer()
      setShowTimeoutWarning(false)

      // Clicking Hint/Give Up (no typed message) still shows a visible
      // action bubble, so it's clear the request went through.
      if (message) {
        pushMessage({ role: 'user', text: message })
      } else if (action && ACTION_LABELS[action]) {
        pushMessage({ role: 'user', text: ACTION_LABELS[action] })
      }

      setIsStreaming(true)
      startNewModelMessage()

      await sendChatTurn(
        { sessionId, activity: activityId, message, action },
        {
          onChunk: appendToLastMessage,
          onNewItem: insertParagraphBreak,
          onDone: () => setIsStreaming(false),
        }
      ).catch((err) => {
        console.error(err)
        appendToLastMessage('Sorry, something went wrong. Please try again.')
        setIsStreaming(false)
      })
    },
    [sessionId, isStreaming, activityId, appendToLastMessage, startNewModelMessage, insertParagraphBreak, resetInactivityTimer, pushMessage]
  )

  const handleSend = () => {
    const trimmed = input.trim()
    if (!trimmed) return
    setInput('')

    // If the child TYPES a hint/give-up request instead of clicking the
    // dedicated button, route it through the same real, live action.
    if (activityId === 'brain_buster') {
      const normalized = trimmed.toLowerCase().replace(/[.!?]+$/, '')
      const hintPhrases = ['hint', 'give hint', 'give me a hint', 'provide hint', 'i need a hint', 'can i have a hint', 'more hint', 'another hint']
      const giveUpPhrases = ['give up', 'i give up', 'skip', 'skip this one', 'i quit', "i don't know", 'idk']

      if (hintPhrases.includes(normalized)) {
        sendTurn({ message: trimmed, action: 'hint' })
        return
      }
      if (giveUpPhrases.includes(normalized)) {
        sendTurn({ message: trimmed, action: 'give_up' })
        return
      }
    }

    sendTurn({ message: trimmed })
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-brand-50 to-indigo-100 flex flex-col">
      <header className="bg-white/80 backdrop-blur px-4 py-3 flex items-center gap-3 shadow-sm sticky top-0 z-10">
        <button
          onClick={() => {
            endSession(sessionId)
            onBack()
          }}
          className="text-slate-600 hover:text-slate-900 font-medium"
        >
          ← Back
        </button>
        <span className="text-2xl">{activity?.emoji}</span>
        <h2 className="font-bold text-lg text-slate-800">{activity?.name}</h2>
      </header>

      {showTimeoutWarning && (
        <div className="bg-amber-100 text-amber-800 text-center py-2 text-sm font-medium">
          Still there? This session will end soon due to inactivity.
        </div>
      )}

      <main className="flex-1 overflow-y-auto p-4 space-y-3 max-w-2xl w-full mx-auto">
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div
              className={`rounded-2xl px-4 py-2.5 max-w-[80%] whitespace-pre-wrap ${
                m.role === 'user'
                  ? 'bg-brand-600 text-white rounded-br-sm'
                  : 'bg-white text-slate-800 shadow-sm rounded-bl-sm'
              }`}
            >
              {m.text || (isStreaming && i === messages.length - 1 ? '···' : '')}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </main>

      <footer className="bg-white/80 backdrop-blur p-4 sticky bottom-0">
        <div className="max-w-2xl mx-auto space-y-2">
          {activityId === 'brain_buster' && (
            <div className="flex gap-2 justify-center">
              <button
                onClick={() => sendTurn({ action: 'hint' })}
                disabled={isStreaming}
                className="px-4 py-1.5 rounded-full bg-amber-100 text-amber-800 font-medium text-sm hover:bg-amber-200 disabled:opacity-50"
              >
                💡 Hint
              </button>
              <button
                onClick={() => sendTurn({ action: 'give_up' })}
                disabled={isStreaming}
                className="px-4 py-1.5 rounded-full bg-slate-100 text-slate-600 font-medium text-sm hover:bg-slate-200 disabled:opacity-50"
              >
                🏳️ Give Up
              </button>
            </div>
          )}

          <div className="flex gap-2">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isStreaming}
              placeholder={
                activityId === 'ask_explore' ? 'Ask me anything...' : 'Type your answer...'
              }
              className="flex-1 border border-slate-300 rounded-full px-4 py-2 outline-none focus:ring-2 focus:ring-brand-400 disabled:opacity-50"
            />
            <button
              onClick={handleSend}
              disabled={isStreaming || !input.trim()}
              className="bg-brand-600 text-white rounded-full px-5 py-2 font-medium disabled:opacity-40"
            >
              Send
            </button>
          </div>
        </div>
      </footer>
    </div>
  )
}