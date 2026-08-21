import { useState } from 'react'
import HomeScreen from './pages/Home.jsx'
import ActivityChat from './pages/ActivityChat.jsx'

export default function App() {
  const [activeActivity, setActiveActivity] = useState(null)

  if (activeActivity) {
    return <ActivityChat activityId={activeActivity} onBack={() => setActiveActivity(null)} />
  }

  return <HomeScreen onSelectActivity={setActiveActivity} />
}
