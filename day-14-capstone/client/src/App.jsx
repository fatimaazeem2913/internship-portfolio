/**
 * App.jsx
 * ----------
 * Top-level screen routing: home <-> one of three activity chat screens
 * (requirement #1). Deliberately simple useState-based routing rather
 * than a full router library -- with only 4 possible screens and no
 * deep-linking requirement, react-router would be genuine overkill.
 */

import { useState } from "react";
import HomeScreen from "./components/HomeScreen";
import ActivityChat from "./components/ActivityChat";

export default function App() {
  const [activeActivity, setActiveActivity] = useState(null);

  function handleSelectActivity(activityId) {
    setActiveActivity(activityId);
  }

  function handleBackToHome() {
    setActiveActivity(null);
  }

  if (activeActivity) {
    // key={activeActivity} forces a full remount when switching activities
    // directly, so each activity always starts with fully fresh component
    // state -- no leftover messages/timers from a previous activity could
    // ever bleed into a new one.
    return <ActivityChat key={activeActivity} activity={activeActivity} onBack={handleBackToHome} />;
  }

  return <HomeScreen onSelectActivity={handleSelectActivity} />;
}
