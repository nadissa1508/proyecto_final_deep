import { useState } from 'react'
import Landing from './Landing'
import Dashboard from './Dashboard'

export default function App() {
  const [view, setView] = useState('landing')

  if (view === 'dashboard') {
    return <Dashboard onBackToLanding={() => setView('landing')} />
  }
  return <Landing onOpenApp={() => setView('dashboard')} />
}
