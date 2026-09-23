import { useState } from 'react'
import './App.css'
import AskQuestion from './components/AskQuestion'
import AuditAnswer from './components/AuditAnswer'
import ReportViewer from './components/ReportViewer'

function App() {
  const [activeTab, setActiveTab] = useState('ask') // 'ask' or 'audit'
  const [report, setReport] = useState(null)

  return (
    <div className="app-container">
      <header className="header animate-in">
        <h1 className="gradient-text">RAG Answer Verifier</h1>
        <p>
          Given a query and an LLM-generated answer, this system checks each factual claim against the retrieved source documents and flags anything unsupported or contradicted — <strong>before you act on it.</strong>
        </p>
      </header>

      <main>
        <div className="tabs-container animate-in" style={{ animationDelay: '0.1s' }}>
          <div className="tabs-list">
            <button 
              className={`tab-button ${activeTab === 'ask' ? 'active' : ''}`}
              onClick={() => { setActiveTab('ask'); setReport(null); }}
            >
              🔍 Ask a Question
            </button>
            <button 
              className={`tab-button ${activeTab === 'audit' ? 'active' : ''}`}
              onClick={() => { setActiveTab('audit'); setReport(null); }}
            >
              🛡️ Audit an Existing Answer
            </button>
          </div>

          <div className="tab-content glass-panel" style={{ padding: '2rem' }}>
            {activeTab === 'ask' ? (
              <AskQuestion onResult={setReport} />
            ) : (
              <AuditAnswer onResult={setReport} />
            )}
          </div>
        </div>

        {report && (
          <div className="animate-in" style={{ animationDelay: '0.2s' }}>
            <ReportViewer report={report} />
          </div>
        )}
      </main>
    </div>
  )
}

export default App
