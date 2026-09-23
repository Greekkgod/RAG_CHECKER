import { useState } from 'react';

const API_BASE = "https://rag-verifier.onrender.com";

export default function AskQuestion({ onResult }) {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const sampleQueries = [
    "How long are deleted files kept in Trash?",
    "How many days of paid leave per year?",
    "What's the storage limit on the Free plan?"
  ];

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query) return;

    setLoading(true);
    setError('');
    onResult(null);

    try {
      const response = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_k: 4 })
      });

      if (!response.ok) {
        throw new Error(`Backend error: ${response.statusText}`);
      }

      const data = await response.json();
      onResult(data);
    } catch (err) {
      setError("Can't reach the backend. Ensure it is running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="form-section">
      <div>
        <h3 style={{ marginBottom: '0.5rem' }}>Test the full RAG pipeline</h3>
        <p className="caption">Corpus: sample company policy + product FAQ documents.</p>
      </div>

      <div className="form-group">
        <label>Try a sample question:</label>
        <div className="sample-queries">
          {sampleQueries.map((q, idx) => (
            <button key={idx} type="button" onClick={() => setQuery(q)}>
              {idx === 0 ? "Deleted files?" : idx === 1 ? "Paid leave?" : "Free plan limit?"}
            </button>
          ))}
        </div>
      </div>

      <form onSubmit={handleSubmit} className="form-group">
        <label htmlFor="query">Your question</label>
        <input 
          id="query"
          type="text" 
          className="input-field" 
          placeholder="e.g. How many days of paid leave do employees get per year?" 
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={loading}
        />
        
        {error && <div className="error-message">{error}</div>}

        <button 
          type="submit" 
          className="btn-primary" 
          style={{ marginTop: '1rem' }}
          disabled={!query || loading}
        >
          {loading ? 'Retrieving & Verifying...' : 'Generate & Verify'}
        </button>
      </form>

      {loading && (
        <div className="spinner-container">
          <div className="spinner"></div>
          <p style={{ color: 'var(--muted-foreground)' }}>Retrieving context, generating answer, verifying claims...</p>
        </div>
      )}
    </div>
  );
}
