import { useState } from 'react';

const API_BASE = "https://rag-verifier.onrender.com";

export default function AuditAnswer({ onResult }) {
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const loadExample = () => {
    setQuery("Can I use SMS-based 2FA on the Free plan, what is my storage limit, and what happens if I downgrade my account?");
    setAnswer("SMS-based 2FA is fully supported on all CloudSync plans, including the Free plan. On the Free plan, your total storage limit is 5 GB, and your individual file upload limit is also 5 GB. If you downgrade your account and exceed your new storage limit, your account goes into read-only mode and any data over the limit will be automatically deleted after 30 days.");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query || !answer) return;

    setLoading(true);
    setError('');
    onResult(null);

    try {
      const response = await fetch(`${API_BASE}/verify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, answer, top_k: 4 })
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
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h3 style={{ marginBottom: '0.5rem' }}>Audit any answer against the corpus</h3>
          <p className="caption">Paste an answer from anywhere to verify its claims.</p>
        </div>
        <button type="button" onClick={loadExample} className="btn-secondary">
          ✨ Load Example
        </button>
      </div>

      <form onSubmit={handleSubmit} className="form-section">
        <div className="form-group">
          <label htmlFor="audit-query">Original question</label>
          <input 
            id="audit-query"
            type="text" 
            className="input-field" 
            placeholder="e.g. What's the file upload limit?" 
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={loading}
          />
        </div>

        <div className="form-group">
          <label htmlFor="audit-answer">Answer to audit</label>
          <textarea 
            id="audit-answer"
            className="input-field" 
            placeholder="e.g. The maximum file upload size is 5 GB..." 
            rows={5}
            value={answer}
            onChange={(e) => setAnswer(e.target.value)}
            disabled={loading}
          />
        </div>
        
        {error && <div className="error-message">{error}</div>}

        <button 
          type="submit" 
          className="btn-primary" 
          disabled={!query || !answer || loading}
        >
          {loading ? 'Verifying claims...' : 'Verify this answer'}
        </button>
      </form>

      {loading && (
        <div className="spinner-container">
          <div className="spinner"></div>
          <p style={{ color: 'var(--muted-foreground)' }}>Verifying claims against corpus...</p>
        </div>
      )}
    </div>
  );
}
