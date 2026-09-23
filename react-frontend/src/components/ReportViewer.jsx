import { useState } from 'react';

const BADGE_STYLE = {
  green: { emoji: "🟢", color: "#1a7f37", desc: "High confidence — claims are well-supported by sources.", bgClass: "verdict-supported" },
  yellow: { emoji: "🟡", color: "#9a6700", desc: "Mixed confidence — some claims lack strong support.", bgClass: "verdict-unverifiable" },
  red: { emoji: "🔴", color: "#cf222e", desc: "Low confidence — likely hallucinated or contradicted claims.", bgClass: "verdict-contradicted" },
};

function ClaimItem({ claim, index }) {
  const [isOpen, setIsOpen] = useState(claim.verdict !== 'supported');
  
  const vClass = `verdict-${claim.verdict}`;
  const verdictEmoji = claim.verdict === 'supported' ? '✅' : claim.verdict === 'contradicted' ? '❌' : '❓';

  return (
    <div className={`claim-item ${isOpen ? 'open' : ''}`}>
      <div className="claim-header" onClick={() => setIsOpen(!isOpen)}>
        <div className="claim-icon">{verdictEmoji}</div>
        <div className="claim-title">
          <span style={{opacity: 0.7, marginRight: 8, fontSize: '0.9em'}}>Claim {index}:</span>
          {claim.claim}
        </div>
        <div className={`claim-toggle ${isOpen ? 'open' : ''}`}>▼</div>
      </div>
      
      <div className="claim-details" style={{ display: isOpen ? 'block' : 'none' }}>
        <div className="claim-meta">
          <div className="meta-item">
            <span className="meta-label">Verdict:</span>
            <span className={`verdict-tag ${vClass}`}>{claim.verdict}</span>
          </div>
          <div className="meta-item">
            <span className="meta-label">Method:</span>
            <code style={{background: 'var(--muted)', padding: '2px 6px', borderRadius: 4}}>{claim.method}</code>
          </div>
          <div className="meta-item">
            <span className="meta-label">Confidence:</span>
            <strong>{claim.confidence.toFixed(2)}</strong>
          </div>
        </div>

        {claim.reasoning && (
          <div className="reasoning-box">
            <div style={{fontSize: '1.2rem'}}>🤖</div>
            <div>
              <strong>Judge Reasoning:</strong>
              <div style={{marginTop: 4}}>{claim.reasoning}</div>
            </div>
          </div>
        )}

        <div className="evidence-list">
          {claim.evidence && claim.evidence.length > 0 ? (
            <>
              <h4>📄 Matched Evidence</h4>
              {claim.evidence.map((ev, i) => (
                <div key={i} className="evidence-item">{ev}</div>
              ))}
            </>
          ) : (
            <div style={{color: 'var(--muted-foreground)', fontSize: '0.9rem', fontStyle: 'italic'}}>
              No evidence was matched for this claim.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function ReportViewer({ report }) {
  if (!report) return null;

  const style = BADGE_STYLE[report.badge] || BADGE_STYLE.yellow;

  return (
    <div className="report-viewer">
      <div className="report-header">
        <div className="badge-card" style={{
          background: `color-mix(in srgb, ${style.color} 15%, transparent)`,
          border: `1px solid color-mix(in srgb, ${style.color} 40%, transparent)`
        }}>
          <h3 style={{color: style.color}}>
            <span style={{fontSize: '1.2em'}}>{style.emoji}</span> {report.badge.toUpperCase()}
          </h3>
          <p style={{color: 'var(--foreground)'}}>{style.desc}</p>
        </div>

        <div className="metric-card">
          <div className="metric-label">Faithfulness Score</div>
          <div className="metric-value">{(report.faithfulness_score * 100).toFixed(0)}%</div>
          <div className="progress-bar-container">
            <div className="progress-bar" style={{
              width: `${report.faithfulness_score * 100}%`,
              background: style.color
            }}></div>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Latency</div>
          <div className="metric-value">{report.latency_seconds}s</div>
        </div>
      </div>

      <div className="answer-section">
        <h3 style={{marginBottom: '1rem'}}>📝 Answer Analyzed</h3>
        <div className="answer-box">
          {report.answer}
        </div>
      </div>

      <div className="claims-section">
        <h3>🔍 Claim-by-Claim Breakdown</h3>
        
        {!report.claims || report.claims.length === 0 ? (
          <div className="reasoning-box" style={{background: 'rgba(154, 103, 0, 0.1)'}}>
            No checkable factual claims were found in this answer.
          </div>
        ) : (
          <div className="claims-list">
            {report.claims.map((claim, idx) => (
              <ClaimItem key={idx} claim={claim} index={idx + 1} />
            ))}
          </div>
        )}
      </div>

      {report.retrieved_sources && report.retrieved_sources.length > 0 && (
        <div className="sources-list">
          <strong>Sources retrieved:</strong> {report.retrieved_sources.join(', ')}
        </div>
      )}
    </div>
  );
}
