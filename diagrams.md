# RAG Verifier Diagrams

Here are the system architecture and workflow diagrams. You can take screenshots of these to include in your presentation, or include them directly in your GitHub repo since GitHub natively renders Mermaid diagrams!

## 1. System Architecture
This diagram shows the high-level components of your system and how they communicate.

```mermaid
graph TD
    User([User]) -->|Interacts with| UI
    
    subgraph Frontend
        UI[Streamlit UI<br/>app.py]
    end
    
    subgraph Backend [FastAPI Backend]
        API[API Router<br/>/query & /verify]
        Verifier[Verification Engine<br/>verifier.py]
        Indexer[Local Embedding Index<br/>indexing.py]
    end
    
    subgraph External Services
        Gemini[Google Gemini API<br/>models/gemini-3.6-flash]
    end
    
    UI -->|HTTP POST| API
    API --> Verifier
    API --> Indexer
    
    Verifier -->|Embeddings| Indexer
    Verifier -->|Generate/Decompose/Judge| Gemini
    
    style User fill:#f9f9f9,stroke:#333,stroke-width:2px
    style Frontend fill:#e1f5fe,stroke:#03a9f4,stroke-width:2px
    style Backend fill:#e8f5e9,stroke:#4caf50,stroke-width:2px
    style External Services fill:#fff3e0,stroke:#ff9800,stroke-width:2px
```

<br>

## 2. Verification Workflow (The Tiered System)
This diagram maps out exactly how your clever two-tier hybrid verification system processes claims to balance cost/latency with accuracy.

```mermaid
flowchart TD
    Start([Generated Answer]) --> Decompose[LLM: Decompose into Atomic Claims]
    Decompose --> ClaimLoop{For each claim}
    
    ClaimLoop --> EmbedClaim[Embed Claim via Sentence Transformers]
    EmbedClaim --> Compare[Compute Cosine Similarity against Source Evidence]
    
    Compare --> CheckThreshold{Similarity Score?}
    
    CheckThreshold -->|Score >= 0.75| Supported[Auto-mark: Supported<br/>*Fast, No LLM*]
    CheckThreshold -->|Score < 0.35| Unverifiable[Auto-mark: Unverifiable<br/>*Fast, No LLM*]
    CheckThreshold -->|0.35 <= Score < 0.75| LLMJudge[Tier 2: Escalated to LLM Judge]
    
    LLMJudge --> LLMDecision{LLM Verdict}
    LLMDecision -->|Supported| Supported
    LLMDecision -->|Contradicted| Contradicted[Mark: Contradicted]
    LLMDecision -->|Unverifiable| Unverifiable
    
    Supported --> Aggregate
    Unverifiable --> Aggregate
    Contradicted --> Aggregate
    
    Aggregate[Aggregate into Faithfulness Score] --> End([Return Report to UI])

    style Supported fill:#c8e6c9,stroke:#2e7d32
    style Contradicted fill:#ffcdd2,stroke:#c62828
    style Unverifiable fill:#ffe0b2,stroke:#ef6c00
    style LLMJudge fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
```
