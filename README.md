# GraphRAG on Microsoft Fabric + Neo4j

> Demo notebook for **WeAreDevelopers Berlin 2026**  
> Talk: *"Your RAG has a relationship problem"*

---

## What this demo does

This notebook builds a **GraphRAG pipeline** over a pharma supply chain dataset to answer a complex relational question that standard vector RAG cannot solve:

> *"Which HIGH-priority orders are DELAYED because their batch FAILED quality checks AND was manufactured by a HIGH-risk supplier? Which warehouse holds the batch?"*

It demonstrates side-by-side:
- **Vector RAG** — finds semantically similar batches but fails to traverse relationships → incomplete answer
- **GraphRAG** — combines vector search with Cypher graph traversal → correct, complete answer

The full stack: **Microsoft Fabric Lakehouse** (data) → **Neo4j AuraDB** (knowledge graph) → **Azure OpenAI** (embeddings + LLM) → **neo4j-graphrag** (pipeline)

---

## Architecture

```
Microsoft Fabric Lakehouse (Delta Tables)
        ↓  Python neo4j Bolt driver (zero-ETL)
Neo4j AuraDB (Knowledge Graph)
        ↓  VectorCypherRetriever (neo4j-graphrag)
Azure OpenAI
  ├── text-embedding-3-large  →  vector search entry point
  └── gpt-4o                  →  answer generation
```

**Graph schema:**
```
(Order) -[:CONTAINS]->      (Batch)
(Batch) -[:MANUFACTURED_BY]->(Supplier)
(Batch) -[:STORED_IN]->     (Warehouse)
```

---

## Prerequisites

| Requirement | Details |
|---|---|
| Microsoft Fabric | Workspace with Lakehouse, Fabric capacity (F2+) |
| Neo4j AuraDB | Free instance at [console.neo4j.io](https://console.neo4j.io) |
| Azure OpenAI | Deployments: `gpt-4o` + `text-embedding-3-large` |
| Azure Key Vault | Linked to your Fabric workspace |
| Python packages | `neo4j`, `neo4j-graphrag==1.7.0`, `openai` (installed in notebook) |

---

## Setup

### 1. Store secrets in Azure Key Vault

Create the following secrets in your Key Vault:

| Secret name | Value |
|---|---|
| `neo4juri` | Your AuraDB connection URI (`neo4j+s://...`) |
| `neo4j` | Your AuraDB username (default: `neo4j`) |
| `neo4j-pw` | Your AuraDB password |
| `aoai-endpoint` | Your Azure OpenAI endpoint URL |
| `aoai-key` | Your Azure OpenAI API key |

Grant your user (or the Fabric workspace managed identity) the **Key Vault Secrets User** role via IAM.

### 2. Update the config cell

In **Step 1** of the notebook, set your Key Vault URL:

```python
KV_NAME = 'https://your-keyvault-name.vault.azure.net/'
```

### 3. Run the notebook

| Step | Description | When to run |
|---|---|---|
| Step 1 | Configuration (Key Vault secrets)
| Step 2 | Install libraries
| Step 3 | Generate demo data → Delta Tables
| Step 4 | Build Neo4j graph + constraints 
| Step 5 | Create vector index + embeddings
| Step 6 | Vector RAG demo
| Step 7 | **GraphRAG demo** 
| Step 8 | Ground truth Cypher verification

---

## Key concepts

**Why vector RAG fails here:**  
Vector search finds semantically similar text chunks. Each chunk describes one entity (a batch). But the question requires traversing 3 hops: Batch → Order (priority + status), Batch → Supplier (risk tier), Batch → Warehouse (location). No single text chunk contains all of this — vector search hands the LLM incomplete fragments.

**Why GraphRAG works:**  
After the vector search identifies the relevant Batch nodes, a Cypher query traverses the graph to collect all connected entities. The LLM receives a complete relational context per batch — and can answer precisely.

---

## Author

**Hanna Schwab**  
[GitHub](https://github.com/hannaschwab) · [LinkedIn](https://linkedin.com/in/hannaschwab)
