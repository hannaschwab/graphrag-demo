# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "ab264d80-4288-4a82-9b04-004a2d988af2",
# META       "default_lakehouse_name": "supply_chain_lh",
# META       "default_lakehouse_workspace_id": "5eda1118-9c7b-45fc-b3cb-05a04ec165eb",
# META       "known_lakehouses": [
# META         {
# META           "id": "ab264d80-4288-4a82-9b04-004a2d988af2"
# META         }
# META       ]
# META     },
# META     "environment": {
# META       "environmentId": "8d552e4a-191e-8a1f-45fd-a256f09ccae6",
# META       "workspaceId": "00000000-0000-0000-0000-000000000000"
# META     }
# META   }
# META }

# MARKDOWN ********************

# # GraphRAG on Microsoft Fabric + Neo4j
# ## WeAreDevelopers World Congress 2026, Berlin
# 
# **Hanna Schwab** | Data & AI Lead, teccle group
# 
# Self-contained demo: generates pharma supply chain data, projects it into Neo4j,
# then shows exactly why vector RAG fails on relational queries and how GraphRAG fixes it.
# 
# | Step | What happens | Run live? |
# |---|---|---|
# | 0 | Install dependencies | Pre-run |
# | 1 | Configure connections | Pre-run |
# | 2 | Generate demo data (Delta tables) | Pre-run |
# | 3 | Project into Neo4j graph | Pre-run |
# | 4 | Vector RAG - the failure | Pre-run |
# | 5 | GraphRAG - the added value | **RUN LIVE** |
# | 6 | Ground truth verification | Pre-run |


# CELL ********************

#%pip install 'neo4j>=5.17,<6.0' 'neo4j-graphrag==1.7.0' openai --quiet
#print('Dependencies ready')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Step 1 - Configuration
# 
# Neo4j AuraDB: free at https://console.neo4j.io
# Azure OpenAI: use your tenant endpoint, or set USE_AZURE=False for plain OpenAI key.
# Store secrets via notebookutils.credentials.getSecret() in production.


# CELL ********************

## Step 1 — Configuration
import notebookutils

KV_NAME = 'https://kv-fabricdemo-hs.vault.azure.net/'

# Neo4j AuraDB
NEO4J_URI      = notebookutils.credentials.getSecret(KV_NAME, 'neo4juri')
NEO4J_USER     = notebookutils.credentials.getSecret(KV_NAME, 'neo4j')
NEO4J_PASSWORD = notebookutils.credentials.getSecret(KV_NAME, 'neo4j-pw')

# Azure OpenAI
USE_AZURE            = True
AZURE_ENDPOINT       = notebookutils.credentials.getSecret(KV_NAME, 'aoai-endpoint')
AZURE_API_KEY        = notebookutils.credentials.getSecret(KV_NAME, 'aoai-key')
AZURE_API_VERSION    = '2024-10-21'
AZURE_EMB_DEPLOYMENT = 'text-embedding-3-large'
AZURE_LLM_DEPLOYMENT = 'gpt-4o'

print(f'Config loaded ✓ | LLM: {AZURE_LLM_DEPLOYMENT} | Embedder: {AZURE_EMB_DEPLOYMENT}')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Step 2 - Generate Demo Data
# 
# Pharma supply chain - 4 entity types written as Delta tables to the Lakehouse:
# 
# ```
# (:Order)-[:CONTAINS]->(:Batch)-[:MANUFACTURED_BY]->(:Supplier)
#                           \-[:STORED_IN]---------->(:Warehouse)
# ```
# 
# The killer question requires traversing all 4 nodes across 3 hops.


# CELL ********************

from pyspark.sql import Row
from datetime import date

SUPPLIERS = [
    Row(supplier_id='S001', name='ChemSynth GmbH',       country='Germany', risk_tier='HIGH',   active_deliveries=3),
    Row(supplier_id='S002', name='PharmaBase Italia',    country='Italy',   risk_tier='LOW',    active_deliveries=7),
    Row(supplier_id='S003', name='Nordic API Solutions', country='Sweden',  risk_tier='MEDIUM', active_deliveries=2),
    Row(supplier_id='S004', name='EastChem Ltd',         country='Poland',  risk_tier='HIGH',   active_deliveries=5),
]

WAREHOUSES = [
    Row(warehouse_id='W001', location='Berlin',   region='EU-West',  capacity=5000),
    Row(warehouse_id='W002', location='Hamburg',  region='EU-North', capacity=3000),
    Row(warehouse_id='W003', location='Munich',   region='EU-South', capacity=4000),
    Row(warehouse_id='W004', location='Warsaw',   region='EU-East',  capacity=2500),
]

# quality_status: PASSED | FAILED | ON_HOLD
BATCHES = [
    Row(batch_id='B001', product='Aspirin 500mg',     quality_status='FAILED',  mfg_date=str(date(2024,1,10)), supplier_id='S001', warehouse_id='W002'),
    Row(batch_id='B002', product='Ibuprofen 200mg',   quality_status='PASSED',  mfg_date=str(date(2024,1,15)), supplier_id='S002', warehouse_id='W001'),
    Row(batch_id='B003', product='Paracetamol 500mg', quality_status='PASSED',  mfg_date=str(date(2024,1,20)), supplier_id='S003', warehouse_id='W003'),
    Row(batch_id='B004', product='Aspirin 500mg',     quality_status='ON_HOLD', mfg_date=str(date(2024,2,1)),  supplier_id='S004', warehouse_id='W004'),
    Row(batch_id='B005', product='Metformin 1000mg',  quality_status='FAILED',  mfg_date=str(date(2024,2,5)),  supplier_id='S001', warehouse_id='W002'),
    Row(batch_id='B006', product='Lisinopril 10mg',   quality_status='PASSED',  mfg_date=str(date(2024,2,8)),  supplier_id='S002', warehouse_id='W001'),
]

# status: SHIPPED | DELAYED | PENDING | priority: HIGH | MEDIUM | LOW
ORDERS = [
    Row(order_id='O001', customer='Charite Berlin',    batch_id='B001', status='DELAYED', priority='HIGH',   order_date=str(date(2024,2,10))),
    Row(order_id='O002', customer='UKE Hamburg',       batch_id='B001', status='DELAYED', priority='HIGH',   order_date=str(date(2024,2,11))),
    Row(order_id='O003', customer='Klinikum Muenchen', batch_id='B002', status='SHIPPED', priority='MEDIUM', order_date=str(date(2024,2,12))),
    Row(order_id='O004', customer='Szpital Krakow',    batch_id='B004', status='PENDING', priority='HIGH',   order_date=str(date(2024,2,13))),
    Row(order_id='O005', customer='Karolinska',        batch_id='B005', status='DELAYED', priority='HIGH',   order_date=str(date(2024,2,14))),
    Row(order_id='O006', customer='Charite Berlin',    batch_id='B003', status='SHIPPED', priority='LOW',    order_date=str(date(2024,2,15))),
    Row(order_id='O007', customer='AZ Sint-Jan',       batch_id='B004', status='PENDING', priority='MEDIUM', order_date=str(date(2024,2,16))),
]

for name, data in [('suppliers',SUPPLIERS),('warehouses',WAREHOUSES),('batches',BATCHES),('orders',ORDERS)]:
    spark.createDataFrame(data).write.format('delta').mode('overwrite').saveAsTable(name)
    print(f'  {name}: {len(data)} rows -> Delta table')

print('All demo data in Lakehouse')


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Step 3 - Project OneLake Data into Neo4j
# 
# Zero-ETL pattern: native Python driver pulls Lakehouse rows into graph nodes + relationships.
# No pipeline. No copy. No middleware.
# 
# ```python
# # Three lines to connect:
# from neo4j import GraphDatabase
# driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
# # Then: for row in spark.table('batches').collect(): session.run(CYPHER, **row)
# ```


# CELL ********************

from neo4j import GraphDatabase

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
with driver.session() as s:
    print(s.run('RETURN "Connected" AS msg').single()['msg'], '- Neo4j AuraDB')
    s.run('MATCH (n) DETACH DELETE n')
    for label, key in [('Batch','batch_id'),('Warehouse','warehouse_id'),
                        ('Supplier','supplier_id'),('Order','order_id')]:
        s.run(f'CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{key} IS UNIQUE')
    print('Graph cleared + constraints ready')


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Project Suppliers
with driver.session() as s:
    for r in spark.table('suppliers').collect():
        s.run('MERGE (n:Supplier {supplier_id:$id}) SET n.name=$name, n.country=$country, n.risk_tier=$risk, n.active_deliveries=$adel',
              id=r.supplier_id, name=r.name, country=r.country, risk=r.risk_tier, adel=r.active_deliveries)

# Project Warehouses
with driver.session() as s:
    for r in spark.table('warehouses').collect():
        s.run('MERGE (n:Warehouse {warehouse_id:$id}) SET n.location=$loc, n.region=$reg, n.capacity=$cap',
              id=r.warehouse_id, loc=r.location, reg=r.region, cap=r.capacity)

# Project Batches + STORED_IN + MANUFACTURED_BY relationships
with driver.session() as s:
    for r in spark.table('batches').collect():
        desc = f"{r.product} batch {r.batch_id} quality {r.quality_status} supplier {r.supplier_id}"
        s.run('''
            MERGE (b:Batch {batch_id:$bid})
            SET b.product=$prod, b.quality_status=$qs, b.description=$desc
            WITH b
            MATCH (w:Warehouse {warehouse_id:$wid}) MERGE (b)-[:STORED_IN]->(w)
            WITH b
            MATCH (sup:Supplier {supplier_id:$sid}) MERGE (b)-[:MANUFACTURED_BY]->(sup)
        ''', bid=r.batch_id, prod=r.product, qs=r.quality_status, desc=desc,
             wid=r.warehouse_id, sid=r.supplier_id)

# Project Orders + CONTAINS relationship
with driver.session() as s:
    for r in spark.table('orders').collect():
        s.run('''
            MERGE (o:Order {order_id:$oid})
            SET o.customer=$cust, o.status=$st, o.priority=$pri, o.order_date=$od
            WITH o
            MATCH (b:Batch {batch_id:$bid}) MERGE (o)-[:CONTAINS]->(b)
        ''', oid=r.order_id, cust=r.customer, st=r.status, pri=r.priority,
             od=r.order_date, bid=r.batch_id)

# Verify
print('Graph structure:')
with driver.session() as s:
    for r in s.run('MATCH (n) RETURN labels(n)[0] AS l, count(n) AS c ORDER BY l'):
        print(f'  {r["l"]}: {r["c"]} nodes')
    for r in s.run('MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS c ORDER BY t'):
        print(f'  {r["t"]}: {r["c"]} rels')


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# ## The Question
# 
# Which HIGH-priority orders are DELAYED because their batch FAILED quality checks,
# was manufactured by a HIGH-risk supplier - and where is that batch stored?
# 
# Traversal path:
# ```
# (Order {priority:HIGH, status:DELAYED})
#   |--[:CONTAINS]-->(Batch {quality_status:FAILED})
#                      |--[:MANUFACTURED_BY]-->(Supplier {risk_tier:HIGH})
#                      |--[:STORED_IN]-------->(Warehouse)
# ```
# 
# 3 hops. 4 filters. SQL: 3 JOINs + 4 WHERE clauses. Vector search: 0 hops.
# 
# ---


# MARKDOWN ********************

# ## Step 4 - Vector RAG (The Failure)
# 
# Semantic similarity retrieves text that looks relevant.
# It cannot traverse entity relationships. Watch what breaks.


# CELL ********************

# ============================================================
# STEP 4 — Vector RAG (pre-run before talk)
# Shows WHY pure vector search fails for this question
# ============================================================

from neo4j import GraphDatabase
import openai

# Reconnect (driver may have timed out)
driver = GraphDatabase.driver(
    NEO4J_URI, 
    auth=(NEO4J_USER, NEO4J_PASSWORD),
    notifications_min_severity='OFF'
)

if USE_AZURE:
    client = openai.AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION)
    emb_model, llm_model = AZURE_EMB_DEPLOYMENT, AZURE_LLM_DEPLOYMENT
else:
    client = openai.OpenAI(api_key=OPENAI_KEY)
    emb_model, llm_model = EMBEDDING_MODEL, LLM_MODEL

def embed(text):
    return client.embeddings.create(input=text, model=emb_model).data[0].embedding

QUESTION = (
    'Which HIGH-priority orders are DELAYED because their batch FAILED quality checks '
    'AND was manufactured by a HIGH-risk supplier? Which warehouse holds the batch?'
)

# Create vector index + embed batch descriptions
with driver.session() as s:
    s.run('''
        CREATE VECTOR INDEX batch_embeddings IF NOT EXISTS
        FOR (b:Batch) ON (b.embedding)
        OPTIONS {indexConfig: {`vector.dimensions`: 3072, `vector.similarity_function`: 'cosine'}}
    ''')
    for r in spark.table('batches').collect():
        text = f"{r.product} batch {r.batch_id} quality {r.quality_status}"
        s.run('MATCH (b:Batch {batch_id:$bid}) SET b.embedding=$emb',
              bid=r.batch_id, emb=embed(text))
print('Vector index ready')

# Vector search — finds similar batches by embedding only
# NO knowledge of Order priority, Supplier risk, or Warehouse
q_emb = embed(QUESTION)
with driver.session() as s:
    hits = s.run('''
        CALL db.index.vector.queryNodes('batch_embeddings', 3, $emb)
        YIELD node, score
        RETURN node.batch_id + ' | ' + node.product + ' | quality: ' + node.quality_status AS ctx, score
        ORDER BY score DESC
    ''', emb=q_emb).data()

context = '\n'.join(f"- {h['ctx']} (score: {h['score']:.3f})" for h in hits)
print('Vector context retrieved:')
print(context)

resp = client.chat.completions.create(
    model=llm_model,
    messages=[
        {'role': 'system', 'content': 'Pharma supply chain analyst. Answer ONLY from context provided. Never invent data.'},
        {'role': 'user',   'content': f'Context:\n{context}\n\nQuestion: {QUESTION}'}
    ]
)

print('QUESTION:', QUESTION)
print('\n' + '='*60)
print('VECTOR RAG ANSWER (incomplete — this is the failure):')
print('='*60)
print(resp.choices[0].message.content)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# ## Step 5 - GraphRAG  <<< RUN THIS CELL LIVE ON STAGE
# 
# VectorCypherRetriever: vector finds the Batch entry node, Cypher walks the full chain.
# Same question. Same LLM. Complete relational context.


# CELL ********************

# ============================================================
# STEP 5 — GraphRAG  (RUN LIVE ON STAGE)
# Expected execution time: ~8 seconds
# ============================================================

from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.types import RetrieverResultItem

# Reconnect (driver may have timed out)
driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD),
    notifications_min_severity='OFF'
)

if USE_AZURE:
    from neo4j_graphrag.llm import AzureOpenAILLM
    from neo4j_graphrag.embeddings import AzureOpenAIEmbeddings
    embedder = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_ENDPOINT, api_key=AZURE_API_KEY,
        azure_deployment=AZURE_EMB_DEPLOYMENT, api_version=AZURE_API_VERSION)
    llm = AzureOpenAILLM(
        model_name=AZURE_LLM_DEPLOYMENT,
        azure_endpoint=AZURE_ENDPOINT, api_key=AZURE_API_KEY,
        azure_deployment=AZURE_LLM_DEPLOYMENT, api_version=AZURE_API_VERSION,
        model_params={'temperature': 0})
else:
    from neo4j_graphrag.llm import OpenAILLM
    from neo4j_graphrag.embeddings import OpenAIEmbeddings
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_KEY)
    llm = OpenAILLM(model_name=LLM_MODEL, api_key=OPENAI_KEY, model_params={'temperature': 0})

QUESTION = (
    'Which HIGH-priority orders are DELAYED because their batch FAILED quality checks '
    'AND was manufactured by a HIGH-risk supplier? Which warehouse holds the batch?'
)

retriever = VectorCypherRetriever(
    driver=driver,
    index_name='batch_embeddings',
    embedder=embedder,
    retrieval_query='''
        MATCH (o:Order)-[:CONTAINS]->(node)
        OPTIONAL MATCH (node)-[:MANUFACTURED_BY]->(sup:Supplier)
        OPTIONAL MATCH (node)-[:STORED_IN]->(w:Warehouse)
        RETURN  
            o.order_id          AS order_id,
            o.customer          AS customer,
            o.status            AS status,
            o.priority          AS priority,
            node.batch_id       AS batch_id,
            node.product        AS product,
            node.quality_status AS quality_status,
            sup.name            AS supplier_name,
            sup.risk_tier       AS supplier_risk,
            w.location          AS warehouse_location
    ''',
    result_formatter=lambda r: RetrieverResultItem(
        content=(
            f"Order {r['order_id']} ({r['customer']}, {r['status']}, priority:{r['priority']}) | "
            f"Batch {r['batch_id']} ({r['product']}) | quality: {r['quality_status']} | "
            f"supplier: {r['supplier_name']} [risk:{r['supplier_risk']}] | "
            f"warehouse: {r['warehouse_location']}"
        ),
        metadata={'order_id': r['order_id']}
    )
)

# Single pipeline run
pipeline = GraphRAG(retriever=retriever, llm=llm)

# Get retriever context separately (display only — no extra LLM call)
retriever_result = retriever.search(query_text=QUESTION, top_k=5)

# One LLM call
result = pipeline.search(query_text=QUESTION, retriever_config={'top_k': 5})

print(f'QUESTION: {QUESTION}')
print('='*60)
print('GRAPHRAG ANSWER:')
print('='*60)
print(result.answer)
print()
print('Graph context the LLM received:')
for item in retriever_result.items:
    print(f'  {item.content}')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# ## Step 6 - Ground Truth Verification (Direct Cypher)
# 
# The precise answer - what the LLM should have said:


# CELL ********************

print('GROUND TRUTH (direct Cypher, 3-hop traversal):')
print('-'*60)
with driver.session() as s:
    rows = s.run('''
        MATCH (o:Order)-[:CONTAINS]->(b:Batch)-[:MANUFACTURED_BY]->(sup:Supplier)
        MATCH (b)-[:STORED_IN]->(w:Warehouse)
        WHERE o.priority = 'HIGH'
          AND o.status   = 'DELAYED'
          AND b.quality_status = 'FAILED'
          AND sup.risk_tier    = 'HIGH'
        RETURN
            o.order_id AS oid, o.customer AS cust,
            b.batch_id AS bid, b.product AS prod,
            sup.name AS sup_name, sup.risk_tier AS risk,
            w.location AS warehouse
        ORDER BY o.order_id
    ''').data()
    for r in rows:
        print(f"  {r['oid']} | {r['cust']}")
        print(f"    Batch {r['bid']} ({r['prod']}) -- {r['sup_name']} [{r['risk']} risk] -- stored at {r['warehouse']}")

driver.close()
print('\nDemo complete.')


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
