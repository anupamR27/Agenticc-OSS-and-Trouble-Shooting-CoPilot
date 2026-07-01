# Full Live Pipeline README

This project includes a final end-to-end live telecom RCA pipeline in:

```bash
full_live_pipeline.py
```

It combines:

1. Member 1 anomaly detection
2. Member 2 root cause analysis
3. Member 3 recommendations, RAG context, and LLM incident reporting
4. Final clean report formatting

## Required Python Files

To run `full_live_pipeline.py`, keep these Python files available:

```text
full_live_pipeline.py
final_report_formatter.py
live_rca_pipeline.py
Member 2/rca_poc.py
Member 3/agents/recommendation_agent.py
Member 3/agents/llm_agent.py
Member 3/rag/retriever.py
Member 3/rag/vector_store.py
Member 3/rag/config.py
```

These Member 3 files are needed if you need to rebuild the RAG index:

```text
Member 3/rag/build_index.py
Member 3/rag/loader.py
Member 3/rag/chunker.py
```

## Required Non-Python Files

The pipeline also depends on these artifacts:

```text
Member 1/member1/models/isolation_forest_production.pkl
Member 1/member1/data/ds1_processed.csv
Member 1/member1/outputs/anomaly_scores_production.csv
Member 3/vector_db/index.faiss
Member 3/vector_db/index.pkl
Member 3/knowledge_base/Huawei_RAN_TG.pdf
Member 3/knowledge_base/3gpp.pdf
Member 3/.env
```

`Member 3/.env` must contain:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

Do not push the real `.env` file to GitHub.

## Run Command

From the project root:

```bash
python3 full_live_pipeline.py
```

Then paste KPI JSON and press Enter twice.

For quick testing:

```bash
python3 full_live_pipeline.py --sample
```

For debug output:

```bash
python3 full_live_pipeline.py --sample --debug
```

## Output Files

Running the final pipeline creates:

```text
final_incident_report.txt
full_pipeline_output.json
```

`final_incident_report.txt` is the clean human-readable report.

`full_pipeline_output.json` is the raw developer/debug output.

These two files are generated outputs and usually do not need to be pushed.
