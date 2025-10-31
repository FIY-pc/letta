
## Nested K/V (`nested_kv_task`)
This task runs K/V lookups on synthetic data. You can run it with `paper_experiments/nested_kv_task/run.sh`.

## Document Q/A (`doc_qa_task`)
This task runs question answering on a set of embedded wikipedia passages.

### Setup

install dependencies:
```
uv sync --all-extras
```

easy way to launch a pgvector
```bash
# set your variables in docker-compose.yaml first, then
docker compose up -d
```

You need a a running postgres database to run this experiment and an OpenAI account. Set your enviornment variables:
```
export PGVECTOR_TEST_DB_URL=postgresql+pg8000://{username}:{password}@localhost:5432/{db}
export OPENAI_API_KEY={key}
```

## Download data
Download the wikipedia embedding at:
```
hf download Upstash/wikipedia-2024-06-bge-m3 --repo-type dataset --include "data/zh/*" --cache-dir {YOUR_DIR}
```

Download the questions at

```
hf download MemGPT/qa_data --repo-type dataset --cache-dir {YOUR_DIR}
```

## Loading embeddings
Run the script `./0_load_embeddings.sh`.

This step will take a while. You can check the status of the loading by connecting to `psql`:
```
> psql -h localhost -p {password} -U {username} -d {db}
> SELECT COUNT(*) FROM memgpt_passages;
```
Once completed, there will be ~19 million rows in the database.

### Creating an index
To avoid extremeley slow queries, you need to create an index:
```
CREATE INDEX ON memgpt_passages USING hnsw (embedding vector_l2_ops);
```
You can check to see if the index was created successfully with:
```
> SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'memgpt_passages';

memgpt_passages_embedding_idx | CREATE INDEX memgpt_passages_embedding_idx ON public.memgpt_passages USING hnsw (embedding vector_cosine_ops) WITH (m='24', ef_construction='100')
```

## Running Document Q/A
Run the script `./1_run_docqa.sh {model_name} {n_docs} {memgpt/model_name}`.

## Evaluation
Run the script `./2_run_eval.sh`.
