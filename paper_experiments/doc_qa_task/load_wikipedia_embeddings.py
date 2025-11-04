import json
import hashlib
import os
import uuid
import copy
import pandas as pd
from tqdm import tqdm
from memgpt.cli.cli_config import delete
from memgpt.data_types import Passage
from memgpt.agent_store.storage import StorageConnector, TableType
from dotenv import load_dotenv

from paper_experiments.utils import get_experiment_config
from concurrent.futures import ThreadPoolExecutor, as_completed
from absl import app, flags
import time

load_dotenv()

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM").strip())
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL").strip()
PGVECTOR_TEST_DB_URL = os.getenv("PGVECTOR_TEST_DB_URL").strip()

# Batch size for inserting passages into database (to control memory usage)
# Adjust this value based on your memory constraints
INSERT_BATCH_SIZE = 5000

# Create an empty list to store the JSON objects
source_name = "wikipedia"
config = get_experiment_config(PGVECTOR_TEST_DB_URL, endpoint_type="openai")
config.save()  # save config to file
user_id = uuid.UUID(config.anon_clientid)

FLAGS = flags.FLAGS
flags.DEFINE_boolean("drop_db", default=False, required=False, help="Drop existing source DB")
flags.DEFINE_string("file", default=None, required=True, help="File to parse")
flags.DEFINE_integer("batch_size", default=INSERT_BATCH_SIZE, required=False, help=f"Batch size for inserting passages (default: {INSERT_BATCH_SIZE})")


def create_uuid_from_string(val: str):
    """
    Generate consistent UUID from a string
    from: https://samos-it.com/posts/python-create-uuid-from-random-string-of-words.html
    """
    hex_string = hashlib.md5(val.encode("UTF-8")).hexdigest()
    return uuid.UUID(hex=hex_string)


def process_parquet_chunk(chunk_df, conn, batch_size, added, model, filename, show_progress=False):
    """Process a chunk of parquet DataFrame and insert into database
    
    Args:
        chunk_df: DataFrame chunk to process
        conn: Database connection
        batch_size: Number of passages to accumulate before inserting
        added: Set of already processed passage IDs (for deduplication)
        model: Embedding model name
        filename: Filename for progress display
        show_progress: Whether to show progress bar
    """
    passages = []
    total_insert_time = 0.0
    
    total_rows = len(chunk_df)
    
    if show_progress and total_rows > 0:
        row_iterator = tqdm(
            chunk_df.itertuples(index=False),
            total=total_rows,
            desc=f"Loading {filename}",
            leave=False,
            unit="row",
            unit_scale=False,
            miniters=max(1, total_rows // 100)  # Update every 1% for better performance
        )
    else:
        row_iterator = chunk_df.itertuples(index=False)
    
    for row in row_iterator:
        # row is a namedtuple from itertuples(), access by attribute name
        text = row.text
        embedding = row.embedding
        
        # Handle embedding: could be list, numpy array, or already a list
        if isinstance(embedding, (list, tuple)):
            embedding_list = list(embedding)
        elif hasattr(embedding, 'tolist'):
            embedding_list = embedding.tolist()
        else:
            embedding_list = list(embedding)
        
        embedding_dim = len(embedding_list)
        assert embedding_dim == EMBEDDING_DIM, f"Wrong embedding dim: {embedding_dim}, expected {EMBEDDING_DIM}"

        passage_id = create_uuid_from_string(text)  # consistent hash for text (prevent duplicates)
        if passage_id in added:
            continue
        else:
            added.add(passage_id)

        passage = Passage(
            id=passage_id,
            user_id=user_id,
            text=text,
            embedding_model=model,
            embedding_dim=embedding_dim,
            embedding=embedding_list,
            data_source=source_name,
        )
        passages.append(passage)
        
        # Insert batch when it reaches the batch size
        if len(passages) >= batch_size:
            st = time.time()
            conn.insert_many(passages, exists_ok=True)
            total_insert_time += time.time() - st
            passages = []  # Clear batch to free memory
    
    # Insert remaining passages
    if passages:
        st = time.time()
        conn.insert_many(passages, exists_ok=True)
        total_insert_time += time.time() - st
        passages = []
    
    return total_insert_time


def insert_lines(file_paths, conn, show_progress=False, batch_size=INSERT_BATCH_SIZE, chunk_rows=10000):
    """Parse and insert list of parquet file paths into source database
    
    Args:
        file_paths: List of parquet file paths to process
        conn: Database connection
        show_progress: Whether to show progress bars
        batch_size: Number of passages to accumulate before inserting (default: INSERT_BATCH_SIZE)
        chunk_rows: Number of rows to process in each chunk for large files (default: 10000)
    """
    added = set()
    total_insert_time = 0.0
    
    # Show file-level progress only if there are multiple files
    use_file_progress = show_progress and len(file_paths) > 1
    file_iterator = tqdm(file_paths, desc="Processing files", unit="file") if use_file_progress else file_paths
    
    model = EMBEDDING_MODEL
    
    for file_path in file_iterator:
        # Strip whitespace and newlines from file path
        file_path = file_path.strip() if isinstance(file_path, str) else file_path
        if not file_path:
            continue
            
        # Get filename for progress display
        filename = os.path.basename(file_path)
        
        # Read parquet file
        d = pd.read_parquet(file_path)
        assert len(d) > 0, f"Parquet file is empty: {file_path}"
        assert "text" in d.columns, f"Missing 'text' column in {file_path}"
        assert "embedding" in d.columns, f"Missing 'embedding' column in {file_path}"
        
        total_rows = len(d)
        
        # Process file in chunks if it's large (similar to original approach)
        # This allows parallel processing of chunks and better memory management
        if total_rows > chunk_rows:
            # Process large files in chunks
            num_chunks = (total_rows + chunk_rows - 1) // chunk_rows
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_rows
                end_idx = min((chunk_idx + 1) * chunk_rows, total_rows)
                chunk_df = d.iloc[start_idx:end_idx]
                
                chunk_time = process_parquet_chunk(
                    chunk_df, conn, batch_size, added, model,
                    f"{filename} [{chunk_idx+1}/{num_chunks}]",
                    show_progress
                )
                total_insert_time += chunk_time
        else:
            # Process small files directly
            chunk_time = process_parquet_chunk(
                d, conn, batch_size, added, model, filename, show_progress
            )
            total_insert_time += chunk_time
    
    return total_insert_time


def main(argv):
    # clear out existing source
    if FLAGS.drop_db:
        delete("source", source_name)
        try:
            passages_table = StorageConnector.get_storage_connector(TableType.PASSAGES, config, user_id)
            passages_table.delete_table()

        except Exception as e:
            print("Failed to delete source")
            print(e)

    # Check if FLAGS.file is a parquet file or a text file with parquet file paths
    input_file = FLAGS.file
    parquet_files = []
    
    if input_file.lower().endswith('.parquet'):
        # Direct parquet file
        print(f"Processing parquet file: {input_file}")
        parquet_files = [input_file]
    else:
        # Text file containing parquet file paths (one per line)
        print(f"Reading parquet file paths from: {input_file}")
        with open(input_file, "r", encoding="utf-8") as f:
            parquet_files = [line.strip() for line in f if line.strip()]
    
    count = 0
    chunk_size = 1000
    conn = StorageConnector.get_storage_connector(TableType.PASSAGES, config, user_id)
    
    futures = []
    
    # Show file-level progress only if there are multiple files
    use_file_progress = len(parquet_files) > 1
    file_list_iterator = tqdm(parquet_files, desc="Processing parquet files", unit="file") if use_file_progress else parquet_files
    
    with ThreadPoolExecutor(max_workers=64) as p:
        file_batch = []

        # Process parquet files in chunks
        for file_path in file_list_iterator:
            file_batch.append(file_path)
            if len(file_batch) >= chunk_size:
                if count == 0:
                    print("Await first result (hack to avoid concurrency issues)")
                    t = insert_lines(file_batch, conn, True, FLAGS.batch_size)
                    print("Finished first result", t)
                else:
                    # For background tasks, don't show progress to avoid cluttering output
                    future = p.submit(insert_lines, copy.deepcopy(file_batch), conn, False, FLAGS.batch_size)
                    futures.append(future)
                count += len(file_batch)
                file_batch = []

        # Insert remaining files
        if len(file_batch) > 0:
            # Show progress for the remaining batch if it's the first batch or if it's a single file
            show_progress_for_remaining = (count == 0 or len(parquet_files) == 1)
            if show_progress_for_remaining:
                # Process synchronously with progress bar
                t = insert_lines(file_batch, conn, True, FLAGS.batch_size)
                print("Finished processing", t)
            else:
                # Process in background
                future = p.submit(insert_lines, copy.deepcopy(file_batch), conn, False, FLAGS.batch_size)
                futures.append(future)
            count += len(file_batch)
            file_batch = []


    if futures:
        print(f"Waiting for {len(futures)} background batches")
        # wait for futures
        for future in tqdm(as_completed(futures), desc="Waiting for batches", unit="batch"):
            succ = future.result()

    size = conn.size()
    print("Number of passages", size)


if __name__ == "__main__":
    app.run(main)
