$docs = @(1, 5, 10, 20, 50, 100, 200, 700)
$models = @("deepseek_deepseek-v3.2-exp")

# run memgpt eval
foreach ($model in $models) {
    uv run llm_judge_doc_qa.py --file results/doc_qa_results_model_${model}.json
}

# Iterate over each model
foreach ($model in $models) {
    # Iterate over each doc
    foreach ($doc in $docs) {
        # Construct and run the command
        Write-Host "Running for model $model with $doc docs..."
        uv run llm_judge_doc_qa.py --file results/doc_qa_baseline_model_${model}_num_docs_${doc}.json --baseline
    }
}
