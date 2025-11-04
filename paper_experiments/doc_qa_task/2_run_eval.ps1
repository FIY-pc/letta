$docs = @(1, 5, 10, 20, 50, 100, 200, 700)
$models = @("gpt-4-0613", "gpt-3.5-turbo-1106", "gpt-4-1106-preview")

# run memgpt eval
foreach ($model in $models) {
    python llm_judge_doc_qa.py --file results/doc_qa_results_model_${model}.json
}

# Iterate over each model
foreach ($model in $models) {
    # Iterate over each doc
    foreach ($doc in $docs) {
        # Construct and run the command
        Write-Host "Running for model $model with $doc docs..."
        python llm_judge_doc_qa.py --file results/doc_qa_baseline_model_${model}_num_docs_${doc}.json --baseline
    }
}
