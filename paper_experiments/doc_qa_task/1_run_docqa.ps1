param(
    [string]$model,
    [string]$docs,
    [string]$baseline,
    [string]$data_file
)

python doc_qa.py --model $model --baseline $baseline --num_docs $docs --data_file $data_file
