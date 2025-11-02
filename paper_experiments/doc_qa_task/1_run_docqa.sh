model=$1
docs=$2
baseline=$3
data_file=$4
python doc_qa.py --model $model --baseline $baseline --num_docs $docs --data_file $data_file
