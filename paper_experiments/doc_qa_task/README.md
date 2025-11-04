# 多文档QA实验

## 环境准备

### 基本信息
- 嵌入模型：bge-m3
- 嵌入维度：1024
- 数据集大小：14G

### 基本配置

1. 装依赖，pip install .[postgres,experiments]
2. 准备embedding API，可以使用自己的api渠道或者使用[xinference](https://inference.readthedocs.io/zh-cn/latest/getting_started/using_xinference.html#run-on-nvidia-gpu-host)进行部署
 
3. 拉起pgvector服务
```bash
cd paper_experiments
docker compose up -d
```

4. 将.env.example复制为.env并设置

### 数据集准备

信息：
- 完整文档集一共340w行，建议为实验预留200G以上磁盘
- 加载完整文档集需要约一个半小时(受磁盘性能影响)
- 为完整文档集建立索引需要时间预计在7小时以上
- 若是仅测试，建议修改bash脚本只加载其中一个分块(10w行)，此时加载时间约2min，建立索引时间约7min

1. 下数据集
```bash
# 问题集 156M
hf download MemGPT/qa_data --repo-type dataset --cache-dir { DIR }
# 文档集 14G
hf download Upstash/wikipedia-2024-06-bge-m3 --repo-type dataset --include "data/zh/*" --cache-dir { DIR }
```

2. 加载数据集

首先将0_load_embeddings.sh中所有file参数修正为你的路径，然后

```bash
cd paper_experiments/doc_qa_task
bash ./0_load_embeddings.sh
```

3. 建立HNSW索引

连接到pgvector，执行
```sql
CREATE INDEX ON memgpt_passages USING hnsw (embedding vector_l2_ops);
```

## 运行实验

### 文档QA

运行
```bash
cd paper_experiments/doc_qa_task
./1_run_docqa.sh {model_name} {n_docs} {memgpt/model_name} {data_file_path}
```

其中
- model_name：测试的大模型名称
- n_docs：指定每次从achieve_memory检索top_n份文档
- {memgpt/model_name}：若指定为memgpt则使用memgpt agent来运行实验，若指定为model_name则运行baseline
- data_file_path：qa_data的分块文件名

实验变量
- n_docs
- model_name
这两者可指定为不同值以测得多份数据

若是运行时发现目录创建相关的报错且results目录没有被创建，请手动创建results目录

### LLM Judge

更改2_run_eval.sh开头的两个参数列表

- docs(对应n_docs)：如果跑了baseline，填入所有测过的n_docs
- models(对应model_name)：填入所有测过的模型名称

运行
```bash
cd paper_experiments/doc_qa_task
bash ./2_run_eval.sh
```

## 实验结果示例

- 嵌入：bge-m3
- 维度：1024
- 模型：deepseek/deepseek-v3-0324
- 评估模型：deepseek/deepseek-v3-0324
- n_docs: 5
- 仅memgpt，仅qa_data第一分块，仅使用20w行wikipedia


[qa结果示例](./example_results/doc_qa_results_model_deepseek_deepseek-v3-0324.json)

[评估结果示例](./example_results/results_deepseek_deepseek-v3-0324_None_memgpt.json)

## 附加说明

如果要更换数据集
1. memgpt\constants.py 中的MAX_EMBEDDING_DIM改成新数据集嵌入模型的维度
2. paper_experiments\doc_qa_task\load_wikipedia_embeddings.py开头的EMBEDDING_DIM改为新维度,EMBEDDING_MODEL改为新嵌入模型名称
3. 若有字段及格式变更，对load_wikipedia_embeddings.py进行相应修改

> 当前版本letta，使用的嵌入模型维度不宜超过2000，否则会导致HNSW索引建立失败