# 信息系统项目管理师 RAG 知识库讲义

这个项目的目标，是把《信息系统项目管理师辅导教程第3版全版》做成一个可以问答的教材型 RAG 知识库。

你可以把整个系统想成四层：

1. 原始教材层：保存 Word 文档，不直接让大模型全文阅读。
2. 文本块层：把 Word 拆成很多带章节信息的 chunk。
3. 检索索引层：同时建立向量索引和关键词索引。
4. LangGraph 问答层：用户提问后，先检索原文，再基于原文回答。

## 为什么要先切 chunk

大模型不能每次都读取整本教材。

所以我们先把教材拆成小块。

每个 chunk 都保留：

- 书名
- 章节
- 小节
- 原始 Word 路径
- chunk_id
- 原文内容

这样用户提问时，系统可以快速找出最相关的几段，而不是把整本书塞给模型。

## embedding 在这里干什么

embedding 不是用来回答问题的。

embedding 的作用是把文本变成向量。

建库时：

```text
教材 chunk -> embedding 模型 -> chunk 向量 -> Chroma 向量库
```

查询时：

```text
用户问题 -> embedding 模型 -> 问题向量 -> 和 chunk 向量比较相似度
```

所以 embedding 是“语义检索工具”。

真正回答问题的是后面的聊天模型。

## 为什么还要 BM25

考试教材里有很多固定术语：

- WBS
- 挣值管理
- 关键路径法
- 范围基准
- 定性风险分析
- 配置管理

这些词用关键词检索非常准。

所以本项目不只做向量检索，还做 BM25 关键词检索。

最终查询时会同时跑：

```text
向量检索：找语义相关
BM25 检索：找关键词精确命中
```

然后把两边结果合并、去重、重排。

## 为什么中文 BM25 要分词

英文句子天然用空格隔开单词。

中文没有空格。

如果直接把中文原文丢给默认 BM25，它可能不知道“范围基准”是一个要匹配的术语。

所以本项目加入了：

```text
jieba 中文分词
中文 2 到 4 字 n-gram
英文数字术语提取
```

这样像下面这些考试术语会更容易命中：

- 范围基准
- 需求基线
- 成本基线
- 关键路径
- 挣值管理
- 工作分解结构

这是中文 RAG 很重要的一点：检索质量不是只靠大模型，分词和索引策略也很关键。

## 文件结构

```text
信息系统项目管理师RAG知识库/
  source_docs/
    信息系统项目管理师辅导教程第3版全版/
      这里是复制出来的 Word 源文档
  code/
    00_setup.sh
    01_build_chunks.py
    02_build_indexes.py
    03_query_graph.py
    04_keyword_search_demo.py
    05_retrieval_graph_demo.py
    06_answer_graph_no_key.py
    07_eval_retrieval.py
    08_interactive_cli.py
    query_understanding.py
    config.py
    requirements.txt
  indexes/
    chunks.jsonl
    manifest.json
    chroma/
    bm25_retriever.pkl
  notes/
    老师讲义_信息系统项目管理师RAG知识库.md
  outputs/
    预留给查询结果和调试报告
```

## 串行执行顺序

第一步，安装依赖：

```bash
cd <项目根目录>/code
zsh 00_setup.sh
```

第二步，配置 OpenAI Key：

```bash
export OPENAI_API_KEY='你的 key'
```

第三步，读取 Word 并切 chunk：

```bash
python 01_build_chunks.py
```

这一步会生成：

```text
indexes/chunks.jsonl
indexes/manifest.json
```

第四步，建立索引：

```bash
python 02_build_indexes.py
```

这一步会生成：

```text
indexes/chroma/
indexes/bm25_retriever.pkl
```

第五步，提问：

```bash
python 03_query_graph.py "什么是范围基准？包括哪些内容？"
```

如果你还没有配置 `OPENAI_API_KEY`，可以先用 BM25 演示脚本验证关键词检索：

```bash
python 04_keyword_search_demo.py "范围基准"
```

这个脚本不会调用 embedding，也不会调用大模型，只会把命中的教材原文片段打印出来。

第六步，用 LangGraph 跑一个“不依赖大模型”的检索图：

```bash
python 05_retrieval_graph_demo.py "范围基准"
```

这个脚本会执行：

```text
prepare_query
retrieve_bm25
build_report
save_report
END
```

它的意义是：先把 RAG 的“检索和引用”链路跑通，再去接大模型生成答案。

## 问题理解层

`query_understanding.py` 是一个不依赖大模型的规则版问题理解器。

它做两件事：

1. 判断用户问题类型。
2. 给检索 query 补充增强词。

例如：

```text
用户问题：什么是范围基准？
问题类型：定义解释
增强 query：什么是范围基准？ 定义 概念 含义 是指 包括
```

再例如：

```text
用户问题：定性风险分析和定量风险分析有什么区别？
问题类型：区别对比
增强 query：定性风险分析和定量风险分析有什么区别？ 区别 比较 不同点 相同点 特点
```

这一步的作用是让检索器更容易命中教材里的定义段、对比段、流程段或 ITTO 表格。

## 无 Key 抽取式回答

`06_answer_graph_no_key.py` 是一个不调用大模型的回答图。

它的流程是：

```text
prepare_query
retrieve_bm25
generate_extractive_answer
save_answer
END
```

它不会“自由发挥”，只会从教材原文里挑出相关句子，组成一个抽取式回答。

这个功能的意义是：在没有 `OPENAI_API_KEY` 的情况下，也能验证 RAG 的核心能力：

```text
能不能找到依据
能不能返回引用
能不能形成一个可读的回答雏形
```

## 批量评估

`07_eval_retrieval.py` 用 `notes/sample_questions.json` 里的样例问题做批量评估。

每个样例问题都写了一个期望命中的章节关键词。

脚本会统计：

```text
Top1 命中数
Top5 命中数
每个问题的 Top1 来源
```

这一步很重要，因为 RAG 不能只靠感觉说“好像准”。

我们需要用一组固定问题反复测试，每次改分词、chunk 或增强词，都看命中率有没有变好。

当前样例评估结果：

```text
问题数：6
Top1 命中：6/6
Top5 命中：6/6
```

这不是说系统已经“永远准确”，而是说明当前这批代表性问题已经能稳定命中预期章节。

后续你可以继续往 `sample_questions.json` 里加题，题越多，评估越可信。

## 交互式模式

`08_interactive_cli.py` 是一个简单的终端问答入口。

运行：

```bash
python 08_interactive_cli.py
```

然后你可以连续输入问题。

它目前调用的是无 Key 抽取式回答图，适合先做本地学习和检索调试。

## 三个核心代码文件分别讲什么

### 01_build_chunks.py

它负责做“资料入库前处理”。

它做四件事：

1. 递归扫描所有 Word。
2. 读取段落和表格文字。
3. 从路径和文件名推断章节元数据。
4. 用 LangChain 的 RecursiveCharacterTextSplitter 切成 chunk。

这里最重要的设计是：每个 chunk 不是只有正文，而是会加上：

```text
书名
章节
小节
正文
```

这样 embedding 时，模型知道这段话属于哪一章、哪一节。

### 02_build_indexes.py

它负责建立两个索引。

第一个是 Chroma 向量索引：

```text
chunk -> OpenAIEmbeddings -> vector -> Chroma
```

第二个是 BM25 关键词索引：

```text
chunk -> BM25Retriever
```

两个索引都保留 chunk 的 metadata。

这就是以后能返回引用来源的原因。

### 03_query_graph.py

它负责在线问答。

LangGraph 的节点顺序是：

```text
rewrite_query
retrieve_vector
retrieve_keyword
merge_and_score
generate_answer
END
```

第一版的 query rewrite 很保守，直接用原问题。

后续可以升级成：

- 自动提取术语
- 自动补充同义词
- 自动判断问题类型
- 检索不足时自动重试

## 后续可以优化什么

第一版完成后，建议你重点观察检索是否准。

可以优化的地方有：

1. chunk_size 是否过大或过小。
2. 是否要按“定义、流程、输入输出工具技术、公式、例题”打标签。
3. 是否要给章节标题命中更高权重。
4. 是否要加入 reranker 模型。
5. 是否要做考试模式，比如“请按考点解释”。
6. 是否要输出引用到具体 Word 文件和 chunk_id。

我建议先别急着上复杂 Agent。

先把“切得准、检得准、引用清楚”做好，这就是 RAG 的地基。
