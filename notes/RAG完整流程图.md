# RAG 知识库完整流程图

## 总览

```mermaid
flowchart TD
    A["源文档 Word/PDF"] --> B["文档加载 Loader"]
    B --> C["文本清洗 Normalization"]
    C --> D["结构识别 章节/小节/标题"]
    D --> E["Chunk 切分"]
    E --> F["写入 chunks.jsonl"]
    E --> G["元数据 Metadata"]
    F --> H["BM25 关键词索引"]
    F --> I["Embedding 模型"]
    I --> J["向量 Vector"]
    J --> K["Chroma/Qdrant 向量库"]
    G --> H
    G --> K

    L["用户问题"] --> M["问题理解/分类"]
    M --> N["Query 增强/术语扩展"]
    N --> O["BM25 检索"]
    N --> P["Embedding 模型"]
    P --> Q["问题向量"]
    Q --> R["向量检索"]
    K --> R
    H --> O
    O --> S["合并去重"]
    R --> S
    S --> T["重排/打分"]
    T --> U["选 Top-K 原文"]
    U --> V["LLM 回答模型"]
    V --> W["答案 + 引用原文 + 来源"]
```

## 离线建库阶段

```mermaid
flowchart LR
    A["208 个 Word 源文档"] --> B["读取段落和表格"]
    B --> C["清洗空白/乱码/断行"]
    C --> D["识别章节元数据"]
    D --> E["切成 1125 个 chunk"]
    E --> F["保存 chunks.jsonl"]
    F --> G["BM25 索引"]
    F --> H["Embedding"]
    H --> I["向量索引"]
```

## 在线问答阶段

```mermaid
flowchart LR
    A["用户问题"] --> B["判断问题类型"]
    B --> C["增强 query"]
    C --> D["BM25 关键词检索"]
    C --> E["Embedding 问题向量"]
    E --> F["向量检索"]
    D --> G["合并候选"]
    F --> G
    G --> H["重排"]
    H --> I["引用原文"]
    I --> J["LLM 生成答案"]
    J --> K["返回答案和引用"]
```

## 模型分工

```text
Embedding 模型：
  负责把 chunk 和用户问题变成向量。
  用于语义检索。

LLM 回答模型：
  负责阅读检索到的原文。
  用于组织答案、解释、总结、按考试方式讲解。
```

## 当前项目状态

```text
已完成：
  Word 读取
  chunk 切分
  BM25 中文索引
  问题理解
  无 Key 检索报告
  无 Key 抽取式回答
  批量评估

未完成：
  embedding 向量索引
  Chroma 语义检索
  LLM 最终回答
```
