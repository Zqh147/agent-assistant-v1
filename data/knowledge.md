# Agent Assistant Knowledge Base

RAG 是 Retrieval Augmented Generation 的缩写，即检索增强生成。系统先从外部知识库检索与问题相关的片段，再把片段作为上下文交给语言模型生成答案。

向量数据库保存文本的 embedding 向量，并通过向量相似度完成语义检索。BM25 是基于关键词和词频的稀疏检索方法。混合检索可以结合语义匹配和关键词匹配的优点。

Agent 不等于普通聊天机器人。Agent 会根据用户任务进行规划，并在需要时调用计算器、搜索、数据库或其他外部工具，再根据工具结果继续推理。

LangGraph 使用 State 表示共享状态，使用 Node 表示处理步骤，使用 Edge 表示流程连接，使用条件路由根据状态决定下一步。Checkpointer 可以把某个 thread 的状态保存下来，使多轮调用能够恢复之前的消息和执行上下文。

MCP 是一种让模型应用以统一方式发现和调用外部工具、资源与提示的协议。MCP Server 提供能力，MCP Client 负责连接、发现和调用。
