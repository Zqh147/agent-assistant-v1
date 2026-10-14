from .graph import AgentGraph


def main() -> None:
    agent = AgentGraph()
    print("网络设备售后咨询与工单处理 Agent（输入 exit 退出）")
    print("本地演示订单/规则/工单；产品资料由项目一提供。")
    print("模型模式：", "在线" if agent.llm.online else "离线规则演示")
    while True:
        question = input("> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            print("请输入问题")
            continue
        result = agent.invoke(question)
        print(result["answer"])
        if result["tool_trace"]:
            print("[trace]", result["tool_trace"])


if __name__ == "__main__":
    main()
