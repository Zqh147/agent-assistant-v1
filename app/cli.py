from .graph import AgentGraph


def main() -> None:
    agent = AgentGraph()
    print("Agent Assistant V1.0（输入 exit 退出）")
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
