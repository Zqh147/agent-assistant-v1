from .contracts import AgentResult, AgentTask


SCOPE_FIELDS = (
    "product_model",
    "hardware_version",
    "region",
)


def extract_order_scope(
    order_result: AgentResult,
    expected_order_id: str,
) -> dict[str, str]:

    # 1. 检查 Order Agent 是否成功完成任务
    if order_result.status != "success":
        raise ValueError("Order Agent did not complete successfully")

    matches = []

    # 2. 从实际工具记录中查找订单查询结果
    for item in order_result.data.get("tool_results", []):

        if item.get("tool") != "lookup_order":
            continue

        result = item.get("result")

        if not isinstance(result, dict):
            continue

        if result.get("status") != "found":
            continue

        order = result.get("order", {})
        scope = result.get("product_scope", {})

        # 3. 确认返回的是预期订单
        if order.get("order_id") != expected_order_id:
            continue

        # 4. 校验设备范围是否完整
        if not isinstance(scope, dict) or not all(
            isinstance(scope.get(key), str) and scope[key].strip()
            for key in SCOPE_FIELDS
        ):
            raise ValueError("Incomplete product scope")

        matches.append({
            key: scope[key]
            for key in SCOPE_FIELDS
        })

    if not matches:
        raise ValueError("No verified order scope found")

    # 5. 同一订单不允许出现相互矛盾的设备范围
    if any(scope != matches[0] for scope in matches):
        raise ValueError("Conflicting product scopes")

    return matches[0]


class Supervisor:
    def __init__(self, order_agent, knowledge_agent):
        self.order_agent = order_agent
        self.knowledge_agent = knowledge_agent

    def run(
        self,
        order_id: str,
        technical_question: str,
        *,
        trusted_context: dict,
    ) -> AgentResult:

        if not order_id.strip() or not technical_question.strip():
            return AgentResult(
                agent="supervisor",
                status="needs_clarification",
                summary="请提供订单号和需要咨询的设备技术问题。",
            )

        # 第一步：调用 Order Agent
        try:
            order_result = self.order_agent.run(
                f"请查询订单 {order_id} 的订单状态和设备信息。",
                trusted_context=trusted_context,
            )

        except Exception as exc:
            return AgentResult(
                agent="supervisor",
                status="error",
                summary="订单子任务执行异常，已停止后续检索。",
                error=f"order_agent_failed: {type(exc).__name__}",
            )

        if order_result.status != "success":
            return AgentResult(
                agent="supervisor",
                status=order_result.status,
                summary=order_result.summary,
                data={"order_result": order_result.to_dict()},
                error=order_result.error,
            )

        # 第二步：从真实订单工具结果中提取设备范围
        try:
            scope = extract_order_scope(
                order_result,
                expected_order_id=order_id,
            )
        except ValueError as exc:
            return AgentResult(
                agent="supervisor",
                status="no_evidence",
                summary="未取得与指定订单匹配的完整设备信息，已停止技术检索。",
                data={"order_result": order_result.to_dict()},
                error=str(exc),
            )

        # 第三步：创建 Knowledge Agent 任务
        knowledge_task = AgentTask(
            task=technical_question,
            product_scope=scope,
        )

        # 第四步：调用 Knowledge Agent
        try:
            knowledge_result = self.knowledge_agent.run(
                knowledge_task
            )

        except Exception as exc:
            return AgentResult(
                agent="supervisor",
                status="error",
                summary="知识检索子任务执行异常。",
                data={
                    "order_id": order_id,
                    "product_scope": scope,
                    "order_result": order_result.to_dict(),
                },
                error=f"knowledge_agent_failed: {type(exc).__name__}",
            )

        # 第五步：汇总两个子 Agent 的结果
        data = {
            "order_id": order_id,
            "product_scope": scope,
            "order_result": order_result.to_dict(),
            "knowledge_result": knowledge_result.to_dict(),
        }

        if knowledge_result.status != "success":
            return AgentResult(
                agent="supervisor",
                status=knowledge_result.status,
                summary=knowledge_result.summary,
                evidence=knowledge_result.evidence,
                data=data,
                error=knowledge_result.error,
            )

        return AgentResult(
            agent="supervisor",
            status="success",
            summary=(
                f"订单 {order_id} 的设备为 "
                f"{scope['product_model']} "
                f"{scope['hardware_version']} "
                f"{scope['region']}。\n\n"
                f"技术检索结果：{knowledge_result.summary}"
            ),
            evidence=knowledge_result.evidence,
            data=data,
        )
