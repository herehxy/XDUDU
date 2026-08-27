//! Agent 领域事件；核心只发布事件，终端表现由调用方决定。

use async_trait::async_trait;
use serde::Serialize;

use crate::{
    plan::CompletionEvidence, provider::TokenUsage, session::AgentLoopState, tools::ToolResult,
};
use serde_json::{Value, json};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    StateChanged {
        state: AgentLoopState,
    },
    AssistantDelta {
        text: String,
    },
    ToolStarted {
        call_id: String,
        name: String,
    },
    ToolProgress {
        call_id: String,
        name: String,
        phase: String,
        completed: Option<u64>,
        total: Option<u64>,
        unit: Option<String>,
        message: Option<String>,
    },
    ToolFinished {
        call_id: String,
        name: String,
        result: ToolResult,
    },
    UsageUpdated {
        usage: TokenUsage,
    },
    /// 压缩/投影指标：layer 为 L1-projection / L2-deterministic / L3-llm。
    CompactionApplied {
        layer: String,
        saved_tokens: u64,
    },
    /// 思维链增量：渲染层折叠为浅灰摘要，不与正文混排。
    ReasoningDelta {
        text: String,
    },
    Warning {
        code: String,
        message: String,
    },
    /// 只包含运行时元数据，不包含模型隐藏推理、助手正文或工具输入输出。
    DebugTrace {
        phase: String,
        summary: String,
        details: Value,
    },
    PlanStarted {
        plan_id: Uuid,
        revision: u32,
    },
    PlanStepStarted {
        plan_id: Uuid,
        step_id: Uuid,
        title: String,
        attempt: u32,
    },
    PlanStepCompleted {
        plan_id: Uuid,
        step_id: Uuid,
        summary: String,
        evidence: Vec<CompletionEvidence>,
    },
    PlanStepFailed {
        plan_id: Uuid,
        step_id: Uuid,
        error: String,
    },
    PlanPaused {
        plan_id: Uuid,
        reason: String,
    },
    PlanCompleted {
        plan_id: Uuid,
    },
    /// 检测到主循环停滞：重复失败或无进展。
    StalledRecovery {
        repeats: usize,
        tool_names: Vec<String>,
        recovery: String,
    },
    /// 段轮次预算用尽，任务自动续跑进入下一段。
    Continuing {
        segment: u32,
        note: String,
    },
    /// 技能加载成功（索引与正文注入当前轮系统提示词）。
    SkillLoaded {
        name: String,
    },
    /// 子代理开始执行（隔离上下文，不写父会话历史）。
    SubagentStarted {
        agent_id: String,
        prompt: String,
    },
    /// 子代理执行结束（结果转为普通 ToolResult）。
    SubagentFinished {
        agent_id: String,
        result: ToolResult,
    },
    /// 子代理任务图完成预检并开始调度。
    SubagentGraphStarted {
        graph_id: Uuid,
        total: usize,
        max_concurrency: usize,
    },
    /// 图节点依赖已满足并开始执行。
    SubagentGraphNodeStarted {
        graph_id: Uuid,
        node_id: String,
        agent_id: String,
    },
    /// 图节点进入终态；不携带节点正文或工具结果。
    SubagentGraphNodeFinished {
        graph_id: Uuid,
        node_id: String,
        agent_id: String,
        status: String,
        duration_ms: u64,
    },
    /// 整张任务图结束。
    SubagentGraphFinished {
        graph_id: Uuid,
        success: bool,
        succeeded: usize,
        failed: usize,
        blocked: usize,
        cancelled: usize,
    },
}

impl AgentEvent {
    /// 将领域事件映射成可公开的结构化调试轨迹。
    ///
    /// 映射刻意排除助手正文、工具参数、工具输出和证据正文；Renderer
    /// 仍会对整个值执行统一脱敏，形成第二道保护。
    pub fn structured_trace(&self) -> Option<Value> {
        if let Self::DebugTrace {
            phase,
            summary,
            details,
        } = self
        {
            return Some(json!({
                "type": "debug_trace",
                "phase": phase,
                "summary": summary,
                "details": details,
            }));
        }
        let (phase, summary, details) = match self {
            Self::StateChanged { state } => {
                ("agent_state", "Agent 运行状态变化", json!({"state": state}))
            }
            Self::AssistantDelta { .. } | Self::ReasoningDelta { .. } => return None,
            Self::ToolStarted { call_id, name } => (
                "tool_started",
                "工具调用开始",
                json!({"callId": call_id, "name": name}),
            ),
            Self::ToolProgress {
                call_id,
                name,
                phase,
                completed,
                total,
                unit,
                ..
            } => (
                "tool_progress",
                "工具报告执行进度",
                json!({
                    "callId": call_id,
                    "name": name,
                    "toolPhase": phase,
                    "completed": completed,
                    "total": total,
                    "unit": unit,
                }),
            ),
            Self::ToolFinished {
                call_id,
                name,
                result,
            } => (
                "tool_finished",
                "工具调用结束",
                json!({
                    "callId": call_id,
                    "name": name,
                    "success": result.success,
                    "durationMs": result.duration_ms,
                    "errorCode": result.error.as_ref().map(|error| error.code.as_str()),
                    "approvalScope": result.approval.as_ref().map(|approval| approval.scope),
                }),
            ),
            Self::UsageUpdated { usage } => (
                "usage",
                "模型用量更新",
                json!({
                    "inputTokens": usage.input_tokens,
                    "outputTokens": usage.output_tokens,
                    "cacheReadTokens": usage.cache_read_tokens,
                    "cacheWriteTokens": usage.cache_write_tokens,
                }),
            ),
            Self::Warning { code, .. } => ("warning", "运行时警告", json!({"code": code})),
            Self::DebugTrace { .. } => unreachable!("已在映射前处理调试轨迹事件"),
            Self::PlanStarted { plan_id, revision } => (
                "plan_started",
                "计划开始执行",
                json!({"planId": plan_id, "revision": revision}),
            ),
            Self::PlanStepStarted {
                plan_id,
                step_id,
                attempt,
                ..
            } => (
                "plan_step_started",
                "计划步骤开始",
                json!({"planId": plan_id, "stepId": step_id, "attempt": attempt}),
            ),
            Self::PlanStepCompleted {
                plan_id,
                step_id,
                evidence,
                ..
            } => (
                "plan_step_completed",
                "计划步骤完成",
                json!({
                    "planId": plan_id,
                    "stepId": step_id,
                    "evidenceCount": evidence.len(),
                    "criterionIndexes": evidence.iter().map(|item| item.criterion_index).collect::<Vec<_>>(),
                }),
            ),
            Self::PlanStepFailed {
                plan_id, step_id, ..
            } => (
                "plan_step_failed",
                "计划步骤失败",
                json!({"planId": plan_id, "stepId": step_id}),
            ),
            Self::PlanPaused { plan_id, .. } => {
                ("plan_paused", "计划暂停", json!({"planId": plan_id}))
            }
            Self::PlanCompleted { plan_id } => {
                ("plan_completed", "计划完成", json!({"planId": plan_id}))
            }
            Self::CompactionApplied {
                layer,
                saved_tokens,
            } => (
                "compaction_applied",
                "上下文压缩/投影已应用",
                json!({"layer": layer, "savedTokens": saved_tokens}),
            ),
            Self::StalledRecovery {
                repeats,
                tool_names,
                ..
            } => (
                "stalled_recovery",
                "检测到停滞并注入恢复指令",
                json!({
                    "repeats": repeats,
                    "toolNames": tool_names,
                }),
            ),
            Self::Continuing { segment, .. } => (
                "continuing",
                "段轮次预算用尽，自动续跑",
                json!({"segment": segment}),
            ),
            Self::SkillLoaded { name } => ("skill_loaded", "技能加载成功", json!({"name": name})),
            Self::SubagentStarted { agent_id, prompt } => (
                "subagent_started",
                "子代理开始执行",
                json!({"agentId": agent_id, "promptChars": prompt.chars().count()}),
            ),
            Self::SubagentFinished { agent_id, result } => (
                "subagent_finished",
                "子代理执行结束",
                json!({
                    "agentId": agent_id,
                    "success": result.success,
                    "errorCode": result.error.as_ref().map(|error| error.code.as_str()),
                }),
            ),
            Self::SubagentGraphStarted {
                graph_id,
                total,
                max_concurrency,
            } => (
                "subagent_graph_started",
                "子代理任务图开始调度",
                json!({
                    "graphId": graph_id,
                    "total": total,
                    "maxConcurrency": max_concurrency,
                }),
            ),
            Self::SubagentGraphNodeStarted {
                graph_id,
                node_id,
                agent_id,
            } => (
                "subagent_graph_node_started",
                "子代理任务图节点开始",
                json!({
                    "graphId": graph_id,
                    "nodeId": node_id,
                    "agentId": agent_id,
                }),
            ),
            Self::SubagentGraphNodeFinished {
                graph_id,
                node_id,
                agent_id,
                status,
                duration_ms,
            } => (
                "subagent_graph_node_finished",
                "子代理任务图节点结束",
                json!({
                    "graphId": graph_id,
                    "nodeId": node_id,
                    "agentId": agent_id,
                    "status": status,
                    "durationMs": duration_ms,
                }),
            ),
            Self::SubagentGraphFinished {
                graph_id,
                success,
                succeeded,
                failed,
                blocked,
                cancelled,
            } => (
                "subagent_graph_finished",
                "子代理任务图结束",
                json!({
                    "graphId": graph_id,
                    "success": success,
                    "succeeded": succeeded,
                    "failed": failed,
                    "blocked": blocked,
                    "cancelled": cancelled,
                }),
            ),
        };
        Some(json!({
            "type": "debug_trace",
            "phase": phase,
            "summary": summary,
            "details": details,
        }))
    }
}

#[async_trait]
pub trait EventSink: Send + Sync {
    async fn emit(&self, event: AgentEvent);
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NoopEventSink;

#[async_trait]
impl EventSink for NoopEventSink {
    async fn emit(&self, _event: AgentEvent) {}
}

pub(crate) async fn emit(sink: Option<&dyn EventSink>, event: AgentEvent) {
    if let Some(sink) = sink {
        sink.emit(event).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn 事件序列化为稳定的_json_类型() {
        let value = serde_json::to_value(AgentEvent::AssistantDelta {
            text: "你好".into(),
        })
        .unwrap();
        assert_eq!(value["type"], "assistant_delta");
        assert_eq!(value["text"], "你好");
    }

    #[test]
    fn 工具进度序列化为稳定事件() {
        let value = serde_json::to_value(AgentEvent::ToolProgress {
            call_id: "call-1".into(),
            name: "search_text".into(),
            phase: "scanning".into(),
            completed: Some(1000),
            total: None,
            unit: Some("files".into()),
            message: None,
        })
        .unwrap();
        assert_eq!(value["type"], "tool_progress");
        assert_eq!(value["call_id"], "call-1");
        assert_eq!(value["completed"], 1000);
        assert_eq!(value["unit"], "files");
    }

    #[test]
    fn 子代理任务图事件不携带提示词或节点结果() {
        let value = serde_json::to_value(AgentEvent::SubagentGraphNodeFinished {
            graph_id: Uuid::nil(),
            node_id: "inspect".into(),
            agent_id: "explore".into(),
            status: "succeeded".into(),
            duration_ms: 42,
        })
        .unwrap();
        assert_eq!(value["type"], "subagent_graph_node_finished");
        assert_eq!(value["node_id"], "inspect");
        assert!(value.get("prompt").is_none());
        assert!(value.get("result").is_none());
    }

    #[test]
    fn 结构化轨迹不包含工具输出或错误正文() {
        let mut result = ToolResult::success(
            json!({"apiKey":"sk-should-never-enter-trace"}),
            Utc::now(),
            json!({}),
        );
        result.error = None;
        let trace = AgentEvent::ToolFinished {
            call_id: "call-1".into(),
            name: "file_read".into(),
            result,
        }
        .structured_trace()
        .unwrap();
        let raw = serde_json::to_string(&trace).unwrap();
        assert_eq!(trace["type"], "debug_trace");
        assert_eq!(trace["details"]["success"], true);
        assert!(!raw.contains("sk-should-never-enter-trace"));
        assert!(!raw.contains("apiKey"));
    }

    #[test]
    fn 步骤完成事件携带公开证据而调试轨迹只保留索引() {
        let event = AgentEvent::PlanStepCompleted {
            plan_id: Uuid::new_v4(),
            step_id: Uuid::new_v4(),
            summary: "完成".into(),
            evidence: vec![CompletionEvidence {
                criterion_index: 1,
                evidence: "测试通过".into(),
            }],
        };
        let serialized = serde_json::to_value(&event).unwrap();
        assert_eq!(serialized["evidence"][0]["evidence"], "测试通过");
        let trace = event.structured_trace().unwrap();
        assert_eq!(trace["details"]["evidenceCount"], 1);
        assert_eq!(trace["details"]["criterionIndexes"][0], 1);
        assert!(!serde_json::to_string(&trace).unwrap().contains("测试通过"));
    }
}
