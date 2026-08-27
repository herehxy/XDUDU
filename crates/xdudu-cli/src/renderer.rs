//! 终端与 JSON Lines Renderer；核心事件不直接依赖 stdout。

use std::{
    io::{self, Write},
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use async_trait::async_trait;
use serde_json::json;
use xdudu_core::{AgentEvent, AgentRunResult, EventSink, XduduError, redact_text, redact_value};

use crate::markdown::{MarkdownLineKind, terminal_markdown};
use crate::ui::{TerminalTheme, tool_display_name, tool_phase_display};

pub struct ConsoleRenderer {
    json: bool,
    stream: bool,
    color: bool,
    debug_trace: bool,
    emitted_assistant: AtomicBool,
    assistant_buffer: Mutex<String>,
    output_lock: Mutex<()>,
}

impl ConsoleRenderer {
    pub fn new(json: bool, stream: bool, color: bool, debug_trace: bool) -> Self {
        Self {
            json,
            stream,
            color,
            debug_trace,
            emitted_assistant: AtomicBool::new(false),
            assistant_buffer: Mutex::new(String::new()),
            output_lock: Mutex::new(()),
        }
    }

    pub fn begin_run(&self) {
        self.emitted_assistant.store(false, Ordering::Release);
        self.assistant_buffer.lock().unwrap().clear();
    }

    pub fn finish_run(&self, result: &AgentRunResult) -> Result<(), XduduError> {
        let _guard = self.output_lock.lock().unwrap();
        if self.json {
            let value = redact_value(&json!({
                "type": "run_completed",
                "sessionId": result.session_id,
                "status": result.status,
                "turns": result.turns,
                "finalMessage": result.final_message,
                "exitCode": result.exit_code,
            }));
            println!(
                "{}",
                serde_json::to_string(&value).map_err(XduduError::from)?
            );
        } else {
            let remaining = std::mem::take(&mut *self.assistant_buffer.lock().unwrap());
            if !remaining.trim().is_empty() {
                print_markdown(&remaining, TerminalTheme::new(self.color));
                self.emitted_assistant.store(true, Ordering::Release);
            } else if !self.emitted_assistant.load(Ordering::Acquire)
                && !result.final_message.is_empty()
            {
                print_markdown(
                    &redact_text(&result.final_message),
                    TerminalTheme::new(self.color),
                );
            }
            println!();
        }
        Ok(())
    }
}

#[async_trait]
impl EventSink for ConsoleRenderer {
    async fn emit(&self, event: AgentEvent) {
        let _guard = self.output_lock.lock().unwrap();
        let trace = self
            .debug_trace
            .then(|| event.structured_trace())
            .flatten()
            .map(|value| redact_value(&value));
        if self.json {
            if let Some(trace) = trace
                && let Ok(line) = serde_json::to_string(&trace)
            {
                println!("{line}");
            }
            if matches!(event, AgentEvent::DebugTrace { .. }) {
                return;
            }
            let event = serde_json::to_value(&event)
                .map(|value| redact_value(&value))
                .and_then(|value| serde_json::to_string(&value));
            if let Ok(line) = event {
                println!("{line}");
            }
            return;
        }
        let theme = TerminalTheme::new(self.color);
        if let Some(trace) = trace
            && let Ok(line) = serde_json::to_string(&trace)
        {
            eprintln!("  {} {line}", theme.muted("trace"));
        }
        match event {
            AgentEvent::AssistantDelta { text } if self.stream => {
                let mut buffer = self.assistant_buffer.lock().unwrap();
                buffer.push_str(&redact_text(&text));
                if let Some(boundary) = ready_markdown_boundary(&buffer) {
                    let remaining = buffer.split_off(boundary);
                    let ready = std::mem::replace(&mut *buffer, remaining);
                    drop(buffer);
                    print_markdown(&ready, theme);
                    self.emitted_assistant.store(true, Ordering::Release);
                    let _ = io::stdout().flush();
                }
            }
            AgentEvent::ToolStarted { name, .. } => {
                eprintln!(
                    "\n  {} {}",
                    theme.accent("●"),
                    theme.strong(tool_display_name(&name))
                );
            }
            AgentEvent::ToolProgress {
                name: _,
                phase,
                completed,
                total,
                unit,
                message,
                ..
            } => {
                let count = match (completed, total, unit.as_deref()) {
                    (Some(completed), Some(total), Some(unit)) => {
                        format!(" {completed}/{total} {unit}")
                    }
                    (Some(completed), None, Some(unit)) => format!(" {completed} {unit}"),
                    _ => String::new(),
                };
                let message = message
                    .as_deref()
                    .map(redact_text)
                    .map(|message| format!("：{message}"))
                    .unwrap_or_default();
                eprintln!(
                    "  {} {}{count}{message}",
                    theme.muted("│"),
                    tool_phase_display(&phase)
                );
            }
            AgentEvent::ToolFinished { name, result, .. } => {
                let marker = if result.success {
                    theme.success("✓")
                } else {
                    theme.danger("✗")
                };
                eprintln!(
                    "  {} {marker} {} {}",
                    theme.muted("└"),
                    tool_display_name(&name),
                    theme.muted(&format!("{} ms", result.duration_ms))
                );
            }
            AgentEvent::Warning { message, .. } => {
                eprintln!("  {} {}", theme.warning("⚠ 警告"), redact_text(&message));
            }
            AgentEvent::Continuing { note, .. } => {
                eprintln!("  {} {}", theme.accent("↻"), redact_text(&note));
            }
            AgentEvent::CompactionApplied {
                layer,
                saved_tokens,
            } => {
                eprintln!(
                    "  {} 上下文压缩（{layer}）：节省约 {saved_tokens} tokens",
                    theme.accent("↻")
                );
            }
            AgentEvent::ReasoningDelta { .. } => {}
            AgentEvent::PlanStarted { revision, .. } => {
                eprintln!("  {} 开始执行计划 revision {revision}", theme.accent("◆"));
            }
            AgentEvent::PlanStepStarted { title, attempt, .. } => {
                eprintln!(
                    "  {} {} {}",
                    theme.accent("→"),
                    redact_text(&title),
                    theme.muted(&format!("（第 {attempt} 次尝试）"))
                );
            }
            AgentEvent::PlanStepCompleted {
                summary, evidence, ..
            } => {
                eprintln!("  {} {}", theme.success("✓"), redact_text(&summary));
                for item in evidence {
                    eprintln!(
                        "    {} {}",
                        theme.muted(&format!("证据 {}：", item.criterion_index)),
                        redact_text(&item.evidence)
                    );
                }
            }
            AgentEvent::PlanStepFailed { error, .. } => {
                eprintln!("  {} {}", theme.danger("✗"), redact_text(&error));
            }
            AgentEvent::PlanPaused { reason, .. } => {
                eprintln!(
                    "  {} {}",
                    theme.warning("Ⅱ 计划已暂停"),
                    redact_text(&reason)
                );
            }
            AgentEvent::PlanCompleted { .. } => {
                eprintln!("  {} 计划全部完成", theme.success("✓"));
            }
            AgentEvent::SubagentGraphStarted {
                total,
                max_concurrency,
                ..
            } => {
                eprintln!(
                    "  {} 子代理任务图：{total} 个节点，并发上限 {max_concurrency}",
                    theme.accent("◆")
                );
            }
            AgentEvent::SubagentGraphNodeStarted {
                node_id, agent_id, ..
            } => {
                eprintln!("  {} {node_id} · {agent_id}", theme.accent("→"));
            }
            AgentEvent::SubagentGraphNodeFinished {
                node_id,
                status,
                duration_ms,
                ..
            } => {
                let marker = if status == "succeeded" {
                    theme.success("✓")
                } else {
                    theme.warning("•")
                };
                eprintln!("  {marker} {node_id} · {status} · {duration_ms} ms");
            }
            AgentEvent::SubagentGraphFinished {
                success,
                succeeded,
                failed,
                blocked,
                cancelled,
                ..
            } => {
                let marker = if success {
                    theme.success("✓")
                } else {
                    theme.warning("Ⅱ")
                };
                eprintln!(
                    "  {marker} 任务图：成功 {succeeded} · 失败 {failed} · 阻塞 {blocked} · 取消 {cancelled}"
                );
            }
            _ => {}
        }
    }
}

fn ready_markdown_boundary(source: &str) -> Option<usize> {
    let mut offset = 0;
    let mut boundary = None;
    let mut fence: Option<&str> = None;
    for line in source.split_inclusive('\n') {
        offset += line.len();
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            fence = if fence == Some("```") {
                None
            } else if fence.is_none() {
                Some("```")
            } else {
                fence
            };
            if fence.is_none() {
                boundary = Some(offset);
            }
        } else if trimmed.starts_with("~~~") {
            fence = if fence == Some("~~~") {
                None
            } else if fence.is_none() {
                Some("~~~")
            } else {
                fence
            };
            if fence.is_none() {
                boundary = Some(offset);
            }
        } else if trimmed.is_empty() && fence.is_none() {
            boundary = Some(offset);
        }
    }
    boundary
}

fn print_markdown(source: &str, theme: TerminalTheme) {
    for line in terminal_markdown(source) {
        match line.kind {
            MarkdownLineKind::Heading => println!("{}", theme.strong(&line.text)),
            MarkdownLineKind::Code => println!("{}", theme.muted(&line.text)),
            MarkdownLineKind::DiffAdd => println!("{}", theme.success(&line.text)),
            MarkdownLineKind::DiffRemove => println!("{}", theme.danger(&line.text)),
            MarkdownLineKind::DiffContext => println!("{}", theme.muted(&line.text)),
            MarkdownLineKind::Body => println!("{}", line.text),
        }
    }
}
