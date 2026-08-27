//! 主循环停滞检测与恢复策略。

use std::collections::VecDeque;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::{XduduError, XduduResult};
use crate::session::{ToolCallRecord, ToolCallStatus};

/// 重复失败判定阈值：同一工具连续失败次数。
pub const REPEAT_FAILURE_THRESHOLD: usize = 3;
/// 无进展判定窗口：最近多少轮模型输出被考察。
pub const NO_PROGRESS_WINDOW: usize = 4;
/// 无进展判定：助手文本低于该字符数视为极短。
pub const SHORT_OUTPUT_CHARS: usize = 16;
/// 滑动窗口大小（最近工具动作记录数）。
pub const STALL_WINDOW: usize = 8;

/// 停滞后的恢复策略。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum StalledRecoveryMode {
    /// 附加恢复提示到系统提示词并继续；达到上限仍停滞则以 `Incomplete` 收尾。
    #[default]
    Auto,
    /// 切换为暂停，向用户请求下一步。
    Ask,
    /// 完全关闭。
    Off,
}

impl StalledRecoveryMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Ask => "ask",
            Self::Off => "off",
        }
    }
}

impl FromStr for StalledRecoveryMode {
    type Err = XduduError;

    fn from_str(value: &str) -> XduduResult<Self> {
        match value {
            "auto" => Ok(Self::Auto),
            "ask" => Ok(Self::Ask),
            "off" => Ok(Self::Off),
            _ => Err(XduduError::validation(format!(
                "非法停滞恢复模式：{value}。可选值：auto、ask、off。"
            ))),
        }
    }
}

/// 一个已识别的停滞信号（滑动窗口内）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StallSignal {
    /// 已重复的失败次数（重复失败）/ 无进展轮数。
    pub repeats: usize,
    /// 涉及的工具名（用于展示与恢复提示）。
    pub tool_names: Vec<String>,
    /// 提供给模型与用户恢复提示。
    pub recovery: String,
}

/// 恢复提示中携带的最近错误详情最大字符数。
const RECOVERY_ERROR_DETAIL_CHARS: usize = 200;

/// 构建恢复提示：携带具体失败次数与最近一次错误详情，让模型能定位
/// 失败原因并调整策略，而不是拿着套话盲目重试。
fn build_recovery(tool_name: &str, repeats: usize, last_error: Option<&str>) -> String {
    let detail = last_error
        .map(|error| error.trim())
        .filter(|error| !error.is_empty())
        .map(|error| format!("最近一次失败原因：{error}。"))
        .unwrap_or_default();
    format!(
        "工具「{tool_name}」已连续失败 {repeats} 次。请不要再重试相同的参数或相同的命令写法；{detail}建议先读取相关文件确认前置条件（如路径是否存在、命令语法是否匹配当前系统），改用其它工具或方式，必要时直接向用户说明阻碍。"
    )
}

fn truncate_error_detail(error: &str) -> String {
    error.chars().take(RECOVERY_ERROR_DETAIL_CHARS).collect()
}

/// 停滞检测器：维护最近工具动作的滑动窗口。
#[derive(Debug, Default)]
pub struct StallDetector {
    records: VecDeque<ToolCallRecord>,
}

impl StallDetector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, record: ToolCallRecord) {
        self.records.push_back(record);
        while self.records.len() > STALL_WINDOW {
            self.records.pop_front();
        }
    }

    pub fn clear(&mut self) {
        self.records.clear();
    }

    /// 滑动窗口内已完成的工具记录快照。
    pub fn window(&self) -> &VecDeque<ToolCallRecord> {
        &self.records
    }

    /// 检测连续失败。连续 ≥ `REPEAT_FAILURE_THRESHOLD` 次相同工具失败（含被拒）。
    pub fn consecutive_failures(&self) -> Option<StallSignal> {
        let mut counts = Vec::<(String, usize)>::new();
        for record in self.records.iter().rev() {
            let failed = !matches!(
                record.status,
                ToolCallStatus::Succeeded | ToolCallStatus::Cancelled
            );
            if !failed {
                break;
            }
            if let Some(last) = counts.last_mut()
                && last.0 == record.tool_name
            {
                last.1 += 1;
            } else {
                counts.push((record.tool_name.clone(), 1));
            }
        }
        let (name, repeats) = counts.first()?;
        if *repeats < REPEAT_FAILURE_THRESHOLD {
            return None;
        }
        // 窗口末尾即最近一次失败：携带其错误详情，恢复提示才有纠偏价值。
        let last_error = self
            .records
            .back()
            .and_then(|record| record.error.as_deref())
            .map(truncate_error_detail);
        Some(StallSignal {
            repeats: *repeats,
            tool_names: vec![name.clone()],
            recovery: build_recovery(name, *repeats, last_error.as_deref()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::ToolCallStatus;
    use chrono::Utc;
    use serde_json::Value;

    fn failed(name: &str) -> ToolCallRecord {
        ToolCallRecord {
            id: "call".into(),
            tool_name: name.into(),
            input: Value::Null,
            output: Some(Value::Null),
            error: Some("boom".into()),
            status: ToolCallStatus::Failed,
            duration_ms: Some(1),
            started_at: Utc::now(),
            ended_at: Some(Utc::now()),
            approval: None,
        }
    }

    fn ok(name: &str) -> ToolCallRecord {
        ToolCallRecord {
            status: ToolCallStatus::Succeeded,
            ..failed(name)
        }
    }

    #[test]
    fn 连续三次相同失败触发停滞信号() {
        let mut detector = StallDetector::new();
        for _ in 0..3 {
            detector.push(failed("file_write"));
        }
        let signal = detector.consecutive_failures().unwrap();
        assert_eq!(signal.repeats, 3);
        assert_eq!(signal.tool_names, vec!["file_write"]);
        assert!(signal.recovery.contains("file_write"));
        // 提示携带真实失败次数与最近错误详情，而非泛化套话。
        assert!(signal.recovery.contains("连续失败 3 次"));
        assert!(signal.recovery.contains("boom"));
    }

    #[test]
    fn 超长错误详情在恢复提示中被截断() {
        let mut detector = StallDetector::new();
        let mut record = failed("terminal_exec");
        record.error = Some("x".repeat(1_000));
        for _ in 0..2 {
            detector.push(failed("terminal_exec"));
        }
        detector.push(record);
        let signal = detector.consecutive_failures().unwrap();
        let detail = signal
            .recovery
            .split("最近一次失败原因：")
            .nth(1)
            .and_then(|text| text.split('。').next())
            .unwrap_or("");
        assert_eq!(detail.chars().count(), RECOVERY_ERROR_DETAIL_CHARS);
    }

    #[test]
    fn 两次失败不触发() {
        let mut detector = StallDetector::new();
        detector.push(failed("file_write"));
        detector.push(failed("file_write"));
        assert!(detector.consecutive_failures().is_none());
    }

    #[test]
    fn 成功操作打断失败序列() {
        let mut detector = StallDetector::new();
        detector.push(failed("file_write"));
        detector.push(failed("file_write"));
        detector.push(ok("file_read"));
        detector.push(failed("file_write"));
        detector.push(failed("file_write"));
        assert!(detector.consecutive_failures().is_none());
    }

    #[test]
    fn 拒绝未知恢复模式() {
        assert!("unknown".parse::<StalledRecoveryMode>().is_err());
    }
}
