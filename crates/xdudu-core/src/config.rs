//! 分层配置加载、来源追踪与安全写入。

use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use toml::Value;
use uuid::Uuid;

use crate::{
    approval::ApprovalMode,
    error::{ErrorKind, XduduError, XduduResult},
    permission::PermissionMode,
    stall::StalledRecoveryMode,
    subagent::{AgentProfile, ProfileMode, validate_profile},
};

const DEFAULT_ANTHROPIC_MODEL: &str = "claude-sonnet-4-5-20250929";
const DEFAULT_DEEPSEEK_MODEL: &str = "deepseek-v4-flash";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigSource {
    Default,
    User,
    Project,
    Environment,
    Cli,
}

impl ConfigSource {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::User => "user",
            Self::Project => "project",
            Self::Environment => "environment",
            Self::Cli => "cli",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderConfig {
    pub name: String,
    pub model: String,
    pub base_url: Option<String>,
    pub timeout_seconds: u64,
    pub max_attempts: u32,
    pub retry_base_ms: u64,
    pub min_request_interval_ms: u64,
    /// 模型请求采样温度，默认 0.2。
    pub temperature: f32,
    /// 单次模型请求最大输出 Token，默认 8192。
    pub max_output_tokens: u32,
    /// 是否启用内部思考闭环（推理内容持久化并回传，不进入公开输出）。
    pub reasoning: bool,
    /// 模型上下文窗口上限（Token 能力声明）：0 表示按内置模型表推导，
    /// openai-compatible 等未知模型可用此覆盖。上下文压缩预算由此推导。
    pub max_context_tokens: u32,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            name: "deepseek".into(),
            model: DEFAULT_DEEPSEEK_MODEL.into(),
            base_url: None,
            timeout_seconds: 180,
            max_attempts: 3,
            retry_base_ms: 500,
            min_request_interval_ms: 0,
            temperature: 0.2,
            max_output_tokens: 8192,
            reasoning: false,
            max_context_tokens: 0,
        }
    }
}

impl ProviderConfig {
    /// 模型上下文窗口（Token）：显式 `max_context_tokens` 优先，否则按内置表。
    pub fn context_window(&self) -> XduduResult<usize> {
        let window = if self.max_context_tokens == 0 {
            model_context_window(&self.name, &self.model)
        } else {
            self.max_context_tokens
        } as usize;
        if !(16_384..=1_048_576).contains(&window) {
            return Err(config_error(
                "provider.max_context_tokens 必须是 16384 到 1048576 之间的整数。",
            ));
        }
        Ok(window)
    }

    /// 软阈值（对齐 Codex auto_compact_token_limit）：窗口的 90%，真实用量
    /// 达到即触发自动上下文压缩；`max_context_tokens` 显式设置时基于该窗口计算。
    pub fn context_budget(&self) -> XduduResult<usize> {
        Ok((self.context_window()? * 9 / 10).max(8_000))
    }

    /// 硬顶（对齐 Codex effective_context_window_percent）：窗口的 95%，
    /// 真实用量达到时在 L3 之外强制 L2 确定性截断，防止突破模型上下文窗口。
    pub fn context_hard_limit(&self) -> XduduResult<usize> {
        Ok((self.context_window()? * 95 / 100).max(8_000))
    }
}

/// 内置模型上下文窗口表（Token，来源：各厂商公开规格）。
/// deepseek-chat/deepseek-reasoner 及 deepseek-v4 系列为 128K；
/// Anthropic Claude 系列 200K；openai-compatible 未知模型保守 64K。
fn model_context_window(provider: &str, model: &str) -> u32 {
    match provider {
        "deepseek" if model.starts_with("deepseek") => 128_000,
        "anthropic" if model.starts_with("claude") => 200_000,
        _ => 64_000,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentConfig {
    pub max_turns: u32,
    pub permission: String,
    pub approval: String,
    /// 停滞后的恢复策略：auto | ask | off。
    pub stalled_recovery: String,
    /// auto 模式下允许附加恢复提示继续的最大次数。
    pub stalled_max_recovery: u32,
    /// `terminal_exec` 三档前缀规则：deny > allow > ask。
    pub commands: CommandRules,
    /// 技能加载策略：allow | ask | deny（默认 allow）。
    pub skills: String,
    /// 段轮次预算用尽时自动续跑：开启后打满 max_turns 不终止任务，
    /// 注入续跑指令继续下一段；只有总预算或用户中断能停止。
    pub auto_continue: bool,
    /// 总轮次预算硬上限：所有段（含续跑）累计；达到后注入收尾指令交接。
    pub max_total_turns: u32,
    /// 自定义 Agent 档案（不含内置；最终由调用方与内置档案合并）。
    pub profiles: Vec<AgentProfile>,
}

/// `terminal_exec` 命令三档策略的前缀规则表。
///
/// 每条规则为「可执行名」或「可执行名 + 首个参数」（如 `cargo check`）。
/// 匹配顺序固定为 deny > allow > ask：命中 deny 立即拒绝；命中 allow
/// 直接执行；其余进入审批门。项目配置只能追加 deny 与 ask，不能追加
/// allow（项目不可信原则）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CommandRules {
    pub allow: Vec<String>,
    pub ask: Vec<String>,
    pub deny: Vec<String>,
}

impl CommandRules {
    /// auto-safe 内置默认 allow 白名单（对齐主流编码 Agent 体验）。
    fn default_allow() -> Vec<String> {
        [
            "pwd",
            "echo",
            "ls",
            "git status",
            "git log",
            "git diff",
            "git show",
            "git branch",
            "git stash",
            "git rev-parse",
            "cargo check",
            "cargo build",
            "cargo test",
            "cargo fmt",
            "cargo clippy",
            "npm run",
            "npm test",
            "npm run lint",
            "python3 -m pytest",
            "python3 -m unittest",
            "make -n",
            "gofmt -l",
            "go test",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    }
}

impl Default for CommandRules {
    fn default() -> Self {
        Self {
            allow: Self::default_allow(),
            ask: Vec::new(),
            deny: ["sudo", "mkfs", "rm"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
        }
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_turns: 25,
            permission: "auto-safe".into(),
            approval: "ask".into(),
            stalled_recovery: "auto".into(),
            stalled_max_recovery: 3,
            commands: CommandRules::default(),
            skills: "allow".into(),
            auto_continue: true,
            max_total_turns: 200,
            profiles: Vec::new(),
        }
    }
}
impl AgentConfig {
    pub fn permission_mode(&self) -> XduduResult<PermissionMode> {
        self.permission.parse()
    }

    pub fn approval_mode(&self) -> XduduResult<ApprovalMode> {
        self.approval.parse()
    }

    pub fn stalled_recovery_mode(&self) -> XduduResult<StalledRecoveryMode> {
        self.stalled_recovery.parse()
    }

    /// 技能加载策略：allow | ask | deny。
    pub fn skills_mode(&self) -> XduduResult<SkillMode> {
        match self.skills.as_str() {
            "allow" => Ok(SkillMode::Allow),
            "ask" => Ok(SkillMode::Ask),
            "deny" => Ok(SkillMode::Deny),
            _ => Err(config_error("agent.skills 只能是 allow、ask 或 deny。")),
        }
    }

    /// 总轮次预算：1–1,000，默认 200。
    pub fn max_total_turns(&self) -> XduduResult<u32> {
        if !(1..=1_000).contains(&self.max_total_turns) {
            return Err(config_error(
                "agent.max_total_turns 必须是 1 到 1000 之间的整数。",
            ));
        }
        Ok(self.max_total_turns)
    }
}

/// 技能加载策略。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkillMode {
    /// 直接加载。
    Allow,
    /// 加载前进入审批门。
    Ask,
    /// 索引不暴露，加载被拒绝。
    Deny,
}

impl SkillMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Allow => "allow",
            Self::Ask => "ask",
            Self::Deny => "deny",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OutputConfig {
    pub json: bool,
    pub no_stream: bool,
    pub color: bool,
    pub debug_trace: bool,
    /// TUI 配色主题：dark | light | auto（默认按终端背景探测）。
    pub theme: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AppConfig {
    pub provider: ProviderConfig,
    pub agent: AgentConfig,
    pub output: OutputConfig,
    pub telemetry: TelemetryConfig,
    pub memory: MemoryConfig,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryConfig {
    /// 任务完成后是否由 Agent 自动提炼长期记忆；默认开启，可由用户关闭。
    pub suggest_enabled: bool,
    /// 记忆注入的最大条数（召回后排序去重，默认 8）。
    pub top_k: usize,
    /// 记忆注入的 Token 预算（默认 1500）。
    pub injection_token_budget: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TelemetryConfig {
    /// 遥测默认关闭；XDUDU 当前不发送任何数据，未来若引入必须保持
    /// 默认关闭并显式授权。
    pub enabled: bool,
}

#[derive(Debug, Clone)]
pub struct ResolvedConfig {
    pub config: AppConfig,
    pub sources: BTreeMap<String, ConfigSource>,
    pub user_path: PathBuf,
    pub project_path: PathBuf,
}

impl ResolvedConfig {
    pub fn source(&self, key: &str) -> Option<ConfigSource> {
        self.sources.get(key).copied()
    }
}

#[derive(Debug, Clone, Default)]
pub struct ConfigOverrides {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub base_url: Option<String>,
    pub max_turns: Option<u32>,
    pub permission: Option<String>,
    pub approval: Option<String>,
    pub json: Option<bool>,
    pub no_stream: Option<bool>,
    pub color: Option<bool>,
    pub debug_trace: Option<bool>,
    pub telemetry_enabled: Option<bool>,
    pub memory_suggest_enabled: Option<bool>,
    pub temperature: Option<f32>,
    pub max_output_tokens: Option<u32>,
    pub reasoning: Option<bool>,
    pub stalled_recovery: Option<String>,
    pub stalled_max_recovery: Option<u32>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct FileConfig {
    #[serde(default)]
    provider: FileProvider,
    #[serde(default)]
    agent: FileAgent,
    #[serde(default)]
    output: FileOutput,
    #[serde(default)]
    telemetry: FileTelemetry,
    #[serde(default)]
    memory: FileMemory,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct FileMemory {
    suggest_enabled: Option<bool>,
    top_k: Option<usize>,
    injection_token_budget: Option<usize>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct FileTelemetry {
    enabled: Option<bool>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct FileProvider {
    name: Option<String>,
    model: Option<String>,
    base_url: Option<String>,
    timeout_seconds: Option<u64>,
    max_attempts: Option<u32>,
    retry_base_ms: Option<u64>,
    min_request_interval_ms: Option<u64>,
    temperature: Option<f32>,
    max_output_tokens: Option<u32>,
    reasoning: Option<bool>,
    max_context_tokens: Option<u32>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct FileAgent {
    max_turns: Option<u32>,
    permission: Option<String>,
    approval: Option<String>,
    stalled_recovery: Option<String>,
    stalled_max_recovery: Option<u32>,
    auto_continue: Option<bool>,
    max_total_turns: Option<u32>,
    commands: Option<FileCommandRules>,
    skills: Option<String>,
    profiles: Option<std::collections::BTreeMap<String, FileProfile>>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct FileProfile {
    description: Option<String>,
    mode: Option<String>,
    model: Option<String>,
    permission: Option<String>,
    allowed_tools: Option<Vec<String>>,
    max_turns: Option<u32>,
    system_extra: Option<String>,
}

impl FileProfile {
    fn into_profile(self, id: String) -> XduduResult<AgentProfile> {
        let mode = match self.mode.as_deref().unwrap_or("subagent") {
            "primary" => ProfileMode::Primary,
            "subagent" => ProfileMode::Subagent,
            "all" => ProfileMode::All,
            other => {
                return Err(config_error(format!(
                    "档案 {id} 的 mode“{other}”无效：primary、subagent 或 all。"
                )));
            }
        };
        let permission = match self.permission.as_deref().unwrap_or("read-only") {
            "read-only" => PermissionMode::ReadOnly,
            "auto-safe" => PermissionMode::AutoSafe,
            "full-access" => PermissionMode::FullAccess,
            other => {
                return Err(config_error(format!(
                    "档案 {id} 的 permission“{other}”无效：read-only、auto-safe 或 full-access。"
                )));
            }
        };
        let max_turns = self.max_turns.unwrap_or(8);
        let profile = AgentProfile {
            id,
            description: self.description.unwrap_or_default(),
            mode,
            model: self.model,
            permission,
            allowed_tools: self.allowed_tools,
            max_turns,
            system_extra: self.system_extra,
        };
        validate_profile(&profile)?;
        Ok(profile)
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct FileCommandRules {
    allow: Option<Vec<String>>,
    ask: Option<Vec<String>>,
    deny: Option<Vec<String>>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct FileOutput {
    json: Option<bool>,
    no_stream: Option<bool>,
    color: Option<bool>,
    debug_trace: Option<bool>,
    theme: Option<String>,
}

fn config_error(message: impl Into<String>) -> XduduError {
    XduduError::new(ErrorKind::ConfigError, message)
}

/// 命令规则格式校验：「可执行名」加 0..=3 个参数。前缀匹配语义：
/// 匹配时只比较可执行名与首个参数，多余参数用于描述更具体的命令。
fn valid_command_rule(rule: &str) -> bool {
    let mut parts = rule.split_whitespace();
    let Some(executable) = parts.next() else {
        return false;
    };
    let executable_valid = !executable.is_empty()
        && executable.len() <= 128
        && executable
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._+-".contains(&byte));
    if !executable_valid {
        return false;
    }
    let mut count = 0;
    for argument in parts {
        count += 1;
        let argument_valid = !argument.is_empty()
            && argument.len() <= 64
            && argument
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"._-+/".contains(&byte));
        if count > 3 || !argument_valid {
            return false;
        }
    }
    true
}

fn user_config_path() -> XduduResult<PathBuf> {
    if let Some(path) = env::var_os("XDUDU_CONFIG_HOME") {
        return Ok(PathBuf::from(path).join("config.toml"));
    }
    if let Some(path) = env::var_os("XDG_CONFIG_HOME") {
        return Ok(PathBuf::from(path).join("xdudu/config.toml"));
    }
    if cfg!(windows)
        && let Some(path) = env::var_os("APPDATA")
    {
        return Ok(PathBuf::from(path).join("xdudu/config.toml"));
    }
    let home =
        env::var_os("HOME").ok_or_else(|| config_error("无法确定用户配置目录：HOME 未设置。"))?;
    Ok(PathBuf::from(home).join(".config/xdudu/config.toml"))
}

fn legacy_user_config_path() -> XduduResult<PathBuf> {
    if let Some(path) = env::var_os("XYCLI_CONFIG_HOME") {
        return Ok(PathBuf::from(path).join("config.toml"));
    }
    if let Some(path) = env::var_os("XDG_CONFIG_HOME") {
        return Ok(PathBuf::from(path).join("xycli/config.toml"));
    }
    if cfg!(windows)
        && let Some(path) = env::var_os("APPDATA")
    {
        return Ok(PathBuf::from(path).join("xycli/config.toml"));
    }
    let home =
        env::var_os("HOME").ok_or_else(|| config_error("无法确定用户配置目录：HOME 未设置。"))?;
    Ok(PathBuf::from(home).join(".config/xycli/config.toml"))
}

pub fn config_paths(cwd: &Path) -> XduduResult<(PathBuf, PathBuf)> {
    Ok((user_config_path()?, cwd.join(".xdudu/config.toml")))
}

pub fn approval_rules_path() -> XduduResult<PathBuf> {
    let config = user_config_path()?;
    let parent = config
        .parent()
        .ok_or_else(|| config_error("用户配置路径缺少父目录。"))?;
    Ok(parent.join("approval-rules.json"))
}

fn contains_secret_key(value: &Value) -> bool {
    match value {
        Value::Table(table) => table.iter().any(|(key, value)| {
            matches!(
                key.to_ascii_lowercase().replace('-', "_").as_str(),
                "api_key" | "apikey" | "token" | "secret"
            ) || contains_secret_key(value)
        }),
        Value::Array(values) => values.iter().any(contains_secret_key),
        _ => false,
    }
}

fn read_file(path: &Path) -> XduduResult<Option<FileConfig>> {
    let raw = match fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(config_error(format!(
                "无法读取配置 {}：{error}",
                path.display()
            )));
        }
    };
    let value: Value = toml::from_str(&raw)
        .map_err(|error| config_error(format!("配置 {} 格式无效：{error}", path.display())))?;
    if contains_secret_key(&value) {
        return Err(config_error(format!(
            "配置 {} 包含明文密钥字段；请使用环境变量或 xdudu auth login。",
            path.display()
        )));
    }
    value
        .try_into()
        .map(Some)
        .map_err(|error| config_error(format!("配置 {} 内容无效：{error}", path.display())))
}

fn set<T: Clone>(
    value: &mut T,
    incoming: &Option<T>,
    key: &str,
    source: ConfigSource,
    sources: &mut BTreeMap<String, ConfigSource>,
) {
    if let Some(incoming) = incoming {
        value.clone_from(incoming);
        sources.insert(key.to_owned(), source);
    }
}

fn apply_file(
    config: &mut AppConfig,
    file: &FileConfig,
    source: ConfigSource,
    sources: &mut BTreeMap<String, ConfigSource>,
) -> XduduResult<()> {
    set(
        &mut config.provider.name,
        &file.provider.name,
        "provider.name",
        source,
        sources,
    );
    set(
        &mut config.provider.model,
        &file.provider.model,
        "provider.model",
        source,
        sources,
    );
    set(
        &mut config.provider.base_url,
        &file.provider.base_url.clone().map(Some),
        "provider.base_url",
        source,
        sources,
    );
    set(
        &mut config.provider.timeout_seconds,
        &file.provider.timeout_seconds,
        "provider.timeout_seconds",
        source,
        sources,
    );
    set(
        &mut config.provider.max_attempts,
        &file.provider.max_attempts,
        "provider.max_attempts",
        source,
        sources,
    );
    set(
        &mut config.provider.retry_base_ms,
        &file.provider.retry_base_ms,
        "provider.retry_base_ms",
        source,
        sources,
    );
    set(
        &mut config.provider.min_request_interval_ms,
        &file.provider.min_request_interval_ms,
        "provider.min_request_interval_ms",
        source,
        sources,
    );
    set(
        &mut config.provider.temperature,
        &file.provider.temperature,
        "provider.temperature",
        source,
        sources,
    );
    set(
        &mut config.provider.max_output_tokens,
        &file.provider.max_output_tokens,
        "provider.max_output_tokens",
        source,
        sources,
    );
    set(
        &mut config.provider.reasoning,
        &file.provider.reasoning,
        "provider.reasoning",
        source,
        sources,
    );
    set(
        &mut config.provider.max_context_tokens,
        &file.provider.max_context_tokens,
        "provider.max_context_tokens",
        source,
        sources,
    );
    set(
        &mut config.agent.max_turns,
        &file.agent.max_turns,
        "agent.max_turns",
        source,
        sources,
    );
    set(
        &mut config.agent.permission,
        &file.agent.permission,
        "agent.permission",
        source,
        sources,
    );
    set(
        &mut config.agent.approval,
        &file.agent.approval,
        "agent.approval",
        source,
        sources,
    );
    set(
        &mut config.agent.stalled_recovery,
        &file.agent.stalled_recovery,
        "agent.stalled_recovery",
        source,
        sources,
    );
    set(
        &mut config.agent.stalled_max_recovery,
        &file.agent.stalled_max_recovery,
        "agent.stalled_max_recovery",
        source,
        sources,
    );
    set(
        &mut config.agent.skills,
        &file.agent.skills,
        "agent.skills",
        source,
        sources,
    );
    set(
        &mut config.agent.auto_continue,
        &file.agent.auto_continue,
        "agent.auto_continue",
        source,
        sources,
    );
    set(
        &mut config.agent.max_total_turns,
        &file.agent.max_total_turns,
        "agent.max_total_turns",
        source,
        sources,
    );
    if let Some(commands) = &file.agent.commands {
        // 项目配置只能追加 ask/deny（validate_project_trust 已拒绝项目 allow）；
        // 其余来源整表覆盖。
        if source == ConfigSource::Project {
            if let Some(ask) = &commands.ask {
                config.agent.commands.ask.extend(ask.iter().cloned());
                sources.insert("agent.commands.ask".into(), source);
            }
            if let Some(deny) = &commands.deny {
                config.agent.commands.deny.extend(deny.iter().cloned());
                sources.insert("agent.commands.deny".into(), source);
            }
        } else {
            if let Some(allow) = &commands.allow {
                config.agent.commands.allow.clone_from(allow);
                sources.insert("agent.commands.allow".into(), source);
            }
            if let Some(ask) = &commands.ask {
                config.agent.commands.ask.clone_from(ask);
                sources.insert("agent.commands.ask".into(), source);
            }
            if let Some(deny) = &commands.deny {
                config.agent.commands.deny.clone_from(deny);
                sources.insert("agent.commands.deny".into(), source);
            }
        }
    }
    set(
        &mut config.output.json,
        &file.output.json,
        "output.json",
        source,
        sources,
    );
    set(
        &mut config.output.no_stream,
        &file.output.no_stream,
        "output.no_stream",
        source,
        sources,
    );
    set(
        &mut config.output.color,
        &file.output.color,
        "output.color",
        source,
        sources,
    );
    set(
        &mut config.output.debug_trace,
        &file.output.debug_trace,
        "output.debug_trace",
        source,
        sources,
    );
    set(
        &mut config.output.theme,
        &file.output.theme,
        "output.theme",
        source,
        sources,
    );
    set(
        &mut config.telemetry.enabled,
        &file.telemetry.enabled,
        "telemetry.enabled",
        source,
        sources,
    );
    set(
        &mut config.memory.suggest_enabled,
        &file.memory.suggest_enabled,
        "memory.suggest_enabled",
        source,
        sources,
    );
    set(
        &mut config.memory.top_k,
        &file.memory.top_k,
        "memory.top_k",
        source,
        sources,
    );
    set(
        &mut config.memory.injection_token_budget,
        &file.memory.injection_token_budget,
        "memory.injection_token_budget",
        source,
        sources,
    );
    if let Some(profiles) = &file.agent.profiles {
        let mut converted = Vec::new();
        for (id, file_profile) in profiles {
            converted.push(file_profile.clone().into_profile(id.clone())?);
        }
        if source == ConfigSource::Project {
            config.agent.profiles.extend(converted);
        } else {
            config.agent.profiles = converted;
        }
        sources.insert("agent.profiles".into(), source);
    }
    Ok(())
}

fn validate_project_trust(
    config: &AppConfig,
    project: &FileConfig,
    path: &Path,
) -> XduduResult<()> {
    if project.provider.base_url.is_some() {
        return Err(config_error(format!(
            "项目配置 {} 不能设置 provider.base_url；请使用 CLI、环境变量或用户配置，避免仓库重定向系统凭据。",
            path.display()
        )));
    }
    if let Some(permission) = &project.agent.permission {
        let current = config.agent.permission_mode()?;
        let requested = permission.parse::<PermissionMode>()?;
        if requested
            .allowed_levels()
            .iter()
            .any(|level| !current.allows(*level))
        {
            return Err(config_error(format!(
                "项目配置 {} 不能把 agent.permission 从 {} 提升到 {}。",
                path.display(),
                current.as_str(),
                requested.as_str()
            )));
        }
    }
    if let Some(approval) = &project.agent.approval {
        let current = config.agent.approval_mode()?;
        let requested = approval.parse::<ApprovalMode>()?;
        let rank = |mode| match mode {
            ApprovalMode::Never => 0,
            ApprovalMode::Ask => 1,
            ApprovalMode::AcceptEdits => 2,
            ApprovalMode::Always => 3,
        };
        if rank(requested) > rank(current) {
            return Err(config_error(format!(
                "项目配置 {} 不能把 agent.approval 从 {} 提升到 {}。",
                path.display(),
                current.as_str(),
                requested.as_str()
            )));
        }
    }
    if let Some(commands) = &project.agent.commands
        && commands
            .allow
            .as_ref()
            .is_some_and(|allow| !allow.is_empty())
    {
        return Err(config_error(format!(
            "项目配置 {} 不能设置 agent.commands.allow；项目只能追加 deny 与 ask 规则。",
            path.display()
        )));
    }
    if let Some(skills) = &project.agent.skills
        && skills != "deny"
    {
        return Err(config_error(format!(
            "项目配置 {} 只能把 agent.skills 收紧为 deny。",
            path.display()
        )));
    }
    if let Some(profiles) = &project.agent.profiles {
        for (id, profile) in profiles {
            if profile
                .permission
                .as_deref()
                .is_some_and(|value| value != "read-only")
            {
                return Err(config_error(format!(
                    "项目配置 {} 的档案 {id} 只能使用只读权限（permission = \"read-only\"）。",
                    path.display()
                )));
            }
        }
    }
    Ok(())
}

fn env_file(provider: &str) -> FileConfig {
    let value =
        |current: &str, legacy: &str| env::var(current).ok().or_else(|| env::var(legacy).ok());
    let provider_name = value("XDUDU_PROVIDER", "XYCLI_PROVIDER");
    let endpoint_provider = provider_name.clone().unwrap_or_else(|| provider.to_owned());
    FileConfig {
        provider: FileProvider {
            name: provider_name,
            model: value("XDUDU_MODEL", "XYCLI_MODEL"),
            base_url: value("XDUDU_BASE_URL", "XYCLI_BASE_URL").or_else(|| {
                env::var(format!(
                    "{}_BASE_URL",
                    endpoint_provider.to_ascii_uppercase()
                ))
                .ok()
            }),
            timeout_seconds: value("XDUDU_TIMEOUT_SECONDS", "XYCLI_TIMEOUT_SECONDS")
                .and_then(|value| value.parse().ok()),
            max_attempts: value("XDUDU_MAX_ATTEMPTS", "XYCLI_MAX_ATTEMPTS")
                .and_then(|value| value.parse().ok()),
            retry_base_ms: value("XDUDU_RETRY_BASE_MS", "XYCLI_RETRY_BASE_MS")
                .and_then(|value| value.parse().ok()),
            min_request_interval_ms: value(
                "XDUDU_MIN_REQUEST_INTERVAL_MS",
                "XYCLI_MIN_REQUEST_INTERVAL_MS",
            )
            .and_then(|value| value.parse().ok()),
            temperature: value("XDUDU_TEMPERATURE", "XYCLI_TEMPERATURE")
                .and_then(|value| value.parse().ok()),
            max_output_tokens: value("XDUDU_MAX_OUTPUT_TOKENS", "XYCLI_MAX_OUTPUT_TOKENS")
                .and_then(|value| value.parse().ok()),
            reasoning: value("XDUDU_REASONING", "XYCLI_REASONING")
                .and_then(|value| value.parse().ok()),
            max_context_tokens: value("XDUDU_MAX_CONTEXT_TOKENS", "XYCLI_MAX_CONTEXT_TOKENS")
                .and_then(|value| value.parse().ok()),
        },
        agent: FileAgent {
            max_turns: value("XDUDU_MAX_TURNS", "XYCLI_MAX_TURNS")
                .and_then(|value| value.parse().ok()),
            permission: value("XDUDU_PERMISSION", "XYCLI_PERMISSION"),
            approval: value("XDUDU_APPROVAL", "XYCLI_APPROVAL"),
            stalled_recovery: value("XDUDU_STALLED_RECOVERY", "XYCLI_STALLED_RECOVERY"),
            stalled_max_recovery: value("XDUDU_STALLED_MAX_RECOVERY", "XYCLI_STALLED_MAX_RECOVERY")
                .and_then(|value| value.parse().ok()),
            commands: None,
            skills: value("XDUDU_SKILLS", "XYCLI_SKILLS"),
            auto_continue: value("XDUDU_AUTO_CONTINUE", "XYCLI_AUTO_CONTINUE")
                .and_then(|value| value.parse().ok()),
            max_total_turns: value("XDUDU_MAX_TOTAL_TURNS", "XYCLI_MAX_TOTAL_TURNS")
                .and_then(|value| value.parse().ok()),
            profiles: None,
        },
        output: FileOutput {
            json: value("XDUDU_JSON", "XYCLI_JSON").and_then(|value| value.parse().ok()),
            no_stream: value("XDUDU_NO_STREAM", "XYCLI_NO_STREAM")
                .and_then(|value| value.parse().ok()),
            color: env::var("NO_COLOR").ok().map(|_| false),
            debug_trace: value("XDUDU_DEBUG_TRACE", "XYCLI_DEBUG_TRACE")
                .and_then(|value| value.parse().ok()),
            theme: value("XDUDU_THEME", "XYCLI_THEME"),
        },
        telemetry: FileTelemetry {
            enabled: value("XDUDU_TELEMETRY_ENABLED", "XYCLI_TELEMETRY_ENABLED")
                .and_then(|value| value.parse().ok()),
        },
        memory: FileMemory {
            suggest_enabled: value(
                "XDUDU_MEMORY_SUGGEST_ENABLED",
                "XYCLI_MEMORY_SUGGEST_ENABLED",
            )
            .and_then(|value| value.parse().ok()),
            top_k: value("XDUDU_MEMORY_TOP_K", "XYCLI_MEMORY_TOP_K")
                .and_then(|value| value.parse().ok()),
            injection_token_budget: value(
                "XDUDU_MEMORY_INJECTION_BUDGET",
                "XYCLI_MEMORY_INJECTION_BUDGET",
            )
            .and_then(|value| value.parse().ok()),
        },
    }
}

fn validate(config: &mut AppConfig, model_was_explicit: bool) -> XduduResult<()> {
    config.provider.name = config.provider.name.to_ascii_lowercase();
    if !matches!(
        config.provider.name.as_str(),
        "anthropic" | "deepseek" | "openai-compatible"
    ) {
        return Err(config_error(format!(
            "不支持的 Provider：{}。可选值：anthropic、deepseek、openai-compatible。",
            config.provider.name
        )));
    }
    if !model_was_explicit {
        config.provider.model = match config.provider.name.as_str() {
            "deepseek" => DEFAULT_DEEPSEEK_MODEL,
            _ => DEFAULT_ANTHROPIC_MODEL,
        }
        .to_owned();
    }
    if config.provider.model.trim().is_empty() {
        return Err(config_error("provider.model 不能为空。"));
    }
    if !(1..=100).contains(&config.agent.max_turns) {
        return Err(config_error("agent.max_turns 必须是 1 到 100。"));
    }
    if !(1..=600).contains(&config.provider.timeout_seconds) {
        return Err(config_error("provider.timeout_seconds 必须是 1 到 600。"));
    }
    if !(1..=10).contains(&config.provider.max_attempts) {
        return Err(config_error("provider.max_attempts 必须是 1 到 10。"));
    }
    if !(config.provider.temperature.is_finite()
        && (0.0..=2.0).contains(&config.provider.temperature))
    {
        return Err(config_error("provider.temperature 必须是 0.0 到 2.0。"));
    }
    if !(1..=32_768).contains(&config.provider.max_output_tokens) {
        return Err(config_error(
            "provider.max_output_tokens 必须是 1 到 32768。",
        ));
    }
    if !(10..=30_000).contains(&config.provider.retry_base_ms) {
        return Err(config_error("provider.retry_base_ms 必须是 10 到 30000。"));
    }
    if config.provider.min_request_interval_ms > 60_000 {
        return Err(config_error(
            "provider.min_request_interval_ms 必须是 0 到 60000。",
        ));
    }
    config.agent.permission.parse::<PermissionMode>()?;
    config.agent.approval.parse::<ApprovalMode>()?;
    config
        .agent
        .stalled_recovery
        .parse::<StalledRecoveryMode>()?;
    if !(1..=10).contains(&config.agent.stalled_max_recovery) {
        return Err(config_error("agent.stalled_max_recovery 必须是 1 到 10。"));
    }
    for (tier, rules) in [
        ("allow", &config.agent.commands.allow),
        ("ask", &config.agent.commands.ask),
        ("deny", &config.agent.commands.deny),
    ] {
        for rule in rules {
            if !valid_command_rule(rule) {
                return Err(config_error(format!(
                    "agent.commands.{tier} 规则“{rule}”无效：必须是「可执行名」或「可执行名 + 单个参数」，且不能包含路径、空格或 shell 元字符。"
                )));
            }
        }
    }
    config.agent.skills_mode()?;
    if !(1..=32).contains(&config.memory.top_k) {
        return Err(config_error("memory.top_k 必须是 1 到 32。"));
    }
    if !(128..=8192).contains(&config.memory.injection_token_budget) {
        return Err(config_error(
            "memory.injection_token_budget 必须是 128 到 8192。",
        ));
    }
    if let Some(base_url) = &config.provider.base_url
        && !(base_url.starts_with("https://")
            || base_url.starts_with("http://127.0.0.1")
            || base_url.starts_with("http://localhost"))
    {
        return Err(config_error(
            "provider.base_url 必须使用 HTTPS；只有本机测试地址允许 HTTP。",
        ));
    }
    Ok(())
}

pub fn load_config(cwd: &Path, overrides: ConfigOverrides) -> XduduResult<ResolvedConfig> {
    let (user_path, project_path) = config_paths(cwd)?;
    let legacy_user_path = legacy_user_config_path()?;
    let legacy_project_path = cwd.join(".xycli/config.toml");
    let read_user_path = if user_path.exists() {
        user_path.clone()
    } else {
        legacy_user_path
    };
    let read_project_path = if project_path.exists() {
        project_path.clone()
    } else {
        legacy_project_path
    };
    let mut resolved =
        load_config_from_paths(cwd, read_user_path, read_project_path, overrides, None)?;
    resolved.user_path = user_path;
    resolved.project_path = project_path;
    Ok(resolved)
}

fn load_config_from_paths(
    _cwd: &Path,
    user_path: PathBuf,
    project_path: PathBuf,
    overrides: ConfigOverrides,
    fixed_environment: Option<FileConfig>,
) -> XduduResult<ResolvedConfig> {
    let mut config = AppConfig {
        provider: ProviderConfig {
            name: "deepseek".into(),
            model: DEFAULT_DEEPSEEK_MODEL.into(),
            base_url: None,
            timeout_seconds: 180,
            max_attempts: 3,
            retry_base_ms: 500,
            min_request_interval_ms: 0,
            temperature: 0.2,
            max_output_tokens: 8192,
            reasoning: false,
            max_context_tokens: 0,
        },
        agent: AgentConfig::default(),
        output: OutputConfig {
            json: false,
            no_stream: false,
            color: env::var_os("NO_COLOR").is_none(),
            debug_trace: false,
            theme: "auto".into(),
        },
        telemetry: TelemetryConfig { enabled: false },
        memory: MemoryConfig {
            suggest_enabled: true,
            top_k: 8,
            injection_token_budget: 1500,
        },
    };
    let mut sources = [
        ("provider.name", ConfigSource::Default),
        ("provider.model", ConfigSource::Default),
        ("provider.base_url", ConfigSource::Default),
        ("provider.timeout_seconds", ConfigSource::Default),
        ("provider.max_attempts", ConfigSource::Default),
        ("provider.retry_base_ms", ConfigSource::Default),
        ("provider.min_request_interval_ms", ConfigSource::Default),
        ("provider.temperature", ConfigSource::Default),
        ("provider.max_output_tokens", ConfigSource::Default),
        ("provider.reasoning", ConfigSource::Default),
        ("provider.max_context_tokens", ConfigSource::Default),
        ("agent.max_turns", ConfigSource::Default),
        ("agent.permission", ConfigSource::Default),
        ("agent.approval", ConfigSource::Default),
        ("agent.stalled_recovery", ConfigSource::Default),
        ("agent.stalled_max_recovery", ConfigSource::Default),
        ("agent.commands.allow", ConfigSource::Default),
        ("agent.commands.ask", ConfigSource::Default),
        ("agent.commands.deny", ConfigSource::Default),
        ("agent.skills", ConfigSource::Default),
        ("agent.auto_continue", ConfigSource::Default),
        ("agent.max_total_turns", ConfigSource::Default),
        ("agent.profiles", ConfigSource::Default),
        ("output.json", ConfigSource::Default),
        ("output.no_stream", ConfigSource::Default),
        ("output.color", ConfigSource::Default),
        ("output.debug_trace", ConfigSource::Default),
        ("output.theme", ConfigSource::Default),
        ("telemetry.enabled", ConfigSource::Default),
        ("memory.suggest_enabled", ConfigSource::Default),
        ("memory.top_k", ConfigSource::Default),
        ("memory.injection_token_budget", ConfigSource::Default),
    ]
    .into_iter()
    .map(|(key, source)| (key.to_owned(), source))
    .collect::<BTreeMap<_, _>>();

    if let Some(file) = read_file(&user_path)? {
        apply_file(&mut config, &file, ConfigSource::User, &mut sources)?;
    }
    if let Some(file) = read_file(&project_path)? {
        validate_project_trust(&config, &file, &project_path)?;
        apply_file(&mut config, &file, ConfigSource::Project, &mut sources)?;
    }
    let environment = fixed_environment.unwrap_or_else(|| env_file(&config.provider.name));
    apply_file(
        &mut config,
        &environment,
        ConfigSource::Environment,
        &mut sources,
    )?;
    let model_was_explicit = overrides.model.is_some()
        || environment.provider.model.is_some()
        || sources.get("provider.model") != Some(&ConfigSource::Default);
    let cli = FileConfig {
        provider: FileProvider {
            name: overrides.provider,
            model: overrides.model,
            base_url: overrides.base_url,
            timeout_seconds: None,
            max_attempts: None,
            retry_base_ms: None,
            min_request_interval_ms: None,
            temperature: overrides.temperature,
            max_output_tokens: overrides.max_output_tokens,
            reasoning: overrides.reasoning,
            max_context_tokens: None,
        },
        agent: FileAgent {
            max_turns: overrides.max_turns,
            permission: overrides.permission,
            approval: overrides.approval,
            stalled_recovery: overrides.stalled_recovery,
            stalled_max_recovery: overrides.stalled_max_recovery,
            auto_continue: None,
            max_total_turns: None,
            commands: None,
            skills: None,
            profiles: None,
        },
        output: FileOutput {
            json: overrides.json,
            no_stream: overrides.no_stream,
            color: overrides.color,
            debug_trace: overrides.debug_trace,
            theme: None,
        },
        telemetry: FileTelemetry {
            enabled: overrides.telemetry_enabled,
        },
        memory: FileMemory {
            suggest_enabled: overrides.memory_suggest_enabled,
            top_k: None,
            injection_token_budget: None,
        },
    };
    apply_file(&mut config, &cli, ConfigSource::Cli, &mut sources)?;
    validate(&mut config, model_was_explicit)?;
    Ok(ResolvedConfig {
        config,
        sources,
        user_path,
        project_path,
    })
}

pub fn write_config_value(
    cwd: &Path,
    user: bool,
    key: &str,
    raw_value: &str,
) -> XduduResult<PathBuf> {
    if matches!(
        key.to_ascii_lowercase().replace('-', "_").as_str(),
        "api_key" | "apikey" | "token" | "secret"
    ) {
        return Err(config_error(
            "密钥不能写入配置文件，请使用 xdudu auth login。",
        ));
    }
    if !user && key == "provider.base_url" {
        return Err(config_error(
            "项目配置不能设置 provider.base_url；请使用 --user、环境变量或 CLI 参数。",
        ));
    }
    if !user && key == "agent.permission" && raw_value != "read-only" {
        return Err(config_error(
            "项目配置只能把 agent.permission 收紧为 read-only。",
        ));
    }
    if !user && key == "agent.approval" && raw_value != "never" {
        return Err(config_error("项目配置只能把 agent.approval 收紧为 never。"));
    }
    let (user_path, project_path) = config_paths(cwd)?;
    let path = if user { user_path } else { project_path };
    let mut value = match fs::read_to_string(&path) {
        Ok(raw) => toml::from_str::<Value>(&raw)
            .map_err(|error| config_error(format!("配置 {} 格式无效：{error}", path.display())))?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Value::Table(Default::default())
        }
        Err(error) => {
            return Err(config_error(format!(
                "无法读取配置 {}：{error}",
                path.display()
            )));
        }
    };
    let parts = key.split('.').collect::<Vec<_>>();
    if parts.len() != 2
        || !matches!(
            key,
            "provider.name"
                | "provider.model"
                | "provider.base_url"
                | "provider.timeout_seconds"
                | "provider.max_attempts"
                | "provider.retry_base_ms"
                | "provider.min_request_interval_ms"
                | "provider.temperature"
                | "provider.max_output_tokens"
                | "provider.reasoning"
                | "provider.max_context_tokens"
                | "agent.max_turns"
                | "agent.permission"
                | "agent.approval"
                | "agent.stalled_recovery"
                | "agent.stalled_max_recovery"
                | "agent.skills"
                | "agent.auto_continue"
                | "agent.max_total_turns"
                | "output.json"
                | "output.no_stream"
                | "output.color"
                | "output.debug_trace"
                | "output.theme"
                | "telemetry.enabled"
                | "memory.suggest_enabled"
                | "memory.top_k"
                | "memory.injection_token_budget"
        )
    {
        return Err(config_error(format!("不支持的配置项：{key}")));
    }
    if !user && key == "agent.skills" && raw_value != "deny" {
        return Err(config_error("项目配置只能把 agent.skills 收紧为 deny。"));
    }
    match key {
        "provider.name" if !matches!(raw_value, "anthropic" | "deepseek" | "openai-compatible") => {
            return Err(config_error(
                "provider.name 只能是 anthropic、deepseek 或 openai-compatible。",
            ));
        }
        "provider.model" if raw_value.trim().is_empty() => {
            return Err(config_error("provider.model 不能为空。"));
        }
        "provider.base_url"
            if !(raw_value.starts_with("https://")
                || raw_value.starts_with("http://127.0.0.1")
                || raw_value.starts_with("http://localhost")) =>
        {
            return Err(config_error(
                "provider.base_url 必须使用 HTTPS；只有本机测试地址允许 HTTP。",
            ));
        }
        "agent.permission" => {
            raw_value.parse::<PermissionMode>()?;
        }
        "agent.approval" => {
            raw_value.parse::<ApprovalMode>()?;
        }
        "agent.skills" if !matches!(raw_value, "allow" | "ask" | "deny") => {
            return Err(config_error("agent.skills 只能是 allow、ask 或 deny。"));
        }
        _ => {}
    }
    let parsed = if matches!(
        key,
        "provider.timeout_seconds"
            | "provider.max_attempts"
            | "provider.retry_base_ms"
            | "provider.min_request_interval_ms"
            | "provider.max_output_tokens"
            | "provider.max_context_tokens"
            | "agent.max_turns"
            | "agent.stalled_max_recovery"
            | "agent.max_total_turns"
            | "memory.top_k"
            | "memory.injection_token_budget"
    ) {
        let number = raw_value
            .parse::<i64>()
            .map_err(|_| config_error(format!("{key} 必须是整数。")))?;
        let valid = match key {
            "agent.max_turns" => (1..=100).contains(&number),
            "provider.timeout_seconds" => (1..=600).contains(&number),
            "provider.max_attempts" => (1..=10).contains(&number),
            "provider.retry_base_ms" => (10..=30_000).contains(&number),
            "provider.min_request_interval_ms" => (0..=60_000).contains(&number),
            "provider.max_output_tokens" => (1..=32_768).contains(&number),
            "provider.max_context_tokens" => (16_384..=1_048_576).contains(&number),
            "agent.stalled_max_recovery" => (1..=10).contains(&number),
            "agent.max_total_turns" => (1..=1_000).contains(&number),
            "memory.top_k" => (1..=32).contains(&number),
            "memory.injection_token_budget" => (128..=8192).contains(&number),
            _ => false,
        };
        if !valid {
            return Err(config_error(format!("{key} 超出允许范围。")));
        }
        Value::Integer(number)
    } else if key == "provider.temperature" {
        let number = raw_value
            .parse::<f64>()
            .map_err(|_| config_error(format!("{key} 必须是数字。")))?;
        if !number.is_finite() || !(0.0..=2.0).contains(&number) {
            return Err(config_error(format!("{key} 必须在 0.0 到 2.0 之间。")));
        }
        Value::Float(number)
    } else if key == "provider.reasoning" {
        Value::Boolean(
            raw_value
                .parse()
                .map_err(|_| config_error(format!("{key} 必须是 true 或 false。")))?,
        )
    } else if key.starts_with("output.")
        || key == "telemetry.enabled"
        || key == "memory.suggest_enabled"
        || key == "agent.auto_continue"
    {
        Value::Boolean(
            raw_value
                .parse()
                .map_err(|_| config_error(format!("{key} 必须是 true 或 false。")))?,
        )
    } else {
        Value::String(raw_value.to_owned())
    };
    let table = value
        .as_table_mut()
        .ok_or_else(|| config_error("配置根节点必须是表。"))?;
    let section = table
        .entry(parts[0])
        .or_insert_with(|| Value::Table(Default::default()))
        .as_table_mut()
        .ok_or_else(|| config_error(format!("配置段 {} 必须是表。", parts[0])))?;
    section.insert(parts[1].to_owned(), parsed);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            config_error(format!("无法创建配置目录 {}：{error}", parent.display()))
        })?;
    }
    let data = toml::to_string_pretty(&value).map_err(|error| config_error(error.to_string()))?;
    let temporary = path.with_extension(format!("toml.{}.tmp", Uuid::new_v4()));
    fs::write(&temporary, data).map_err(|error| {
        config_error(format!("无法写入临时配置 {}：{error}", temporary.display()))
    })?;
    if let Err(error) = fs::rename(&temporary, &path) {
        let _ = fs::remove_file(&temporary);
        return Err(config_error(format!(
            "无法替换配置 {}：{error}",
            path.display()
        )));
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn 默认使用_deepseek() {
        let root = tempdir().unwrap();
        let resolved = load_config_from_paths(
            root.path(),
            root.path().join("missing-user.toml"),
            root.path().join("missing-project.toml"),
            ConfigOverrides::default(),
            Some(FileConfig::default()),
        )
        .unwrap();
        assert_eq!(resolved.config.provider.name, "deepseek");
        assert_eq!(resolved.config.provider.model, DEFAULT_DEEPSEEK_MODEL);
        assert!(!resolved.config.output.debug_trace);
    }

    #[test]
    fn 项目配置覆盖用户配置且_cli_优先() {
        let root = tempdir().unwrap();
        let config_home = tempdir().unwrap();
        fs::write(
            config_home.path().join("config.toml"),
            "[provider]\nname='deepseek'\nmodel='user-model'\n",
        )
        .unwrap();
        fs::create_dir_all(root.path().join(".xdudu")).unwrap();
        fs::write(
            root.path().join(".xdudu/config.toml"),
            "[provider]\nmodel='project-model'\n",
        )
        .unwrap();
        let resolved = load_config_from_paths(
            root.path(),
            config_home.path().join("config.toml"),
            root.path().join(".xdudu/config.toml"),
            ConfigOverrides {
                model: Some("cli-model".into()),
                ..Default::default()
            },
            Some(FileConfig::default()),
        )
        .unwrap();
        assert_eq!(resolved.config.provider.name, "deepseek");
        assert_eq!(resolved.config.provider.model, "cli-model");
        assert_eq!(resolved.source("provider.name"), Some(ConfigSource::User));
        assert_eq!(resolved.source("provider.model"), Some(ConfigSource::Cli));
    }

    #[test]
    fn 配置拒绝明文密钥和不安全远端_http() {
        let root = tempdir().unwrap();
        fs::create_dir_all(root.path().join(".xdudu")).unwrap();
        fs::write(
            root.path().join(".xdudu/config.toml"),
            "[provider]\napi_key='secret'\n",
        )
        .unwrap();
        assert!(load_config(root.path(), ConfigOverrides::default()).is_err());
        fs::write(
            root.path().join(".xdudu/config.toml"),
            "[provider]\nbase_url='http://example.com'\n",
        )
        .unwrap();
        assert!(load_config(root.path(), ConfigOverrides::default()).is_err());
    }

    #[test]
    fn 写配置只允许已知非秘密字段() {
        let root = tempdir().unwrap();
        write_config_value(root.path(), false, "agent.max_turns", "30").unwrap();
        let resolved = load_config(root.path(), ConfigOverrides::default()).unwrap();
        assert_eq!(resolved.config.agent.max_turns, 30);
        write_config_value(root.path(), false, "output.debug_trace", "true").unwrap();
        let resolved = load_config(root.path(), ConfigOverrides::default()).unwrap();
        assert!(resolved.config.output.debug_trace);
        assert!(write_config_value(root.path(), false, "api_key", "secret").is_err());
    }

    #[test]
    fn 项目配置不能追加命令allow但可追加ask与deny() {
        let root = tempdir().unwrap();
        let project = root.path().join(".xdudu/config.toml");
        fs::create_dir_all(root.path().join(".xdudu")).unwrap();
        let load = || {
            load_config_from_paths(
                root.path(),
                root.path().join("missing-user.toml"),
                project.clone(),
                ConfigOverrides::default(),
                Some(FileConfig::default()),
            )
        };

        fs::write(&project, "[agent.commands]\nallow=['cargo run']\n").unwrap();
        assert!(load().is_err());

        fs::write(
            &project,
            "[agent.commands]\nask=['cargo run']\ndeny=['dd']\n",
        )
        .unwrap();
        let resolved = load().unwrap();
        assert!(
            resolved
                .config
                .agent
                .commands
                .ask
                .contains(&"cargo run".to_owned())
        );
        assert!(
            resolved
                .config
                .agent
                .commands
                .deny
                .contains(&"dd".to_owned())
        );
        // 默认 deny 保留。
        assert!(
            resolved
                .config
                .agent
                .commands
                .deny
                .contains(&"sudo".to_owned())
        );
    }

    #[test]
    fn 非法命令规则被拒绝() {
        let root = tempdir().unwrap();
        fs::create_dir_all(root.path().join(".xdudu")).unwrap();
        fs::write(
            root.path().join(".xdudu/config.toml"),
            "[agent.commands]\nallow=['cargo run --release']\n",
        )
        .unwrap();
        assert!(load_config(root.path(), ConfigOverrides::default()).is_err());
        fs::write(
            root.path().join(".xdudu/config.toml"),
            "[agent.commands]\nallow=['/bin/ls']\n",
        )
        .unwrap();
        assert!(load_config(root.path(), ConfigOverrides::default()).is_err());
    }

    #[test]
    fn 项目配置不能重定向凭据或提升权限审批() {
        let root = tempdir().unwrap();
        fs::create_dir_all(root.path().join(".xdudu")).unwrap();
        for content in [
            "[provider]\nbase_url='https://attacker.example'\n",
            "[agent]\npermission='full-access'\n",
            "[agent]\napproval='always'\n",
        ] {
            fs::write(root.path().join(".xdudu/config.toml"), content).unwrap();
            assert!(
                load_config_from_paths(
                    root.path(),
                    root.path().join("missing-user.toml"),
                    root.path().join(".xdudu/config.toml"),
                    ConfigOverrides::default(),
                    Some(FileConfig::default()),
                )
                .is_err()
            );
        }
    }

    #[test]
    fn 上下文预算按模型窗口推导() {
        // deepseek 128K → 软阈值 115200（90%）/ 硬顶 121600（95%）。
        let deepseek = ProviderConfig {
            name: "deepseek".into(),
            model: "deepseek-chat".into(),
            max_context_tokens: 0,
            ..ProviderConfig::default()
        };
        assert_eq!(deepseek.context_budget().unwrap(), 115_200);
        assert_eq!(deepseek.context_hard_limit().unwrap(), 121_600);
        // anthropic claude 200K → 软 180K / 硬 190K。
        let claude = ProviderConfig {
            name: "anthropic".into(),
            model: "claude-sonnet-4-5".into(),
            max_context_tokens: 0,
            ..ProviderConfig::default()
        };
        assert_eq!(claude.context_budget().unwrap(), 180_000);
        assert_eq!(claude.context_hard_limit().unwrap(), 190_000);
        // openai-compatible 未知模型 → 保守 64K → 软 57600 / 硬 60800。
        let custom = ProviderConfig {
            name: "openai-compatible".into(),
            model: "my-model".into(),
            max_context_tokens: 0,
            ..ProviderConfig::default()
        };
        assert_eq!(custom.context_budget().unwrap(), 57_600);
        assert_eq!(custom.context_hard_limit().unwrap(), 60_800);
        // 显式能力声明覆盖内置表。
        let overridden = ProviderConfig {
            name: "openai-compatible".into(),
            model: "my-model".into(),
            max_context_tokens: 200_000,
            ..ProviderConfig::default()
        };
        assert_eq!(overridden.context_budget().unwrap(), 180_000);
        assert_eq!(overridden.context_hard_limit().unwrap(), 190_000);
        // 非法窗口拒绝。
        let invalid = ProviderConfig {
            name: "openai-compatible".into(),
            model: "x".into(),
            max_context_tokens: 1_000,
            ..ProviderConfig::default()
        };
        assert!(invalid.context_budget().is_err());
        assert!(invalid.context_hard_limit().is_err());
    }

    #[test]
    fn max_total_turns_默认值与范围校验() {
        assert_eq!(AgentConfig::default().max_total_turns, 200);
        assert!(AgentConfig::default().auto_continue);
        let mut agent = AgentConfig::default();
        assert_eq!(agent.max_total_turns().unwrap(), 200);
        agent.max_total_turns = 1_000;
        assert_eq!(agent.max_total_turns().unwrap(), 1_000);
        for invalid in [0, 1_001] {
            agent.max_total_turns = invalid;
            assert!(agent.max_total_turns().is_err(), "预算 {invalid} 应被拒绝");
        }
    }

    #[test]
    fn auto_continue与max_total_turns_支持配置文件与写白名单() {
        let root = tempdir().unwrap();
        fs::create_dir_all(root.path().join(".xdudu")).unwrap();
        fs::write(
            root.path().join(".xdudu/config.toml"),
            "[agent]\nauto_continue = false\nmax_total_turns = 80\n",
        )
        .unwrap();
        let resolved = load_config(root.path(), ConfigOverrides::default()).unwrap();
        assert!(!resolved.config.agent.auto_continue);
        assert_eq!(resolved.config.agent.max_total_turns, 80);
        assert_eq!(
            resolved.sources.get("agent.auto_continue"),
            Some(&ConfigSource::Project)
        );
        write_config_value(root.path(), false, "agent.auto_continue", "true").unwrap();
        write_config_value(root.path(), false, "agent.max_total_turns", "300").unwrap();
        let resolved = load_config(root.path(), ConfigOverrides::default()).unwrap();
        assert!(resolved.config.agent.auto_continue);
        assert_eq!(resolved.config.agent.max_total_turns, 300);
        assert!(write_config_value(root.path(), false, "agent.max_total_turns", "1001").is_err());
        assert!(write_config_value(root.path(), false, "agent.auto_continue", "maybe").is_err());
    }

    #[test]
    fn max_context_tokens_支持配置文件与写白名单() {
        let root = tempdir().unwrap();
        fs::create_dir_all(root.path().join(".xdudu")).unwrap();
        fs::write(
            root.path().join(".xdudu/config.toml"),
            "[provider]\nmax_context_tokens = 200000\n",
        )
        .unwrap();
        let resolved = load_config(root.path(), ConfigOverrides::default()).unwrap();
        assert_eq!(resolved.config.provider.max_context_tokens, 200_000);
        assert_eq!(
            resolved.sources.get("provider.max_context_tokens"),
            Some(&ConfigSource::Project)
        );
        write_config_value(root.path(), false, "provider.max_context_tokens", "128000").unwrap();
        let resolved = load_config(root.path(), ConfigOverrides::default()).unwrap();
        assert_eq!(resolved.config.provider.max_context_tokens, 128_000);
        assert!(
            write_config_value(root.path(), false, "provider.max_context_tokens", "100").is_err()
        );
    }
}
