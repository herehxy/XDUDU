//! XDUDU Rust 命令行入口。

mod approval_prompt;
mod doctor;
mod inline_terminal;
mod input_editor;
mod input_queue;
mod markdown;
mod renderer;
mod tui;
mod ui;
mod version_check;

use std::io::Write as _;
use std::{
    collections::{BTreeSet, VecDeque},
    env,
    future::Future,
    io::{self, IsTerminal},
    path::{Path, PathBuf},
    pin::Pin,
    process::ExitCode,
    sync::Arc,
};

use async_trait::async_trait;
use clap::{Args, Parser, Subcommand};
use crossterm::event::{Event, EventStream, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use futures_util::StreamExt;
use tokio::io::AsyncReadExt;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;
use xdudu_core::{
    AgentLoopState, AgentProfile, AgentRunConfig, AgentRunResult, AllowAllApprovalGate,
    ApprovalDecision, ApprovalGate, ApprovalMode, ApprovalRequest, ApprovalRule, ApprovalScope,
    ConfigOverrides, DefaultProviderFactory, DenyAllApprovalGate, EventSink, JsonApprovalRuleStore,
    JsonChangeLedger, KeyringSecretStore, McpConfigFile, McpServerConfig, McpServerRuntime,
    McpTransportKind, MemoryConsolidationConfig, MemoryRecord, MemoryStore, MemorySuggestionConfig,
    PermissionMode, Plan, PlanExecutorConfig, PlanGenerationConfig, PlanRevisionConfig, PlanStatus,
    PlanStore, PluginManifest, ProfileMode, Provider, ProviderFactory, ResolvedConfig,
    SecretSource, SecretStore, SecretString, Session, SessionStatus, SessionStore, SideEffectKind,
    Skill, SkillMode, SkillTool, SqliteSessionStore, StalledRecoveryMode, ToolRegistry,
    WebReadTool, WorkspaceLock, XduduError, approval_rules_path, approve_plan, config_paths,
    consolidate_memory_document, discover_skills, generate_plan, load_config, load_instructions,
    load_mcp_config, load_plugin_manifests, mcp_config_path, memory_document_path, merge_profiles,
    plugin_directory, read_memory_document, redact_text, register_builtins,
    register_configured_mcp_tools, reject_plan, resolve_secret, revise_plan, run_agent, run_plan,
    save_mcp_config, save_plugin_manifest, submit_plan_for_review, write_config_value,
    write_memory_document,
};

use crate::approval_prompt::{ApprovalMenuChoice, format_approval_prompt, read_approval_menu};
use crate::doctor::run_doctor;
use crate::input_editor::{InputEditor, ReadResult};
use crate::input_queue::{InputFocus, InputRouter};
use crate::renderer::ConsoleRenderer;
use crate::tui::{
    InputOutcome, PlanRecoveryChoice, PlanReviewChoice, SessionChoice, TuiApp, TuiContext,
    TuiRenderer,
};
use crate::ui::TerminalTheme;

/// TUI 运行任务的 Future 类型：通过克隆捕获运行所需片段，不借用 Runtime。
type RunFuture = Pin<Box<dyn Future<Output = Result<AgentRunResult, XduduError>> + Send>>;

/// 生成自定义指令加载摘要（来源、数量与警告），供 `/instructions` 与 TUI 展示。
fn instruction_summary(cwd: &Path) -> String {
    let (files, warnings) = load_instructions(cwd);
    if files.is_empty() && warnings.is_empty() {
        return "未加载自定义指令。可通过 ~/.config/xdudu/instructions/、.xdudu/instructions/ 或仓库 AGENTS.md / CLAUDE.md 添加。".into();
    }
    let mut lines = Vec::new();
    if files.is_empty() {
        lines.push("未加载自定义指令。".to_owned());
    } else {
        for file in &files {
            lines.push(format!(
                "{} · {}（{} 字符）",
                file.source.as_str(),
                file.file_name,
                file.content.chars().count()
            ));
        }
    }
    for warning in &warnings {
        lines.push(format!("警告：{warning}"));
    }
    lines.join("\n")
}

/// 生成技能摘要：可用技能索引、加载策略与来源。
fn skills_summary(runtime: &Runtime) -> String {
    if runtime.skills_mode == SkillMode::Deny {
        return "技能加载已被 agent.skills=deny 禁用。".into();
    }
    let mode = format!("加载策略：{}", runtime.skills_mode.as_str());
    if runtime.skills.is_empty() {
        return format!(
            "{mode}\n未发现技能。可将 SKILL.md 放入 .xdudu/skills/、.claude/skills/ 或 ~/.config/xdudu/skills/ 等目录。"
        );
    }
    let mut lines = vec![mode];
    for skill in &runtime.skills {
        lines.push(format!(
            "- {}：{}（{}）",
            skill.name, skill.description, skill.source_label
        ));
    }
    lines.join("\n")
}

/// 生成 Agent 档案摘要：id、模式与定位（`task` 工具可委派的子代理）。
fn agent_summary(runtime: &Runtime) -> String {
    let mut lines = Vec::new();
    for profile in &runtime.profiles {
        let mode = match profile.mode {
            ProfileMode::Primary => "primary",
            ProfileMode::Subagent => "subagent",
            ProfileMode::All => "all",
        };
        lines.push(format!(
            "- {}（{mode}）：{}",
            profile.id, profile.description
        ));
    }
    lines.join("\n")
}

#[derive(Debug, Parser)]
#[command(name = "xdudu", version, about = "终端原生 AI 编程助手")]
struct Cli {
    /// 自然语言指令；省略时进入交互模式，管道输入则作为一次性指令。
    prompt: Option<String>,

    #[command(subcommand)]
    command: Option<Command>,

    /// 模型名称；默认值由分层配置决定。
    #[arg(long, global = true)]
    model: Option<String>,

    /// Provider：anthropic 或 deepseek。
    #[arg(long, global = true)]
    provider: Option<String>,

    /// 自定义 Provider Base URL。
    #[arg(long, global = true)]
    base_url: Option<String>,

    /// 单次任务最大 Agent 循环次数。
    #[arg(long, global = true, value_parser = clap::value_parser!(u32).range(1..=100))]
    max_turns: Option<u32>,

    /// 强制进入交互模式。
    #[arg(short, long, global = true)]
    interactive: bool,

    /// 权限模式：read-only、auto-safe 或 full-access。
    #[arg(long, global = true)]
    permission: Option<String>,

    /// 副作用审批模式：ask、never 或 always。
    #[arg(long, global = true)]
    approval: Option<String>,

    /// 继续已有会话。
    #[arg(long, global = true)]
    session: Option<Uuid>,

    /// 以 JSON Lines 输出机器可读事件。
    #[arg(long, global = true)]
    json: bool,

    /// 禁用流式终端渲染。
    #[arg(long, global = true)]
    no_stream: bool,

    /// 禁用颜色。
    #[arg(long, global = true)]
    no_color: bool,

    /// 输出不含模型思维链、且经过脱敏的结构化运行时调试轨迹。
    #[arg(long, global = true)]
    debug_trace: bool,

    /// 模型请求采样温度（覆盖配置）。
    #[arg(long, global = true)]
    temperature: Option<f32>,

    /// 单次模型请求最大输出 Token（覆盖配置）。
    #[arg(long, global = true)]
    max_output_tokens: Option<u32>,

    /// 启用内部思考闭环（覆盖配置）。
    #[arg(long, global = true)]
    reasoning: bool,

    /// 停滞后的恢复策略：auto、ask 或 off（覆盖配置）。
    #[arg(long, global = true)]
    stalled_recovery: Option<String>,

    /// 停滞恢复模式下允许的最大恢复尝试次数（覆盖配置）。
    #[arg(long, global = true)]
    stalled_max_recovery: Option<u32>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// 执行一次任务；省略 prompt 时从 stdin 读取或进入交互模式。
    Run(RunArgs),
    /// 管理系统凭据中的 Provider API Key。
    Auth {
        #[command(subcommand)]
        command: AuthCommand,
    },
    /// 查看、解释或修改分层配置。
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    /// 查看或撤销永久工具审批规则。
    Approval {
        #[command(subcommand)]
        command: ApprovalCommand,
    },
    /// 检查安装、配置、凭据和工作区状态。
    Doctor,
    /// 安全撤销最近一次或指定的 Agent 文件变更。
    Undo(UndoArgs),
    /// 查看或编辑整理后的长期记忆文档。
    Memory {
        #[command(subcommand)]
        command: MemoryCommand,
    },
    /// 查询或恢复本地会话。
    Session {
        #[command(subcommand)]
        command: SessionCommand,
    },
    /// 创建、审阅、执行和恢复计划。
    Plan {
        #[command(subcommand)]
        command: PlanCommand,
    },
    /// 管理 Model Context Protocol 服务器。
    Mcp {
        #[command(subcommand)]
        command: McpCommand,
    },
    /// 管理只声明 MCP Server 的隔离插件。
    Plugin {
        #[command(subcommand)]
        command: PluginCommand,
    },
}

#[derive(Debug, Subcommand)]
enum PluginCommand {
    /// 列出本地插件清单。
    List,
    /// 显示插件的脱敏清单。
    Show { id: String },
    /// 启用插件。
    Enable { id: String },
    /// 禁用插件。
    Disable { id: String },
    /// 校验插件并连接其中的 MCP Server。
    Doctor { id: Option<String> },
}

#[derive(Debug, Subcommand)]
enum McpCommand {
    /// 列出已配置的 MCP Server。
    List,
    /// 显示一个 MCP Server 的脱敏配置。
    Show { name: String },
    /// 添加本地 stdio MCP Server；额外参数直接放在命令之后。
    AddStdio {
        name: String,
        command: String,
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },
    /// 添加 Streamable HTTP MCP Server。
    AddHttp {
        name: String,
        url: String,
        /// 使用系统凭据 mcp:<name> 作为 Bearer Token。
        #[arg(long)]
        auth: bool,
    },
    /// 启用 MCP Server。
    Enable { name: String },
    /// 禁用 MCP Server。
    Disable { name: String },
    /// 删除 MCP Server 配置（不会删除系统凭据）。
    Remove { name: String },
    /// 把远程 MCP Bearer Token 保存到系统凭据。
    Login { name: String },
    /// 删除远程 MCP Bearer Token。
    Logout { name: String },
    /// 启动并检查 Server、协议和工具列表。
    Doctor { name: Option<String> },
}

#[derive(Debug, Args)]
struct RunArgs {
    /// 要执行的自然语言任务。
    prompt: Option<String>,
}

#[derive(Debug, Args)]
struct UndoArgs {
    /// 指定变更记录 ID；省略时撤销符合条件的最近一次变更。
    #[arg(long)]
    change: Option<Uuid>,
}

#[derive(Debug, Subcommand)]
enum SessionCommand {
    /// 按更新时间列出会话。
    List {
        /// 最多显示的会话数量。
        #[arg(long, default_value_t = 20, value_parser = clap::value_parser!(u16).range(1..=200))]
        limit: u16,
    },
    /// 显示一个会话的完整记录。
    Show { id: Uuid },
    /// 恢复一个会话；省略指令时进入交互模式。
    Resume { id: Uuid, prompt: Option<String> },
}

#[derive(Debug, Subcommand)]
enum PlanCommand {
    Create {
        goal: String,
    },
    List {
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    Show {
        id: Uuid,
    },
    Revisions {
        id: Uuid,
    },
    Approve {
        id: Uuid,
        #[arg(long, default_value = "用户通过命令行批准计划。")]
        reason: String,
    },
    Reject {
        id: Uuid,
        #[arg(long, default_value = "用户通过命令行拒绝计划。")]
        reason: String,
    },
    Revise {
        id: Uuid,
        request: String,
    },
    Run {
        id: Uuid,
    },
    Retry {
        id: Uuid,
    },
    Cancel {
        id: Uuid,
    },
}

#[derive(Debug, Subcommand)]
enum AuthCommand {
    /// 通过隐藏输入把 API Key 保存到系统凭据存储。
    Login { provider: Option<String> },
    /// 查看环境变量或系统凭据是否已配置。
    Status { provider: Option<String> },
    /// 从系统凭据存储删除 API Key。
    Logout { provider: Option<String> },
}

#[derive(Debug, Subcommand)]
enum ConfigCommand {
    /// 显示脱敏后的最终配置和每项来源。
    Show,
    /// 显示某项配置的最终值与来源。
    Explain { key: String },
    /// 写入非秘密配置项。
    Set {
        key: String,
        value: String,
        /// 写入用户配置；默认写入当前项目配置。
        #[arg(long, conflicts_with = "project")]
        user: bool,
        /// 明确写入当前项目配置。
        #[arg(long)]
        project: bool,
    },
    /// 显示用户配置与项目配置路径。
    Path,
}

#[derive(Debug, Subcommand)]
enum ApprovalCommand {
    /// 列出永久允许的同类工具。
    List,
    /// 按工具名撤销永久审批规则。
    Revoke { tool: String },
    /// 清除全部永久审批规则。
    Clear,
}

#[derive(Debug, Subcommand)]
enum MemoryCommand {
    /// 查看整理后的 MEMORY.md。
    List,
    /// 使用 $VISUAL 或 $EDITOR 编辑 MEMORY.md。
    Edit,
    /// 显示 MEMORY.md 的本地路径。
    Path,
}

struct Runtime {
    provider: Arc<dyn Provider>,
    provider_display: String,
    model: String,
    max_turns: u32,
    /// 上下文软阈值（窗口×90%）：超出触发自动压缩。
    context_budget: usize,
    /// 上下文硬顶（窗口×95%）：真实用量超过时强制压缩。
    context_hard_limit: usize,
    /// 段轮次预算用尽时是否自动续跑。
    auto_continue: bool,
    /// 总轮次预算硬上限（所有续跑段累计）。
    max_total_turns: u32,
    cwd: PathBuf,
    permission_mode: PermissionMode,
    /// 运行中可变的共享权限模式（Shift+Tab 切换即时生效）。
    shared_permission: Arc<std::sync::Mutex<PermissionMode>>,
    /// /plan 生成期间锁定只读时保存的原权限模式，执行/取消时恢复。
    plan_restore_mode: Option<PermissionMode>,
    /// 任务完成后是否由 Agent 自动提炼并保存长期记忆。
    memory_suggest_enabled: bool,
    registry: ToolRegistry,
    store: Arc<SqliteSessionStore>,
    renderer: ConsoleRenderer,
    stream: bool,
    color: bool,
    /// TUI 配色主题：dark | light | auto。
    theme: String,
    debug_trace: bool,
    startup_notices: Vec<String>,
    /// TUI 会话共享输入路由；非 TUI 模式保持未激活。
    input_router: Arc<InputRouter>,
    /// 模型请求采样温度。
    temperature: f32,
    /// 单次模型请求最大输出 Token。
    max_output_tokens: u32,
    /// 是否启用内部思考闭环。
    reasoning: bool,
    /// 停滞后的恢复策略。
    stalled_recovery: StalledRecoveryMode,
    /// 停滞恢复模式下允许的最大恢复尝试次数。
    stalled_max_recovery: u32,
    /// 可用技能索引（来自目录发现）。
    skills: Vec<Skill>,
    /// 技能加载策略（allow | ask | deny）。
    skills_mode: SkillMode,
    /// 记忆注入的最大条数与 Token 预算。
    memory_top_k: usize,
    memory_injection_budget: usize,
    /// 强制压缩标志：`/compact` 置位，运行中的 Agent 在下一轮请求前压缩。
    force_compact: Arc<std::sync::atomic::AtomicBool>,
    /// Agent 档案（内置 + 自定义），供 `task` 委派子代理。
    profiles: Vec<AgentProfile>,
}

fn overrides(cli: &Cli) -> ConfigOverrides {
    ConfigOverrides {
        provider: cli.provider.clone(),
        model: cli.model.clone(),
        base_url: cli.base_url.clone(),
        max_turns: cli.max_turns,
        permission: cli.permission.clone(),
        approval: cli.approval.clone(),
        json: cli.json.then_some(true),
        no_stream: cli.no_stream.then_some(true),
        color: cli.no_color.then_some(false),
        debug_trace: cli.debug_trace.then_some(true),
        telemetry_enabled: None,
        memory_suggest_enabled: None,
        temperature: cli.temperature,
        max_output_tokens: cli.max_output_tokens,
        reasoning: cli.reasoning.then_some(true),
        stalled_recovery: cli.stalled_recovery.clone(),
        stalled_max_recovery: cli.stalled_max_recovery,
    }
}

fn provider_label(name: &str) -> String {
    match name {
        "anthropic" => "Anthropic",
        "deepseek" => "DeepSeek",
        other => other,
    }
    .to_owned()
}

#[derive(Debug)]
struct ConsoleApprovalGate {
    can_prompt: bool,
    fullscreen: bool,
    theme: TerminalTheme,
    session_rules: tokio::sync::Mutex<BTreeSet<(Uuid, ApprovalRule)>>,
    persistent_rules: JsonApprovalRuleStore,
    /// TUI 会话安装的共享输入路由；激活时审批菜单从路由消费事件。
    router: Option<Arc<InputRouter>>,
    /// accept-edits 模式：工作区文件编辑自动接受（本会话），
    /// 命令与网络访问仍走 ask 流程。
    accept_edits: bool,
}

impl ConsoleApprovalGate {
    fn new(
        can_prompt: bool,
        persistent_rules: JsonApprovalRuleStore,
        theme: TerminalTheme,
        fullscreen: bool,
        router: Option<Arc<InputRouter>>,
        accept_edits: bool,
    ) -> Self {
        Self {
            can_prompt,
            fullscreen,
            theme,
            session_rules: tokio::sync::Mutex::new(BTreeSet::new()),
            persistent_rules,
            router,
            accept_edits,
        }
    }
}

#[async_trait]
impl ApprovalGate for ConsoleApprovalGate {
    async fn review(&self, request: &ApprovalRequest) -> ApprovalDecision {
        let rule = ApprovalRule::from_request(request);
        // accept-edits：工作区文件编辑自动接受（本会话），其余仍询问。
        if self.accept_edits && request.side_effect == SideEffectKind::WorkspaceWrite {
            self.session_rules
                .lock()
                .await
                .insert((request.session_id, rule.clone()));
            return ApprovalDecision::approve_with_scope(
                "accept-edits 模式自动接受工作区文件编辑。",
                ApprovalScope::Session,
            );
        }
        if self.persistent_rules.contains(&rule).await {
            return ApprovalDecision::approve_with_scope(
                "命中用户永久审批规则。",
                ApprovalScope::Always,
            );
        }
        if self
            .session_rules
            .lock()
            .await
            .contains(&(request.session_id, rule.clone()))
        {
            return ApprovalDecision::approve_with_scope(
                "命中当前会话审批规则。",
                ApprovalScope::Session,
            );
        }
        if !self.can_prompt {
            return ApprovalDecision::deny("当前运行方式无法交互审批，且没有匹配的永久审批规则。");
        }
        let theme = self.theme;
        let fullscreen = self.fullscreen;
        let router = self.router.clone();
        let prompt = format_approval_prompt(theme, request);
        let choice = match read_approval_menu(theme, &prompt, router, fullscreen).await {
            Ok(Some(choice)) => Some(choice),
            _ => None,
        };
        match choice {
            Some(ApprovalMenuChoice::Once) => {
                ApprovalDecision::approve_with_scope("用户批准当前工具调用。", ApprovalScope::Once)
            }
            Some(ApprovalMenuChoice::Session) => {
                self.session_rules
                    .lock()
                    .await
                    .insert((request.session_id, rule));
                ApprovalDecision::approve_with_scope(
                    "用户批准本会话中的同类工具调用。",
                    ApprovalScope::Session,
                )
            }
            Some(ApprovalMenuChoice::Always) => match self.persistent_rules.allow(rule).await {
                Ok(()) => ApprovalDecision::approve_with_scope(
                    format!(
                        "用户永久批准同类工具调用；规则已保存到 {}。",
                        self.persistent_rules.path().display()
                    ),
                    ApprovalScope::Always,
                ),
                Err(error) => ApprovalDecision::deny(format!(
                    "无法保存永久审批规则，本次未执行：{}",
                    error.message
                )),
            },
            Some(ApprovalMenuChoice::Deny) => ApprovalDecision::deny("用户拒绝或未明确批准。"),
            None => ApprovalDecision::deny("审批界面已关闭或无法读取审批输入。"),
        }
    }
}

async fn create_runtime(
    cwd: PathBuf,
    resolved: &ResolvedConfig,
    interactive: bool,
) -> Result<Runtime, XduduError> {
    let change_ledger = Arc::new(JsonChangeLedger::new(&cwd));
    change_ledger.recover_incomplete().await?;
    let store = KeyringSecretStore;
    let (secret, _) = resolve_secret(&resolved.config.provider.name, &store).await?;
    let provider = Arc::from(DefaultProviderFactory.create(&resolved.config.provider, secret)?);
    let approval_mode = resolved.config.agent.approval_mode()?;
    let color = resolved.config.output.color && io::stdout().is_terminal();
    let theme = TerminalTheme::new(color);
    let rich_terminal = interactive
        && io::stdin().is_terminal()
        && io::stdout().is_terminal()
        && env::var("TERM").is_ok_and(|term| term != "dumb");
    let input_router = Arc::new(InputRouter::new());
    let approval_gate: Arc<dyn ApprovalGate> = match approval_mode {
        ApprovalMode::Always => Arc::new(AllowAllApprovalGate),
        ApprovalMode::Never => Arc::new(DenyAllApprovalGate),
        ApprovalMode::AcceptEdits | ApprovalMode::Ask => Arc::new(ConsoleApprovalGate::new(
            interactive && !resolved.config.output.json,
            JsonApprovalRuleStore::open(approval_rules_path()?).await?,
            theme,
            rich_terminal,
            Some(Arc::clone(&input_router)),
            approval_mode == ApprovalMode::AcceptEdits,
        )),
    };
    let mut registry = ToolRegistry::with_runtime(approval_gate, change_ledger);
    registry.set_command_rules(resolved.config.agent.commands.clone());
    let skills_mode = resolved.config.agent.skills_mode()?;
    let (skills, skill_warnings) = discover_skills(&cwd);
    if skills_mode != SkillMode::Deny {
        registry.register(SkillTool::new(skills.clone(), skills_mode))?;
    }
    register_builtins(&mut registry)?;
    // web_read 提炼复用当前 Provider（独立请求、不进会话历史）。
    registry.register(WebReadTool::new(
        Some(Arc::clone(&provider)),
        resolved.config.provider.model.clone(),
        resolved.config.provider.temperature,
        resolved.config.provider.max_output_tokens,
        resolved.config.provider.reasoning,
    ))?;
    let mcp = register_configured_mcp_tools(&mut registry).await?;
    // 合并内置与自定义 Agent 档案（自定义不与内置同名）。
    let (profiles, profile_conflicts) = merge_profiles(resolved.config.agent.profiles.clone());
    let profile_warnings = profile_conflicts
        .into_iter()
        .map(|id| format!("自定义档案 {id} 与内置档案重名，已忽略。"))
        .collect::<Vec<_>>();
    Ok(Runtime {
        provider,
        provider_display: provider_label(&resolved.config.provider.name),
        model: resolved.config.provider.model.clone(),
        max_turns: resolved.config.agent.max_turns,
        context_budget: resolved.config.provider.context_budget()?,
        context_hard_limit: resolved.config.provider.context_hard_limit()?,
        auto_continue: resolved.config.agent.auto_continue,
        max_total_turns: resolved.config.agent.max_total_turns()?,
        cwd: cwd.clone(),
        permission_mode: resolved.config.agent.permission_mode()?,
        shared_permission: Arc::new(std::sync::Mutex::new(
            resolved.config.agent.permission_mode()?,
        )),
        plan_restore_mode: None,
        memory_suggest_enabled: resolved.config.memory.suggest_enabled,
        registry,
        store: Arc::new(SqliteSessionStore::new(&cwd)?),
        renderer: ConsoleRenderer::new(
            resolved.config.output.json,
            !resolved.config.output.no_stream,
            color,
            resolved.config.output.debug_trace,
        ),
        stream: !resolved.config.output.no_stream,
        color,
        theme: resolved.config.output.theme.clone(),
        debug_trace: resolved.config.output.debug_trace,
        startup_notices: mcp
            .failures
            .into_iter()
            .chain(skill_warnings)
            .chain(profile_warnings)
            .collect(),
        input_router,
        temperature: resolved.config.provider.temperature,
        max_output_tokens: resolved.config.provider.max_output_tokens,
        reasoning: resolved.config.provider.reasoning,
        stalled_recovery: resolved.config.agent.stalled_recovery_mode()?,
        stalled_max_recovery: resolved.config.agent.stalled_max_recovery,
        skills,
        skills_mode,
        memory_top_k: resolved.config.memory.top_k,
        memory_injection_budget: resolved.config.memory.injection_token_budget,
        force_compact: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        profiles,
    })
}

async fn execute_prompt(
    runtime: &Runtime,
    prompt: String,
    session_id: Option<Uuid>,
) -> Result<AgentRunResult, XduduError> {
    runtime.renderer.begin_run();
    let result =
        execute_prompt_with_sink(runtime, prompt, session_id, &runtime.renderer, true).await?;
    if result.status == SessionStatus::Completed {
        auto_capture_memories(runtime, result.session_id).await;
    }
    Ok(result)
}

async fn execute_prompt_with_sink(
    runtime: &Runtime,
    prompt: String,
    session_id: Option<Uuid>,
    event_sink: &dyn EventSink,
    print_interrupt: bool,
) -> Result<AgentRunResult, XduduError> {
    let cancellation = CancellationToken::new();
    let run = execute_prompt_with_cancellation(
        runtime,
        prompt,
        session_id,
        event_sink,
        cancellation.clone(),
    );
    tokio::pin!(run);
    tokio::select! {
        result = &mut run => result,
        signal = tokio::signal::ctrl_c() => {
            if signal.is_ok() {
                if print_interrupt {
                    eprintln!("\n  ⏸  已中断，正在保存...");
                }
                cancellation.cancel();
            }
            run.await
        }
    }
}

async fn execute_prompt_with_cancellation(
    runtime: &Runtime,
    prompt: String,
    session_id: Option<Uuid>,
    event_sink: &dyn EventSink,
    cancellation: CancellationToken,
) -> Result<AgentRunResult, XduduError> {
    let memories = relevant_memories(runtime, &prompt, session_id).await;
    run_agent(AgentRunConfig {
        prompt,
        model: runtime.model.clone(),
        max_turns: runtime.max_turns,
        auto_continue: runtime.auto_continue,
        max_total_turns: runtime.max_total_turns,
        cwd: runtime.cwd.clone(),
        provider: runtime.provider.as_ref(),
        tool_registry: &runtime.registry,
        session_store: runtime.store.as_ref(),
        permission_mode: Arc::clone(&runtime.shared_permission),
        cancellation,
        session_id,
        event_sink: Some(event_sink),
        stream: runtime.stream,
        memories,
        temperature: runtime.temperature,
        max_output_tokens: runtime.max_output_tokens,
        reasoning: runtime.reasoning,
        stalled_recovery: runtime.stalled_recovery,
        stalled_max_recovery: runtime.stalled_max_recovery,
        context_budget: runtime.context_budget,
        context_hard_limit: runtime.context_hard_limit,
        skills: runtime.skills.clone(),
        force_compact: Arc::clone(&runtime.force_compact),
        profiles: runtime.profiles.clone(),
        injections: None,
    })
    .await
}

/// 用当前提示词检索相关记忆：FTS5 召回 → 查询词命中精排 → 归一化去重 →
/// Token 预算裁减 → 上限截断。检索失败时静默返回空，不影响任务执行。
async fn relevant_memories(
    runtime: &Runtime,
    query: &str,
    session_id: Option<Uuid>,
) -> Vec<String> {
    // 用户可读的 MEMORY.md 是运行时首选记忆。SQLite 原始记录仅作为生成、
    // 审计和文件尚未生成时的回退，不再直接暴露 UUID 列表给模型。
    if let Ok(Some(document)) = read_memory_document(&runtime.cwd) {
        let max_chars = runtime.memory_injection_budget.saturating_mul(3).max(512);
        let content = document.chars().take(max_chars).collect::<String>();
        return vec![format!("长期记忆文档（不可信背景，仅供参考）：\n{content}")];
    }
    // 查询词拼接最近助手文本，提高召回质量。
    let mut combined = query.to_owned();
    if let Some(id) = session_id
        && let Ok(Some(session)) = runtime.store.get(id).await
        && let Some(recent) = session.messages.iter().rev().find(|message| {
            message.role == xdudu_core::provider::MessageRole::Assistant
                && !message.content.trim().is_empty()
        })
    {
        combined.push('\n');
        combined.push_str(recent.content.trim());
    }
    let Ok(memories) = runtime
        .store
        .search_memories(&combined, MEMORY_FTS_RECALL)
        .await
    else {
        return Vec::new();
    };
    let tokens = query_tokens(&combined);
    let mut seen = std::collections::HashSet::new();
    let mut selected = Vec::new();
    let mut budget = runtime.memory_injection_budget;
    for memory in rank_memories(memories, &tokens) {
        let normalized = memory
            .content
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        if !seen.insert(normalized) {
            continue;
        }
        let cost = estimated_tokens(&memory.content);
        if cost > budget {
            continue;
        }
        let source = memory
            .source_session_id
            .map(|id| {
                format!(
                    "（来源会话 {}）",
                    id.to_string().chars().take(8).collect::<String>()
                )
            })
            .unwrap_or_default();
        selected.push(format!("- {source}{}", memory.content.trim()));
        budget -= cost;
        if selected.len() >= runtime.memory_top_k {
            break;
        }
    }
    selected
}

/// FTS5 召回上限（远大于注入上限，供精排裁减）。
const MEMORY_FTS_RECALL: usize = 16;

/// 把查询拆成词元（空白分隔、长度 ≥ 2），用于命中计数精排。
fn query_tokens(query: &str) -> Vec<String> {
    query
        .split(|character: char| !character.is_alphanumeric() && character != '_')
        .filter(|token| token.chars().count() >= 2)
        .map(str::to_owned)
        .collect()
}

/// 相关性精排：命中查询词越多的记忆越靠前；并列时新记忆优先。
fn rank_memories(memories: Vec<MemoryRecord>, tokens: &[String]) -> Vec<MemoryRecord> {
    let mut scored = memories
        .into_iter()
        .map(|memory| {
            let hits = tokens
                .iter()
                .filter(|token| memory.content.contains(token.as_str()))
                .count();
            (memory, hits)
        })
        .collect::<Vec<_>>();
    scored.sort_by(|left, right| {
        right
            .1
            .cmp(&left.1)
            .then(right.0.updated_at.cmp(&left.0.updated_at))
    });
    scored.into_iter().map(|(memory, _)| memory).collect()
}

/// 字符加权 Token 估算：`ceil((ascii 字节 + 2×CJK 字符) / 3.5) + 8`。
fn estimated_tokens(text: &str) -> usize {
    let ascii_bytes = text.bytes().filter(|byte| byte.is_ascii()).count();
    let cjk_chars = text
        .chars()
        .filter(|character| ('\u{4e00}'..='\u{9fff}').contains(character))
        .count();
    (((ascii_bytes + 2 * cjk_chars) as f64) / 3.5).ceil() as usize + 8
}

fn print_banner(runtime: &Runtime, interactive: bool) {
    if !interactive {
        println!(
            "{}",
            ui::compact_banner(
                TerminalTheme::new(runtime.color),
                env!("CARGO_PKG_VERSION"),
                &runtime.provider_display,
                &runtime.model,
            )
        );
    }
}

async fn interactive_loop(
    runtime: Runtime,
    initial_prompt: Option<String>,
    initial_session: Option<Uuid>,
) -> Result<u8, XduduError> {
    if interactive_terminal() {
        tui_interactive_loop(runtime, initial_prompt, initial_session).await
    } else {
        plain_interactive_loop(runtime, initial_prompt, initial_session).await
    }
}

fn interactive_terminal() -> bool {
    io::stdin().is_terminal()
        && io::stdout().is_terminal()
        && env::var("TERM").is_ok_and(|term| term != "dumb")
}

async fn tui_interactive_loop(
    mut runtime: Runtime,
    initial_prompt: Option<String>,
    initial_session: Option<Uuid>,
) -> Result<u8, XduduError> {
    let available_tools = runtime
        .registry
        .definitions()
        .into_iter()
        .map(|definition| definition.name.to_owned())
        .collect();
    let skills = runtime
        .skills
        .iter()
        .map(|skill| skill.name.clone())
        .collect();
    let context = TuiContext {
        provider: runtime.provider_display.clone(),
        model: runtime.model.clone(),
        cwd: runtime.cwd.clone(),
        permission: runtime.permission_mode.as_str().to_owned(),
        available_tools,
        skills,
        color: runtime.color,
        debug_trace: runtime.debug_trace,
        theme: runtime.theme.clone(),
    };
    let router = Arc::clone(&runtime.input_router);
    let (app, _screen) = TuiApp::enter(context, Arc::clone(&router)).map_err(XduduError::from)?;
    let renderer = app.renderer();

    // 启动终端输入生产者：raw 模式会话期间唯一读取终端事件的任务。
    router.set_active(true);
    let shutdown = router.shutdown_token();
    let producer_router = Arc::clone(&router);
    let producer_shutdown = shutdown.clone();
    let producer = tokio::spawn(async move {
        let mut events = EventStream::new();
        loop {
            tokio::select! {
                () = producer_shutdown.cancelled() => break,
                event = events.next() => match event {
                    Some(Ok(event)) => producer_router.produce(event),
                    Some(Err(_)) | None => break,
                },
            }
        }
        producer_router.close();
    });

    for notice in &runtime.startup_notices {
        app.notice(notice).map_err(XduduError::from)?;
    }

    let mut session_id = initial_session;
    if let Some(id) = session_id {
        let session = session_for_resume(&runtime, id).await?;
        app.load_session(&session).map_err(XduduError::from)?;
        if let Some(plan) = pending_plan_for_session(&runtime, Some(id)).await? {
            review_plan_in_tui(&runtime, &app, plan).await?;
        } else if let Some(plan) = paused_plan_for_session(&runtime, Some(id)).await? {
            recover_plan_in_tui(&runtime, &app, plan).await?;
        }
    }

    // 启动时静默检查 GitHub Releases 新版本；结果在空闲时提示。
    let update_available: Arc<std::sync::Mutex<Option<String>>> =
        Arc::new(std::sync::Mutex::new(None));
    {
        let slot = Arc::clone(&update_available);
        tokio::spawn(async move {
            if let Some(latest) = version_check::fetch_latest_version().await
                && version_check::is_newer(env!("CARGO_PKG_VERSION"), &latest)
            {
                *slot.lock().unwrap() = Some(latest);
            }
        });
    }

    let mut pending: VecDeque<String> = VecDeque::new();
    let mut run_future: Option<RunFuture> = None;
    let mut run_cancel: Option<CancellationToken> = None;
    let mut run_inject: Option<tokio::sync::mpsc::UnboundedSender<String>> = None;
    if let Some(prompt) = initial_prompt {
        start_tui_run(
            &app,
            &runtime,
            &renderer,
            prompt,
            session_id,
            &mut run_future,
            &mut run_cancel,
            &mut run_inject,
        )
        .await?;
    }

    loop {
        if let Some(future) = run_future.as_mut() {
            let outcome = tokio::select! {
                result = future => TuiSelect::Finished(result),
                event = router.next_for(InputFocus::Composer) => {
                    match event {
                        Some(event) => TuiSelect::Event(handle_tui_running_event(
                            &app, event, &mut pending, run_cancel.as_ref(),
                            run_inject.as_ref(),
                        ).await?),
                        None => {
                            if let Some(cancel) = run_cancel.as_ref() {
                                cancel.cancel();
                            }
                            TuiSelect::Event(TuiLoopAction::Exit)
                        },
                    }
                }
            };
            match outcome {
                TuiSelect::Finished(result) => {
                    let result = result?;
                    run_future = None;
                    run_cancel = None;
                    run_inject = None;
                    app.finish_prompt(&result).map_err(XduduError::from)?;
                    session_id = Some(result.session_id);
                    // 任务完成后由模型自主判断是否存在值得长期保留的信息；
                    // TUI 中放入后台，避免额外模型请求阻塞下一次输入。
                    if result.status == xdudu_core::SessionStatus::Completed {
                        spawn_auto_capture_memories(&runtime, result.session_id);
                    }
                }
                TuiSelect::Event(TuiLoopAction::Exit) => break,
                TuiSelect::Event(TuiLoopAction::Continue) => {}
            }
            // 运行结束后接续排队任务（在 select 之外）。
            if run_future.is_none()
                && let Some(next) = pending.pop_front()
            {
                match handle_queued_tui_input(
                    &app,
                    &mut runtime,
                    &renderer,
                    next,
                    &mut session_id,
                    &mut run_future,
                    &mut run_cancel,
                    &mut run_inject,
                )
                .await?
                {
                    TuiLoopAction::Exit => break,
                    TuiLoopAction::Continue => {}
                }
            }
        } else {
            if let Some(latest) = update_available.lock().unwrap().take() {
                app.notice(format!(
                    "发现新版本 v{latest}：可通过 cargo install --path crates/xdudu-cli --locked --force 升级。"
                ))
                .map_err(XduduError::from)?;
            }
            if let Some(next) = pending.pop_front() {
                match handle_queued_tui_input(
                    &app,
                    &mut runtime,
                    &renderer,
                    next,
                    &mut session_id,
                    &mut run_future,
                    &mut run_cancel,
                    &mut run_inject,
                )
                .await?
                {
                    TuiLoopAction::Exit => break,
                    TuiLoopAction::Continue => {}
                }
                continue;
            }
            let Some(event) = router.next_for(InputFocus::Composer).await else {
                break;
            };
            match handle_tui_idle_event(
                &app,
                &mut runtime,
                &renderer,
                event,
                &mut session_id,
                &mut run_future,
                &mut run_cancel,
                &mut run_inject,
            )
            .await?
            {
                TuiLoopAction::Exit => break,
                TuiLoopAction::Continue => {}
            }
        }
    }

    router.set_active(false);
    shutdown.cancel();
    producer.abort();
    let _ = producer.await;
    Ok(0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TuiLoopAction {
    Continue,
    Exit,
}

enum TuiSelect {
    Finished(Result<AgentRunResult, XduduError>),
    Event(TuiLoopAction),
}

/// 运行中的事件处理：Ctrl+C 取消当前任务；普通提示注入当前任务下一轮，
/// 斜杠命令排队等任务结束后执行；其余按键编辑 Composer。
async fn handle_tui_running_event(
    app: &TuiApp,
    event: Event,
    pending: &mut VecDeque<String>,
    run_cancel: Option<&CancellationToken>,
    run_inject: Option<&tokio::sync::mpsc::UnboundedSender<String>>,
) -> Result<TuiLoopAction, XduduError> {
    match event {
        Event::Paste(text) => {
            app.handle_paste(&text).map_err(XduduError::from)?;
        }
        Event::Resize(_, _) => {
            app.renderer().draw_dynamic().map_err(XduduError::from)?;
        }
        Event::Key(key) => {
            if is_ctrl_c(&key) || is_escape(&key) {
                if let Some(cancel) = run_cancel {
                    cancel.cancel();
                }
                app.notice("已中断当前任务，正在停止…")
                    .map_err(XduduError::from)?;
                return Ok(TuiLoopAction::Continue);
            }
            if is_ctrl_d(&key) {
                return Ok(TuiLoopAction::Continue);
            }
            if is_backtab(&key) {
                app.notice("权限模式将在下一个任务生效：请在任务结束后按 Shift+Tab 切换。")
                    .map_err(XduduError::from)?;
                return Ok(TuiLoopAction::Continue);
            }
            match app.handle_key(key).map_err(XduduError::from)? {
                Some(InputOutcome::Submit(line)) => {
                    // 普通提示：注入当前任务的下一轮，立即在历史区可见。
                    if let Some(inject) = run_inject {
                        app.inject_prompt(&line).map_err(XduduError::from)?;
                        let _ = inject.send(line);
                        app.notice("已并入当前任务：将在下一轮送达模型。")
                            .map_err(XduduError::from)?;
                    } else {
                        pending.push_back(line);
                        app.notice("已排队，当前任务结束后自动执行。")
                            .map_err(XduduError::from)?;
                    }
                }
                Some(InputOutcome::Command(line)) => {
                    pending.push_back(line);
                    app.notice("已排队，当前任务结束后自动执行。")
                        .map_err(XduduError::from)?;
                }
                Some(InputOutcome::ToggleFold) => {
                    app.renderer()
                        .toggle_last_group()
                        .map_err(XduduError::from)?;
                }
                Some(InputOutcome::Interrupted | InputOutcome::Exit) | None => {}
            }
        }
        _ => {}
    }
    Ok(TuiLoopAction::Continue)
}

/// 空闲事件处理：正常 Composer 交互，提交后启动任务或执行命令。
#[allow(clippy::too_many_arguments)]
async fn handle_tui_idle_event(
    app: &TuiApp,
    runtime: &mut Runtime,
    renderer: &TuiRenderer,
    event: Event,
    session_id: &mut Option<Uuid>,
    run_future: &mut Option<RunFuture>,
    run_cancel: &mut Option<CancellationToken>,
    run_inject: &mut Option<tokio::sync::mpsc::UnboundedSender<String>>,
) -> Result<TuiLoopAction, XduduError> {
    match event {
        Event::Paste(text) => {
            app.handle_paste(&text).map_err(XduduError::from)?;
        }
        Event::Resize(_, _) => {
            app.renderer().draw_dynamic().map_err(XduduError::from)?;
        }
        Event::Key(key) => {
            if is_backtab(&key) {
                let next = next_permission_mode(runtime.permission_mode);
                runtime.permission_mode = next;
                *runtime.shared_permission.lock().unwrap() = next;
                app.set_permission(next.as_str())
                    .map_err(XduduError::from)?;
                return Ok(TuiLoopAction::Continue);
            }
            match app.handle_key(key).map_err(XduduError::from)? {
                Some(InputOutcome::Submit(line)) => {
                    start_tui_run(
                        app,
                        runtime,
                        renderer,
                        line,
                        *session_id,
                        run_future,
                        run_cancel,
                        run_inject,
                    )
                    .await?;
                }
                Some(InputOutcome::Command(input)) => {
                    handle_tui_command(app, runtime, renderer, &input, session_id).await?;
                }
                Some(InputOutcome::Interrupted) => {
                    app.notice("再次按 Ctrl+D 或输入 /exit 可退出。")
                        .map_err(XduduError::from)?;
                }
                Some(InputOutcome::ToggleFold) => {
                    app.renderer()
                        .toggle_last_group()
                        .map_err(XduduError::from)?;
                }
                Some(InputOutcome::Exit) => return Ok(TuiLoopAction::Exit),
                None => {}
            }
        }
        _ => {}
    }
    Ok(TuiLoopAction::Continue)
}

/// 应用权限模式到运行时与共享引用，并同步界面徽标。
fn apply_permission_mode(
    app: &TuiApp,
    runtime: &mut Runtime,
    mode: PermissionMode,
) -> Result<(), XduduError> {
    runtime.permission_mode = mode;
    *runtime.shared_permission.lock().unwrap() = mode;
    app.set_permission(mode.as_str())
        .map_err(XduduError::from)?;
    Ok(())
}

/// 审阅结束后若计划被拒绝，恢复生成期锁定的原权限模式。
async fn maybe_restore_after_review(
    app: &TuiApp,
    runtime: &mut Runtime,
    plan_id: Uuid,
) -> Result<(), XduduError> {
    if runtime.plan_restore_mode.is_none() {
        return Ok(());
    }
    if let Some(plan) = runtime.store.get_plan(plan_id).await?
        && plan.status == PlanStatus::Rejected
    {
        restore_plan_mode(app, runtime)?;
    }
    Ok(())
}

/// 恢复 plan 生成期间锁定的原权限模式（执行/取消/拒绝/失败时调用）。
fn restore_plan_mode(app: &TuiApp, runtime: &mut Runtime) -> Result<(), XduduError> {
    if let Some(mode) = runtime.plan_restore_mode.take() {
        apply_permission_mode(app, runtime, mode)?;
        app.notice(format!("已恢复权限模式：{}。", mode.as_str()))
            .map_err(XduduError::from)?;
    }
    Ok(())
}

/// 权限模式循环：read-only → auto-safe → full-access。
fn next_permission_mode(current: PermissionMode) -> PermissionMode {
    match current {
        PermissionMode::ReadOnly => PermissionMode::AutoSafe,
        PermissionMode::AutoSafe => PermissionMode::FullAccess,
        PermissionMode::FullAccess => PermissionMode::ReadOnly,
    }
}

fn is_backtab(key: &KeyEvent) -> bool {
    key.kind == KeyEventKind::Press && key.code == KeyCode::BackTab
}

/// 排队输入：命令直接执行，普通提示启动新任务。
#[allow(clippy::too_many_arguments)]
async fn handle_queued_tui_input(
    app: &TuiApp,
    runtime: &mut Runtime,
    renderer: &TuiRenderer,
    input: String,
    session_id: &mut Option<Uuid>,
    run_future: &mut Option<RunFuture>,
    run_cancel: &mut Option<CancellationToken>,
    run_inject: &mut Option<tokio::sync::mpsc::UnboundedSender<String>>,
) -> Result<TuiLoopAction, XduduError> {
    if input.trim_start().starts_with('/') {
        handle_tui_command(app, runtime, renderer, input.trim(), session_id).await
    } else {
        start_tui_run(
            app,
            runtime,
            renderer,
            input,
            *session_id,
            run_future,
            run_cancel,
            run_inject,
        )
        .await?;
        Ok(TuiLoopAction::Continue)
    }
}

/// 启动一次 Agent 任务：composer 状态更新 + 可取消的 Future。
///
/// Future 通过克隆捕获运行所需的全部片段，不借用 `runtime`，因此任务
/// 运行期间 `/model`、`/turns` 等命令仍可修改运行时配置。
#[allow(clippy::too_many_arguments)]
async fn start_tui_run(
    app: &TuiApp,
    runtime: &Runtime,
    renderer: &TuiRenderer,
    prompt: String,
    session_id: Option<Uuid>,
    run_future: &mut Option<RunFuture>,
    run_cancel: &mut Option<CancellationToken>,
    run_inject: &mut Option<tokio::sync::mpsc::UnboundedSender<String>>,
) -> Result<(), XduduError> {
    app.begin_prompt(&prompt).map_err(XduduError::from)?;
    let cancellation = CancellationToken::new();
    let (inject_tx, inject_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    *run_inject = Some(inject_tx);
    let provider = Arc::clone(&runtime.provider);
    let registry = runtime.registry.clone();
    let store = Arc::clone(&runtime.store);
    let model = runtime.model.clone();
    let max_turns = runtime.max_turns;
    let cwd = runtime.cwd.clone();
    let permission_mode = Arc::clone(&runtime.shared_permission);
    let stream = runtime.stream;
    let run_renderer = renderer.clone();
    let task_cancel = cancellation.clone();
    let temperature = runtime.temperature;
    let max_output_tokens = runtime.max_output_tokens;
    let reasoning = runtime.reasoning;
    let stalled_recovery = runtime.stalled_recovery;
    let stalled_max_recovery = runtime.stalled_max_recovery;
    let context_budget = runtime.context_budget;
    let context_hard_limit = runtime.context_hard_limit;
    let auto_continue = runtime.auto_continue;
    let max_total_turns = runtime.max_total_turns;
    let skills = runtime.skills.clone();
    let force_compact = Arc::clone(&runtime.force_compact);
    let profiles = runtime.profiles.clone();
    let memories = relevant_memories(runtime, &prompt, session_id).await;
    let future: RunFuture = Box::pin(async move {
        run_agent(AgentRunConfig {
            prompt,
            model,
            max_turns,
            auto_continue,
            max_total_turns,
            cwd,
            provider: provider.as_ref(),
            tool_registry: &registry,
            session_store: store.as_ref(),
            permission_mode,
            cancellation: task_cancel,
            session_id,
            event_sink: Some(&run_renderer),
            stream,
            memories,
            temperature,
            max_output_tokens,
            reasoning,
            stalled_recovery,
            stalled_max_recovery,
            context_budget,
            context_hard_limit,
            skills,
            force_compact,
            profiles,
            injections: Some(inject_rx),
        })
        .await
    });
    *run_future = Some(future);
    *run_cancel = Some(cancellation);
    Ok(())
}

/// 计划执行期间保持 Composer 可用：Ctrl+C 取消当前计划执行。
async fn run_plan_in_tui(
    app: &TuiApp,
    runtime: &Runtime,
    renderer: &TuiRenderer,
    plan_id: Uuid,
) -> Result<xdudu_core::PlanExecutionResult, XduduError> {
    let router = Arc::clone(&runtime.input_router);
    let cancellation = CancellationToken::new();
    let plan_future =
        execute_plan_with_cancellation(runtime, plan_id, renderer, cancellation.clone());
    tokio::pin!(plan_future);
    loop {
        tokio::select! {
            result = &mut plan_future => return result,
            event = router.next_for(InputFocus::Composer) => {
                match event {
                    Some(Event::Paste(text)) => {
                        app.handle_paste(&text).map_err(XduduError::from)?;
                    }
                    Some(Event::Resize(_, _)) => {
                        app.renderer().draw_dynamic().map_err(XduduError::from)?;
                    }
                    Some(Event::Key(key)) if is_ctrl_c(&key) => {
                        cancellation.cancel();
                        app.notice("已中断计划执行，正在保存现场…")
                            .map_err(XduduError::from)?;
                    }
                    Some(Event::Key(key)) if is_plain_enter(&key) => {
                        app.notice("计划执行中：输入已保留，计划结束后再按 Enter 发送。")
                            .map_err(XduduError::from)?;
                    }
                    Some(Event::Key(key)) => {
                        app.handle_key(key).map_err(XduduError::from)?;
                    }
                    None => {
                        cancellation.cancel();
                        return Err(XduduError::validation("终端输入已关闭。"));
                    }
                    _ => {}
                }
            }
        }
    }
}

fn is_ctrl_c(key: &KeyEvent) -> bool {
    key.kind == KeyEventKind::Press
        && key.code == KeyCode::Char('c')
        && key.modifiers.contains(KeyModifiers::CONTROL)
}

/// Claude Code 风格：Esc 与 Ctrl+C 同样用于打断当前任务。
fn is_escape(key: &KeyEvent) -> bool {
    key.kind == KeyEventKind::Press && key.code == KeyCode::Esc
}

fn is_ctrl_d(key: &KeyEvent) -> bool {
    key.kind == KeyEventKind::Press
        && key.code == KeyCode::Char('d')
        && key.modifiers.contains(KeyModifiers::CONTROL)
}

fn is_plain_enter(key: &KeyEvent) -> bool {
    key.kind == KeyEventKind::Press
        && key.code == KeyCode::Enter
        && !key
            .modifiers
            .intersects(KeyModifiers::SHIFT | KeyModifiers::ALT)
}

/// TUI 内确认对话框：等待用户在 Composer 输入 YES 并回车。
async fn confirm_in_tui(app: &TuiApp) -> Result<bool, XduduError> {
    match read_line_in_tui(app).await? {
        Some(value) => Ok(value.trim() == "YES"),
        None => Ok(false),
    }
}

/// TUI 内读取一行输入；Ctrl+C/Esc 或退出返回 None。
async fn read_line_in_tui(app: &TuiApp) -> Result<Option<String>, XduduError> {
    loop {
        let Some(event) = app.next_input().await else {
            return Ok(None);
        };
        if let Event::Key(key) = event {
            match app.handle_key(key).map_err(XduduError::from)? {
                Some(InputOutcome::Submit(value)) | Some(InputOutcome::Command(value)) => {
                    return Ok(Some(value));
                }
                Some(InputOutcome::ToggleFold) => {
                    app.renderer()
                        .toggle_last_group()
                        .map_err(XduduError::from)?;
                }
                Some(InputOutcome::Interrupted) | Some(InputOutcome::Exit) => return Ok(None),
                None => {}
            }
        }
    }
}

/// 任务完成后由 Agent 自主判断是否需要形成长期记忆。
///
/// Provider 可以返回空集合；候选在本地脱敏、归一化去重后静默保存。该辅助
/// 流程的任何失败都不能改变主任务结果，也不会弹出审批界面打断用户。
async fn auto_capture_memories(runtime: &Runtime, session_id: Uuid) {
    capture_memories(
        runtime.memory_suggest_enabled,
        Arc::clone(&runtime.store),
        Arc::clone(&runtime.provider),
        runtime.model.clone(),
        runtime.cwd.clone(),
        session_id,
    )
    .await;
}

/// TUI 专用后台入口：记忆提炼不能阻塞 Composer 或排队任务。
fn spawn_auto_capture_memories(runtime: &Runtime, session_id: Uuid) {
    let enabled = runtime.memory_suggest_enabled;
    let store = Arc::clone(&runtime.store);
    let provider = Arc::clone(&runtime.provider);
    let model = runtime.model.clone();
    let cwd = runtime.cwd.clone();
    tokio::spawn(async move {
        capture_memories(enabled, store, provider, model, cwd, session_id).await;
    });
}

async fn capture_memories(
    enabled: bool,
    store: Arc<SqliteSessionStore>,
    provider: Arc<dyn Provider>,
    model: String,
    cwd: PathBuf,
    session_id: Uuid,
) {
    if !enabled {
        return;
    }
    let Ok(Some(session)) = store.get(session_id).await else {
        return;
    };
    let cancellation = CancellationToken::new();
    let Ok(suggestions) = xdudu_core::suggest_memories(MemorySuggestionConfig {
        session: &session,
        model: model.clone(),
        cwd: cwd.clone(),
        provider: provider.as_ref(),
        cancellation,
    })
    .await
    else {
        return;
    };
    let existing = store.list_memories(500).await.unwrap_or_default();
    let mut seen = existing
        .into_iter()
        .map(|memory| normalize_memory(&memory.content))
        .collect::<std::collections::HashSet<_>>();
    for suggestion in suggestions {
        let normalized = normalize_memory(&suggestion.content);
        if normalized.is_empty() || !seen.insert(normalized) {
            continue;
        }
        let _ = store
            .add_memory(&suggestion.content, Some(session_id))
            .await;
    }
    refresh_memory_document(&store, provider.as_ref(), &model, &cwd).await;
}

async fn refresh_memory_document(
    store: &SqliteSessionStore,
    provider: &dyn Provider,
    model: &str,
    cwd: &Path,
) {
    let Ok(raw_memories) = store.list_memories(500).await else {
        return;
    };
    let current = read_memory_document(cwd).ok().flatten();
    let Ok(content) = consolidate_memory_document(MemoryConsolidationConfig {
        raw_memories: &raw_memories,
        current_document: current.as_deref(),
        model: model.to_owned(),
        cwd: cwd.to_path_buf(),
        provider,
        cancellation: CancellationToken::new(),
    })
    .await
    else {
        return;
    };
    let _ = write_memory_document(cwd, &content);
}

fn normalize_memory(content: &str) -> String {
    content.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// TUI 斜杠命令处理；"/exit" 返回 [`TuiLoopAction::Exit`]。
async fn handle_tui_command(
    app: &TuiApp,
    runtime: &mut Runtime,
    renderer: &TuiRenderer,
    input: &str,
    session_id: &mut Option<Uuid>,
) -> Result<TuiLoopAction, XduduError> {
    match input {
        "/exit" | "/quit" | "/q" => return Ok(TuiLoopAction::Exit),
        "/help" | "/h" => {
            app.notice(
                "/new  新会话  ·  /resume  恢复会话  ·  /plan  生成/审阅计划  ·  /model  选择模型  ·  /memory  查看记忆  ·  /mcp  外部工具  ·  /plugins  插件  ·  /turns N  最大循环次数  ·  /exit  退出",
            )
            .map_err(XduduError::from)?;
        }
        "/new" => {
            *session_id = None;
            app.notice("已开始新会话。").map_err(XduduError::from)?;
        }
        "/permission" => {
            app.notice(format!(
                "当前权限模式：{} · Shift+Tab 循环切换（read-only → auto-safe → full-access）。",
                runtime.permission_mode.as_str()
            ))
            .map_err(XduduError::from)?;
        }
        "/vim" => {
            app.toggle_vim().map_err(XduduError::from)?;
            app.notice("已切换 Vim 模式。输入 Esc 可进入普通模式，帮助行会显示当前模式。")
                .map_err(XduduError::from)?;
        }
        "/model" => {
            if let Some(model) = app.select_model().await.map_err(XduduError::from)? {
                runtime.model = model;
                let saved =
                    write_config_value(&runtime.cwd, true, "provider.model", &runtime.model);
                app.set_model(&runtime.model).map_err(XduduError::from)?;
                app.notice(format!(
                    "已切换到 {}{}",
                    ui::model_display_name(&runtime.provider_display, &runtime.model),
                    if saved.is_ok() {
                        "，并保存为默认模型。"
                    } else {
                        "（仅当前会话，默认配置保存失败）。"
                    }
                ))
                .map_err(XduduError::from)?;
            }
        }
        "/mcp" => {
            let config = load_mcp_config()?;
            let external_tools = runtime
                .registry
                .definitions()
                .into_iter()
                .filter(|definition| definition.name.starts_with("mcp__"))
                .count();
            let servers = if config.servers.is_empty() {
                "尚未配置 Server".to_owned()
            } else {
                config
                    .servers
                    .iter()
                    .map(|server| {
                        format!(
                            "{} [{}]",
                            server.name,
                            if server.enabled {
                                "enabled"
                            } else {
                                "disabled"
                            }
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("、")
            };
            app.notice(format!(
                "MCP：{servers}\n已加载外部工具：{external_tools}\n管理命令：xdudu mcp --help"
            ))
            .map_err(XduduError::from)?;
        }
        "/plugins" => {
            let plugins = load_plugin_manifests()?;
            let summary = if plugins.is_empty() {
                "尚未安装插件".to_owned()
            } else {
                plugins
                    .iter()
                    .map(|plugin| {
                        format!(
                            "{} [{}] · {} MCP servers",
                            plugin.id,
                            if plugin.enabled {
                                "enabled"
                            } else {
                                "disabled"
                            },
                            plugin.mcp_servers.len()
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            };
            app.notice(format!("插件：\n{summary}\n管理命令：xdudu plugin --help"))
                .map_err(XduduError::from)?;
        }
        "/instructions" => {
            app.notice(instruction_summary(&runtime.cwd))
                .map_err(XduduError::from)?;
        }
        "/skills" => {
            app.notice(skills_summary(runtime))
                .map_err(XduduError::from)?;
        }
        "/agent" => {
            app.notice(agent_summary(runtime))
                .map_err(XduduError::from)?;
        }
        "/memory" => {
            if read_memory_document(&runtime.cwd)?.is_none() {
                refresh_memory_document(
                    runtime.store.as_ref(),
                    runtime.provider.as_ref(),
                    &runtime.model,
                    &runtime.cwd,
                )
                .await;
            }
            let summary = read_memory_document(&runtime.cwd)?
                .unwrap_or_else(|| "# XDUDU 长期记忆\n\n当前没有需要长期保留的信息。".into());
            app.notice(format!(
                "{summary}\n\n文件：{}\n修改：退出 XDUDU 后运行 xdudu memory edit",
                memory_document_path(&runtime.cwd).display()
            ))
            .map_err(XduduError::from)?;
        }
        "/compact" => {
            runtime
                .force_compact
                .store(true, std::sync::atomic::Ordering::Relaxed);
            app.notice("已请求在下一轮请求前触发一次上下文压缩（LLM 不可用时回退确定性截断）。")
                .map_err(XduduError::from)?;
        }
        "/resume" => {
            let sessions = runtime.store.list(20).await?;
            if sessions.is_empty() {
                app.notice("当前工作区还没有可恢复的历史会话。")
                    .map_err(XduduError::from)?;
                return Ok(TuiLoopAction::Continue);
            }
            let choices = sessions.iter().map(session_choice).collect();
            if let Some(id) = app
                .select_session(choices)
                .await
                .map_err(XduduError::from)?
            {
                let session = session_for_resume(runtime, id).await?;
                app.load_session(&session).map_err(XduduError::from)?;
                *session_id = Some(id);
                if let Some(plan) = pending_plan_for_session(runtime, *session_id).await? {
                    review_plan_in_tui(runtime, app, plan).await?;
                } else if let Some(plan) = paused_plan_for_session(runtime, *session_id).await? {
                    recover_plan_in_tui(runtime, app, plan).await?;
                }
            }
        }
        "/plan" => {
            if let Some(plan) = pending_plan_for_session(runtime, *session_id).await? {
                let plan_id = plan.id;
                review_plan_in_tui(runtime, app, plan).await?;
                maybe_restore_after_review(app, runtime, plan_id).await?;
            } else if let Some(plan) = paused_plan_for_session(runtime, *session_id).await? {
                recover_plan_in_tui(runtime, app, plan).await?;
            } else if let Some(id) = *session_id
                && let Some(plan) = runtime.store.latest_plan_for_session(id).await?
            {
                app.notice(plan_summary(&plan)).map_err(XduduError::from)?;
            } else {
                app.notice("用法：/plan <目标>").map_err(XduduError::from)?;
            }
        }
        "/plan status" => {
            if let Some(id) = *session_id
                && let Some(plan) = runtime.store.latest_plan_for_session(id).await?
            {
                app.notice(plan_summary(&plan)).map_err(XduduError::from)?;
            } else {
                app.notice("当前会话没有计划。").map_err(XduduError::from)?;
            }
        }
        "/plan run" | "/plan retry" => {
            let Some(id) = *session_id else {
                app.notice("当前没有活动会话。").map_err(XduduError::from)?;
                return Ok(TuiLoopAction::Continue);
            };
            let Some(plan) = runtime.store.latest_plan_for_session(id).await? else {
                app.notice("当前会话没有计划。").map_err(XduduError::from)?;
                return Ok(TuiLoopAction::Continue);
            };
            if input == "/plan run" && plan.status != PlanStatus::Approved {
                app.notice("/plan run 只执行已批准计划；暂停计划请使用 /plan retry。")
                    .map_err(XduduError::from)?;
                return Ok(TuiLoopAction::Continue);
            }
            if input == "/plan retry" && plan.status != PlanStatus::Paused {
                app.notice("/plan retry 只重试暂停计划。")
                    .map_err(XduduError::from)?;
                return Ok(TuiLoopAction::Continue);
            }
            let result = run_plan_in_tui(app, runtime, renderer, plan.id).await?;
            app.notice(result.message).map_err(XduduError::from)?;
        }
        "/plan revisions" => {
            let Some(id) = *session_id else {
                app.notice("当前没有活动会话。").map_err(XduduError::from)?;
                return Ok(TuiLoopAction::Continue);
            };
            let Some(plan) = runtime.store.latest_plan_for_session(id).await? else {
                app.notice("当前会话没有计划。").map_err(XduduError::from)?;
                return Ok(TuiLoopAction::Continue);
            };
            let revisions = runtime.store.list_plan_revisions(plan.id).await?;
            app.notice(format!(
                "计划修订：{}",
                revisions
                    .iter()
                    .map(|item| item.revision.to_string())
                    .collect::<Vec<_>>()
                    .join("、")
            ))
            .map_err(XduduError::from)?;
        }
        "/plan cancel" => {
            app.notice("取消不会撤销既有副作用。请输入 YES 确认。")
                .map_err(XduduError::from)?;
            let confirmed = confirm_in_tui(app).await?;
            if !confirmed {
                app.notice("已保留计划。").map_err(XduduError::from)?;
                return Ok(TuiLoopAction::Continue);
            }
            let Some(id) = *session_id else {
                app.notice("当前没有活动会话。").map_err(XduduError::from)?;
                return Ok(TuiLoopAction::Continue);
            };
            let Some(plan) = runtime.store.latest_plan_for_session(id).await? else {
                app.notice("当前会话没有计划。").map_err(XduduError::from)?;
                return Ok(TuiLoopAction::Continue);
            };
            cancel_plan(runtime, plan.id).await?;
            restore_plan_mode(app, runtime)?;
            app.notice("计划已取消；既有副作用未撤销。")
                .map_err(XduduError::from)?;
        }
        _ => {
            if let Some(raw_id) = input
                .strip_prefix("/resume ")
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                match Uuid::parse_str(raw_id) {
                    Ok(id) => match session_for_resume(runtime, id).await {
                        Ok(session) => {
                            app.load_session(&session).map_err(XduduError::from)?;
                            *session_id = Some(id);
                            if let Some(plan) =
                                pending_plan_for_session(runtime, *session_id).await?
                            {
                                review_plan_in_tui(runtime, app, plan).await?;
                            } else if let Some(plan) =
                                paused_plan_for_session(runtime, *session_id).await?
                            {
                                recover_plan_in_tui(runtime, app, plan).await?;
                            }
                        }
                        Err(error) => {
                            app.notice(error.message).map_err(XduduError::from)?;
                        }
                    },
                    Err(_) => {
                        app.notice("会话 ID 必须是完整 UUID。")
                            .map_err(XduduError::from)?;
                    }
                }
            } else if let Some(goal) = input
                .strip_prefix("/plan new ")
                .or_else(|| input.strip_prefix("/plan "))
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                app.notice("正在生成结构化计划…")
                    .map_err(XduduError::from)?;
                // 生成与审阅期间锁定只读，批准执行/取消/失败时恢复。
                if runtime.plan_restore_mode.is_none() {
                    runtime.plan_restore_mode = Some(runtime.permission_mode);
                    apply_permission_mode(app, runtime, PermissionMode::ReadOnly)?;
                    app.notice("计划生成与审阅期间已锁定只读；批准执行时将恢复原权限模式。")
                        .map_err(XduduError::from)?;
                }
                match create_plan_for_review(runtime, goal.to_owned(), *session_id).await {
                    Ok((session, plan)) => {
                        *session_id = Some(session.id);
                        let plan_id = plan.id;
                        review_plan_in_tui(runtime, app, plan).await?;
                        maybe_restore_after_review(app, runtime, plan_id).await?;
                    }
                    Err(error) => {
                        restore_plan_mode(app, runtime)?;
                        app.notice(error.message).map_err(XduduError::from)?;
                    }
                }
            } else if let Some(model) = input
                .strip_prefix("/model ")
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                runtime.model = model.to_owned();
                let saved =
                    write_config_value(&runtime.cwd, true, "provider.model", &runtime.model);
                app.set_model(&runtime.model).map_err(XduduError::from)?;
                app.notice(if saved.is_ok() {
                    format!("已切换到 {}，并保存为默认模型。", runtime.model)
                } else {
                    format!("已切换到 {}，但默认配置保存失败。", runtime.model)
                })
                .map_err(XduduError::from)?;
            } else if let Some(turns) = input.strip_prefix("/turns ") {
                match turns.trim().parse::<u32>() {
                    Ok(value) if (1..=100).contains(&value) => {
                        runtime.max_turns = value;
                        app.notice(format!("最大循环次数已设为 {value}。"))
                            .map_err(XduduError::from)?;
                    }
                    _ => app
                        .notice("最大循环次数必须是 1 到 100 之间的整数。")
                        .map_err(XduduError::from)?,
                }
            } else {
                app.notice(format!("未知命令：{input}"))
                    .map_err(XduduError::from)?;
            }
        }
    }
    Ok(TuiLoopAction::Continue)
}

async fn session_for_resume(runtime: &Runtime, id: Uuid) -> Result<Session, XduduError> {
    let session = runtime
        .store
        .get(id)
        .await?
        .ok_or_else(|| XduduError::validation(format!("找不到会话：{id}")))?;
    if session.cwd != runtime.cwd {
        return Err(XduduError::validation("不能在不同工作目录中恢复已有会话。"));
    }
    Ok(session)
}

fn session_choice(session: &Session) -> SessionChoice {
    let status = serde_json::to_string(&session.status)
        .unwrap_or_else(|_| "\"unknown\"".into())
        .trim_matches('"')
        .to_owned();
    SessionChoice {
        id: session.id,
        title: session.title.clone(),
        status,
        updated_at: session.updated_at.format("%m-%d %H:%M").to_string(),
    }
}

fn plan_context(session: &Session) -> Option<String> {
    const LIMIT: usize = 65_536;
    let mut parts = Vec::new();
    if !session.context_summary.trim().is_empty() {
        parts.push(format!(
            "较早会话摘要：\n{}",
            redact_text(&session.context_summary)
        ));
    }
    let recent = session
        .messages
        .iter()
        .rev()
        .filter(|message| {
            matches!(
                message.role,
                xdudu_core::provider::MessageRole::User
                    | xdudu_core::provider::MessageRole::Assistant
            )
        })
        .take(24)
        .collect::<Vec<_>>();
    for message in recent.into_iter().rev() {
        let role = if message.role == xdudu_core::provider::MessageRole::User {
            "用户"
        } else {
            "助手"
        };
        parts.push(format!("{role}：{}", redact_text(&message.content)));
    }
    let mut output = String::new();
    for part in parts {
        let separator = if output.is_empty() { "" } else { "\n\n" };
        if output.len() + separator.len() + part.len() > LIMIT {
            break;
        }
        output.push_str(separator);
        output.push_str(&part);
    }
    (!output.is_empty()).then_some(output)
}

fn active_plan(status: PlanStatus) -> bool {
    matches!(
        status,
        PlanStatus::Draft
            | PlanStatus::PendingApproval
            | PlanStatus::Approved
            | PlanStatus::Running
            | PlanStatus::Paused
    )
}

async fn create_plan_for_review(
    runtime: &Runtime,
    goal: String,
    session_id: Option<Uuid>,
) -> Result<(Session, Plan), XduduError> {
    let mut session = if let Some(id) = session_id {
        session_for_resume(runtime, id).await?
    } else {
        Session::new(
            runtime.cwd.clone(),
            runtime.provider.name(),
            runtime.model.clone(),
            goal.clone(),
        )
    };
    if let Some(plan) = runtime.store.latest_plan_for_session(session.id).await?
        && active_plan(plan.status)
    {
        return Err(XduduError::validation(format!(
            "当前会话已有活动计划（revision {}，状态 {:?}），不能创建第二份计划。",
            plan.revision, plan.status
        )));
    }

    let context = plan_context(&session);
    session.status = SessionStatus::Running;
    session.current_state = AgentLoopState::Planning;
    session.provider_name = runtime.provider.name().to_owned();
    session.model.clone_from(&runtime.model);
    session.completed_at = None;
    if session_id.is_some() {
        session.append_user_message(goal.clone());
        runtime.store.update(&session).await?;
    } else {
        runtime.store.create(&session).await?;
    }

    let cancellation = CancellationToken::new();
    let generation = generate_plan(PlanGenerationConfig {
        session_id: session.id,
        goal,
        context,
        model: runtime.model.clone(),
        cwd: runtime.cwd.clone(),
        provider: runtime.provider.as_ref(),
        plan_store: runtime.store.as_ref(),
        cancellation: cancellation.clone(),
    });
    tokio::pin!(generation);
    let generated = tokio::select! {
        result = &mut generation => result,
        signal = tokio::signal::ctrl_c() => {
            if signal.is_ok() {
                cancellation.cancel();
            }
            generation.await
        }
    };
    let generated = match generated {
        Ok(generated) => generated,
        Err(error) => {
            session.status = SessionStatus::Incomplete;
            session.current_state = AgentLoopState::Incomplete;
            session.touch();
            runtime.store.update(&session).await?;
            return Err(error);
        }
    };
    session.total_input_tokens = session
        .total_input_tokens
        .saturating_add(generated.usage.input_tokens);
    session.total_output_tokens = session
        .total_output_tokens
        .saturating_add(generated.usage.output_tokens);
    let plan = submit_plan_for_review(runtime.store.as_ref(), generated.plan.id, 1).await?;
    session.status = SessionStatus::WaitingApproval;
    session.current_state = AgentLoopState::WaitingApproval;
    session.touch();
    runtime.store.update(&session).await?;
    Ok((session, plan))
}

async fn pending_plan_for_session(
    runtime: &Runtime,
    session_id: Option<Uuid>,
) -> Result<Option<Plan>, XduduError> {
    let Some(session_id) = session_id else {
        return Ok(None);
    };
    Ok(runtime
        .store
        .latest_plan_for_session(session_id)
        .await?
        .filter(|plan| plan.status == PlanStatus::PendingApproval))
}

async fn paused_plan_for_session(
    runtime: &Runtime,
    session_id: Option<Uuid>,
) -> Result<Option<Plan>, XduduError> {
    let Some(session_id) = session_id else {
        return Ok(None);
    };
    Ok(runtime
        .store
        .latest_plan_for_session(session_id)
        .await?
        .filter(|plan| plan.status == PlanStatus::Paused))
}

async fn sync_plan_session(
    runtime: &Runtime,
    session_id: Uuid,
    status: SessionStatus,
    state: AgentLoopState,
) -> Result<(), XduduError> {
    let mut session = session_for_resume(runtime, session_id).await?;
    session.status = status;
    session.current_state = state;
    session.completed_at = None;
    session.touch();
    runtime.store.update(&session).await
}

async fn sync_plan_session_store(
    store: &SqliteSessionStore,
    session_id: Uuid,
    status: SessionStatus,
    state: AgentLoopState,
) -> Result<(), XduduError> {
    let mut session = store
        .get(session_id)
        .await?
        .ok_or_else(|| XduduError::validation(format!("找不到会话：{session_id}")))?;
    session.status = status;
    session.current_state = state;
    session.completed_at = None;
    session.touch();
    store.update(&session).await
}

async fn review_plan_in_tui(
    runtime: &Runtime,
    app: &TuiApp,
    mut plan: Plan,
) -> Result<(), XduduError> {
    loop {
        match app.review_plan(&plan).await.map_err(XduduError::from)? {
            None => {
                app.notice("计划仍保持等待审批，可稍后输入 /plan 重新打开。")
                    .map_err(XduduError::from)?;
                return Ok(());
            }
            Some(PlanReviewChoice::Approve) => {
                let approved = approve_plan(
                    runtime.store.as_ref(),
                    plan.id,
                    plan.revision,
                    "用户在终端审阅界面批准计划。",
                )
                .await?;
                sync_plan_session(
                    runtime,
                    approved.session_id,
                    SessionStatus::PlanReady,
                    AgentLoopState::Completed,
                )
                .await?;
                app.notice("计划已批准。输入 /plan run 开始执行；具体副作用仍会单独审批。")
                    .map_err(XduduError::from)?;
                return Ok(());
            }
            Some(PlanReviewChoice::Reject) => {
                let rejected = reject_plan(
                    runtime.store.as_ref(),
                    plan.id,
                    plan.revision,
                    "用户在终端审阅界面拒绝计划。",
                )
                .await?;
                sync_plan_session(
                    runtime,
                    rejected.session_id,
                    SessionStatus::Incomplete,
                    AgentLoopState::Incomplete,
                )
                .await?;
                app.notice("计划已拒绝，未执行任何步骤。")
                    .map_err(XduduError::from)?;
                return Ok(());
            }
            Some(PlanReviewChoice::RequestChanges) => {
                app.notice("请输入对计划的修改要求；Ctrl+C 取消并保留当前版本。")
                    .map_err(XduduError::from)?;
                let change_request = match read_line_in_tui(app).await? {
                    Some(value) if !value.trim().is_empty() => value,
                    _ => {
                        app.notice("已取消修改，原计划继续等待审批。")
                            .map_err(XduduError::from)?;
                        continue;
                    }
                };
                let session = session_for_resume(runtime, plan.session_id).await?;
                sync_plan_session(
                    runtime,
                    plan.session_id,
                    SessionStatus::Running,
                    AgentLoopState::Planning,
                )
                .await?;
                let cancellation = CancellationToken::new();
                let revision = revise_plan(PlanRevisionConfig {
                    plan_id: plan.id,
                    expected_revision: plan.revision,
                    change_request,
                    context: plan_context(&session),
                    model: runtime.model.clone(),
                    cwd: runtime.cwd.clone(),
                    provider: runtime.provider.as_ref(),
                    plan_store: runtime.store.as_ref(),
                    cancellation: cancellation.clone(),
                });
                tokio::pin!(revision);
                let result = tokio::select! {
                    result = &mut revision => result,
                    signal = tokio::signal::ctrl_c() => {
                        if signal.is_ok() {
                            cancellation.cancel();
                        }
                        revision.await
                    }
                };
                match result {
                    Ok(result) => {
                        plan = result.plan;
                        let mut session = session_for_resume(runtime, plan.session_id).await?;
                        session.total_input_tokens = session
                            .total_input_tokens
                            .saturating_add(result.usage.input_tokens);
                        session.total_output_tokens = session
                            .total_output_tokens
                            .saturating_add(result.usage.output_tokens);
                        session.status = SessionStatus::WaitingApproval;
                        session.current_state = AgentLoopState::WaitingApproval;
                        session.touch();
                        runtime.store.update(&session).await?;
                        app.notice(format!(
                            "计划已修订为 revision {}，请重新审阅。",
                            plan.revision
                        ))
                        .map_err(XduduError::from)?;
                    }
                    Err(error) => {
                        sync_plan_session(
                            runtime,
                            plan.session_id,
                            SessionStatus::WaitingApproval,
                            AgentLoopState::WaitingApproval,
                        )
                        .await?;
                        app.notice(format!("修订失败，原计划保持不变：{}", error.message))
                            .map_err(XduduError::from)?;
                    }
                }
            }
        }
    }
}

async fn review_plan_classic(
    runtime: &Runtime,
    editor: &mut InputEditor,
    mut plan: Plan,
) -> Result<(), XduduError> {
    loop {
        println!("\n{}", plan_summary(&plan));
        for (index, step) in plan.steps.iter().enumerate() {
            println!("  {}. {}", index + 1, step.title);
            if !step.dependencies.is_empty() {
                println!("     依赖 {} 个步骤", step.dependencies.len());
            }
            for criterion in &step.completion_criteria {
                println!("     ✓ {criterion}");
            }
        }
        println!("\n  1 批准计划  ·  2 请求修改  ·  0 拒绝（默认）");
        let choice = match editor.read_line("  选择：").map_err(XduduError::from)? {
            ReadResult::Line(value) => value,
            _ => return Ok(()),
        };
        match choice.trim() {
            "1" => {
                let approved = approve_plan(
                    runtime.store.as_ref(),
                    plan.id,
                    plan.revision,
                    "用户在经典终端审阅界面批准计划。",
                )
                .await?;
                sync_plan_session(
                    runtime,
                    approved.session_id,
                    SessionStatus::PlanReady,
                    AgentLoopState::Completed,
                )
                .await?;
                println!("  计划已批准。输入 /plan run 开始执行；工具副作用仍会单独审批。");
                return Ok(());
            }
            "2" => {
                let request = match editor.read_line("  修改要求：").map_err(XduduError::from)?
                {
                    ReadResult::Line(value) if !value.trim().is_empty() => value,
                    _ => continue,
                };
                let session = session_for_resume(runtime, plan.session_id).await?;
                sync_plan_session(
                    runtime,
                    plan.session_id,
                    SessionStatus::Running,
                    AgentLoopState::Planning,
                )
                .await?;
                let result = revise_plan(PlanRevisionConfig {
                    plan_id: plan.id,
                    expected_revision: plan.revision,
                    change_request: request,
                    context: plan_context(&session),
                    model: runtime.model.clone(),
                    cwd: runtime.cwd.clone(),
                    provider: runtime.provider.as_ref(),
                    plan_store: runtime.store.as_ref(),
                    cancellation: CancellationToken::new(),
                })
                .await;
                match result {
                    Ok(result) => {
                        plan = result.plan;
                        sync_plan_session(
                            runtime,
                            plan.session_id,
                            SessionStatus::WaitingApproval,
                            AgentLoopState::WaitingApproval,
                        )
                        .await?;
                        println!("  计划已修订为 revision {}。", plan.revision);
                    }
                    Err(error) => {
                        sync_plan_session(
                            runtime,
                            plan.session_id,
                            SessionStatus::WaitingApproval,
                            AgentLoopState::WaitingApproval,
                        )
                        .await?;
                        println!("  修订失败，原计划保持不变：{}", error.message);
                    }
                }
            }
            _ => {
                let rejected = reject_plan(
                    runtime.store.as_ref(),
                    plan.id,
                    plan.revision,
                    "用户在经典终端审阅界面拒绝计划。",
                )
                .await?;
                sync_plan_session(
                    runtime,
                    rejected.session_id,
                    SessionStatus::Incomplete,
                    AgentLoopState::Incomplete,
                )
                .await?;
                println!("  计划已拒绝，未执行任何步骤。");
                return Ok(());
            }
        }
    }
}

async fn recover_plan_in_tui(
    runtime: &Runtime,
    app: &TuiApp,
    mut plan: Plan,
) -> Result<(), XduduError> {
    loop {
        match app.recover_plan(&plan).await.map_err(XduduError::from)? {
            None => {
                app.notice("计划保持暂停，未重放任何工具调用。")
                    .map_err(XduduError::from)?;
                return Ok(());
            }
            Some(PlanRecoveryChoice::ViewDetails) => {
                app.notice(plan_summary(&plan)).map_err(XduduError::from)?;
                plan = runtime
                    .store
                    .get_plan(plan.id)
                    .await?
                    .ok_or_else(|| XduduError::validation("计划已不存在。"))?;
            }
            Some(PlanRecoveryChoice::Continue | PlanRecoveryChoice::Retry) => {
                let result = run_plan_in_tui(app, runtime, &app.renderer(), plan.id).await?;
                app.notice(result.message).map_err(XduduError::from)?;
                return Ok(());
            }
            Some(PlanRecoveryChoice::Cancel) => {
                app.notice("取消不会撤销既有副作用。请输入 YES 确认。")
                    .map_err(XduduError::from)?;
                let confirmed = confirm_in_tui(app).await?;
                if confirmed {
                    cancel_plan(runtime, plan.id).await?;
                    app.notice("计划已取消；既有副作用未撤销。")
                        .map_err(XduduError::from)?;
                    return Ok(());
                }
                app.notice("已保留暂停计划。").map_err(XduduError::from)?;
            }
        }
    }
}

async fn execute_plan_with_sink(
    runtime: &Runtime,
    plan_id: Uuid,
    sink: &dyn EventSink,
) -> Result<xdudu_core::PlanExecutionResult, XduduError> {
    let cancellation = CancellationToken::new();
    let execution = execute_plan_with_cancellation(runtime, plan_id, sink, cancellation.clone());
    tokio::pin!(execution);
    tokio::select! {
        result = &mut execution => result,
        signal = tokio::signal::ctrl_c() => {
            if signal.is_ok() {
                cancellation.cancel();
            }
            execution.await
        }
    }
}

async fn execute_plan_with_cancellation(
    runtime: &Runtime,
    plan_id: Uuid,
    sink: &dyn EventSink,
    cancellation: CancellationToken,
) -> Result<xdudu_core::PlanExecutionResult, XduduError> {
    run_plan(PlanExecutorConfig {
        plan_id,
        model: runtime.model.clone(),
        cwd: runtime.cwd.clone(),
        max_turns_per_step: runtime.max_turns,
        provider: runtime.provider.as_ref(),
        tool_registry: &runtime.registry,
        session_store: runtime.store.as_ref(),
        plan_store: runtime.store.as_ref(),
        permission_mode: runtime.permission_mode,
        cancellation,
        event_sink: Some(sink),
    })
    .await
}

fn plan_summary(plan: &Plan) -> String {
    let completed = plan
        .steps
        .iter()
        .filter(|step| step.status == xdudu_core::StepStatus::Completed)
        .count();
    format!(
        "Plan {} · revision {} · {:?} · {}/{} 步完成{}",
        plan.id,
        plan.revision,
        plan.status,
        completed,
        plan.steps.len(),
        plan.paused_reason
            .as_ref()
            .map(|reason| format!(" · {reason}"))
            .unwrap_or_default()
    )
}

async fn cancel_plan(runtime: &Runtime, plan_id: Uuid) -> Result<Plan, XduduError> {
    cancel_plan_with_store(runtime.store.as_ref(), plan_id).await
}

async fn cancel_plan_with_store(
    store: &SqliteSessionStore,
    plan_id: Uuid,
) -> Result<Plan, XduduError> {
    let mut plan = store
        .get_plan(plan_id)
        .await?
        .ok_or_else(|| XduduError::validation(format!("找不到计划：{plan_id}")))?;
    let expected_status = plan.status;
    if matches!(
        expected_status,
        PlanStatus::Completed | PlanStatus::Failed | PlanStatus::Rejected | PlanStatus::Cancelled
    ) {
        return Err(XduduError::validation("终态计划不能取消。"));
    }
    let mut session = store
        .get(plan.session_id)
        .await?
        .ok_or_else(|| XduduError::validation(format!("找不到计划会话：{}", plan.session_id)))?;
    for step in &mut plan.steps {
        if !matches!(
            step.status,
            xdudu_core::StepStatus::Completed | xdudu_core::StepStatus::Skipped
        ) {
            let _ = step.transition_to(xdudu_core::StepStatus::Cancelled);
        }
    }
    plan.transition_to(PlanStatus::Cancelled)?;
    let expected_version = plan.execution_version;
    plan.execution_version += 1;
    session.status = SessionStatus::Incomplete;
    session.current_state = AgentLoopState::Incomplete;
    if !store
        .checkpoint_plan_execution(&plan, &session, expected_version, expected_status)
        .await?
    {
        return Err(XduduError::validation(
            "PLAN_CONFLICT：计划已被其他请求更新。",
        ));
    }
    Ok(plan)
}

async fn plain_interactive_loop(
    mut runtime: Runtime,
    initial_prompt: Option<String>,
    initial_session: Option<Uuid>,
) -> Result<u8, XduduError> {
    let mut session_id = initial_session;
    let mut editor = InputEditor::with_workspace(&runtime.cwd);
    println!(
        "{}",
        ui::compact_banner(
            TerminalTheme::new(runtime.color),
            env!("CARGO_PKG_VERSION"),
            &runtime.provider_display,
            &runtime.model,
        )
    );
    println!(
        "  {} · {} · 输入 /help 查看命令",
        runtime.permission_mode.as_str(),
        runtime.cwd.display()
    );
    for notice in &runtime.startup_notices {
        eprintln!("  警告：{notice}");
    }
    if let Some(prompt) = initial_prompt {
        let result = execute_prompt(&runtime, prompt, session_id).await?;
        runtime.renderer.finish_run(&result)?;
        session_id = Some(result.session_id);
    }

    loop {
        println!();
        let prompt = ui::prompt(TerminalTheme::new(runtime.color));
        let line = match editor.read_line(&prompt).map_err(XduduError::from)? {
            ReadResult::Line(line) => line,
            ReadResult::Interrupted => {
                println!(
                    "  {}",
                    TerminalTheme::new(runtime.color).muted("已取消当前输入。")
                );
                continue;
            }
            ReadResult::Eof => break,
        };
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        match input {
            "/exit" | "/quit" | "/q" => {
                println!(
                    "  {}",
                    TerminalTheme::new(runtime.color).muted("再见，期待下次一起写代码。")
                );
                break;
            }
            "/help" | "/h" => {
                print!("{}", ui::help(TerminalTheme::new(runtime.color)));
                continue;
            }
            "/new" | "/clear" => {
                session_id = None;
                println!("  已开始新会话。");
                continue;
            }
            "/transcript" => {
                if let Some(id) = session_id {
                    let session = session_for_resume(&runtime, id).await?;
                    println!("\n  会话：{} · {}", session.title, session.id);
                    for message in &session.messages {
                        if message.content.trim().is_empty() {
                            continue;
                        }
                        let role = match message.role {
                            xdudu_core::provider::MessageRole::User => "❯",
                            xdudu_core::provider::MessageRole::Assistant => "┊",
                            xdudu_core::provider::MessageRole::Tool => "⏺",
                            xdudu_core::provider::MessageRole::System => "·",
                        };
                        println!("\n  {role} {}", redact_text(&message.content));
                    }
                } else {
                    println!("  当前没有活动会话。");
                }
                continue;
            }
            "/copy" => {
                if let Some(id) = session_id {
                    let session = session_for_resume(&runtime, id).await?;
                    if let Some(message) = session.messages.iter().rev().find(|message| {
                        message.role == xdudu_core::provider::MessageRole::Assistant
                            && !message.content.trim().is_empty()
                    }) {
                        match copy_text(&redact_text(&message.content)) {
                            Ok(()) => println!("  已复制最后一条助手回答。"),
                            Err(error) => println!("  无法访问系统剪贴板：{error}"),
                        }
                    }
                }
                continue;
            }
            "/export" => {
                if let Some(id) = session_id {
                    let session = session_for_resume(&runtime, id).await?;
                    let path = export_session_markdown(&runtime.cwd, &session)?;
                    println!("  会话已导出：{}", path.display());
                } else {
                    println!("  当前没有活动会话。");
                }
                continue;
            }
            "/compact" => {
                runtime
                    .force_compact
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                println!(
                    "  已请求在下一轮请求前触发一次上下文压缩（LLM 不可用时回退确定性截断）。"
                );
                continue;
            }
            "/mcp" => {
                let config = load_mcp_config()?;
                if config.servers.is_empty() {
                    println!("  尚未配置 MCP Server。使用 xdudu mcp --help 查看管理命令。");
                } else {
                    for server in config.servers {
                        println!(
                            "  {} · {} · {:?}",
                            server.name,
                            if server.enabled {
                                "enabled"
                            } else {
                                "disabled"
                            },
                            server.transport
                        );
                    }
                }
                continue;
            }
            "/plugins" => {
                let plugins = load_plugin_manifests()?;
                if plugins.is_empty() {
                    println!("  尚未安装插件。使用 xdudu plugin --help 查看管理命令。");
                } else {
                    for plugin in plugins {
                        println!(
                            "  {} · {} · {} MCP servers",
                            plugin.id,
                            if plugin.enabled {
                                "enabled"
                            } else {
                                "disabled"
                            },
                            plugin.mcp_servers.len()
                        );
                    }
                }
                continue;
            }
            "/instructions" => {
                for line in instruction_summary(&runtime.cwd).lines() {
                    println!("  {line}");
                }
                continue;
            }
            "/skills" => {
                for line in skills_summary(&runtime).lines() {
                    println!("  {line}");
                }
                continue;
            }
            "/agent" => {
                for line in agent_summary(&runtime).lines() {
                    println!("  {line}");
                }
                continue;
            }
            "/model" => {
                let options = ui::model_options(&runtime.provider_display, &runtime.model);
                println!("  当前 Provider 可用模型：");
                for (index, option) in options.iter().enumerate() {
                    let current = if ui::model_matches(&option.id, &runtime.model) {
                        "（当前）"
                    } else {
                        ""
                    };
                    println!(
                        "    {}. {} {}  {}",
                        index + 1,
                        option.label,
                        current,
                        option.description
                    );
                }
                let selection = match editor
                    .read_line("  选择（Enter 保持当前）：")
                    .map_err(XduduError::from)?
                {
                    ReadResult::Line(value) => value,
                    _ => continue,
                };
                if let Some(option) = selection
                    .trim()
                    .parse::<usize>()
                    .ok()
                    .and_then(|index| options.get(index.saturating_sub(1)))
                {
                    runtime.model.clone_from(&option.id);
                    let saved =
                        write_config_value(&runtime.cwd, true, "provider.model", &runtime.model);
                    println!(
                        "  已切换到 {}{}",
                        option.label,
                        if saved.is_ok() {
                            "，并保存为默认模型。"
                        } else {
                            "（仅当前会话）。"
                        }
                    );
                }
                continue;
            }
            "/resume" => {
                let sessions = runtime.store.list(30).await?;
                if sessions.is_empty() {
                    println!("  当前工作区还没有历史会话。");
                    continue;
                }
                println!("  最近会话（输入编号、标题关键词或完整 UUID）：");
                for (index, session) in sessions.iter().enumerate() {
                    println!(
                        "  {:>2}. {} · {} · {:?} · {} 条消息",
                        index + 1,
                        session.updated_at.format("%m-%d %H:%M"),
                        session.title,
                        session.status,
                        session.messages.len()
                    );
                }
                let selection = match editor.read_line("  选择：").map_err(XduduError::from)? {
                    ReadResult::Line(value) => value,
                    _ => continue,
                };
                let selected = selection
                    .trim()
                    .parse::<usize>()
                    .ok()
                    .and_then(|index| sessions.get(index.saturating_sub(1)))
                    .or_else(|| {
                        Uuid::parse_str(selection.trim())
                            .ok()
                            .and_then(|id| sessions.iter().find(|session| session.id == id))
                    })
                    .or_else(|| {
                        let query = selection.trim().to_ascii_lowercase();
                        sessions
                            .iter()
                            .find(|session| session.title.to_ascii_lowercase().contains(&query))
                    });
                if let Some(selected) = selected {
                    let session = session_for_resume(&runtime, selected.id).await?;
                    session_id = Some(session.id);
                    println!("  已恢复会话：{}", session.title);
                    for message in session.messages.iter().rev().take(6).rev() {
                        if !message.content.trim().is_empty() {
                            let role = match message.role {
                                xdudu_core::provider::MessageRole::User => "❯",
                                xdudu_core::provider::MessageRole::Assistant => "┊",
                                _ => "·",
                            };
                            println!("  {role} {}", redact_text(&message.content));
                        }
                    }
                    if let Some(plan) = pending_plan_for_session(&runtime, session_id).await? {
                        review_plan_classic(&runtime, &mut editor, plan).await?;
                    }
                } else {
                    println!("  没有找到匹配会话。");
                }
                continue;
            }
            "/plan" => {
                if let Some(plan) = pending_plan_for_session(&runtime, session_id).await? {
                    review_plan_classic(&runtime, &mut editor, plan).await?;
                } else {
                    if let Some(plan) = paused_plan_for_session(&runtime, session_id).await? {
                        println!("{}", plan_summary(&plan));
                        println!(
                            "  使用 /plan retry 重试，或 /plan cancel 取消。工具不会自动重放。"
                        );
                    } else {
                        println!("  用法：/plan <目标>");
                    }
                }
                continue;
            }
            "/plan status" => {
                if let Some(id) = session_id
                    && let Some(plan) = runtime.store.latest_plan_for_session(id).await?
                {
                    println!("{}", plan_summary(&plan));
                } else {
                    println!("  当前会话没有计划。");
                }
                continue;
            }
            "/plan run" | "/plan retry" => {
                let Some(id) = session_id else {
                    println!("  当前没有活动会话。");
                    continue;
                };
                let Some(plan) = runtime.store.latest_plan_for_session(id).await? else {
                    println!("  当前会话没有计划。");
                    continue;
                };
                if input == "/plan run" && plan.status != PlanStatus::Approved {
                    println!("  /plan run 只执行已批准计划；暂停计划请使用 /plan retry。");
                    continue;
                }
                if input == "/plan retry" && plan.status != PlanStatus::Paused {
                    println!("  /plan retry 只重试暂停计划。");
                    continue;
                }
                let result = execute_plan_with_sink(&runtime, plan.id, &runtime.renderer).await?;
                println!("  {}", result.message);
                continue;
            }
            "/plan revisions" => {
                let Some(id) = session_id else {
                    println!("  当前没有活动会话。");
                    continue;
                };
                let Some(plan) = runtime.store.latest_plan_for_session(id).await? else {
                    println!("  当前会话没有计划。");
                    continue;
                };
                let revisions = runtime.store.list_plan_revisions(plan.id).await?;
                println!(
                    "  计划修订：{}",
                    revisions
                        .iter()
                        .map(|item| item.revision.to_string())
                        .collect::<Vec<_>>()
                        .join("、")
                );
                continue;
            }
            "/plan cancel" => {
                let Some(id) = session_id else {
                    println!("  当前没有活动会话。");
                    continue;
                };
                let Some(plan) = runtime.store.latest_plan_for_session(id).await? else {
                    println!("  当前会话没有计划。");
                    continue;
                };
                println!("  取消不会撤销既有副作用。再次输入 YES 确认。");
                let confirmed = matches!(
                    editor.read_line("  确认：").map_err(XduduError::from)?,
                    ReadResult::Line(value) if value.trim() == "YES"
                );
                if confirmed {
                    cancel_plan(&runtime, plan.id).await?;
                    println!("  计划已取消；既有副作用未撤销。");
                } else {
                    println!("  已保留计划。");
                }
                continue;
            }
            _ => {}
        }
        if let Some(title) = input
            .strip_prefix("/rename ")
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            let Some(id) = session_id else {
                println!("  当前没有活动会话。");
                continue;
            };
            let mut session = session_for_resume(&runtime, id).await?;
            session.title = redact_text(&title.chars().take(120).collect::<String>());
            session.touch();
            runtime.store.update(&session).await?;
            println!("  会话已重命名：{}", session.title);
            continue;
        }
        if let Some(raw_id) = input
            .strip_prefix("/resume ")
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            let id = Uuid::parse_str(raw_id)
                .map_err(|_| XduduError::validation("会话 ID 必须是完整 UUID。"))?;
            let session = session_for_resume(&runtime, id).await?;
            session_id = Some(id);
            println!("  已恢复会话：{}", session.title);
            if let Some(plan) = pending_plan_for_session(&runtime, session_id).await? {
                review_plan_classic(&runtime, &mut editor, plan).await?;
            } else if paused_plan_for_session(&runtime, session_id)
                .await?
                .is_some()
            {
                println!("  此会话有暂停计划；使用 /plan status 查看，/plan retry 重试。");
            }
            continue;
        }
        if let Some(goal) = input
            .strip_prefix("/plan new ")
            .or_else(|| input.strip_prefix("/plan "))
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            let (session, plan) =
                create_plan_for_review(&runtime, goal.to_owned(), session_id).await?;
            session_id = Some(session.id);
            if io::stdin().is_terminal() {
                review_plan_classic(&runtime, &mut editor, plan).await?;
            } else {
                println!(
                    "  计划 revision {} 已生成并等待审批。会话：{}",
                    plan.revision, session.id
                );
            }
            continue;
        }
        if let Some(model) = input
            .strip_prefix("/model ")
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            runtime.model = model.to_owned();
            let saved = write_config_value(&runtime.cwd, true, "provider.model", &runtime.model);
            println!(
                "  模型已切换：{}{}",
                runtime.model,
                if saved.is_ok() {
                    "（已保存为默认模型）"
                } else {
                    "（默认配置保存失败）"
                }
            );
            continue;
        }
        if let Some(turns) = input.strip_prefix("/turns ") {
            match turns.trim().parse::<u32>() {
                Ok(value) if (1..=100).contains(&value) => {
                    runtime.max_turns = value;
                    println!("  最大循环次数：{value}");
                }
                _ => println!("  最大循环次数必须是 1 到 100 之间的整数。"),
            }
            continue;
        }
        let result = execute_prompt(&runtime, input.to_owned(), session_id).await?;
        runtime.renderer.finish_run(&result)?;
        session_id = Some(result.session_id);
    }
    Ok(0)
}

fn copy_text(text: &str) -> io::Result<()> {
    #[cfg(target_os = "macos")]
    let mut child = std::process::Command::new("pbcopy")
        .stdin(std::process::Stdio::piped())
        .spawn()?;
    #[cfg(target_os = "windows")]
    let mut child = std::process::Command::new("clip")
        .stdin(std::process::Stdio::piped())
        .spawn()?;
    #[cfg(all(unix, not(target_os = "macos")))]
    let mut child = std::process::Command::new("xclip")
        .args(["-selection", "clipboard"])
        .stdin(std::process::Stdio::piped())
        .spawn()?;
    if let Some(stdin) = child.stdin.as_mut() {
        stdin.write_all(text.as_bytes())?;
    }
    let status = child.wait()?;
    if status.success() {
        Ok(())
    } else {
        Err(io::Error::other("剪贴板命令执行失败"))
    }
}

fn export_session_markdown(
    cwd: &std::path::Path,
    session: &Session,
) -> Result<PathBuf, XduduError> {
    let directory = cwd.join(".xdudu/exports");
    std::fs::create_dir_all(&directory).map_err(XduduError::from)?;
    let path = directory.join(format!("{}.md", session.id));
    let mut output = format!(
        "# {}\n\n- 会话：{}\n- 更新时间：{}\n\n",
        redact_text(&session.title),
        session.id,
        session.updated_at
    );
    for message in &session.messages {
        if message.content.trim().is_empty() {
            continue;
        }
        let heading = match message.role {
            xdudu_core::provider::MessageRole::User => "用户",
            xdudu_core::provider::MessageRole::Assistant => "XDUDU",
            xdudu_core::provider::MessageRole::Tool => "工具",
            xdudu_core::provider::MessageRole::System => "系统",
        };
        output.push_str(&format!(
            "## {heading}\n\n{}\n\n",
            redact_text(&message.content)
        ));
    }
    std::fs::write(&path, output).map_err(XduduError::from)?;
    Ok(path)
}

fn print_config(resolved: &ResolvedConfig) -> Result<(), XduduError> {
    let value = serde_json::json!({
        "config": resolved.config,
        "sources": resolved.sources.iter().map(|(key, source)| {
            (key.clone(), source.as_str())
        }).collect::<std::collections::BTreeMap<_, _>>(),
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&value).map_err(XduduError::from)?
    );
    Ok(())
}

fn config_value(resolved: &ResolvedConfig, key: &str) -> Option<String> {
    match key {
        "provider.name" => Some(resolved.config.provider.name.clone()),
        "provider.model" => Some(resolved.config.provider.model.clone()),
        "provider.base_url" => Some(
            resolved
                .config
                .provider
                .base_url
                .clone()
                .unwrap_or_else(|| "<默认端点>".into()),
        ),
        "provider.timeout_seconds" => Some(resolved.config.provider.timeout_seconds.to_string()),
        "provider.max_attempts" => Some(resolved.config.provider.max_attempts.to_string()),
        "provider.retry_base_ms" => Some(resolved.config.provider.retry_base_ms.to_string()),
        "provider.min_request_interval_ms" => {
            Some(resolved.config.provider.min_request_interval_ms.to_string())
        }
        "agent.max_turns" => Some(resolved.config.agent.max_turns.to_string()),
        "agent.permission" => Some(resolved.config.agent.permission.clone()),
        "agent.approval" => Some(resolved.config.agent.approval.clone()),
        "output.json" => Some(resolved.config.output.json.to_string()),
        "output.no_stream" => Some(resolved.config.output.no_stream.to_string()),
        "output.color" => Some(resolved.config.output.color.to_string()),
        "output.debug_trace" => Some(resolved.config.output.debug_trace.to_string()),
        _ => None,
    }
}

async fn handle_config(
    command: ConfigCommand,
    cwd: &std::path::Path,
    cli_overrides: ConfigOverrides,
) -> Result<u8, XduduError> {
    match command {
        ConfigCommand::Show => print_config(&load_config(cwd, cli_overrides)?)?,
        ConfigCommand::Explain { key } => {
            let resolved = load_config(cwd, cli_overrides)?;
            let value = config_value(&resolved, &key)
                .ok_or_else(|| XduduError::validation(format!("未知配置项：{key}")))?;
            let source = resolved
                .source(&key)
                .map(|source| source.as_str())
                .unwrap_or("unknown");
            println!("{key} = {value}\n来源：{source}");
        }
        ConfigCommand::Set {
            key,
            value,
            user,
            project: _,
        } => {
            let path = write_config_value(cwd, user, &key, &value)?;
            println!("已写入：{}", path.display());
        }
        ConfigCommand::Path => {
            let (user, project) = config_paths(cwd)?;
            println!(
                "用户配置：{}\n项目配置：{}\n永久审批规则：{}",
                user.display(),
                project.display(),
                approval_rules_path()?.display()
            );
        }
    }
    Ok(0)
}

fn auth_provider(
    explicit: Option<String>,
    resolved: &ResolvedConfig,
) -> Result<String, XduduError> {
    let provider = explicit.unwrap_or_else(|| resolved.config.provider.name.clone());
    if !matches!(
        provider.as_str(),
        "anthropic" | "deepseek" | "openai-compatible"
    ) {
        return Err(XduduError::validation(format!(
            "不支持的 Provider：{provider}"
        )));
    }
    Ok(provider)
}

async fn handle_auth(command: AuthCommand, resolved: &ResolvedConfig) -> Result<u8, XduduError> {
    let store = KeyringSecretStore;
    match command {
        AuthCommand::Login { provider } => {
            let provider = auth_provider(provider, resolved)?;
            let value = rpassword::prompt_password(format!("{provider} API Key："))
                .map_err(XduduError::from)?;
            store.set(&provider, SecretString::new(value)?).await?;
            println!("已将 {provider} API Key 保存到系统凭据存储。");
        }
        AuthCommand::Status { provider } => {
            let provider = auth_provider(provider, resolved)?;
            match resolve_secret(&provider, &store).await {
                Ok((secret, source)) => {
                    let source = match source {
                        SecretSource::Environment => "环境变量",
                        SecretSource::SystemStore => "系统凭据",
                    };
                    println!("{provider}：已配置（{source}，{}）", secret.masked());
                }
                Err(_) => println!("{provider}：未配置"),
            }
        }
        AuthCommand::Logout { provider } => {
            let provider = auth_provider(provider, resolved)?;
            if store.delete(&provider).await? {
                println!("已从系统凭据存储删除 {provider} API Key。");
            } else {
                println!("系统凭据中没有 {provider} API Key。");
            }
        }
    }
    Ok(0)
}

async fn handle_memory(cwd: &Path, command: MemoryCommand) -> Result<u8, XduduError> {
    match command {
        MemoryCommand::List => {
            let content = read_memory_document(cwd)?.unwrap_or_else(|| {
                "# XDUDU 长期记忆\n\n尚未生成。完成一次有效任务后会自动整理。".into()
            });
            println!("{content}");
            Ok(0)
        }
        MemoryCommand::Edit => {
            let path = memory_document_path(cwd);
            if read_memory_document(cwd)?.is_none() {
                write_memory_document(
                    cwd,
                    "# XDUDU 长期记忆\n\n## 用户偏好\n\n## 项目与工作区\n\n## 长期目标与约定",
                )?;
            }
            let editor = env::var("VISUAL")
                .or_else(|_| env::var("EDITOR"))
                .unwrap_or_else(|_| {
                    if cfg!(windows) {
                        "notepad".into()
                    } else {
                        "vi".into()
                    }
                });
            let status = std::process::Command::new(editor).arg(&path).status()?;
            Ok(if status.success() { 0 } else { 1 })
        }
        MemoryCommand::Path => {
            println!("{}", memory_document_path(cwd).display());
            Ok(0)
        }
    }
}

async fn handle_approval(command: ApprovalCommand) -> Result<u8, XduduError> {
    let store = JsonApprovalRuleStore::open(approval_rules_path()?).await?;
    match command {
        ApprovalCommand::List => {
            let rules = store.list().await;
            if rules.is_empty() {
                println!("没有永久审批规则。\n规则文件：{}", store.path().display());
            } else {
                println!("工具                 副作用");
                for rule in rules {
                    println!("{:<20} {}", rule.tool_name, rule.side_effect.as_str());
                }
                println!("规则文件：{}", store.path().display());
            }
        }
        ApprovalCommand::Revoke { tool } => {
            let removed = store.revoke(&tool).await?;
            if removed == 0 {
                println!("没有找到工具“{tool}”的永久审批规则。");
            } else {
                println!("已撤销工具“{tool}”的永久审批规则。");
            }
        }
        ApprovalCommand::Clear => {
            let removed = store.clear().await?;
            println!("已清除 {removed} 条永久审批规则。");
        }
    }
    Ok(0)
}

async fn handle_mcp(command: McpCommand) -> Result<u8, XduduError> {
    let mut config = load_mcp_config()?;
    match command {
        McpCommand::List => {
            if config.servers.is_empty() {
                println!(
                    "尚未配置 MCP Server。配置文件：{}",
                    mcp_config_path()?.display()
                );
            } else {
                for server in &config.servers {
                    let endpoint = match server.transport {
                        McpTransportKind::Stdio => server.command.as_deref().unwrap_or_default(),
                        McpTransportKind::StreamableHttp => {
                            server.url.as_deref().unwrap_or_default()
                        }
                    };
                    println!(
                        "{}\t{}\t{}\t{}",
                        server.name,
                        if server.enabled {
                            "enabled"
                        } else {
                            "disabled"
                        },
                        match server.transport {
                            McpTransportKind::Stdio => "stdio",
                            McpTransportKind::StreamableHttp => "streamable-http",
                        },
                        endpoint
                    );
                }
            }
        }
        McpCommand::Show { name } => {
            let server = find_mcp_server(&config, &name)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "name":server.name,
                    "enabled":server.enabled,
                    "transport":server.transport,
                    "command":server.command,
                    "args":server.args,
                    "environmentKeys":server.env.keys().collect::<Vec<_>>(),
                    "url":server.url,
                    "credential":server.credential.as_ref().map(|_| "[系统凭据]"),
                    "timeoutSeconds":server.timeout_seconds,
                }))?
            );
        }
        McpCommand::AddStdio {
            name,
            command,
            args,
        } => {
            ensure_new_mcp_name(&config, &name)?;
            config.servers.push(McpServerConfig {
                name: name.clone(),
                enabled: true,
                transport: McpTransportKind::Stdio,
                command: Some(command),
                args,
                env: Default::default(),
                url: None,
                credential: None,
                timeout_seconds: 30,
            });
            let path = save_mcp_config(&config)?;
            println!(
                "已添加并启用 stdio MCP Server：{name}\n配置：{}",
                path.display()
            );
        }
        McpCommand::AddHttp { name, url, auth } => {
            ensure_new_mcp_name(&config, &name)?;
            config.servers.push(McpServerConfig {
                name: name.clone(),
                enabled: true,
                transport: McpTransportKind::StreamableHttp,
                command: None,
                args: Vec::new(),
                env: Default::default(),
                url: Some(url),
                credential: auth.then(|| name.clone()),
                timeout_seconds: 30,
            });
            let path = save_mcp_config(&config)?;
            println!("已添加并启用 Streamable HTTP MCP Server：{name}");
            if auth {
                println!("请继续运行：xdudu mcp login {name}");
            }
            println!("配置：{}", path.display());
        }
        McpCommand::Enable { name } => {
            let server = find_mcp_server_mut(&mut config, &name)?;
            server.enabled = true;
            save_mcp_config(&config)?;
            println!("MCP Server {name} 已启用。");
        }
        McpCommand::Disable { name } => {
            let server = find_mcp_server_mut(&mut config, &name)?;
            server.enabled = false;
            save_mcp_config(&config)?;
            println!("MCP Server {name} 已禁用。");
        }
        McpCommand::Remove { name } => {
            let before = config.servers.len();
            config.servers.retain(|server| server.name != name);
            if config.servers.len() == before {
                return Err(XduduError::validation(format!("找不到 MCP Server：{name}")));
            }
            save_mcp_config(&config)?;
            println!("已删除 MCP Server 配置：{name}");
        }
        McpCommand::Login { name } => {
            let server = find_mcp_server(&config, &name)?;
            if server.transport != McpTransportKind::StreamableHttp {
                return Err(XduduError::validation("只有 HTTP MCP 使用 Bearer Token。"));
            }
            let account = server
                .credential_account()
                .ok_or_else(|| XduduError::validation("该 Server 未启用认证引用。"))?;
            let token = rpassword::prompt_password(format!("请输入 {name} 的 Bearer Token："))?;
            KeyringSecretStore
                .set(&account, SecretString::new(token)?)
                .await?;
            println!("已保存到系统凭据：{account}");
        }
        McpCommand::Logout { name } => {
            let account = format!("mcp:{name}");
            let removed = KeyringSecretStore.delete(&account).await?;
            println!(
                "{}",
                if removed {
                    format!("已删除系统凭据：{account}")
                } else {
                    format!("系统凭据不存在：{account}")
                }
            );
        }
        McpCommand::Doctor { name } => {
            let selected = config
                .servers
                .iter()
                .filter(|server| name.as_ref().is_none_or(|value| &server.name == value))
                .cloned()
                .collect::<Vec<_>>();
            if selected.is_empty() {
                return Err(XduduError::validation("没有匹配的 MCP Server。"));
            }
            let store = KeyringSecretStore;
            for server in selected {
                if !server.enabled {
                    println!("{}\tdisabled", server.name);
                    continue;
                }
                let runtime = McpServerRuntime::new(server.clone(), &store).await?;
                let tools = runtime.list_tools(CancellationToken::new()).await?;
                println!("{}\tok\t{} tools", server.name, tools.len());
            }
        }
    }
    Ok(0)
}

async fn handle_plugin(command: PluginCommand) -> Result<u8, XduduError> {
    let mut manifests = load_plugin_manifests()?;
    match command {
        PluginCommand::List => {
            if manifests.is_empty() {
                println!("尚未安装插件。目录：{}", plugin_directory()?.display());
            } else {
                for plugin in manifests {
                    println!(
                        "{}\t{}\t{}\t{} MCP servers",
                        plugin.id,
                        if plugin.enabled {
                            "enabled"
                        } else {
                            "disabled"
                        },
                        plugin.version,
                        plugin.mcp_servers.len()
                    );
                }
            }
        }
        PluginCommand::Show { id } => {
            let plugin = find_plugin(&manifests, &id)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "schemaVersion": plugin.schema_version,
                    "id": plugin.id,
                    "name": plugin.name,
                    "version": plugin.version,
                    "description": plugin.description,
                    "enabled": plugin.enabled,
                    "homepage": plugin.homepage,
                    "sha256": plugin.sha256,
                    "signature": plugin.signature.as_ref().map(|signature| serde_json::json!({
                        "algorithm": signature.algorithm,
                        "keyId": signature.key_id,
                        "value": "[已隐藏]",
                    })),
                    "mcpServers": plugin.mcp_servers.iter().map(|server| serde_json::json!({
                        "name": server.name,
                        "enabled": server.enabled,
                        "transport": server.transport,
                        "command": server.command,
                        "args": server.args,
                        "environmentKeys": server.env.keys().collect::<Vec<_>>(),
                        "url": server.url,
                        "credential": server.credential.as_ref().map(|_| "[系统凭据]"),
                    })).collect::<Vec<_>>(),
                }))?
            );
        }
        PluginCommand::Enable { id } => {
            let plugin = find_plugin_mut(&mut manifests, &id)?;
            plugin.enabled = true;
            save_plugin_manifest(plugin)?;
            println!("插件 {id} 已启用。");
        }
        PluginCommand::Disable { id } => {
            let plugin = find_plugin_mut(&mut manifests, &id)?;
            plugin.enabled = false;
            save_plugin_manifest(plugin)?;
            println!("插件 {id} 已禁用。");
        }
        PluginCommand::Doctor { id } => {
            let selected = manifests
                .into_iter()
                .filter(|plugin| id.as_ref().is_none_or(|value| &plugin.id == value))
                .collect::<Vec<_>>();
            if selected.is_empty() {
                return Err(XduduError::validation("没有匹配的插件。"));
            }
            let store = KeyringSecretStore;
            for plugin in selected {
                plugin.validate()?;
                if !plugin.enabled {
                    println!("{}\tdisabled", plugin.id);
                    continue;
                }
                let mut tool_count = 0usize;
                for server in plugin.mcp_servers {
                    if !server.enabled {
                        continue;
                    }
                    let runtime = McpServerRuntime::new(server, &store).await?;
                    tool_count += runtime.list_tools(CancellationToken::new()).await?.len();
                }
                println!("{}\tok\t{} tools", plugin.id, tool_count);
            }
        }
    }
    Ok(0)
}

fn find_plugin<'a>(
    manifests: &'a [PluginManifest],
    id: &str,
) -> Result<&'a PluginManifest, XduduError> {
    manifests
        .iter()
        .find(|plugin| plugin.id == id)
        .ok_or_else(|| XduduError::validation(format!("找不到插件：{id}")))
}

fn find_plugin_mut<'a>(
    manifests: &'a mut [PluginManifest],
    id: &str,
) -> Result<&'a mut PluginManifest, XduduError> {
    manifests
        .iter_mut()
        .find(|plugin| plugin.id == id)
        .ok_or_else(|| XduduError::validation(format!("找不到插件：{id}")))
}

fn find_mcp_server<'a>(
    config: &'a McpConfigFile,
    name: &str,
) -> Result<&'a McpServerConfig, XduduError> {
    config
        .servers
        .iter()
        .find(|server| server.name == name)
        .ok_or_else(|| XduduError::validation(format!("找不到 MCP Server：{name}")))
}

fn find_mcp_server_mut<'a>(
    config: &'a mut McpConfigFile,
    name: &str,
) -> Result<&'a mut McpServerConfig, XduduError> {
    config
        .servers
        .iter_mut()
        .find(|server| server.name == name)
        .ok_or_else(|| XduduError::validation(format!("找不到 MCP Server：{name}")))
}

fn ensure_new_mcp_name(config: &McpConfigFile, name: &str) -> Result<(), XduduError> {
    if config.servers.iter().any(|server| server.name == name) {
        Err(XduduError::validation(format!("MCP Server 已存在：{name}")))
    } else {
        Ok(())
    }
}

async fn handle_session(command: SessionCommand, cwd: &std::path::Path) -> Result<u8, XduduError> {
    let store = SqliteSessionStore::new(cwd)?;
    match command {
        SessionCommand::List { limit } => {
            let sessions = store.list(limit as usize).await?;
            if sessions.is_empty() {
                println!("当前工作区还没有会话。");
                return Ok(0);
            }
            println!(
                "会话 ID                              状态               更新时间                 标题"
            );
            for session in sessions {
                let status = serde_json::to_string(&session.status)
                    .unwrap_or_else(|_| "\"unknown\"".into())
                    .trim_matches('"')
                    .to_owned();
                println!(
                    "{}  {:<18} {}  {}",
                    session.id,
                    status,
                    session.updated_at.format("%Y-%m-%d %H:%M:%S"),
                    session.title.replace(['\r', '\n'], " ")
                );
            }
        }
        SessionCommand::Show { id } => {
            let session = store
                .get(id)
                .await?
                .ok_or_else(|| XduduError::validation(format!("找不到会话：{id}")))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&session.public_snapshot())?
            );
        }
        SessionCommand::Resume { .. } => {
            return Err(XduduError::validation(
                "session resume 应由 Agent 运行入口处理。",
            ));
        }
    }
    Ok(0)
}

async fn handle_plan_command(runtime: &Runtime, command: PlanCommand) -> Result<u8, XduduError> {
    match command {
        PlanCommand::Create { goal } => {
            let (session, plan) = create_plan_for_review(runtime, goal, None).await?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "sessionId": session.id,
                    "plan": plan
                }))?
            );
        }
        PlanCommand::List { limit } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&runtime.store.list_plans(limit).await?)?
            );
        }
        PlanCommand::Show { id } => {
            let plan = runtime
                .store
                .get_plan(id)
                .await?
                .ok_or_else(|| XduduError::validation(format!("找不到计划：{id}")))?;
            println!("{}", serde_json::to_string_pretty(&plan)?);
        }
        PlanCommand::Revisions { id } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&runtime.store.list_plan_revisions(id).await?)?
            );
        }
        PlanCommand::Approve { id, reason } => {
            let plan = runtime
                .store
                .get_plan(id)
                .await?
                .ok_or_else(|| XduduError::validation(format!("找不到计划：{id}")))?;
            let approved = approve_plan(runtime.store.as_ref(), id, plan.revision, reason).await?;
            sync_plan_session(
                runtime,
                approved.session_id,
                SessionStatus::PlanReady,
                AgentLoopState::Completed,
            )
            .await?;
            println!("{}", serde_json::to_string_pretty(&approved)?);
        }
        PlanCommand::Reject { id, reason } => {
            let plan = runtime
                .store
                .get_plan(id)
                .await?
                .ok_or_else(|| XduduError::validation(format!("找不到计划：{id}")))?;
            let rejected = reject_plan(runtime.store.as_ref(), id, plan.revision, reason).await?;
            sync_plan_session(
                runtime,
                rejected.session_id,
                SessionStatus::Incomplete,
                AgentLoopState::Incomplete,
            )
            .await?;
            println!("{}", serde_json::to_string_pretty(&rejected)?);
        }
        PlanCommand::Revise { id, request } => {
            let plan = runtime
                .store
                .get_plan(id)
                .await?
                .ok_or_else(|| XduduError::validation(format!("找不到计划：{id}")))?;
            let session = session_for_resume(runtime, plan.session_id).await?;
            let revised = revise_plan(PlanRevisionConfig {
                plan_id: id,
                expected_revision: plan.revision,
                change_request: request,
                context: plan_context(&session),
                model: runtime.model.clone(),
                cwd: runtime.cwd.clone(),
                provider: runtime.provider.as_ref(),
                plan_store: runtime.store.as_ref(),
                cancellation: CancellationToken::new(),
            })
            .await?;
            sync_plan_session(
                runtime,
                revised.plan.session_id,
                SessionStatus::WaitingApproval,
                AgentLoopState::WaitingApproval,
            )
            .await?;
            println!("{}", serde_json::to_string_pretty(&revised.plan)?);
        }
        PlanCommand::Run { id } => {
            let plan = runtime
                .store
                .get_plan(id)
                .await?
                .ok_or_else(|| XduduError::validation(format!("找不到计划：{id}")))?;
            if plan.status != PlanStatus::Approved {
                return Err(XduduError::validation(
                    "plan run 只执行已批准计划；暂停计划请使用 plan retry。",
                ));
            }
            let result = execute_plan_with_sink(runtime, id, &runtime.renderer).await?;
            println!("{}", serde_json::to_string_pretty(&result.plan)?);
            return Ok((!result.completed) as u8);
        }
        PlanCommand::Retry { id } => {
            let plan = runtime
                .store
                .get_plan(id)
                .await?
                .ok_or_else(|| XduduError::validation(format!("找不到计划：{id}")))?;
            if plan.status != PlanStatus::Paused {
                return Err(XduduError::validation("plan retry 只重试暂停计划。"));
            }
            let result = execute_plan_with_sink(runtime, id, &runtime.renderer).await?;
            println!("{}", serde_json::to_string_pretty(&result.plan)?);
            return Ok((!result.completed) as u8);
        }
        PlanCommand::Cancel { id } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&cancel_plan(runtime, id).await?)?
            );
        }
    }
    Ok(0)
}

async fn handle_plan_local(cwd: &std::path::Path, command: PlanCommand) -> Result<u8, XduduError> {
    let store = SqliteSessionStore::new(cwd)?;
    match command {
        PlanCommand::List { limit } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&store.list_plans(limit).await?)?
            );
        }
        PlanCommand::Show { id } => {
            let plan = store
                .get_plan(id)
                .await?
                .ok_or_else(|| XduduError::validation(format!("找不到计划：{id}")))?;
            println!("{}", serde_json::to_string_pretty(&plan)?);
        }
        PlanCommand::Revisions { id } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&store.list_plan_revisions(id).await?)?
            );
        }
        PlanCommand::Approve { id, reason } => {
            let plan = store
                .get_plan(id)
                .await?
                .ok_or_else(|| XduduError::validation(format!("找不到计划：{id}")))?;
            let approved = approve_plan(&store, id, plan.revision, reason).await?;
            sync_plan_session_store(
                &store,
                approved.session_id,
                SessionStatus::PlanReady,
                AgentLoopState::Completed,
            )
            .await?;
            println!("{}", serde_json::to_string_pretty(&approved)?);
        }
        PlanCommand::Reject { id, reason } => {
            let plan = store
                .get_plan(id)
                .await?
                .ok_or_else(|| XduduError::validation(format!("找不到计划：{id}")))?;
            let rejected = reject_plan(&store, id, plan.revision, reason).await?;
            sync_plan_session_store(
                &store,
                rejected.session_id,
                SessionStatus::Incomplete,
                AgentLoopState::Incomplete,
            )
            .await?;
            println!("{}", serde_json::to_string_pretty(&rejected)?);
        }
        PlanCommand::Cancel { id } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&cancel_plan_with_store(&store, id).await?)?
            );
        }
        _ => unreachable!("本地计划入口只接收无需 Provider 的命令"),
    }
    Ok(0)
}

async fn run() -> Result<u8, XduduError> {
    let cli = Cli::parse();
    let cwd = env::current_dir().map_err(XduduError::from)?;
    let cli_overrides = overrides(&cli);

    match cli.command {
        Some(Command::Config { command }) => {
            return handle_config(command, &cwd, cli_overrides).await;
        }
        Some(Command::Auth { command }) => {
            let resolved = load_config(&cwd, cli_overrides)?;
            return handle_auth(command, &resolved).await;
        }
        Some(Command::Approval { command }) => {
            return handle_approval(command).await;
        }
        Some(Command::Mcp { command }) => {
            return handle_mcp(command).await;
        }
        Some(Command::Plugin { command }) => {
            return handle_plugin(command).await;
        }
        Some(Command::Doctor) => {
            return run_doctor(&cwd, cli_overrides, cli.json).await;
        }
        Some(Command::Memory { command }) => {
            return handle_memory(&cwd, command).await;
        }
        Some(Command::Undo(args)) => {
            let _workspace_lock = WorkspaceLock::acquire(&cwd)?;
            let result = JsonChangeLedger::new(&cwd)
                .undo(args.change, cli.session)
                .await?;
            let action = if result.removed_created_files == result.paths.len() {
                "已删除由 Agent 创建的文件"
            } else {
                "已恢复变更事务"
            };
            println!(
                "{action}：{}\n变更记录：{}",
                result
                    .paths
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>()
                    .join("、"),
                result.change_id
            );
            return Ok(0);
        }
        Some(Command::Session {
            command: command @ (SessionCommand::List { .. } | SessionCommand::Show { .. }),
        }) => {
            return handle_session(command, &cwd).await;
        }
        Some(Command::Plan {
            command:
                command @ (PlanCommand::List { .. }
                | PlanCommand::Show { .. }
                | PlanCommand::Revisions { .. }
                | PlanCommand::Approve { .. }
                | PlanCommand::Reject { .. }
                | PlanCommand::Cancel { .. }),
        }) => {
            return handle_plan_local(&cwd, command).await;
        }
        _ => {}
    }

    let command_prompt = match &cli.command {
        Some(Command::Run(args)) => args.prompt.clone(),
        Some(Command::Session {
            command: SessionCommand::Resume { prompt, .. },
        }) => prompt.clone(),
        _ => None,
    };
    let requested_session = match &cli.command {
        Some(Command::Session {
            command: SessionCommand::Resume { id, .. },
        }) => Some(*id),
        _ => cli.session,
    };
    let resolved = load_config(&cwd, cli_overrides)?;
    if let Some(Command::Plan { command }) = cli.command {
        let runtime = create_runtime(cwd, &resolved, false).await?;
        return handle_plan_command(&runtime, command).await;
    }
    let piped = !io::stdin().is_terminal();
    let prompt = command_prompt.or(cli.prompt);
    let interactive = cli.interactive || (prompt.is_none() && !piped);
    if resolved.config.output.json && interactive {
        return Err(XduduError::validation(
            "--json 仅支持非交互模式，请同时提供 prompt 或管道输入。",
        ));
    }
    let runtime = create_runtime(cwd, &resolved, interactive).await?;
    if !resolved.config.output.json && !interactive {
        print_banner(&runtime, interactive);
    }
    if interactive {
        return interactive_loop(runtime, prompt, requested_session).await;
    }
    let prompt = if let Some(prompt) = prompt {
        prompt
    } else {
        let mut input = String::new();
        tokio::io::stdin()
            .read_to_string(&mut input)
            .await
            .map_err(XduduError::from)?;
        input.trim().to_owned()
    };
    if prompt.is_empty() {
        return Err(XduduError::validation("prompt 不能为空。"));
    }
    let result = execute_prompt(&runtime, prompt, requested_session).await?;
    runtime.renderer.finish_run(&result)?;
    Ok(result.exit_code)
}

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(code) => ExitCode::from(code),
        Err(error) => {
            eprintln!("\n  错误：{}", redact_text(&error.message));
            ExitCode::from(error.exit_code())
        }
    }
}
