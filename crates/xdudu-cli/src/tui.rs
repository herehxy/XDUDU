//! XDUDU 全屏终端界面。
//!
//! 界面运行在正常屏幕（不使用备用屏幕，也不捕获鼠标）：已提交的对话内容
//! 一次性打印进终端滚动区，滚轮、搜索和复制都由终端原生处理；底部活动区
//! （流式尾巴、实时工具活动、状态线、Composer）在每次变化时重绘。
//!
//! 输入由共享路由（[`crate::input_queue::InputRouter`]）统一分发：raw 模式
//! 在会话期间常驻，运行中 Composer 仍然可用，Ctrl+C 直接取消当前任务，
//! 提交的内容进入排队队列，当前任务结束后自动执行。

use std::{
    collections::VecDeque,
    io::{self, Write},
    path::PathBuf,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use crossterm::{
    cursor::{Hide, MoveTo, Show},
    event::{
        DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEvent, KeyEventKind,
        KeyModifiers,
    },
    execute, queue,
    style::{Attribute, Color, Print, ResetColor, SetAttribute, SetForegroundColor},
    terminal::{Clear, ClearType, disable_raw_mode, enable_raw_mode, size},
};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};
use xdudu_core::{
    AgentEvent, AgentLoopState, AgentRunResult, EventSink, Plan, Session, provider::MessageRole,
    redact_text, redact_value,
};

use crate::{
    inline_terminal::InlineActivityDiff,
    input_queue::{InputFocus, InputRouter},
    markdown::{MarkdownLineKind, terminal_markdown},
    ui::{
        ModelOption, model_display_name, model_matches, model_options, supports_true_color,
        tool_display_name, tool_phase_display,
    },
};

const MAX_COMMAND_SUGGESTIONS: usize = 5;
const MAX_INPUT_CHARS: usize = 262_144;
/// 流式尾巴在活动区最多显示的行数；更早的行已提交到滚动区。
const STREAMING_WINDOW: usize = 4;
/// 语义化配色（按主题解析，替代硬编码常量）。
#[derive(Debug, Clone, Copy)]
pub struct Palette {
    pub primary: Color,
    pub text: Color,
    pub muted: Color,
    pub border: Color,
    pub accent: Color,
    pub success: Color,
    pub warning: Color,
    pub recovery: Color,
}

/// 按 `output.theme`（dark | light | auto）解析语义色板。
/// auto 通过 `COLORFGBG` 环境变量探测终端背景（";15"/";7" 等亮背景 → light）。
pub fn resolve_palette(theme: &str) -> Palette {
    let light = match theme {
        "light" => true,
        "dark" => false,
        _ => std::env::var("COLORFGBG").ok().is_some_and(|value| {
            value
                .split(';')
                .next_back()
                .is_some_and(|bg| matches!(bg.trim(), "7" | "15" | "231" | "255"))
        }),
    };
    if light {
        // 浅色终端：深色文字 + 高对比语义色。
        Palette {
            primary: Color::Rgb {
                r: 176,
                g: 84,
                b: 0,
            },
            text: Color::Rgb {
                r: 30,
                g: 34,
                b: 42,
            },
            muted: Color::Rgb {
                r: 100,
                g: 106,
                b: 115,
            },
            border: Color::Rgb {
                r: 120,
                g: 128,
                b: 140,
            },
            accent: Color::Rgb {
                r: 0,
                g: 112,
                b: 168,
            },
            success: Color::Rgb {
                r: 22,
                g: 130,
                b: 62,
            },
            warning: Color::Rgb {
                r: 196,
                g: 96,
                b: 0,
            },
            recovery: Color::Rgb {
                r: 0,
                g: 112,
                b: 168,
            },
        }
    } else {
        // 深色终端：亮色文字 + 高对比语义色（palette().warning 与 palette().primary 区分色相）。
        Palette {
            primary: Color::Rgb {
                r: 255,
                g: 179,
                b: 71,
            },
            text: Color::Rgb {
                r: 232,
                g: 236,
                b: 242,
            },
            muted: Color::Rgb {
                r: 148,
                g: 156,
                b: 166,
            },
            border: Color::Rgb {
                r: 110,
                g: 118,
                b: 128,
            },
            accent: Color::Rgb {
                r: 94,
                g: 180,
                b: 255,
            },
            success: Color::Rgb {
                r: 90,
                g: 210,
                b: 130,
            },
            warning: Color::Rgb {
                r: 240,
                g: 150,
                b: 60,
            },
            recovery: Color::Rgb {
                r: 94,
                g: 180,
                b: 255,
            },
        }
    }
}

/// 进程级已解析色板（启动时按配置设置一次）。
static PALETTE: std::sync::OnceLock<Palette> = std::sync::OnceLock::new();

/// 按配置初始化全局色板；重复调用忽略。
pub fn init_palette(theme: &str) {
    let _ = PALETTE.set(resolve_palette(theme));
}

/// 取全局色板；未初始化时按 auto 探测兜底。
fn palette() -> &'static Palette {
    PALETTE.get_or_init(|| resolve_palette("auto"))
}

/// 单次提交行数上限：超过时折叠，避免长输出平铺滚屏。
const MAX_COMMIT_LINES: usize = 40;
/// 折叠时保留的前段行数。
const MAX_COMMIT_HEAD: usize = 20;
/// 折叠时保留的末段行数。
const MAX_COMMIT_TAIL: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    User,
    Assistant,
    AssistantHeading,
    AssistantCode,
    AssistantDiffAdd,
    AssistantDiffRemove,
    Tool,
    /// 工具折叠头行：`- 名称 入参摘要`。
    ToolHead,
    /// Claude 式工具折叠尾行：`⎿ 状态 · 耗时（· 失败原因）`。
    ToolDetail,
    /// 思考叙述块：工具调用之间的模型旁白，浅灰单模块，与正式回答区分。
    Thinking,
    System,
    Warning,
    Recovery,
}

#[derive(Debug, Clone)]
struct TranscriptBlock {
    role: Role,
    text: String,
    /// 折叠组条目（展开状态显示）；None 表示普通块。
    group_entries: Option<Vec<(Role, String)>>,
    /// 折叠状态：true 仅显示组头，false 展开显示全部条目。
    collapsed: bool,
}

#[derive(Debug, Clone)]
struct ToolActivity {
    call_id: String,
    name: String,
    detail: String,
    finished: Option<bool>,
    duration_ms: Option<u64>,
}

/// 活动区缓冲中的同类工具折叠：同键（工具名+入参摘要）累计计数。
#[derive(Debug, Clone, PartialEq)]
struct PendingToolFold {
    key: String,
    head: String,
    count: usize,
    success: usize,
    failed: usize,
    total_ms: u64,
    last_reason: String,
    /// 组内每个工具的明细行（展开状态显示）。
    entries: Vec<(Role, String)>,
}

/// 完成工具的折叠素材：头行、入参摘要、成败、耗时与失败原因。
struct CompletedToolParts {
    head: String,
    summary: String,
    success: bool,
    duration_ms: u64,
    reason: String,
    /// 该工具完整的明细行（展开状态显示）。
    line: String,
}

#[derive(Debug, Clone, Copy)]
struct SlashCommand {
    name: &'static str,
    usage: &'static str,
    description: &'static str,
    requires_argument: bool,
}

const SLASH_COMMANDS: [SlashCommand; 19] = [
    SlashCommand {
        name: "/help",
        usage: "/help",
        description: "显示交互命令",
        requires_argument: false,
    },
    SlashCommand {
        name: "/new",
        usage: "/new",
        description: "开始新会话",
        requires_argument: false,
    },
    SlashCommand {
        name: "/resume",
        usage: "/resume [id]",
        description: "浏览并恢复历史会话",
        requires_argument: false,
    },
    SlashCommand {
        name: "/plan",
        usage: "/plan <目标>",
        description: "生成或审阅执行计划",
        requires_argument: false,
    },
    SlashCommand {
        name: "/plan new",
        usage: "/plan new <目标>",
        description: "创建结构化计划",
        requires_argument: true,
    },
    SlashCommand {
        name: "/plan status",
        usage: "/plan status",
        description: "查看当前计划状态",
        requires_argument: false,
    },
    SlashCommand {
        name: "/plan run",
        usage: "/plan run",
        description: "执行已批准计划",
        requires_argument: false,
    },
    SlashCommand {
        name: "/plan retry",
        usage: "/plan retry",
        description: "重试暂停步骤",
        requires_argument: false,
    },
    SlashCommand {
        name: "/plan cancel",
        usage: "/plan cancel",
        description: "取消当前计划",
        requires_argument: false,
    },
    SlashCommand {
        name: "/plan revisions",
        usage: "/plan revisions",
        description: "查看修订版本",
        requires_argument: false,
    },
    SlashCommand {
        name: "/model",
        usage: "/model [name]",
        description: "选择或切换当前模型",
        requires_argument: false,
    },
    SlashCommand {
        name: "/mcp",
        usage: "/mcp",
        description: "查看 MCP Server 与外部工具",
        requires_argument: false,
    },
    SlashCommand {
        name: "/plugins",
        usage: "/plugins",
        description: "查看声明式插件",
        requires_argument: false,
    },
    SlashCommand {
        name: "/instructions",
        usage: "/instructions",
        description: "查看自定义指令加载情况",
        requires_argument: false,
    },
    SlashCommand {
        name: "/skills",
        usage: "/skills",
        description: "查看可用技能与加载策略",
        requires_argument: false,
    },
    SlashCommand {
        name: "/agent",
        usage: "/agent",
        description: "查看 Agent 档案与子代理",
        requires_argument: false,
    },
    SlashCommand {
        name: "/memory",
        usage: "/memory",
        description: "查看和管理长期记忆",
        requires_argument: false,
    },
    SlashCommand {
        name: "/turns",
        usage: "/turns <n>",
        description: "设置最大循环次数",
        requires_argument: true,
    },
    SlashCommand {
        name: "/exit",
        usage: "/exit",
        description: "退出 XDUDU",
        requires_argument: false,
    },
];

#[derive(Debug)]
struct TuiState {
    version: &'static str,
    provider: String,
    model: String,
    cwd: PathBuf,
    permission: String,
    status: String,
    /// 已提交内容累计打印的行数（单调递增），用于定位内容末尾。
    printed_lines: usize,
    /// 内容末尾在视口中的行：每次提交后更新，清除操作从该行开始，
    /// 与动态变化的滚动区底无关，保证已提交内容永不被清除。
    content_end_row: u16,
    /// 上一次绘制的底部活动区起始行，用于完整清除尺寸变化后的残影。
    dynamic_top: u16,
    /// 启动页尚未被会话内容提交时保持为 true；窗口变化时可安全重新居中。
    intro_active: bool,
    transcript: VecDeque<TranscriptBlock>,
    streaming: String,
    /// 当前轮是否收到过流式增量；用于区分非流式最终消息和已提交的流式回答。
    assistant_received_delta: bool,
    /// 流式文本当前是否位于未闭合的 Markdown 代码栅栏内。
    code_fence_open: bool,
    tools: Vec<ToolActivity>,
    /// 待提交的同类工具折叠：连续相同（工具名+入参摘要）的完成调用先缓冲，
    /// 遇到其它事件时以 ×N 一次性提交。
    tool_fold: Option<PendingToolFold>,
    input: Vec<char>,
    cursor: usize,
    history: Vec<String>,
    history_index: Option<usize>,
    draft: Vec<char>,
    command_selection: usize,
    input_hint: String,
    input_active: bool,
    /// Vim 模式开关（/vim 切换）。
    vim_enabled: bool,
    /// Vim 普通模式（true）或插入模式（false）。
    vim_normal: bool,
    /// 记录普通模式上次按键，用于 dd 等组合键。
    vim_last: Option<char>,
    usage: Option<(u64, u64)>,
    available_tools: Vec<String>,
    skills: Vec<String>,
    session_picker: Option<SessionPicker>,
    plan_review: Option<PlanReviewView>,
    model_picker: Option<ModelPicker>,
    debug_trace: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct SessionChoice {
    pub(crate) id: uuid::Uuid,
    pub(crate) title: String,
    pub(crate) status: String,
    pub(crate) updated_at: String,
}

#[derive(Debug)]
struct SessionPicker {
    choices: Vec<SessionChoice>,
    selected: usize,
}

#[derive(Debug)]
struct ModelPicker {
    choices: Vec<ModelOption>,
    selected: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlanReviewChoice {
    Approve,
    RequestChanges,
    Reject,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlanRecoveryChoice {
    Continue,
    Retry,
    ViewDetails,
    Cancel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlanDialogMode {
    Review,
    Recovery,
}

#[derive(Debug)]
struct PlanReviewView {
    plan: Plan,
    selected: usize,
    scroll: usize,
    mode: PlanDialogMode,
}

#[derive(Clone)]
pub(crate) struct TuiRenderer {
    state: Arc<Mutex<TuiState>>,
    color: bool,
    router: Arc<InputRouter>,
    /// 活动区的上一帧；只输出变化行，避免全屏重绘。
    activity: Arc<Mutex<InlineActivityDiff>>,
}

pub(crate) struct TuiApp {
    renderer: TuiRenderer,
    router: Arc<InputRouter>,
}

pub(crate) struct TuiContext {
    pub(crate) provider: String,
    pub(crate) model: String,
    pub(crate) cwd: PathBuf,
    pub(crate) permission: String,
    pub(crate) available_tools: Vec<String>,
    pub(crate) skills: Vec<String>,
    pub(crate) color: bool,
    pub(crate) debug_trace: bool,
    pub(crate) theme: String,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum InputOutcome {
    Submit(String),
    Command(String),
    Interrupted,
    Exit,
    /// 输入框为空时按 Tab：切换最近一个折叠组的展开/收起。
    ToggleFold,
}

/// 会话期间常驻：raw 模式 + bracketed paste，不进入备用屏幕、不捕获鼠标。
pub(crate) struct ScreenGuard;

impl ScreenGuard {
    fn enter() -> io::Result<Self> {
        enable_raw_mode()?;
        execute!(
            io::stdout(),
            ResetColor,
            SetAttribute(Attribute::Reset),
            EnableBracketedPaste,
            ResetColor,
            SetAttribute(Attribute::Reset),
            Hide
        )?;
        Ok(Self)
    }
}

impl Drop for ScreenGuard {
    fn drop(&mut self) {
        let _ = execute!(
            io::stdout(),
            ResetColor,
            SetAttribute(Attribute::Reset),
            Show,
            DisableBracketedPaste,
            // 恢复全屏滚动区域。
            Print("\x1b[r"),
            ResetColor,
            SetAttribute(Attribute::Reset)
        );
        let _ = disable_raw_mode();
    }
}

impl TuiApp {
    pub(crate) fn enter(
        context: TuiContext,
        router: Arc<InputRouter>,
    ) -> io::Result<(Self, ScreenGuard)> {
        let guard = ScreenGuard::enter()?;
        let state = TuiState {
            version: env!("CARGO_PKG_VERSION"),
            provider: context.provider,
            model: context.model,
            cwd: context.cwd,
            permission: context.permission,
            status: "就绪".into(),
            printed_lines: 0,
            content_end_row: 0,
            dynamic_top: 0,
            intro_active: true,
            transcript: VecDeque::new(),
            streaming: String::new(),
            assistant_received_delta: false,
            code_fence_open: false,
            tools: Vec::new(),
            tool_fold: None,
            input: Vec::new(),
            cursor: 0,
            history: Vec::new(),
            history_index: None,
            draft: Vec::new(),
            command_selection: 0,
            input_hint: "Enter 发送 · Shift+Enter 换行 · / 命令 · ↑↓ 历史 · Ctrl+D 退出".into(),
            input_active: true,
            vim_enabled: false,
            vim_normal: false,
            vim_last: None,
            usage: None,
            available_tools: context.available_tools,
            skills: context.skills,
            session_picker: None,
            plan_review: None,
            model_picker: None,
            debug_trace: context.debug_trace,
        };
        init_palette(&context.theme);
        let app = Self {
            renderer: TuiRenderer {
                state: Arc::new(Mutex::new(state)),
                color: context.color,
                router: Arc::clone(&router),
                activity: Arc::new(Mutex::new(InlineActivityDiff::default())),
            },
            router,
        };
        app.renderer.print_intro()?;
        app.renderer.draw_dynamic()?;
        Ok((app, guard))
    }

    pub(crate) fn renderer(&self) -> TuiRenderer {
        self.renderer.clone()
    }

    /// 更新显示的权限模式（Shift+Tab 循环切换后同步到界面）。
    pub(crate) fn set_permission(&self, permission: &str) -> io::Result<()> {
        {
            let mut state = self.renderer.state.lock().unwrap();
            state.permission = permission.to_owned();
            state.status = format!("权限：{permission}");
        }
        self.renderer.draw_dynamic()
    }

    pub(crate) fn set_model(&self, model: &str) -> io::Result<()> {
        let mut state = self.renderer.state.lock().unwrap();
        state.model = model.to_owned();
        state.status = "模型已切换".into();
        drop(state);
        self.renderer.draw_dynamic()
    }

    pub(crate) fn notice(&self, message: impl Into<String>) -> io::Result<()> {
        let text = message.into();
        {
            let mut state = self.renderer.state.lock().unwrap();
            push_block(&mut state, Role::System, text.clone());
        }
        self.renderer.commit_block(Role::System, &text)
    }

    pub(crate) fn load_session(&self, session: &Session) -> io::Result<()> {
        {
            let mut state = self.renderer.state.lock().unwrap();
            load_session_state(&mut state, session);
        }
        self.renderer.restore_viewport()
    }

    /// 运行中注入的提示词：仅追加用户块到历史区，不清空流式/工具状态。
    pub(crate) fn inject_prompt(&self, prompt: &str) -> io::Result<()> {
        {
            let mut state = self.renderer.state.lock().unwrap();
            push_block(&mut state, Role::User, prompt.to_owned());
        }
        self.renderer.commit_block(Role::User, prompt)
    }

    pub(crate) fn begin_prompt(&self, prompt: &str) -> io::Result<()> {
        {
            let mut state = self.renderer.state.lock().unwrap();
            state.status = "正在处理".into();
            push_block(&mut state, Role::User, prompt.to_owned());
            state.streaming.clear();
            state.assistant_received_delta = false;
            state.code_fence_open = false;
            state.tools.clear();
            state.tool_fold = None;
            state.input.clear();
            state.cursor = 0;
            state.history_index = None;
        }
        self.renderer.commit_block(Role::User, prompt)
    }

    pub(crate) fn finish_prompt(&self, result: &AgentRunResult) -> io::Result<()> {
        let (fold_blocks, assistant) = {
            let mut state = self.renderer.state.lock().unwrap();
            // 任务收尾：先提交缓冲中未归并完的同类工具折叠。
            let mut fold_blocks = Vec::new();
            flush_pending_tool_fold(&mut state, &mut fold_blocks);
            let assistant = take_final_assistant_tail(&mut state, &result.final_message);
            if !assistant.trim().is_empty() {
                push_block(&mut state, Role::Assistant, redact_text(&assistant));
            }
            state.status = if result.exit_code == 0 {
                "就绪".into()
            } else {
                "未完成".into()
            };
            state.tools.clear();
            state.tool_fold = None;
            (fold_blocks, assistant)
        };
        if !fold_blocks.is_empty() {
            let _ = self.renderer.commit_blocks(&fold_blocks);
        }
        if assistant.trim().is_empty() {
            self.renderer.draw_dynamic()
        } else {
            self.renderer.commit_block(Role::Assistant, &assistant)
        }
    }

    /// 把按键交给 Composer 处理，返回可能的交互结果。
    pub(crate) fn handle_key(&self, key: KeyEvent) -> io::Result<Option<InputOutcome>> {
        // Codex 风格的 Ctrl+L：只重绘当前视口，不清理终端原生滚动历史，
        // 也不丢弃 Composer 草稿或正在排队的输入。
        if key.kind == KeyEventKind::Press
            && key.code == KeyCode::Char('l')
            && key.modifiers.contains(KeyModifiers::CONTROL)
        {
            self.renderer.restore_viewport()?;
            self.renderer.draw_dynamic()?;
            return Ok(None);
        }
        let outcome = {
            let mut state = self.renderer.state.lock().unwrap();
            handle_input_key(&mut state, key)
        };
        self.renderer.draw_dynamic()?;
        Ok(outcome)
    }

    /// 把粘贴内容整段插入 Composer，不触发提交。
    pub(crate) fn handle_paste(&self, text: &str) -> io::Result<()> {
        {
            let mut state = self.renderer.state.lock().unwrap();
            if insert_paste_text(&mut state, text) {
                state.input_hint = format!("粘贴内容已截断：最多 {MAX_INPUT_CHARS} 个字符");
            }
        }
        self.renderer.draw_dynamic()
    }

    /// 切换 Vim 模式；开启时进入插入模式，关闭时恢复常规键位。
    pub(crate) fn toggle_vim(&self) -> io::Result<()> {
        {
            let mut state = self.renderer.state.lock().unwrap();
            state.vim_enabled = !state.vim_enabled;
            state.vim_normal = false;
            state.vim_last = None;
        }
        self.renderer.draw_dynamic()
    }

    /// 等待下一个 Composer 输入事件（运行中与空闲共用）。
    pub(crate) async fn next_input(&self) -> Option<Event> {
        self.router.next_for(InputFocus::Composer).await
    }

    pub(crate) async fn select_session(
        &self,
        choices: Vec<SessionChoice>,
    ) -> io::Result<Option<uuid::Uuid>> {
        if choices.is_empty() {
            return Ok(None);
        }
        let _focus_guard = self.router.acquire_focus(InputFocus::Picker);
        {
            let mut state = self.renderer.state.lock().unwrap();
            state.input_active = false;
            state.session_picker = Some(SessionPicker {
                choices,
                selected: 0,
            });
        }
        self.renderer.draw_picker()?;

        let selected = loop {
            let Some(event) = self.router.next_for(InputFocus::Picker).await else {
                break None;
            };
            match event {
                Event::Key(key)
                    if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) =>
                {
                    let mut state = self.renderer.state.lock().unwrap();
                    let Some(picker) = state.session_picker.as_mut() else {
                        break None;
                    };
                    match key.code {
                        KeyCode::Up => {
                            picker.selected = if picker.selected == 0 {
                                picker.choices.len() - 1
                            } else {
                                picker.selected - 1
                            };
                        }
                        KeyCode::Down => {
                            picker.selected = (picker.selected + 1) % picker.choices.len();
                        }
                        KeyCode::Enter => break Some(picker.choices[picker.selected].id),
                        KeyCode::Esc => break None,
                        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            break None;
                        }
                        _ => {}
                    }
                    drop(state);
                    self.renderer.draw_picker()?;
                }
                Event::Resize(_, _) => self.renderer.draw_picker()?,
                _ => {}
            }
        };

        {
            let mut state = self.renderer.state.lock().unwrap();
            state.session_picker = None;
            state.input_active = true;
        }
        self.renderer.restore_viewport()?;
        self.renderer.draw_dynamic()?;
        Ok(selected)
    }

    pub(crate) async fn select_model(&self) -> io::Result<Option<String>> {
        let _focus_guard = self.router.acquire_focus(InputFocus::Picker);
        {
            let mut state = self.renderer.state.lock().unwrap();
            let choices = model_options(&state.provider, &state.model);
            let selected = choices
                .iter()
                .position(|choice| model_matches(&choice.id, &state.model))
                .unwrap_or(0);
            state.input_active = false;
            state.model_picker = Some(ModelPicker { choices, selected });
        }
        self.renderer.draw_picker()?;

        let selected = loop {
            let Some(event) = self.router.next_for(InputFocus::Picker).await else {
                break None;
            };
            match event {
                Event::Key(key)
                    if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) =>
                {
                    let mut state = self.renderer.state.lock().unwrap();
                    let Some(picker) = state.model_picker.as_mut() else {
                        break None;
                    };
                    match key.code {
                        KeyCode::Up | KeyCode::Char('k') => {
                            picker.selected = picker
                                .selected
                                .checked_sub(1)
                                .unwrap_or(picker.choices.len() - 1);
                        }
                        KeyCode::Down | KeyCode::Char('j') => {
                            picker.selected = (picker.selected + 1) % picker.choices.len();
                        }
                        KeyCode::Enter => break Some(picker.choices[picker.selected].id.clone()),
                        KeyCode::Esc => break None,
                        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            break None;
                        }
                        _ => {}
                    }
                    drop(state);
                    self.renderer.draw_picker()?;
                }
                Event::Resize(_, _) => self.renderer.draw_picker()?,
                _ => {}
            }
        };

        {
            let mut state = self.renderer.state.lock().unwrap();
            state.model_picker = None;
            state.input_active = true;
        }
        self.renderer.restore_viewport()?;
        self.renderer.draw_dynamic()?;
        Ok(selected)
    }

    pub(crate) async fn review_plan(&self, plan: &Plan) -> io::Result<Option<PlanReviewChoice>> {
        let _focus_guard = self.router.acquire_focus(InputFocus::Picker);
        {
            let mut state = self.renderer.state.lock().unwrap();
            state.input_active = false;
            state.plan_review = Some(PlanReviewView {
                plan: plan.clone(),
                selected: 2,
                scroll: 0,
                mode: PlanDialogMode::Review,
            });
        }
        self.renderer.draw_picker()?;

        let decision = loop {
            let Some(event) = self.router.next_for(InputFocus::Picker).await else {
                break None;
            };
            match event {
                Event::Key(key)
                    if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) =>
                {
                    let mut state = self.renderer.state.lock().unwrap();
                    let Some(review) = state.plan_review.as_mut() else {
                        break None;
                    };
                    match key.code {
                        KeyCode::Up | KeyCode::Char('k') => {
                            review.selected = if review.selected == 0 {
                                2
                            } else {
                                review.selected - 1
                            };
                        }
                        KeyCode::Down | KeyCode::Char('j') => {
                            review.selected = (review.selected + 1) % 3;
                        }
                        KeyCode::PageUp => review.scroll = review.scroll.saturating_sub(5),
                        KeyCode::PageDown => review.scroll = review.scroll.saturating_add(5),
                        KeyCode::Enter => {
                            break Some(match review.selected {
                                0 => PlanReviewChoice::Approve,
                                1 => PlanReviewChoice::RequestChanges,
                                _ => PlanReviewChoice::Reject,
                            });
                        }
                        KeyCode::Esc => break None,
                        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            break None;
                        }
                        _ => {}
                    }
                    drop(state);
                    self.renderer.draw_picker()?;
                }
                Event::Resize(_, _) => self.renderer.draw_picker()?,
                _ => {}
            }
        };

        {
            let mut state = self.renderer.state.lock().unwrap();
            state.plan_review = None;
            state.input_active = true;
        }
        self.renderer.restore_viewport()?;
        self.renderer.draw_dynamic()?;
        Ok(decision)
    }

    pub(crate) async fn recover_plan(&self, plan: &Plan) -> io::Result<Option<PlanRecoveryChoice>> {
        let _focus_guard = self.router.acquire_focus(InputFocus::Picker);
        {
            let mut state = self.renderer.state.lock().unwrap();
            state.input_active = false;
            state.plan_review = Some(PlanReviewView {
                plan: plan.clone(),
                selected: 2,
                scroll: 0,
                mode: PlanDialogMode::Recovery,
            });
        }
        self.renderer.draw_picker()?;

        let decision = loop {
            let Some(event) = self.router.next_for(InputFocus::Picker).await else {
                break None;
            };
            match event {
                Event::Key(key)
                    if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) =>
                {
                    let mut state = self.renderer.state.lock().unwrap();
                    let Some(view) = state.plan_review.as_mut() else {
                        break None;
                    };
                    match key.code {
                        KeyCode::Up | KeyCode::Char('k') => {
                            view.selected = if view.selected == 0 {
                                3
                            } else {
                                view.selected - 1
                            };
                        }
                        KeyCode::Down | KeyCode::Char('j') => {
                            view.selected = (view.selected + 1) % 4;
                        }
                        KeyCode::PageUp => view.scroll = view.scroll.saturating_sub(5),
                        KeyCode::PageDown => view.scroll = view.scroll.saturating_add(5),
                        KeyCode::Enter => {
                            break Some(match view.selected {
                                0 => PlanRecoveryChoice::Continue,
                                1 => PlanRecoveryChoice::Retry,
                                2 => PlanRecoveryChoice::ViewDetails,
                                _ => PlanRecoveryChoice::Cancel,
                            });
                        }
                        KeyCode::Esc => break None,
                        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            break None;
                        }
                        _ => {}
                    }
                    drop(state);
                    self.renderer.draw_picker()?;
                }
                Event::Resize(_, _) => self.renderer.draw_picker()?,
                _ => {}
            }
        };

        {
            let mut state = self.renderer.state.lock().unwrap();
            state.plan_review = None;
            state.input_active = true;
        }
        self.renderer.restore_viewport()?;
        self.renderer.draw_dynamic()?;
        Ok(decision)
    }
}

impl TuiRenderer {
    /// 审批菜单打开时暂停绘制与提交，避免覆盖菜单。
    fn drawing_allowed(&self) -> bool {
        self.router.focus() != InputFocus::Approval
    }

    /// 把一块已完成的对话文本包装后提交到终端滚动区，再重绘活动区。
    fn commit_block(&self, role: Role, text: &str) -> io::Result<()> {
        if !self.drawing_allowed() {
            return Ok(());
        }
        let (columns, _) = size().unwrap_or((80, 24));
        let body_width = usize::from(columns.saturating_sub(4)).max(10);
        let mut lines = Vec::new();
        if role == Role::Assistant {
            append_markdown_wrapped(&mut lines, text, body_width);
        } else if role == Role::User {
            // 用户消息上下各留一行空行：与前后回答拉开间距，问题更突出。
            lines.push((Role::System, String::new()));
            append_wrapped(&mut lines, role, text, body_width);
            lines.push((Role::System, String::new()));
        } else {
            append_wrapped(&mut lines, role, text, body_width);
        }
        self.commit_wrapped(&lines)
    }

    /// 把已包装的行打印进内容滚动区。
    ///
    /// 内容只在滚动区域 `[0, scroll_bottom]` 内流式打印，超出后终端
    /// 自然向上滚动进入滚动缓冲区；底部活动区位于滚动区域之外，
    /// 不会被内容覆盖，也不会反过来覆盖已提交的内容。
    /// 把缓冲中的「过程」行折叠后追加进输出：超过上限保留前段与末段，
    /// 中间以提示行代替；未超限原样追加。
    fn append_folded(out: &mut Vec<(Role, String)>, process: &mut Vec<(Role, String)>) {
        if process.is_empty() {
            return;
        }
        if process.len() > MAX_COMMIT_LINES {
            out.extend(process.iter().take(MAX_COMMIT_HEAD).cloned());
            out.push((
                Role::System,
                format!(
                    "… 已折叠 {} 行，完整内容保留在会话记录中。",
                    process.len() - MAX_COMMIT_HEAD - MAX_COMMIT_TAIL
                ),
            ));
            out.extend(
                process
                    .iter()
                    .skip(process.len().saturating_sub(MAX_COMMIT_TAIL))
                    .cloned(),
            );
        } else {
            out.extend(process.iter().cloned());
        }
        process.clear();
    }

    fn commit_wrapped(&self, lines: &[(Role, String)]) -> io::Result<()> {
        if !self.drawing_allowed() {
            return Ok(());
        }
        let (columns, rows) = size().unwrap_or((80, 24));
        // 折叠只作用于「过程」行（工具/思考/系统/警告等）：超过上限折叠为
        // 「前段 + 提示 + 末段」，避免过程平铺刷屏；助手正式输出（最终答案、
        // 代码等）不折叠，直接完整输出。
        let mut folded: Vec<(Role, String)> = Vec::with_capacity(lines.len());
        let mut process: Vec<(Role, String)> = Vec::new();
        for (role, line) in lines {
            if matches!(
                role,
                Role::Assistant
                    | Role::AssistantHeading
                    | Role::AssistantCode
                    | Role::AssistantDiffAdd
                    | Role::AssistantDiffRemove
            ) {
                Self::append_folded(&mut folded, &mut process);
                folded.push((*role, line.clone()));
            } else {
                process.push((*role, line.clone()));
            }
        }
        Self::append_folded(&mut folded, &mut process);
        let mut stdout = io::stdout();
        {
            let mut state = self.state.lock().unwrap();
            state.intro_active = false;
            let scroll_bottom = scroll_bottom_of(&state, columns, rows);
            // Codex 的 history insertion：先在活动区上沿设置历史滚动区域，
            // 再从底行用换行把旧内容推入终端原生 scrollback，最后写入新行。
            // 这比“写完一行再换行”更可靠，尤其是 Terminal.app/iTerm 的复制和回滚。
            queue_scroll_region(&mut stdout, scroll_bottom)?;
            queue!(stdout, MoveTo(0, scroll_bottom))?;
            let printed = folded.len();
            let mut previous_role: Option<Role> = None;
            for (role, line) in folded {
                queue!(stdout, Print("\r\n"))?;
                if !line.is_empty() {
                    let first_line = previous_role != Some(role);
                    print_role_line_content(
                        &mut stdout,
                        role,
                        &line,
                        columns,
                        self.color,
                        first_line,
                    )?;
                    previous_role = Some(role);
                } else {
                    previous_role = None;
                }
            }
            state.printed_lines = state.printed_lines.saturating_add(printed);
            state.content_end_row = scroll_bottom;
            state.dynamic_top = 0;
        }
        stdout.flush()?;
        self.draw_dynamic()
    }

    /// 把待提交块包装后一次性写入滚动区（emit 内部使用）。
    fn commit_blocks(&self, blocks: &[(Role, String)]) -> io::Result<()> {
        if !self.drawing_allowed() {
            return Ok(());
        }
        let (columns, _) = size().unwrap_or((80, 24));
        let body_width = usize::from(columns.saturating_sub(4)).max(10);
        let mut lines = Vec::new();
        for (role, text) in blocks {
            if *role == Role::Assistant {
                append_markdown_wrapped(&mut lines, text, body_width);
            } else {
                append_wrapped(&mut lines, *role, text, body_width);
            }
        }
        self.commit_wrapped(&lines)
    }

    /// 重绘底部活动区：流式尾巴、工具活动、命令建议、状态线、Composer。
    ///
    /// 与 Codex 的 inline renderer 一样，已提交内容不再参与重绘；这里只
    /// 生成底部活动区帧并交给差量缓冲，只写入发生变化的行。
    pub(crate) fn draw_dynamic(&self) -> io::Result<()> {
        if !self.drawing_allowed() {
            return Ok(());
        }
        let mut state = self.state.lock().unwrap();
        let (columns, rows) = size().unwrap_or((80, 24));
        if rows < 8 {
            return Ok(());
        }
        let mut stdout = io::stdout();
        if state.intro_active {
            draw_intro(&mut stdout, &mut state, columns, rows, self.color)?;
        }
        let body_width = usize::from(columns.saturating_sub(4)).max(10);
        let input_width = usize::from(columns.saturating_sub(3)).max(10);
        let scroll_bottom = scroll_bottom_of(&state, columns, rows);
        let dynamic_top = scroll_bottom.saturating_add(1);
        let input_row = rows.saturating_sub(3);
        let input_text: String = state.input.iter().collect();
        let input_lines = wrap_input(&input_text, input_width);
        let input_vis = input_lines.len().min(3);
        let folded = input_lines.len().saturating_sub(input_vis);
        let input_block_top = input_block_top_of(input_row, input_vis);
        let separator_row = input_block_top.saturating_sub(1);
        let composer_bottom = input_row.saturating_add(1);
        let help_row = rows.saturating_sub(1);
        let mut frame = vec![String::new(); usize::from(rows.saturating_sub(dynamic_top))];
        let mut put = |row: u16, value: String| {
            if row >= dynamic_top && row < rows {
                frame[usize::from(row - dynamic_top)] = value;
            }
        };
        let mut above = separator_row.saturating_sub(1);
        if folded > 0 && above >= dynamic_top {
            put(
                above,
                styled_fragment(
                    self.color,
                    palette().muted,
                    false,
                    &format!("… 已折叠 {folded} 行，Enter 发送全部 …"),
                ),
            );
            above = above.saturating_sub(1);
        }
        let mut space = above.saturating_sub(dynamic_top);
        let status_reserved = if state.status != "就绪" { 1u16 } else { 0u16 };
        space = space.saturating_sub(status_reserved);

        if !state.streaming.is_empty() {
            let mut tail = Vec::new();
            append_wrapped(&mut tail, Role::Assistant, &state.streaming, body_width);
            let n = tail.len().min(STREAMING_WINDOW).min(space as usize);
            if n > 0 {
                let start = tail.len() - n;
                for (offset, (_, line)) in tail.iter().skip(start).take(n).enumerate() {
                    let row = above.saturating_sub(n as u16).saturating_add(1) + offset as u16;
                    put(
                        row,
                        styled_fragment(
                            self.color,
                            palette().muted,
                            false,
                            &truncate_to_width(line, usize::from(columns.saturating_sub(4)).max(1)),
                        ),
                    );
                }
                above = above.saturating_sub(n as u16);
                space = space.saturating_sub(n as u16);
            }
        }

        if !state.tools.is_empty() {
            let n = state.tools.len().min(2).min(space as usize);
            if n > 0 {
                for (offset, tool) in state.tools.iter().take(n).enumerate() {
                    let row = above.saturating_sub(n as u16).saturating_add(1) + offset as u16;
                    let marker = match tool.finished {
                        Some(true) => "✓",
                        Some(false) => "✗",
                        None => "●",
                    };
                    let duration = tool
                        .duration_ms
                        .map(|duration| format!(" · {duration} ms"))
                        .unwrap_or_default();
                    let tone = if tool.finished == Some(false) {
                        palette().warning
                    } else {
                        palette().muted
                    };
                    let text = format!(
                        "{marker} {}  {}{duration}",
                        tool_display_name(&tool.name),
                        tool.detail
                    );
                    put(
                        row,
                        styled_fragment(
                            self.color,
                            tone,
                            false,
                            &truncate_to_width(&text, usize::from(columns.saturating_sub(4))),
                        ),
                    );
                }
                above = above.saturating_sub(n as u16);
                space = space.saturating_sub(n as u16);
            }
        }

        if let Some(fold) = &state.tool_fold {
            if space > 0 {
                let line = if fold.count > 1 {
                    format!("- {} ×{}", fold.head, fold.count)
                } else {
                    format!("- {}", fold.head)
                };
                put(
                    above.saturating_sub(1),
                    styled_fragment(
                        self.color,
                        palette().muted,
                        false,
                        &truncate_to_width(&line, usize::from(columns.saturating_sub(4))),
                    ),
                );
                above = above.saturating_sub(1);
                space = space.saturating_sub(1);
            }
        }

        let suggestions = matching_commands(&state.input);
        if !suggestions.is_empty() {
            let n = suggestions
                .len()
                .min(MAX_COMMAND_SUGGESTIONS)
                .min(space as usize);
            if n > 0 {
                let usage_width = suggestions
                    .iter()
                    .take(n)
                    .map(|command| command.usage.width())
                    .max()
                    .unwrap_or(0);
                for (index, command) in suggestions.into_iter().take(n).enumerate() {
                    let row = above.saturating_sub(n as u16).saturating_add(1) + index as u16;
                    let selected = index == state.command_selection.min(n.saturating_sub(1));
                    let line = command_suggestion_line(
                        if selected { "›" } else { " " },
                        command.usage,
                        command.description,
                        usage_width,
                    );
                    put(
                        row,
                        styled_fragment(
                            self.color,
                            if selected {
                                palette().primary
                            } else {
                                palette().muted
                            },
                            selected,
                            &truncate_to_width(&line, usize::from(columns.saturating_sub(4))),
                        ),
                    );
                }
            }
        }

        if status_reserved == 1 && above > dynamic_top {
            let row = above.saturating_sub(1);
            let tone = if state.status.contains("恢复") {
                palette().recovery
            } else if state.status.contains("失败") || state.status.contains("暂停") {
                palette().warning
            } else {
                palette().primary
            };
            put(
                row,
                styled_fragment(self.color, tone, false, &format!("• {}", state.status)),
            );
        }

        for (offset, line) in input_lines
            .iter()
            .skip(input_lines.len().saturating_sub(input_vis))
            .take(input_vis)
            .enumerate()
        {
            let row = input_block_top + offset as u16;
            let prefix = if offset + 1 == input_vis {
                "❯ "
            } else {
                "  "
            };
            let text = format!(
                "{}{}",
                styled_fragment(self.color, palette().primary, true, prefix),
                styled_fragment(
                    self.color,
                    palette().text,
                    false,
                    &truncate_to_width(line, usize::from(columns.saturating_sub(4)).max(1))
                )
            );
            put(row, text);
        }
        let horizontal = "─".repeat(usize::from(columns.saturating_sub(2)));
        put(
            separator_row,
            styled_fragment(
                self.color,
                palette().border,
                false,
                &format!("╭{}╮", horizontal),
            ),
        );
        put(
            composer_bottom,
            styled_fragment(
                self.color,
                palette().border,
                false,
                &format!("╰{}╯", horizontal),
            ),
        );

        let running = !state.tools.is_empty() || !state.streaming.is_empty();
        let hint_text = if running {
            "Ctrl+C / Esc 打断 · Enter 排队 · 可继续输入".to_owned()
        } else if state.vim_enabled {
            if state.vim_normal {
                "Vim 普通模式 · i/a/A/I 输入 · j/k 历史 · h/l 移动 · Enter 发送".to_owned()
            } else {
                "Vim 插入模式 · Esc 返回 · Enter 发送".to_owned()
            }
        } else if matching_commands(&state.input).is_empty() {
            format!(
                "⏵ {} mode on · Shift+Tab 切换 · {}",
                state.permission,
                state
                    .input_hint
                    .split_once(" · / 命令")
                    .map_or(state.input_hint.as_str(), |(prefix, _)| prefix)
            )
        } else {
            "↑↓ 选择 · Tab 补全 · Enter 确定".to_owned()
        };
        put(
            help_row,
            styled_fragment(
                self.color,
                palette().muted,
                false,
                &format!(
                    "   {}",
                    truncate_to_width(&hint_text, usize::from(columns.saturating_sub(4)))
                ),
            ),
        );

        queue_scroll_region(&mut stdout, scroll_bottom)?;
        {
            let mut activity = self.activity.lock().unwrap();
            activity.render(&mut stdout, dynamic_top, &frame, columns)?;
        }
        if state.input_active {
            let (cursor_line, cursor_col) =
                input_cursor_position(&state.input, state.cursor, input_width);
            let visible_offset = input_lines.len().saturating_sub(input_vis);
            let cursor_row = if cursor_line >= visible_offset {
                input_row.saturating_sub(
                    (input_vis.saturating_sub(1 + cursor_line - visible_offset)) as u16,
                )
            } else {
                input_block_top
            };
            queue!(
                stdout,
                MoveTo(
                    (3 + cursor_col).min(usize::from(columns).saturating_sub(1)) as u16,
                    cursor_row
                ),
                Show
            )?;
        } else {
            queue!(stdout, Hide)?;
        }
        state.dynamic_top = dynamic_top;
        stdout.flush()
    }

    /// 输入框为空按 Tab：切换最近一个折叠组的展开/收起并重绘视口。
    pub(crate) fn toggle_last_group(&self) -> io::Result<()> {
        {
            let mut state = self.state.lock().unwrap();
            if let Some(block) = state
                .transcript
                .iter_mut()
                .rev()
                .find(|block| block.group_entries.is_some())
            {
                block.collapsed = !block.collapsed;
            }
        }
        self.restore_viewport()
    }

    /// 模态界面（选择器/计划审阅）关闭后，把已提交的可见内容重绘回视口。
    pub(crate) fn restore_viewport(&self) -> io::Result<()> {
        if !self.drawing_allowed() {
            return Ok(());
        }
        let (columns, rows) = size().unwrap_or((80, 24));
        let body_width = usize::from(columns.saturating_sub(4)).max(10);
        let (viewport, lines) = {
            let state = self.state.lock().unwrap();
            let dynamic_height = layout_height(&state, columns, rows).min(rows.saturating_sub(2));
            let viewport = rows.saturating_sub(dynamic_height) as usize;
            let mut tail = VecDeque::new();
            for block in state.transcript.iter().rev() {
                let mut block_lines = Vec::new();
                if let Some(entries) = &block.group_entries {
                    // 折叠组：折叠态只显示组头；展开态追加每个工具明细。
                    append_wrapped(&mut block_lines, block.role, &block.text, body_width);
                    if !block.collapsed {
                        for (role, text) in entries {
                            append_wrapped(&mut block_lines, *role, text, body_width);
                        }
                    }
                } else if block.role == Role::Assistant {
                    append_markdown_wrapped(&mut block_lines, &block.text, body_width);
                } else {
                    append_wrapped(&mut block_lines, block.role, &block.text, body_width);
                }
                block_lines.push((Role::System, String::new()));
                for line in block_lines.into_iter().rev() {
                    tail.push_front(line);
                }
                if tail.len() >= viewport {
                    break;
                }
            }
            while tail.len() > viewport {
                tail.pop_front();
            }
            (viewport, tail)
        };
        let mut stdout = io::stdout();
        self.activity.lock().unwrap().reset(&mut stdout)?;
        for row in 0..viewport {
            queue!(stdout, MoveTo(0, row as u16), Clear(ClearType::CurrentLine))?;
        }
        let mut previous_role: Option<Role> = None;
        for (offset, (role, line)) in lines.iter().enumerate() {
            let row = offset as u16;
            if line.is_empty() {
                previous_role = None;
                continue;
            }
            queue!(stdout, MoveTo(0, row))?;
            let first_line = previous_role != Some(*role);
            print_role_line(&mut stdout, *role, line, columns, self.color, first_line)?;
            previous_role = Some(*role);
        }
        {
            let mut state = self.state.lock().unwrap();
            state.printed_lines = viewport;
            state.dynamic_top = viewport as u16;
        }
        stdout.flush()
    }

    /// 启动横幅：会话尚无内容时随终端尺寸变化重新布局。
    fn print_intro(&self) -> io::Result<()> {
        let (columns, rows) = size().unwrap_or((80, 24));
        let mut state = self.state.lock().unwrap();
        let mut stdout = io::stdout();
        draw_intro(&mut stdout, &mut state, columns, rows, self.color)?;
        stdout.flush()
    }

    /// 选择器/计划审阅的全区域覆盖绘制（随后由 [`Self::restore_viewport`] 恢复）。
    fn draw_picker(&self) -> io::Result<()> {
        let state = self.state.lock().unwrap();
        let (columns, rows) = size().unwrap_or((80, 24));
        let mut stdout = io::stdout();
        if state.session_picker.is_some() {
            draw_session_picker(&mut stdout, &state, columns, rows, self.color)?;
        }
        if state.plan_review.is_some() {
            draw_plan_review(&mut stdout, &state, columns, rows, self.color)?;
        }
        if state.model_picker.is_some() {
            draw_model_picker(&mut stdout, &state, columns, rows, self.color)?;
        }
        Ok(())
    }
}

#[async_trait]
impl EventSink for TuiRenderer {
    async fn emit(&self, event: AgentEvent) {
        let mut commits: Vec<(Role, String)> = Vec::new();
        {
            let mut state = self.state.lock().unwrap();
            if state.debug_trace
                && let Some(trace) = event.structured_trace()
                && let Ok(line) = serde_json::to_string(&redact_value(&trace))
            {
                push_block(&mut state, Role::System, format!("trace {line}"));
            }
            // 遇到工具完成以外的事件先提交缓冲中的同类工具折叠，
            // 保证 ×N 计数与后续块的时序正确。
            if !matches!(event, AgentEvent::ToolFinished { .. }) {
                flush_pending_tool_fold(&mut state, &mut commits);
            }
            match event {
                AgentEvent::StateChanged { state: agent_state } => {
                    state.status = state_label(agent_state).into();
                }
                AgentEvent::AssistantDelta { text } => {
                    state.assistant_received_delta = true;
                    // 延迟提交：叙述/回答先在底部浅灰模块实时滚动，遇工具调用
                    // 降为思考块提交；任务结束时剩余部分作为正式回答提交。
                    state.streaming.push_str(&redact_text(&text));
                }
                AgentEvent::ToolStarted { call_id, name } => {
                    // 工具调用前的叙述视为思考：以浅灰块提交，与正式回答区分。
                    let thought = std::mem::take(&mut state.streaming);
                    let thought = thought.trim();
                    state.code_fence_open = false;
                    if !thought.is_empty() {
                        let thought = thought.to_owned();
                        push_block(&mut state, Role::Thinking, thought.clone());
                        commits.push((Role::Thinking, thought));
                    }
                    state.tools.push(ToolActivity {
                        call_id,
                        name,
                        detail: "启动".into(),
                        finished: None,
                        duration_ms: None,
                    });
                }
                AgentEvent::ToolProgress {
                    call_id,
                    phase,
                    completed,
                    total,
                    unit,
                    message,
                    ..
                } => {
                    if let Some(tool) = state.tools.iter_mut().find(|tool| tool.call_id == call_id)
                    {
                        let count = match (completed, total, unit) {
                            (Some(done), Some(total), Some(unit)) => {
                                format!(" · {done}/{total} {unit}")
                            }
                            (Some(done), None, Some(unit)) => format!(" · {done} {unit}"),
                            _ => String::new(),
                        };
                        let detail = message
                            .map(|value| redact_text(&value))
                            .filter(|value| !value.trim().is_empty())
                            .unwrap_or_else(|| tool_phase_display(&phase).into());
                        tool.detail = format!("{detail}{count}");
                    }
                }
                AgentEvent::ToolFinished {
                    call_id, result, ..
                } => {
                    if let Some(index) = state.tools.iter().position(|tool| tool.call_id == call_id)
                    {
                        let tool = state.tools.remove(index);
                        let name = tool.name.clone();
                        let parts = completed_tool_parts(&tool, &result);
                        // 同类（工具名+入参摘要）连续完成合并为 ×N 折叠，延迟到
                        // 下一个异类事件才提交，避免历史区刷出大量重复两行。
                        merge_pending_tool_fold(&mut state, &mut commits, &name, &parts);
                        if let Some(fold) = &state.tool_fold {
                            state.status = if fold.count > 1 {
                                format!("{} ×{}", tool_display_name(&name), fold.count)
                            } else {
                                tool_display_name(&name).to_owned()
                            };
                        }
                    }
                }
                AgentEvent::UsageUpdated { usage } => {
                    state.usage = Some((usage.input_tokens, usage.output_tokens));
                }
                AgentEvent::Warning { message, .. } => {
                    let line = redact_text(&message);
                    push_block(&mut state, Role::Warning, line.clone());
                    commits.push((Role::Warning, line));
                }
                AgentEvent::DebugTrace { .. } => {}
                AgentEvent::StalledRecovery { recovery, .. } => {
                    state.status = "检测到停滞，尝试恢复".into();
                    // recovery 已自带完整纠偏文案（含失败次数与错误详情），
                    // 直接追加在角色前缀 ↻ 之后，避免双重符号与套话包裹。
                    let line = redact_text(&format!("检测到停滞，任务尝试自动恢复：{recovery}"));
                    push_block(&mut state, Role::Recovery, line.clone());
                    commits.push((Role::Recovery, line));
                }
                AgentEvent::CompactionApplied {
                    layer,
                    saved_tokens,
                } => {
                    state.status = "上下文已压缩".into();
                    let line = format!("↻ 上下文压缩（{layer}）：节省约 {saved_tokens} tokens。");
                    push_block(&mut state, Role::System, line.clone());
                    commits.push((Role::System, line));
                }
                AgentEvent::Continuing { note, .. } => {
                    state.status = "自动续跑".into();
                    let line = format!("↻ {}", redact_text(&note));
                    push_block(&mut state, Role::Recovery, line.clone());
                    commits.push((Role::Recovery, line));
                }
                AgentEvent::ReasoningDelta { .. } => {}
                AgentEvent::PlanStarted { .. } => state.status = "执行计划".into(),
                AgentEvent::PlanStepStarted { title, attempt, .. } => {
                    state.status = format!("步骤 {title} · 第 {attempt} 次");
                }
                AgentEvent::PlanStepCompleted {
                    summary, evidence, ..
                } => {
                    let line = format!("步骤完成：{}", redact_text(&summary));
                    push_block(&mut state, Role::System, line.clone());
                    commits.push((Role::System, line));
                    for item in evidence {
                        let line = format!(
                            "证据 {}：{}",
                            item.criterion_index,
                            redact_text(&item.evidence)
                        );
                        push_block(&mut state, Role::System, line.clone());
                        commits.push((Role::System, line));
                    }
                }
                AgentEvent::PlanStepFailed { error, .. } => {
                    let line = format!("步骤失败：{}", redact_text(&error));
                    push_block(&mut state, Role::Warning, line.clone());
                    commits.push((Role::Warning, line));
                }
                AgentEvent::PlanPaused { reason, .. } => {
                    state.status = "计划暂停".into();
                    let line = redact_text(&reason);
                    push_block(&mut state, Role::Warning, line.clone());
                    commits.push((Role::Warning, line));
                }
                AgentEvent::PlanCompleted { .. } => state.status = "计划完成".into(),
                AgentEvent::SkillLoaded { name } => {
                    let line = format!("技能已加载：{name}");
                    push_block(&mut state, Role::System, line.clone());
                    commits.push((Role::System, line));
                }
                AgentEvent::SubagentStarted { agent_id, prompt } => {
                    let line = format!("子代理 {agent_id} 已启动：{}", redact_text(&prompt));
                    push_block(&mut state, Role::System, line.clone());
                    commits.push((Role::System, line));
                }
                AgentEvent::SubagentFinished { agent_id, result } => {
                    let line = if result.success {
                        format!("子代理 {agent_id} 完成")
                    } else {
                        let message = result
                            .error
                            .as_ref()
                            .map(|error| error.message.clone())
                            .unwrap_or_default();
                        format!("子代理 {agent_id} 未完成：{}", redact_text(&message))
                    };
                    push_block(&mut state, Role::Tool, line.clone());
                    commits.push((Role::Tool, line));
                }
                AgentEvent::SubagentGraphStarted {
                    total,
                    max_concurrency,
                    ..
                } => {
                    state.status = format!("任务图 · {total} 节点 · 并发 {max_concurrency}");
                }
                AgentEvent::SubagentGraphNodeStarted {
                    node_id, agent_id, ..
                } => {
                    state.status = format!("任务图节点 {node_id} · {agent_id}");
                }
                AgentEvent::SubagentGraphNodeFinished {
                    node_id,
                    status,
                    duration_ms,
                    ..
                } => {
                    let line = format!("任务图节点 {node_id}：{status} · {duration_ms} ms");
                    let role = if status == "succeeded" {
                        Role::Tool
                    } else {
                        Role::Warning
                    };
                    push_block(&mut state, role, line.clone());
                    commits.push((role, line));
                }
                AgentEvent::SubagentGraphFinished {
                    success,
                    succeeded,
                    failed,
                    blocked,
                    cancelled,
                    ..
                } => {
                    state.status = if success {
                        "任务图完成".into()
                    } else {
                        "任务图未全部完成".into()
                    };
                    let line = format!(
                        "任务图汇总：成功 {succeeded} · 失败 {failed} · 阻塞 {blocked} · 取消 {cancelled}"
                    );
                    let role = if success { Role::Tool } else { Role::Warning };
                    push_block(&mut state, role, line.clone());
                    commits.push((role, line));
                }
            }
        }
        if self.router.focus() == InputFocus::Composer {
            if !commits.is_empty() {
                let _ = self.commit_blocks(&commits);
            } else {
                let _ = self.draw_dynamic();
            }
        }
    }
}

/// Vim 普通模式键位：j/k 历史、h/l 移动、0/$ 行首尾、
/// i/a/A/I 进入插入、x/X 删除字符、D 删到行尾、dd 清空输入。
fn handle_vim_normal(state: &mut TuiState, key: KeyEvent) -> Option<InputOutcome> {
    match key.code {
        KeyCode::Char('i') => {
            state.vim_normal = false;
            state.vim_last = None;
            None
        }
        KeyCode::Char('a') => {
            if state.cursor < state.input.len() {
                state.cursor += 1;
            }
            state.vim_normal = false;
            state.vim_last = None;
            None
        }
        KeyCode::Char('A') => {
            state.cursor = state.input.len();
            state.vim_normal = false;
            state.vim_last = None;
            None
        }
        KeyCode::Char('I') => {
            state.cursor = 0;
            state.vim_normal = false;
            state.vim_last = None;
            None
        }
        KeyCode::Char('h') if state.cursor > 0 => {
            state.cursor -= 1;
            state.vim_last = None;
            None
        }
        KeyCode::Char('l') if state.cursor < state.input.len() => {
            state.cursor += 1;
            state.vim_last = None;
            None
        }
        KeyCode::Char('0') => {
            state.cursor = 0;
            state.vim_last = None;
            None
        }
        KeyCode::Char('$') => {
            state.cursor = state.input.len();
            state.vim_last = None;
            None
        }
        KeyCode::Char('k') => {
            // Vim k=上：向更早的历史移动。
            state.vim_last = None;
            if !state.history.is_empty() {
                let index = match state.history_index {
                    None => {
                        state.draft.clone_from(&state.input);
                        state.history.len() - 1
                    }
                    Some(index) => index.saturating_sub(1),
                };
                load_history(state, index);
            }
            None
        }
        KeyCode::Char('j') => {
            // Vim j=下：向更新的历史/草稿移动。
            state.vim_last = None;
            match state.history_index {
                Some(index) if index + 1 < state.history.len() => load_history(state, index + 1),
                Some(_) => {
                    state.input.clone_from(&state.draft);
                    state.cursor = state.input.len();
                    state.history_index = None;
                }
                None => {}
            }
            None
        }
        KeyCode::Char('x') => {
            if state.cursor < state.input.len() {
                state.input.remove(state.cursor);
            }
            state.vim_last = None;
            None
        }
        KeyCode::Char('X') if state.cursor > 0 => {
            state.cursor -= 1;
            state.input.remove(state.cursor);
            state.vim_last = None;
            None
        }
        KeyCode::Char('D') => {
            state.input.truncate(state.cursor);
            state.vim_last = None;
            None
        }
        KeyCode::Char('d') => {
            if state.vim_last == Some('d') {
                state.input.clear();
                state.cursor = 0;
                state.vim_last = None;
            } else {
                state.vim_last = Some('d');
            }
            None
        }
        KeyCode::Esc => {
            state.vim_last = None;
            // 普通模式 Esc 无操作。
            None
        }
        KeyCode::Enter => {
            state.vim_last = None;
            handle_input_key_regular(state, key)
        }
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            state.vim_last = None;
            handle_input_key_regular(state, key)
        }
        _ => {
            state.vim_last = None;
            None
        }
    }
}

fn handle_input_key(state: &mut TuiState, key: KeyEvent) -> Option<InputOutcome> {
    if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
        return None;
    }
    if state.vim_enabled && state.vim_normal {
        return handle_vim_normal(state, key);
    }
    handle_input_key_regular(state, key)
}

fn handle_input_key_regular(state: &mut TuiState, key: KeyEvent) -> Option<InputOutcome> {
    match key.code {
        KeyCode::Enter
            if key
                .modifiers
                .intersects(KeyModifiers::SHIFT | KeyModifiers::ALT) =>
        {
            state.input.insert(state.cursor, '\n');
            state.cursor += 1;
            state.history_index = None;
            state.command_selection = 0;
            None
        }
        KeyCode::Char('j') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            state.input.insert(state.cursor, '\n');
            state.cursor += 1;
            state.history_index = None;
            state.command_selection = 0;
            None
        }
        KeyCode::Enter => {
            let mut line = state.input.iter().collect::<String>();
            if line.trim().is_empty() {
                return None;
            }
            let suggestions = matching_commands(&state.input);
            if let Some(command) = suggestions.get(state.command_selection) {
                if command.requires_argument {
                    replace_input(state, &format!("{} ", command.name));
                    return None;
                }
                line = command.name.to_owned();
            }
            if state.history.last() != Some(&line) {
                state.history.push(line.clone());
            }
            state.input.clear();
            state.cursor = 0;
            state.history_index = None;
            state.command_selection = 0;
            if line.trim_start().starts_with('/') {
                Some(InputOutcome::Command(line))
            } else {
                Some(InputOutcome::Submit(line))
            }
        }
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            if state.input.is_empty() {
                Some(InputOutcome::Interrupted)
            } else {
                state.input.clear();
                state.cursor = 0;
                state.history_index = None;
                state.command_selection = 0;
                None
            }
        }
        KeyCode::Char('d')
            if key.modifiers.contains(KeyModifiers::CONTROL) && state.input.is_empty() =>
        {
            Some(InputOutcome::Exit)
        }
        KeyCode::Esc => {
            if state.vim_enabled {
                // Vim 插入模式：Esc 返回普通模式，不清空输入。
                state.vim_normal = true;
                state.vim_last = None;
            } else if !state.input.is_empty() {
                // 常规模式：Esc 清空当前输入。
                state.input.clear();
                state.cursor = 0;
                state.history_index = None;
                state.command_selection = 0;
            }
            None
        }
        KeyCode::Char('a') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            state.cursor = 0;
            None
        }
        KeyCode::Char('e') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            state.cursor = state.input.len();
            None
        }
        KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            state.input.drain(..state.cursor);
            state.cursor = 0;
            None
        }
        KeyCode::Char('k') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            state.input.truncate(state.cursor);
            None
        }
        KeyCode::Char('w') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            delete_previous_word(state);
            None
        }
        KeyCode::Char(character)
            if !key
                .modifiers
                .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
        {
            state.input.insert(state.cursor, character);
            state.cursor += 1;
            state.history_index = None;
            state.command_selection = 0;
            None
        }
        KeyCode::Tab if state.input.is_empty() => {
            // 输入框为空：Tab 切换最近折叠组展开/收起（Claude Code 同款交互）。
            Some(InputOutcome::ToggleFold)
        }
        KeyCode::Tab => {
            let suggestions = matching_commands(&state.input);
            if let Some(command) = suggestions.get(state.command_selection) {
                let suffix = if command.requires_argument { " " } else { "" };
                replace_input(state, &format!("{}{suffix}", command.name));
            }
            None
        }
        KeyCode::Left if state.cursor > 0 => {
            state.cursor -= 1;
            None
        }
        KeyCode::Right if state.cursor < state.input.len() => {
            state.cursor += 1;
            None
        }
        KeyCode::Home => {
            state.cursor = 0;
            None
        }
        KeyCode::End => {
            state.cursor = state.input.len();
            None
        }
        KeyCode::Backspace if state.cursor > 0 => {
            state.cursor -= 1;
            state.input.remove(state.cursor);
            state.history_index = None;
            state.command_selection = 0;
            None
        }
        KeyCode::Delete if state.cursor < state.input.len() => {
            state.input.remove(state.cursor);
            state.history_index = None;
            state.command_selection = 0;
            None
        }
        KeyCode::Up if !matching_commands(&state.input).is_empty() => {
            let count = matching_commands(&state.input).len();
            state.command_selection = (state.command_selection + count.saturating_sub(1)) % count;
            None
        }
        KeyCode::Down if !matching_commands(&state.input).is_empty() => {
            let count = matching_commands(&state.input).len();
            state.command_selection = (state.command_selection + 1) % count;
            None
        }
        KeyCode::Up if !state.history.is_empty() => {
            let index = match state.history_index {
                None => {
                    state.draft.clone_from(&state.input);
                    state.history.len() - 1
                }
                Some(index) => index.saturating_sub(1),
            };
            load_history(state, index);
            None
        }
        KeyCode::Down => {
            match state.history_index {
                Some(index) if index + 1 < state.history.len() => load_history(state, index + 1),
                Some(_) => {
                    state.input.clone_from(&state.draft);
                    state.cursor = state.input.len();
                    state.history_index = None;
                }
                None => {}
            }
            None
        }
        _ => None,
    }
}

fn delete_previous_word(state: &mut TuiState) {
    while state.cursor > 0 && state.input[state.cursor - 1].is_whitespace() {
        state.cursor -= 1;
        state.input.remove(state.cursor);
    }
    while state.cursor > 0 && !state.input[state.cursor - 1].is_whitespace() {
        state.cursor -= 1;
        state.input.remove(state.cursor);
    }
}

fn load_history(state: &mut TuiState, index: usize) {
    state.input = state.history[index].chars().collect();
    state.cursor = state.input.len();
    state.history_index = Some(index);
    state.command_selection = 0;
}

/// 插入整段粘贴内容。返回 true 表示达到输入上限并发生截断。
fn insert_paste_text(state: &mut TuiState, text: &str) -> bool {
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    let remaining = MAX_INPUT_CHARS.saturating_sub(state.input.len());
    let mut inserted: Vec<char> = normalized.chars().take(remaining).collect();
    let inserted_len = inserted.len();
    let truncated = normalized.chars().count() > inserted_len;
    state
        .input
        .splice(state.cursor..state.cursor, inserted.drain(..));
    state.cursor += inserted_len;
    state.history_index = None;
    state.command_selection = 0;
    truncated
}

/// 只取本轮尚未提交的助手尾部。非流式 Provider 才使用完整最终消息。
fn take_final_assistant_tail(state: &mut TuiState, final_message: &str) -> String {
    let mut assistant = std::mem::take(&mut state.streaming);
    state.code_fence_open = false;
    if !state.assistant_received_delta && assistant.trim().is_empty() {
        assistant = final_message.to_owned();
    }
    state.assistant_received_delta = false;
    assistant
}

/// 从流式缓冲中取出可以立即提交的块列表（角色 + 文本）。
///
/// 提交粒度保证长文本实时可见，同时保留可用的 Markdown 样式：
/// - 普通完整行到达即提交，长文本实时流入滚动区（标题、列表、
///   引用等行级语义由渲染器按行转换保留）；
/// - 表格行（以 `|` 开头）做块级缓冲，遇到空行或非表格行时整块
///   提交，保证表格能渲染成终端表格样式；
/// - 代码栅栏打开行提交为普通块（渲染出语言标记），栅栏内逐行以
///   代码样式实时提交，闭合行提交为空代码块。
#[cfg(test)]
fn take_ready_segments(state: &mut TuiState) -> Vec<(Role, String)> {
    let text = std::mem::take(&mut state.streaming);
    let mut segments = Vec::new();
    let mut fence = state.code_fence_open;
    let mut table_buf = String::new();
    let mut consumed = 0usize;
    let mut line_start = 0usize;
    for (index, character) in text.char_indices() {
        if character != '\n' {
            continue;
        }
        let line = &text[line_start..index];
        if fence {
            if is_fence_line(line) {
                // 闭合行：作为空代码块提交，终端上不显示 ``` 本身。
                segments.push((Role::Assistant, "```\n".into()));
                fence = false;
            } else {
                // 栅栏内：逐行以代码样式提交，长代码块实时进入滚动区。
                segments.push((Role::AssistantCode, format!("  {line}\n")));
            }
        } else if is_fence_line(line) {
            flush_table(&mut segments, &mut table_buf);
            // 栅栏打开行：作为普通块提交（未闭合代码块渲染出语言标记）。
            segments.push((Role::Assistant, format!("{line}\n")));
            fence = true;
        } else if is_table_line(line) {
            // 表格候选行：累积到表格结束（空行/非表格行）再整块提交。
            table_buf.push_str(line);
            table_buf.push('\n');
        } else {
            flush_table(&mut segments, &mut table_buf);
            // 普通完整行：立即提交，长文本实时流入滚动区。
            segments.push((Role::Assistant, format!("{line}\n")));
        }
        line_start = index + 1;
        consumed = index + 1;
    }
    state.code_fence_open = fence;
    // 未完成行与未结束的表格缓冲保留回流式缓冲，等待后续增量。
    let mut rest = std::mem::take(&mut table_buf);
    rest.push_str(&text[consumed..]);
    state.streaming = rest;
    segments
}

#[cfg(test)]
fn flush_table(segments: &mut Vec<(Role, String)>, table_buf: &mut String) {
    if !table_buf.is_empty() {
        segments.push((Role::Assistant, std::mem::take(table_buf)));
    }
}

#[cfg(test)]
fn is_table_line(line: &str) -> bool {
    line.trim_start().starts_with('|')
}

#[cfg(test)]
fn is_fence_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("```") || trimmed.starts_with("~~~")
}

fn load_session_state(state: &mut TuiState, session: &Session) {
    state.transcript.clear();
    state.streaming.clear();
    state.code_fence_open = false;
    state.tools.clear();
    state.input.clear();
    state.cursor = 0;
    state.history_index = None;
    state.status = "已恢复会话".into();
    state.usage = Some((session.total_input_tokens, session.total_output_tokens));
    for message in &session.messages {
        let role = match message.role {
            MessageRole::User => Role::User,
            MessageRole::Assistant => Role::Assistant,
            MessageRole::System => Role::System,
            MessageRole::Tool => continue,
        };
        if !message.content.trim().is_empty() {
            push_block(state, role, redact_text(&message.content));
        }
    }
}

fn replace_input(state: &mut TuiState, value: &str) {
    state.input = value.chars().collect();
    state.cursor = state.input.len();
    state.history_index = None;
    state.command_selection = 0;
}

fn matching_commands(input: &[char]) -> Vec<&'static SlashCommand> {
    let value = input.iter().collect::<String>();
    if !value.starts_with('/') {
        return Vec::new();
    }
    if value.chars().any(char::is_whitespace) && !value.starts_with("/plan ") {
        return Vec::new();
    }
    SLASH_COMMANDS
        .iter()
        .filter(|command| command.name.starts_with(&value))
        .take(MAX_COMMAND_SUGGESTIONS)
        .collect()
}

fn push_block(state: &mut TuiState, role: Role, text: String) {
    state.transcript.push_back(TranscriptBlock {
        role,
        text,
        group_entries: None,
        collapsed: false,
    });
}

/// 提交一个折叠组块：组头进历史（默认折叠），条目在展开状态显示。
fn push_group_block(
    state: &mut TuiState,
    role: Role,
    text: String,
    group_entries: Vec<(Role, String)>,
) {
    state.transcript.push_back(TranscriptBlock {
        role,
        text,
        group_entries: Some(group_entries),
        collapsed: true,
    });
}

#[allow(dead_code)]
fn draw_header(
    writer: &mut impl Write,
    state: &TuiState,
    columns: u16,
    color: bool,
) -> io::Result<()> {
    let title = format!(" XDUDU v{} ", state.version);
    let width = usize::from(columns);
    let left = width.saturating_sub(UnicodeWidthStr::width(title.as_str())) / 2;
    let right = width.saturating_sub(left + UnicodeWidthStr::width(title.as_str()));
    set_color(writer, color, palette().border)?;
    queue!(
        writer,
        MoveTo(0, 0),
        Print("─".repeat(left)),
        SetForegroundColor(palette().primary),
        SetAttribute(Attribute::Bold),
        Print(title),
        SetAttribute(Attribute::Reset),
        Print("─".repeat(right))
    )?;
    reset_color(writer, color)?;
    Ok(())
}

/// 工具耗时格式化：毫秒不足 1 秒直接显示，否则保留一位小数秒。
fn format_tool_duration(ms: u64) -> String {
    if ms >= 1_000 {
        format!("{:.1} s", ms as f64 / 1_000.0)
    } else {
        format!("{ms} ms")
    }
}

/// Claude 式工具折叠素材：头行 `名称 入参摘要`，尾行状态/耗时/失败原因由提交时拼装。
fn completed_tool_parts(
    tool: &ToolActivity,
    result: &xdudu_core::tools::ToolResult,
) -> CompletedToolParts {
    let name = tool_display_name(&tool.name);
    if tool.name == "web_search" {
        let query = result
            .output
            .as_ref()
            .and_then(|output| output.get("query"))
            .and_then(serde_json::Value::as_str)
            .or_else(|| {
                result
                    .error
                    .as_ref()
                    .and_then(|error| error.details.get("query"))
                    .and_then(serde_json::Value::as_str)
            })
            .map(redact_text)
            .map(|value| value.replace(['\r', '\n'], " "))
            .unwrap_or_else(|| "未提供查询词".into());
        let success = result.success;
        let duration = result.duration_ms;
        let reason = if success {
            result
                .output
                .as_ref()
                .and_then(|output| output.get("resultCount"))
                .and_then(serde_json::Value::as_u64)
                .map(|count| format!("{count} 条结果"))
                .unwrap_or_default()
        } else {
            String::new()
        };
        return CompletedToolParts {
            head: format!("{name}（“{query}”）"),
            summary: query.clone(),
            success,
            duration_ms: duration,
            reason: reason.clone(),
            line: format!(
                "{} {name}（“{query}”） · {}{}",
                if success { "✓" } else { "✗" },
                format_tool_duration(duration),
                if reason.is_empty() {
                    String::new()
                } else {
                    format!(" · {reason}")
                }
            ),
        };
    }
    let summary = truncate_to_width(&tool_input_summary(result), 48);
    let head = if summary.is_empty() {
        name.to_owned()
    } else {
        format!("{name} {summary}")
    };
    let reason = if result.success {
        String::new()
    } else {
        result
            .error
            .as_ref()
            .map(|error| redact_text(&error.message).replace(['\r', '\n'], " "))
            .map(|reason| truncate_to_width(&reason, 60))
            .unwrap_or_default()
    };
    let suffix = if reason.is_empty() {
        String::new()
    } else {
        format!(" · {reason}")
    };
    let line = format!(
        "{} {} · {}{}",
        if result.success { "✓" } else { "✗" },
        head,
        format_tool_duration(result.duration_ms),
        suffix
    );
    CompletedToolParts {
        head,
        summary,
        success: result.success,
        duration_ms: result.duration_ms,
        reason,
        line,
    }
}

/// 完成的工具并入待提交折叠：同键（工具名+入参摘要）累计 ×N，
/// 异键时先提交旧折叠再接管新折叠。
fn merge_pending_tool_fold(
    state: &mut TuiState,
    commits: &mut Vec<(Role, String)>,
    name: &str,
    parts: &CompletedToolParts,
) {
    let key = format!("{name}\u{1f}{}", parts.summary);
    match state.tool_fold.as_mut() {
        Some(fold) if fold.key == key => {
            fold.count += 1;
            if parts.success {
                fold.success += 1;
            } else {
                fold.failed += 1;
                if !parts.reason.is_empty() {
                    fold.last_reason = parts.reason.clone();
                }
            }
            fold.total_ms += parts.duration_ms;
            fold.entries.push((
                if parts.success {
                    Role::Tool
                } else {
                    Role::Warning
                },
                parts.line.clone(),
            ));
        }
        _ => {
            flush_pending_tool_fold(state, commits);
            state.tool_fold = Some(PendingToolFold {
                key,
                head: parts.head.clone(),
                count: 1,
                success: usize::from(parts.success),
                failed: usize::from(!parts.success),
                total_ms: parts.duration_ms,
                last_reason: parts.reason.clone(),
                entries: vec![(
                    if parts.success {
                        Role::Tool
                    } else {
                        Role::Warning
                    },
                    parts.line.clone(),
                )],
            });
        }
    }
}

/// 提交缓冲中的同类工具折叠：两行 `- 头（×N）` + `⎿ 统计尾行`。
fn flush_pending_tool_fold(state: &mut TuiState, commits: &mut Vec<(Role, String)>) {
    let Some(fold) = state.tool_fold.take() else {
        return;
    };
    let duration = format_tool_duration(fold.total_ms);
    let head = if fold.count > 1 {
        format!("{} ×{}", fold.head, fold.count)
    } else {
        fold.head.clone()
    };
    let tail = if fold.count == 1 {
        if fold.failed == 0 {
            if fold.last_reason.is_empty() {
                format!("完成 · {duration}")
            } else {
                // web_search 等成功时的附加信息（结果条数）。
                format!("{} · {duration}", fold.last_reason)
            }
        } else if fold.last_reason.is_empty() {
            format!("失败 · {duration}")
        } else {
            format!("失败 · {duration} · {}", fold.last_reason)
        }
    } else if fold.failed == 0 {
        format!("完成 ×{} · {duration}", fold.count)
    } else {
        let mut tail = format!(
            "×{} · 成功 {} · 失败 {} · {duration}",
            fold.count, fold.success, fold.failed
        );
        if !fold.last_reason.is_empty() {
            tail.push_str(&format!(" · {}", fold.last_reason));
        }
        tail
    };
    // 折叠组进历史（默认折叠）：组头 + 统计尾行 + 每个工具明细。
    let mut group_entries = fold.entries;
    group_entries.push((Role::ToolDetail, tail.clone()));
    push_group_block(state, Role::ToolHead, head.clone(), group_entries);
    commits.push((Role::ToolHead, head));
    commits.push((Role::ToolDetail, tail));
}

/// 从工具结果中提取入参摘要（优先 path/command/query 等键），作折叠头行后缀。
fn tool_input_summary(result: &xdudu_core::tools::ToolResult) -> String {
    const KEYS: [&str; 8] = [
        "path",
        "resolvedPath",
        "command",
        "query",
        "pattern",
        "glob",
        "url",
        "target",
    ];
    let sources = [
        result.output.as_ref(),
        result.error.as_ref().map(|error| &error.details),
        Some(&result.metadata),
    ];
    for source in sources.into_iter().flatten() {
        for key in KEYS {
            if let Some(value) = source.get(key).and_then(serde_json::Value::as_str)
                && !value.trim().is_empty()
            {
                return redact_text(value).replace(['\r', '\n'], " ");
            }
        }
    }
    String::new()
}

fn draw_session_picker(
    writer: &mut impl Write,
    state: &TuiState,
    columns: u16,
    rows: u16,
    color: bool,
) -> io::Result<()> {
    let Some(picker) = &state.session_picker else {
        return Ok(());
    };
    let top = 1;
    let bottom = rows.saturating_sub(2);
    for row in top..=bottom {
        queue!(writer, MoveTo(0, row), Clear(ClearType::CurrentLine))?;
    }

    set_color(writer, color, palette().primary)?;
    queue!(
        writer,
        MoveTo(2, top),
        SetAttribute(Attribute::Bold),
        Print("恢复历史会话"),
        SetAttribute(Attribute::Reset)
    )?;
    set_color(writer, color, palette().muted)?;
    queue!(
        writer,
        MoveTo(2, top + 1),
        Print("↑↓ 选择 · Enter 恢复 · Esc 取消")
    )?;

    let visible = usize::from(rows.saturating_sub(6)).max(1);
    let start = picker
        .selected
        .saturating_sub(visible.saturating_sub(1))
        .min(picker.choices.len().saturating_sub(visible));
    for (offset, choice) in picker.choices.iter().skip(start).take(visible).enumerate() {
        let index = start + offset;
        let row = top + 3 + offset as u16;
        let selected = index == picker.selected;
        set_color(
            writer,
            color,
            if selected {
                palette().primary
            } else {
                palette().text
            },
        )?;
        if selected {
            queue!(writer, SetAttribute(Attribute::Bold))?;
        }
        let id = choice.id.to_string();
        let line = format!(
            "{} {}  {}  {:<11}  {}",
            if selected { "›" } else { " " },
            &id[..8],
            choice.updated_at,
            choice.status,
            choice.title.replace(['\r', '\n'], " ")
        );
        queue!(
            writer,
            MoveTo(2, row),
            Print(truncate_to_width(
                &line,
                usize::from(columns.saturating_sub(4))
            )),
            SetAttribute(Attribute::Reset)
        )?;
    }
    reset_color(writer, color)
}

fn draw_model_picker(
    writer: &mut impl Write,
    state: &TuiState,
    columns: u16,
    rows: u16,
    color: bool,
) -> io::Result<()> {
    let Some(picker) = &state.model_picker else {
        return Ok(());
    };
    let top = 1_u16;
    let bottom = rows.saturating_sub(2);
    for row in top..=bottom {
        queue!(writer, MoveTo(0, row), Clear(ClearType::CurrentLine))?;
    }

    set_color(writer, color, palette().primary)?;
    queue!(
        writer,
        MoveTo(2, top),
        SetAttribute(Attribute::Bold),
        Print("选择模型"),
        SetAttribute(Attribute::Reset)
    )?;
    set_color(writer, color, palette().muted)?;
    queue!(
        writer,
        MoveTo(2, top + 1),
        Print("↑↓/j/k 选择 · Enter 确认 · Esc 取消")
    )?;

    for (index, choice) in picker.choices.iter().enumerate() {
        let row = top + 3 + (index as u16 * 2);
        if row + 1 >= bottom {
            break;
        }
        let selected = index == picker.selected;
        set_color(
            writer,
            color,
            if selected {
                palette().primary
            } else {
                palette().text
            },
        )?;
        if selected {
            queue!(writer, SetAttribute(Attribute::Bold))?;
        }
        let current = model_matches(&choice.id, &state.model);
        let title = format!(
            "{} {}{}",
            if selected { "›" } else { " " },
            choice.label,
            if current { "  当前" } else { "" }
        );
        queue!(
            writer,
            MoveTo(2, row),
            Print(truncate_to_width(
                &title,
                usize::from(columns.saturating_sub(4))
            )),
            SetAttribute(Attribute::Reset)
        )?;
        set_color(writer, color, palette().muted)?;
        let detail = format!("  {} · {}", choice.description, choice.id);
        queue!(
            writer,
            MoveTo(4, row + 1),
            Print(truncate_to_width(
                &detail,
                usize::from(columns.saturating_sub(6))
            ))
        )?;
    }
    reset_color(writer, color)
}

fn draw_plan_review(
    writer: &mut impl Write,
    state: &TuiState,
    columns: u16,
    rows: u16,
    color: bool,
) -> io::Result<()> {
    let Some(review) = &state.plan_review else {
        return Ok(());
    };
    let recovery = review.mode == PlanDialogMode::Recovery;
    for row in 0..rows {
        queue!(writer, MoveTo(0, row), Clear(ClearType::CurrentLine))?;
    }

    let width = usize::from(columns.saturating_sub(4)).max(20);
    let mut lines = Vec::new();
    append_wrapped(
        &mut lines,
        Role::System,
        &format!("目标：{}", review.plan.goal),
        width,
    );
    lines.push((Role::System, format!("修订版本：{}", review.plan.revision)));
    lines.push((Role::System, format!("状态：{:?}", review.plan.status)));
    if let Some(reason) = &review.plan.paused_reason {
        append_wrapped(
            &mut lines,
            Role::Warning,
            &format!("暂停原因：{reason}"),
            width,
        );
    }
    lines.push((Role::System, String::new()));
    let indexes = review
        .plan
        .steps
        .iter()
        .enumerate()
        .map(|(index, step)| (step.id, index + 1))
        .collect::<std::collections::HashMap<_, _>>();
    for (index, step) in review.plan.steps.iter().enumerate() {
        append_wrapped(
            &mut lines,
            Role::Assistant,
            &format!("{}. {} [{:?}]", index + 1, step.title, step.status),
            width,
        );
        if !step.description.trim().is_empty() {
            append_wrapped(
                &mut lines,
                Role::System,
                &format!("   {}", step.description),
                width,
            );
        }
        if !step.dependencies.is_empty() {
            let dependencies = step
                .dependencies
                .iter()
                .filter_map(|id| indexes.get(id))
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("、");
            lines.push((Role::System, format!("   依赖步骤：{dependencies}")));
        }
        for criterion in &step.completion_criteria {
            append_wrapped(
                &mut lines,
                Role::System,
                &format!("   ✓ {criterion}"),
                width,
            );
        }
        if let Some(attempt) = step.attempts.last() {
            lines.push((
                Role::System,
                format!("   执行尝试：#{} [{:?}]", attempt.attempt, attempt.status),
            ));
            if let Some(summary) = &attempt.summary {
                append_wrapped(
                    &mut lines,
                    Role::System,
                    &format!("   结果：{summary}"),
                    width,
                );
            }
            if let Some(error) = &attempt.error {
                append_wrapped(
                    &mut lines,
                    Role::Warning,
                    &format!("   错误：{error}"),
                    width,
                );
            }
            for evidence in &attempt.evidence {
                append_wrapped(
                    &mut lines,
                    Role::System,
                    &format!(
                        "   证据 {}：{}",
                        evidence.criterion_index, evidence.evidence
                    ),
                    width,
                );
            }
        }
        lines.push((Role::System, String::new()));
    }

    set_color(writer, color, palette().primary)?;
    queue!(
        writer,
        MoveTo(2, 0),
        SetAttribute(Attribute::Bold),
        Print(if recovery {
            "恢复暂停计划"
        } else {
            "审阅执行计划"
        }),
        SetAttribute(Attribute::Reset)
    )?;
    set_color(writer, color, palette().muted)?;
    if rows > 1 {
        queue!(
            writer,
            MoveTo(2, 1),
            Print(truncate_to_width(
                if recovery {
                    "↑↓/j/k 选择 · Enter 确认 · PgUp/PgDn 滚动 · Esc 保持暂停"
                } else {
                    "↑↓/j/k 选择 · Enter 确认 · PgUp/PgDn 滚动 · Esc 保留待审"
                },
                width
            ))
        )?;
    }

    let option_rows = 4_u16;
    let content_top = 3_u16;
    let content_height = usize::from(rows.saturating_sub(content_top + option_rows));
    let max_scroll = lines.len().saturating_sub(content_height);
    let start = review.scroll.min(max_scroll);
    for (offset, (role, line)) in lines.iter().skip(start).take(content_height).enumerate() {
        let row = content_top + offset as u16;
        let line_color = match role {
            Role::Assistant => palette().text,
            Role::Warning => palette().warning,
            Role::Recovery => palette().recovery,
            _ => palette().muted,
        };
        set_color(writer, color, line_color)?;
        queue!(
            writer,
            MoveTo(2, row),
            Print(truncate_to_width(line, width))
        )?;
    }

    let options: &[&str] = if recovery {
        &["继续", "重试当前步骤", "查看详情", "取消计划"]
    } else {
        &["批准计划", "请求修改", "拒绝计划"]
    };
    let options_top = rows.saturating_sub(3);
    for (index, option) in options.iter().enumerate() {
        let selected = index == review.selected;
        set_color(
            writer,
            color,
            if selected {
                palette().primary
            } else {
                palette().muted
            },
        )?;
        if selected {
            queue!(writer, SetAttribute(Attribute::Bold))?;
        }
        let column = 2 + (index as u16 * columns.saturating_sub(4) / options.len() as u16);
        queue!(
            writer,
            MoveTo(column, options_top),
            Print(if selected { "› " } else { "  " }),
            Print(option),
            SetAttribute(Attribute::Reset)
        )?;
    }
    reset_color(writer, color)
}

/// 绘制仍处于空闲态的启动页。它只覆盖启动页自己占用的前五行，因此终端
/// Resize 时可以重新计算水平位置，而不会改写已经进入滚动历史的对话。
fn draw_intro(
    writer: &mut impl Write,
    state: &mut TuiState,
    columns: u16,
    rows: u16,
    color: bool,
) -> io::Result<()> {
    // Codex 风格的 session header：一张紧凑信息卡，不占用大块空白。
    let clear_rows = rows.min(8);
    for row in 0..clear_rows {
        queue!(writer, MoveTo(0, row), Clear(ClearType::CurrentLine))?;
    }

    if rows < 12 || columns < 44 {
        let lines = [
            ">_ XDUDU".to_owned(),
            format!(
                "{} · {}",
                model_display_name(&state.provider, &state.model),
                state.provider
            ),
            format!(
                "{} tools · {}",
                state.available_tools.len(),
                state.permission
            ),
        ];
        for (row, line) in lines.iter().enumerate() {
            let visible = truncate_to_width(line, usize::from(columns).saturating_sub(2).max(1));
            let column = centered_column(columns, &visible);
            set_color(
                writer,
                color,
                if row == 0 {
                    palette().primary
                } else {
                    palette().muted
                },
            )?;
            queue!(writer, MoveTo(column, row as u16))?;
            if row == 0 {
                queue!(writer, SetAttribute(Attribute::Bold))?;
            }
            queue!(writer, Print(visible), SetAttribute(Attribute::Reset))?;
        }
        reset_color(writer, color)?;
        state.printed_lines = 3;
        state.content_end_row = 3;
        return Ok(());
    }

    let inner_width = usize::from(columns.saturating_sub(8)).clamp(34, 72);
    let left = columns.saturating_sub((inner_width + 2) as u16) / 2;
    let border = format!("╭{}╮", "─".repeat(inner_width + 2));
    let bottom = format!("╰{}╯", "─".repeat(inner_width + 2));
    let model = model_display_name(&state.provider, &state.model);
    let directory = state.cwd.display().to_string();
    let entries = [
        format!(">_ XDUDU (v{})", state.version),
        format!("model: {}   /model 切换", model),
        format!("directory: {}", directory),
        format!(
            "permissions: {} · {} tools · {} skills",
            state.permission,
            state.available_tools.len(),
            state.skills.len()
        ),
    ];
    set_color(writer, color, palette().border)?;
    queue!(writer, MoveTo(left, 0), Print(&border))?;
    for (offset, entry) in entries.iter().enumerate() {
        let text = truncate_to_width(entry, inner_width);
        let padding = " ".repeat(inner_width.saturating_sub(text.width()));
        let row = offset as u16 + 1;
        queue!(writer, MoveTo(left, row), Print("│ "))?;
        set_color(
            writer,
            color,
            if offset == 0 {
                palette().primary
            } else {
                palette().muted
            },
        )?;
        if offset == 0 {
            queue!(writer, SetAttribute(Attribute::Bold))?;
        }
        queue!(writer, Print(text), SetAttribute(Attribute::Reset))?;
        reset_color(writer, color)?;
        set_color(writer, color, palette().border)?;
        queue!(writer, Print(padding), Print(" │"))?;
    }
    queue!(writer, MoveTo(left, 5), Print(&bottom))?;
    reset_color(writer, color)?;
    state.printed_lines = 6;
    state.content_end_row = 6;
    Ok(())
}

fn centered_column(columns: u16, text: &str) -> u16 {
    columns.saturating_sub(text.width().min(usize::from(u16::MAX)) as u16) / 2
}

/// 活动区完整高度：chrome 4 行 + 输入块 + 折叠提示 + 建议 + 工具 + 流式尾巴，
/// 封顶在 [`MAX_ACTIVITY_HEIGHT`]，避免活动区挤没内容滚动区。
fn layout_height(state: &TuiState, columns: u16, rows: u16) -> u16 {
    layout_height_raw(state, columns, rows).min(MAX_ACTIVITY_HEIGHT)
}

fn layout_height_raw(state: &TuiState, columns: u16, rows: u16) -> u16 {
    let body_width = usize::from(columns.saturating_sub(4)).max(10);
    let input_width = usize::from(columns.saturating_sub(3)).max(10);
    let mut height = 4u16;
    let input_text: String = state.input.iter().collect();
    let input_lines = wrap_input(&input_text, input_width).len();
    let max_input_rows = usize::from(rows.saturating_sub(10)).max(3);
    height += input_lines.min(max_input_rows) as u16;
    if input_lines > max_input_rows {
        height += 1;
    }
    if !matching_commands(&state.input).is_empty() {
        height += matching_commands(&state.input)
            .len()
            .min(MAX_COMMAND_SUGGESTIONS) as u16;
    }
    height += state.tools.len().min(3) as u16;
    if !state.streaming.is_empty() {
        let mut tail = Vec::new();
        append_wrapped(&mut tail, Role::Assistant, &state.streaming, body_width);
        height += tail.len().min(STREAMING_WINDOW) as u16;
    }
    height
}

fn truncate_to_width(value: &str, width: usize) -> String {
    if UnicodeWidthStr::width(value) <= width {
        return value.to_owned();
    }
    let mut output = String::new();
    let mut used = 0;
    for character in value.chars() {
        let character_width = character.width().unwrap_or(0);
        if used + character_width + 1 > width {
            break;
        }
        output.push(character);
        used += character_width;
    }
    output.push('…');
    output
}

fn append_wrapped(lines: &mut Vec<(Role, String)>, role: Role, text: &str, width: usize) {
    for source_line in text.lines() {
        if source_line.is_empty() {
            lines.push((role, String::new()));
            continue;
        }
        let mut current = String::new();
        let mut current_width = 0;
        for character in source_line.chars() {
            let character_width = character.width().unwrap_or(0);
            if current_width + character_width > width && !current.is_empty() {
                lines.push((role, std::mem::take(&mut current)));
                current_width = 0;
            }
            current.push(character);
            current_width += character_width;
        }
        lines.push((role, current));
    }
}

fn append_markdown_wrapped(lines: &mut Vec<(Role, String)>, text: &str, width: usize) {
    for line in terminal_markdown(text) {
        let role = match line.kind {
            MarkdownLineKind::Body => Role::Assistant,
            MarkdownLineKind::Heading => Role::AssistantHeading,
            MarkdownLineKind::Code => Role::AssistantCode,
            MarkdownLineKind::DiffAdd => Role::AssistantDiffAdd,
            MarkdownLineKind::DiffRemove => Role::AssistantDiffRemove,
            MarkdownLineKind::DiffContext => Role::AssistantCode,
        };
        append_wrapped(lines, role, &line.text, width);
    }
}

/// 输入文本按宽度折行，保留显式换行。
fn wrap_input(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    for source_line in text.split('\n') {
        if source_line.is_empty() {
            lines.push(String::new());
            continue;
        }
        let mut current = String::new();
        let mut current_width = 0;
        for character in source_line.chars() {
            let character_width = character.width().unwrap_or(0);
            if current_width + character_width > width && !current.is_empty() {
                lines.push(std::mem::take(&mut current));
                current_width = 0;
            }
            current.push(character);
            current_width += character_width;
        }
        lines.push(current);
    }
    lines
}

/// 提交内容后推进内容末尾行：打印行数超过终端高度时固定在最底行。
/// 发送 DECSTBM 设置内容滚动区域 `[1, scroll_bottom+1]`；
/// 活动区位于滚动区域之外，内容滚动不会覆盖它。
fn queue_scroll_region(writer: &mut impl Write, scroll_bottom: u16) -> io::Result<()> {
    queue!(
        writer,
        Print(format!("\x1b[1;{}r", scroll_bottom.saturating_add(1)))
    )
}

/// 活动区总高度固定：chrome 4 行 + 动态区 6 行。
///
/// 滚动区底必须固定——流式尾巴/工具/输入只在固定活动区内裁剪显示，
/// 不改变滚动区大小。否则 streaming 非空时滚动区变小，清除范围上移，
/// 会反复啃掉滚动区底部刚提交的内容。
const MAX_ACTIVITY_HEIGHT: u16 = 10;

/// 命令建议行：usage 按显示宽度填充到 `usage_width`，description 列对齐。
fn command_suggestion_line(
    marker: &str,
    usage: &str,
    description: &str,
    usage_width: usize,
) -> String {
    let pad = usage_width.saturating_sub(usage.width());
    format!("{marker} {usage}{} {description}", " ".repeat(pad))
}

/// 输入块顶行：底部对齐输入行（rows-3），多行输入向上展开。
/// 单行输入时顶行即输入行本身，永不与状态行（dynamic_top）重叠。
fn input_block_top_of(input_row: u16, input_vis: usize) -> u16 {
    input_row.saturating_sub(input_vis as u16).saturating_add(1)
}

/// 内容滚动区域的底行：固定为活动区高度之上的一行。
fn scroll_bottom_of(_state: &TuiState, _columns: u16, rows: u16) -> u16 {
    rows.saturating_sub(MAX_ACTIVITY_HEIGHT).saturating_sub(1)
}

/// 光标在折行后输入中的（行号、列号）。
fn input_cursor_position(chars: &[char], cursor: usize, width: usize) -> (usize, usize) {
    let mut line = 0usize;
    let mut column = 0usize;
    for &character in chars.iter().take(cursor) {
        if character == '\n' {
            line += 1;
            column = 0;
            continue;
        }
        let character_width = character.width().unwrap_or(0);
        if column + character_width > width {
            line += 1;
            column = 0;
        }
        column += character_width;
    }
    (line, column)
}

fn role_glyph(role: Role) -> &'static str {
    match role {
        // Codex 的历史单元使用 › 表示用户、• 表示助手；后续行只保留两列缩进。
        Role::User => "›",
        Role::Assistant
        | Role::AssistantHeading
        | Role::AssistantCode
        | Role::AssistantDiffAdd
        | Role::AssistantDiffRemove => "•",
        Role::Tool => "•",
        Role::ToolHead => "-",
        Role::ToolDetail => "⎿",
        Role::Thinking => "✻",
        Role::System => "·",
        Role::Warning => "✗",
        Role::Recovery => "↻",
    }
}

fn role_tone(role: Role) -> Color {
    match role {
        Role::User => palette().primary,
        Role::Assistant => palette().text,
        Role::AssistantHeading => palette().accent,
        Role::AssistantCode => palette().muted,
        Role::AssistantDiffAdd => palette().success,
        Role::AssistantDiffRemove => Color::Rgb {
            r: 194,
            g: 126,
            b: 132,
        },
        // 工具过程弱化（浅灰）：只有最终输出才是正常色。
        Role::Tool => palette().muted,
        Role::ToolHead => palette().accent,
        Role::ToolDetail => palette().muted,
        Role::Thinking => palette().muted,
        Role::System => palette().muted,
        Role::Warning => palette().warning,
        Role::Recovery => palette().recovery,
    }
}

/// 打印一行已提交内容（glyph + 内容 + 换行），用于滚动区与视口恢复。
///
/// Codex 风格的历史层级：首行使用 `›`/`•` 标记，续行只保留两列缩进；
/// 用户首行加粗，助手正文使用 palette().text 色，工具和系统行使用低饱和强调色。
fn print_role_line(
    writer: &mut impl Write,
    role: Role,
    line: &str,
    columns: u16,
    color: bool,
    first_line: bool,
) -> io::Result<()> {
    print_role_line_content(writer, role, line, columns, color, first_line)?;
    queue!(writer, Print("\r\n"))
}

/// 助手正式输出角色（正文/标题/代码/diff）：渲染为纯文本，无行首符号。
fn is_assistant_output(role: Role) -> bool {
    matches!(
        role,
        Role::Assistant
            | Role::AssistantHeading
            | Role::AssistantCode
            | Role::AssistantDiffAdd
            | Role::AssistantDiffRemove
    )
}

/// 写入历史行本身，不追加换行。调用方先输出 CRLF，把上一行推入原生历史。
///
/// 仿 Claude Code：助手正文（最终输出）为纯文本，首行与续行均不带任何
/// 前缀符号或缩进；只有用户行与过程行（工具/思考/系统/警告）保留轻量符号。
fn print_role_line_content(
    writer: &mut impl Write,
    role: Role,
    line: &str,
    columns: u16,
    color: bool,
    first_line: bool,
) -> io::Result<()> {
    if is_assistant_output(role) {
        // 助手正文：无前缀、无缩进，直接输出。
    } else if !first_line {
        queue!(writer, Print("  "))?;
    } else {
        set_color(writer, color, role_tone(role))?;
        queue!(writer, Print(role_glyph(role)), Print(" "))?;
        reset_color(writer, color)?;
    }
    set_color(writer, color, role_tone(role))?;
    if matches!(role, Role::User | Role::AssistantHeading) {
        queue!(writer, SetAttribute(Attribute::Bold))?;
    }
    queue!(
        writer,
        Print(truncate_to_width(
            line,
            usize::from(columns.saturating_sub(4)).max(1),
        )),
        SetAttribute(Attribute::Reset)
    )?;
    reset_color(writer, color)
}

/// 将一行渲染为可缓存的 ANSI 字符串。
fn styled_fragment(enabled: bool, color: Color, bold: bool, text: &str) -> String {
    let mut bytes = Vec::new();
    if enabled {
        let _ = queue!(bytes, SetForegroundColor(terminal_color(color)));
        if bold {
            let _ = queue!(bytes, SetAttribute(Attribute::Bold));
        }
    }
    let _ = queue!(bytes, Print(text));
    if enabled {
        let _ = queue!(bytes, SetAttribute(Attribute::Reset), ResetColor);
    }
    String::from_utf8(bytes).unwrap_or_else(|_| text.to_owned())
}

fn set_color(writer: &mut impl Write, enabled: bool, color: Color) -> io::Result<()> {
    if enabled {
        queue!(writer, SetForegroundColor(terminal_color(color)))?;
    }
    Ok(())
}

fn reset_color(writer: &mut impl Write, _enabled: bool) -> io::Result<()> {
    queue!(writer, ResetColor)?;
    Ok(())
}

/// Terminal.app 未声明真彩色时，把 RGB 安全映射到 xterm-256 调色板。
///
/// 这可以避免旧终端把 `38;2;R;G;B` 中的数值误当成独立 SGR 指令，
/// 从而出现亮洋红背景等错误颜色。
fn terminal_color(color: Color) -> Color {
    match color {
        Color::Rgb { r, g, b } if !supports_true_color() => {
            let component =
                |value: u8| -> u8 { ((u16::from(value) * 5 + 127) / 255).try_into().unwrap_or(5) };
            Color::AnsiValue(16 + 36 * component(r) + 6 * component(g) + component(b))
        }
        _ => color,
    }
}

fn state_label(state: AgentLoopState) -> &'static str {
    match state {
        AgentLoopState::Idle => "就绪",
        AgentLoopState::Planning => "规划中",
        AgentLoopState::Acting => "执行中",
        AgentLoopState::Observing => "观察结果",
        AgentLoopState::Reflecting => "继续思考",
        AgentLoopState::WaitingApproval => "等待批准",
        AgentLoopState::Completed => "已完成",
        AgentLoopState::Incomplete => "未完成",
        AgentLoopState::Interrupted => "已中断",
        AgentLoopState::Error => "错误",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> TuiState {
        TuiState {
            version: "0.6.0",
            provider: "DeepSeek".into(),
            model: "deepseek-chat".into(),
            cwd: PathBuf::from("/work"),
            permission: "auto-safe".into(),
            status: "就绪".into(),
            printed_lines: 0,
            content_end_row: 0,
            dynamic_top: 0,
            intro_active: true,
            transcript: VecDeque::new(),
            streaming: String::new(),
            assistant_received_delta: false,
            code_fence_open: false,
            tools: Vec::new(),
            tool_fold: None,
            input: Vec::new(),
            cursor: 0,
            history: vec!["first".into()],
            history_index: None,
            draft: Vec::new(),
            command_selection: 0,
            input_hint: String::new(),
            input_active: true,
            vim_enabled: false,
            vim_normal: false,
            vim_last: None,
            usage: None,
            available_tools: vec!["file_read".into(), "git_status".into()],
            skills: Vec::new(),
            session_picker: None,
            plan_review: None,
            model_picker: None,
            debug_trace: false,
        }
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn 空输入按_tab_切换折叠_有输入时仍为命令补全() {
        let mut state = state();
        // 空输入：Tab 切换折叠组展开。
        assert_eq!(
            handle_input_key(&mut state, key(KeyCode::Tab)),
            Some(InputOutcome::ToggleFold)
        );
        // 有输入：Tab 仍走命令补全（不返回 ToggleFold）。
        handle_input_key(&mut state, key(KeyCode::Char('g')));
        assert_ne!(
            handle_input_key(&mut state, key(KeyCode::Tab)),
            Some(InputOutcome::ToggleFold)
        );
    }

    #[test]
    fn composer_支持编辑提交与历史() {
        let mut state = state();
        handle_input_key(&mut state, key(KeyCode::Char('a')));
        handle_input_key(&mut state, key(KeyCode::Char('c')));
        handle_input_key(&mut state, key(KeyCode::Left));
        handle_input_key(&mut state, key(KeyCode::Char('b')));
        assert_eq!(
            handle_input_key(&mut state, key(KeyCode::Enter)),
            Some(InputOutcome::Submit("abc".into()))
        );
        assert_eq!(handle_input_key(&mut state, key(KeyCode::Up)), None);
        assert_eq!(state.input.iter().collect::<String>(), "abc");
    }

    #[test]
    fn 启动页居中位置随终端宽度变化() {
        assert_eq!(centered_column(80, "1234567890"), 35);
        assert_eq!(centered_column(120, "1234567890"), 55);
        assert_eq!(centered_column(4, "1234567890"), 0);
    }

    #[test]
    fn 斜杠命令支持候选选择补全和执行() {
        let mut state = state();
        handle_input_key(&mut state, key(KeyCode::Char('/')));
        assert_eq!(matching_commands(&state.input).len(), 5);

        handle_input_key(&mut state, key(KeyCode::Down));
        assert_eq!(state.command_selection, 1);
        handle_input_key(&mut state, key(KeyCode::Tab));
        assert_eq!(state.input.iter().collect::<String>(), "/new");

        assert_eq!(
            handle_input_key(&mut state, key(KeyCode::Enter)),
            Some(InputOutcome::Command("/new".into()))
        );
    }

    #[test]
    fn model_命令无需完整参数即可执行() {
        let mut state = state();
        for character in "/mod".chars() {
            handle_input_key(&mut state, key(KeyCode::Char(character)));
        }
        assert_eq!(
            handle_input_key(&mut state, key(KeyCode::Enter)),
            Some(InputOutcome::Command("/model".into()))
        );
    }

    #[test]
    fn vim_普通模式_移动删除与历史() {
        let mut state = state();
        state.vim_enabled = true;
        state.vim_normal = false; // 插入模式
        state.history = vec!["第一条历史".into(), "第二条历史".into()];
        // 插入模式下输入 "abcd"。
        handle_input_key(&mut state, key(KeyCode::Char('a')));
        handle_input_key(&mut state, key(KeyCode::Char('b')));
        handle_input_key(&mut state, key(KeyCode::Char('c')));
        handle_input_key(&mut state, key(KeyCode::Char('d')));
        assert_eq!(state.input.iter().collect::<String>(), "abcd");
        // Esc 进入普通模式（不清空）。
        handle_input_key(&mut state, key(KeyCode::Esc));
        assert!(state.vim_normal);
        assert_eq!(state.input.iter().collect::<String>(), "abcd");
        // h/l 移动、x 删除、D 删尾、dd 清空。
        handle_input_key(&mut state, key(KeyCode::Char('h')));
        assert_eq!(state.cursor, 3);
        // x 删除光标处字符（此处为 'd'），光标移到行尾。
        handle_input_key(&mut state, key(KeyCode::Char('x')));
        assert_eq!(state.input.iter().collect::<String>(), "abc");
        // h 回退到 'c' 前，D 删除光标到行尾。
        handle_input_key(&mut state, key(KeyCode::Char('h')));
        handle_input_key(&mut state, key(KeyCode::Char('D')));
        assert_eq!(state.input.iter().collect::<String>(), "ab");
        handle_input_key(&mut state, key(KeyCode::Char('d')));
        handle_input_key(&mut state, key(KeyCode::Char('d')));
        assert!(state.input.is_empty());
        // k=上：先到最新"第二条历史"，再向上到"第一条历史"；j=下回到草稿。
        handle_input_key(&mut state, key(KeyCode::Char('k')));
        assert_eq!(state.input.iter().collect::<String>(), "第二条历史");
        handle_input_key(&mut state, key(KeyCode::Char('k')));
        assert_eq!(state.input.iter().collect::<String>(), "第一条历史");
        handle_input_key(&mut state, key(KeyCode::Char('j')));
        assert_eq!(state.input.iter().collect::<String>(), "第二条历史");
        handle_input_key(&mut state, key(KeyCode::Char('j')));
        assert!(state.input.is_empty());
        // i 进入插入模式；普通模式未知键无副作用。
        handle_input_key(&mut state, key(KeyCode::Char('i')));
        assert!(!state.vim_normal);
        handle_input_key(&mut state, key(KeyCode::Char('z')));
        assert!(state.input.iter().collect::<String>().ends_with('z'));
    }

    #[test]
    fn vim_插入模式_esc_返回普通且不清空() {
        let mut state = state();
        state.vim_enabled = true;
        handle_input_key(&mut state, key(KeyCode::Char('x')));
        handle_input_key(&mut state, key(KeyCode::Esc));
        assert!(state.vim_normal);
        assert_eq!(state.input.iter().collect::<String>(), "x");
    }

    #[test]
    fn 空闲_esc_清空当前输入() {
        let mut state = state();
        handle_input_key(&mut state, key(KeyCode::Char('a')));
        handle_input_key(&mut state, key(KeyCode::Char('b')));
        handle_input_key(&mut state, key(KeyCode::Esc));
        assert!(state.input.is_empty());
        assert_eq!(state.cursor, 0);
        // 空输入时 Esc 无副作用。
        assert_eq!(handle_input_key(&mut state, key(KeyCode::Esc)), None);
        assert!(state.input.is_empty());
    }

    #[test]
    fn 粘贴文本插入光标处且不触发提交() {
        let mut state = state();
        insert_paste_text(&mut state, "第一行\r\n第二行");
        let input: String = state.input.iter().collect();
        assert_eq!(input, "第一行\n第二行");
        assert_eq!(state.cursor, input.chars().count());
        // 粘贴不会产生提交结果，只有显式 Enter 才提交。
        assert_eq!(
            handle_input_key(&mut state, key(KeyCode::Enter)),
            Some(InputOutcome::Submit(input))
        );
    }

    #[test]
    fn 超长粘贴批量插入并稳定截断() {
        let mut state = state();
        let pasted = "x".repeat(MAX_INPUT_CHARS + 10);
        assert!(insert_paste_text(&mut state, &pasted));
        assert_eq!(state.input.len(), MAX_INPUT_CHARS);
        assert_eq!(state.cursor, MAX_INPUT_CHARS);
    }

    #[test]
    fn 流式内容已提交时不重复输出最终消息() {
        let mut state = state();
        state.assistant_received_delta = true;
        assert_eq!(take_final_assistant_tail(&mut state, "完整回答"), "");

        state.assistant_received_delta = false;
        assert_eq!(
            take_final_assistant_tail(&mut state, "非流式回答"),
            "非流式回答"
        );
    }

    #[test]
    fn shift_enter_插入换行_普通_enter_提交多行() {
        let mut first = state();
        handle_input_key(&mut first, key(KeyCode::Char('a')));
        handle_input_key(
            &mut first,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT),
        );
        handle_input_key(&mut first, key(KeyCode::Char('b')));
        assert_eq!(
            handle_input_key(&mut first, key(KeyCode::Enter)),
            Some(InputOutcome::Submit("a\nb".into()))
        );
        // Ctrl+J 同样插入换行。
        let mut multiline = state();
        handle_input_key(&mut multiline, key(KeyCode::Char('x')));
        handle_input_key(
            &mut multiline,
            KeyEvent::new(KeyCode::Char('j'), KeyModifiers::CONTROL),
        );
        assert!(multiline.input.contains(&'\n'));
    }

    #[test]
    fn 普通完整行到达即提交_长文本实时流入滚动区() {
        let mut state = state();
        state.streaming = "第一行\n".into();
        let segments = take_ready_segments(&mut state);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0], (Role::Assistant, "第一行\n".into()));
        assert!(state.streaming.is_empty());

        // 未完成的尾行保留。
        state.streaming = "没有换行的尾巴".into();
        assert!(take_ready_segments(&mut state).is_empty());
        assert_eq!(state.streaming, "没有换行的尾巴");
    }

    #[test]
    fn 表格行块级缓冲_整块提交保留表格样式() {
        let mut state = state();
        // 输入以表格行结束：缓冲保留待续，等后续增量或消息结束再提交。
        state.streaming = "| 名称 | 状态 |\n| --- | --- |\n| 搜索 | 完成 |\n".into();
        let segments = take_ready_segments(&mut state);
        assert!(segments.is_empty());
        assert_eq!(
            state.streaming,
            "| 名称 | 状态 |\n| --- | --- |\n| 搜索 | 完成 |\n"
        );

        // 表格结束后空行到达：整块提交，空行单独提交。
        state.streaming.push('\n');
        let segments = take_ready_segments(&mut state);
        assert_eq!(segments.len(), 2);
        assert_eq!(
            segments[0].1,
            "| 名称 | 状态 |\n| --- | --- |\n| 搜索 | 完成 |\n"
        );
        assert_eq!(segments[1].1, "\n");
        assert!(state.streaming.is_empty());

        // 表格行后接普通行：表格先整块提交，普通行立即提交。
        state.streaming = "| a | b |\n| 1 | 2 |\n正文\n".into();
        let segments = take_ready_segments(&mut state);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].1, "| a | b |\n| 1 | 2 |\n");
        assert_eq!(segments[1].1, "正文\n");

        // 未结束的表格缓冲保留回流式缓冲。
        state.streaming = "| a | b |\n".into();
        assert!(take_ready_segments(&mut state).is_empty());
        assert_eq!(state.streaming, "| a | b |\n");
    }

    #[test]
    fn 代码栅栏打开行提交_栅栏内逐行以代码样式提交() {
        let mut state = state();
        state.streaming = "```rust\ncode\nmore\n".into();
        let segments = take_ready_segments(&mut state);
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0], (Role::Assistant, "```rust\n".into()));
        assert_eq!(segments[1], (Role::AssistantCode, "  code\n".into()));
        assert_eq!(segments[2], (Role::AssistantCode, "  more\n".into()));
        assert!(state.code_fence_open);
        assert!(state.streaming.is_empty());

        // 闭合行提交为空代码块（终端不显示 ``` 本身），普通行与空行独立提交。
        state.streaming.push_str("```\n末尾\n\n");
        let segments = take_ready_segments(&mut state);
        assert_eq!(segments[0], (Role::Assistant, "```\n".into()));
        assert_eq!(segments[1], (Role::Assistant, "末尾\n".into()));
        assert_eq!(segments[2], (Role::Assistant, "\n".into()));
        assert!(!state.code_fence_open);
        assert!(state.streaming.is_empty());
    }

    #[test]
    fn 多行输入光标定位() {
        let chars = "ab\ncd".chars().collect::<Vec<_>>();
        assert_eq!(input_cursor_position(&chars, 1, 10), (0, 1));
        // 光标在换行符之后即下一行行首。
        assert_eq!(input_cursor_position(&chars, 3, 10), (1, 0));
        assert_eq!(input_cursor_position(&chars, 4, 10), (1, 1));
        // 超宽自动折行。
        let wide = "abcdef".chars().collect::<Vec<_>>();
        assert_eq!(input_cursor_position(&wide, 3, 4), (0, 3));
        // 折行后字符落在下一行行首之后的列。
        assert_eq!(input_cursor_position(&wide, 5, 4), (1, 1));
    }

    #[test]
    fn web_search_完成记录包含查询结果数和耗时() {
        let tool = ToolActivity {
            call_id: "call-1".into(),
            name: "web_search".into(),
            detail: "搜索：爪子刀".into(),
            finished: None,
            duration_ms: None,
        };
        let result = xdudu_core::tools::ToolResult::success(
            serde_json::json!({
                "query": "爪子刀",
                "resultCount": 5,
                "results": [],
            }),
            chrono::Utc::now(),
            serde_json::json!({}),
        );
        let parts = completed_tool_parts(&tool, &result);
        assert!(parts.head.contains("联网搜索（“爪子刀”）"));
        assert_eq!(parts.reason, "5 条结果");
        assert!(parts.success);
    }

    #[test]
    fn 连续同类工具完成合并为计数折叠() {
        let mut state = state();
        let mut commits = Vec::new();
        for _ in 0..3 {
            let parts = CompletedToolParts {
                head: "运行命令 ls".into(),
                summary: "ls".into(),
                success: true,
                duration_ms: 10,
                reason: String::new(),
                line: "✓ 运行命令 ls · 10ms".into(),
            };
            merge_pending_tool_fold(&mut state, &mut commits, "terminal_exec", &parts);
        }
        // 同键合并期间不产生提交，全部缓冲在活动区。
        assert!(commits.is_empty());
        assert_eq!(state.tool_fold.as_ref().map(|fold| fold.count), Some(3));
        flush_pending_tool_fold(&mut state, &mut commits);
        assert_eq!(commits.len(), 2);
        assert!(commits[0].1.contains("运行命令 ls ×3"));
        assert!(commits[1].1.contains("完成 ×3"));
        assert!(state.tool_fold.is_none());
    }

    #[test]
    fn 异键工具完成先提交旧折叠() {
        let mut state = state();
        let mut commits = Vec::new();
        let first = CompletedToolParts {
            head: "运行命令 ls".into(),
            summary: "ls".into(),
            success: false,
            duration_ms: 5,
            reason: "退出码 126".into(),
            line: "✗ 运行命令 ls · 5ms · 退出码 126".into(),
        };
        merge_pending_tool_fold(&mut state, &mut commits, "terminal_exec", &first);
        merge_pending_tool_fold(&mut state, &mut commits, "terminal_exec", &first);
        let second = CompletedToolParts {
            head: "读取文件 pom.xml".into(),
            summary: "pom.xml".into(),
            success: true,
            duration_ms: 1,
            reason: String::new(),
            line: "✓ 读取文件 pom.xml · 1ms".into(),
        };
        merge_pending_tool_fold(&mut state, &mut commits, "file_read", &second);
        // 旧折叠（失败 ×2）已提交，新折叠仍在缓冲。
        assert_eq!(commits.len(), 2);
        assert!(commits[0].1.contains("运行命令 ls ×2"));
        assert!(commits[1].1.contains("失败"));
        assert!(commits[1].1.contains("退出码 126"));
        assert_eq!(
            state.tool_fold.as_ref().map(|fold| fold.head.as_str()),
            Some("读取文件 pom.xml")
        );
    }

    #[test]
    fn resume_可以通过斜杠候选定位() {
        let input = "/r".chars().collect::<Vec<_>>();
        let commands = matching_commands(&input);
        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].name, "/resume");
    }

    #[test]
    fn plan_可以通过斜杠候选定位并直接打开审阅() {
        let commands = matching_commands(&['/', 'p']);
        let plan = commands
            .iter()
            .find(|command| command.name == "/plan")
            .expect("应包含通用 /plan 命令");
        assert!(!plan.requires_argument);

        let commands = matching_commands(&"/plan r".chars().collect::<Vec<_>>());
        assert!(commands.iter().any(|command| command.name == "/plan run"));
        assert!(commands.iter().any(|command| command.name == "/plan retry"));
        assert!(
            commands
                .iter()
                .any(|command| command.name == "/plan revisions")
        );
    }

    #[test]
    fn 恢复会话会重建用户和助手时间线() {
        let session: Session = serde_json::from_value(serde_json::json!({
            "id": uuid::Uuid::new_v4(),
            "title": "历史会话",
            "cwd": "/work",
            "status": "completed",
            "currentState": "COMPLETED",
            "plan": {},
            "providerName": "deepseek",
            "model": "deepseek-chat",
            "messages": [
                {
                    "id": uuid::Uuid::new_v4(),
                    "role": "user",
                    "content": "问题",
                    "sequence": 0,
                    "createdAt": "2026-07-30T00:00:00Z"
                },
                {
                    "id": uuid::Uuid::new_v4(),
                    "role": "assistant",
                    "content": "回答",
                    "sequence": 1,
                    "createdAt": "2026-07-30T00:00:01Z"
                },
                {
                    "id": uuid::Uuid::new_v4(),
                    "role": "tool",
                    "content": "工具结果",
                    "toolCallId": "call-1",
                    "sequence": 2,
                    "createdAt": "2026-07-30T00:00:02Z"
                }
            ],
            "toolCalls": [],
            "totalInputTokens": 12,
            "totalOutputTokens": 8,
            "createdAt": "2026-07-30T00:00:00Z",
            "updatedAt": "2026-07-30T00:00:02Z",
            "completedAt": "2026-07-30T00:00:02Z"
        }))
        .unwrap();
        let mut state = state();
        load_session_state(&mut state, &session);
        assert_eq!(state.transcript.len(), 2);
        assert_eq!(state.transcript[0].role, Role::User);
        assert_eq!(state.transcript[1].role, Role::Assistant);
        assert_eq!(state.usage, Some((12, 8)));
    }

    #[test]
    fn 文本按_unicode_显示宽度换行() {
        let mut lines = Vec::new();
        append_wrapped(&mut lines, Role::Assistant, "你好Rust", 6);
        assert_eq!(lines[0].1, "你好Ru");
        assert_eq!(lines[1].1, "st");
    }

    #[test]
    fn 命令建议description列按显示宽度对齐() {
        let width = "/plan new <目标>".width(); // 含中文，显示宽 > 字符数
        let line1 = command_suggestion_line(" ", "/new", "开始新会话", width);
        let line2 = command_suggestion_line(" ", "/plan new <目标>", "创建结构化计划", width);
        // 按显示宽度比较 description 起始列（find 返回字符索引，中文字符数≠宽度）。
        let desc1 = line1.find('开').unwrap();
        let desc2 = line2.find('创').unwrap();
        let col1 = line1[..desc1].width();
        let col2 = line2[..desc2].width();
        assert_eq!(col1, col2);
        // 对齐列 = marker(1) + 空格(1) + usage_width + 分隔空格(1)。
        assert_eq!(col1, 3 + width);
    }

    #[test]
    fn 输入块顶行不越过状态行且单行对齐输入行() {
        // rows=18：输入行 15，状态行 dynamic_top=8。
        assert_eq!(input_block_top_of(15, 1), 15); // 单行：输入行本身
        assert_eq!(input_block_top_of(15, 2), 14);
        assert_eq!(input_block_top_of(15, 3), 13);
        // 三行输入时顶行 13 > 状态行 8，永不重叠。
        assert!(input_block_top_of(15, 3) > 8);
        // 矮终端 rows=12：输入行 9，状态行 dynamic_top=2。
        assert!(input_block_top_of(9, 3) > 2);
    }

    #[test]
    fn 滚动区底固定不随流式内容变化() {
        let mut state = state();
        state.streaming = "正在输出\n更多\n第三行\n第四行\n第五行".into();
        let bottom_streaming = scroll_bottom_of(&state, 80, 40);
        state.streaming.clear();
        let bottom_idle = scroll_bottom_of(&state, 80, 40);
        // 关键不变量：流式尾巴不改变滚动区大小，避免清除啃掉已提交内容。
        assert_eq!(bottom_streaming, bottom_idle);
        // 矮终端：活动区让步，内容区至少 3 行。
        assert!(scroll_bottom_of(&state, 80, 14) >= 3);
    }

    #[test]
    fn 内容末尾行随提交推进并封顶在滚动区底() {
        let mut state = state();
        state.content_end_row = 4;
        // 提交 3 行：内容末尾推进到 7。
        state.content_end_row = (4u16.saturating_add(3)).min(14);
        assert_eq!(state.content_end_row, 7);
        // 满屏后封顶在滚动区底。
        state.content_end_row = (12u16.saturating_add(10)).min(14);
        assert_eq!(state.content_end_row, 14);
    }

    #[test]
    fn 活动区高度包含多行输入与流式尾巴() {
        let mut state = state();
        state.input = "ab\ncd".chars().collect();
        state.streaming = "正在输出\n更多".into();
        let height = layout_height(&state, 80, 24);
        // chrome 4 + 输入 2 行 + 流式 2 行。
        assert_eq!(height, 8);
    }
}
