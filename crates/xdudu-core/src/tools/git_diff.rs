//! `git_diff`：固定参数、输出有界的工作区或暂存区差异。
//!
//! 支持 `path` 参数：目标项目位于工作区子目录（嵌套仓库）而工作区根不是
//! Git 仓库时，从该目录向上发现仓库根，与 `git_status` 行为对齐。

use std::path::Path;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::{SideEffectKind, permission::PermissionLevel};

use super::git_common::{repository_root_at, run_git, safe_relative_path};
use super::path_policy::resolve_existing;
use super::{Tool, ToolContext, ToolDefinition, ToolResult, object, reject_unknown_fields};

const DEFAULT_MAX_BYTES: u64 = 512 * 1024;
const MAX_BYTES: u64 = 2 * 1024 * 1024;
const MAX_PATHS: usize = 50;

pub struct GitDiffTool;

fn parse_name_status(bytes: &[u8]) -> Vec<Value> {
    let fields = bytes
        .split(|byte| *byte == 0)
        .filter(|value| !value.is_empty())
        .map(|value| String::from_utf8_lossy(value).into_owned())
        .collect::<Vec<_>>();
    let mut files = Vec::new();
    let mut index = 0;
    while index < fields.len() {
        let status = &fields[index];
        let kind = status.chars().next().unwrap_or('?');
        if matches!(kind, 'R' | 'C') && index + 2 < fields.len() {
            files.push(json!({
                "status": kind.to_string(),
                "score": status.get(1..),
                "originalPath": fields[index + 1],
                "path": fields[index + 2],
            }));
            index += 3;
        } else if index + 1 < fields.len() {
            files.push(json!({
                "status": kind.to_string(),
                "score": Value::Null,
                "originalPath": Value::Null,
                "path": fields[index + 1],
            }));
            index += 2;
        } else {
            break;
        }
    }
    files
}

fn paths(input: &Value) -> Vec<String> {
    input
        .get("paths")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn base_args(input: &Value) -> Vec<String> {
    let scope = input
        .get("scope")
        .and_then(Value::as_str)
        .unwrap_or("worktree");
    let context_lines = input
        .get("contextLines")
        .and_then(Value::as_u64)
        .unwrap_or(3);
    let mut args = vec![
        "--no-optional-locks".to_owned(),
        "diff".to_owned(),
        "--no-ext-diff".to_owned(),
        "--no-textconv".to_owned(),
        format!("--unified={context_lines}"),
        "--src-prefix=a/".to_owned(),
        "--dst-prefix=b/".to_owned(),
    ];
    if scope == "staged" {
        args.push("--cached".to_owned());
    }
    args.push("--".to_owned());
    args.extend(paths(input));
    args
}

#[async_trait]
impl Tool for GitDiffTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "git_diff".into(),
            description: "返回工作区或暂存区的 Git unified diff 和结构化文件状态，不执行外部 diff 或 textconv。目标项目位于工作区子目录时，用 path 指定项目目录（在其内向上发现仓库根）；paths 是相对仓库根的文件过滤。".into(),
            input_schema: json!({
                "type":"object",
                "properties":{
                    "scope":{"type":"string","enum":["worktree","staged"]},
                    "path":{"type":"string","minLength":1,"maxLength":4096},
                    "paths":{"type":"array","maxItems":MAX_PATHS,"items":{"type":"string","minLength":1,"maxLength":4096}},
                    "contextLines":{"type":"integer","minimum":0,"maximum":20},
                    "maxBytes":{"type":"integer","minimum":1,"maximum":MAX_BYTES}
                },
                "additionalProperties":false
            }),
            permission_level: PermissionLevel::ReadOnly,
            side_effect: SideEffectKind::None,
            default_timeout: Duration::from_secs(20),
        }
    }

    fn validate(&self, input: &Value) -> Result<(), Vec<String>> {
        let map = object(input)?;
        let mut issues = Vec::new();
        reject_unknown_fields(
            map,
            &["scope", "path", "paths", "contextLines", "maxBytes"],
            &mut issues,
        );
        if let Some(path) = map.get("path")
            && (!path.is_string()
                || path
                    .as_str()
                    .is_some_and(|value| value.is_empty() || value.len() > 4096))
        {
            issues.push("path 必须是 1 到 4096 字符的非空字符串。".into());
        }
        if !map.get("scope").is_none_or(|value| {
            value
                .as_str()
                .is_some_and(|value| matches!(value, "worktree" | "staged"))
        }) {
            issues.push("scope 只能是 worktree 或 staged。".into());
        }
        if let Some(value) = map.get("paths") {
            match value.as_array() {
                Some(values)
                    if values.len() <= MAX_PATHS
                        && values
                            .iter()
                            .all(|value| value.as_str().is_some_and(safe_relative_path)) => {}
                _ => issues.push(format!(
                    "paths 必须是最多 {MAX_PATHS} 个安全工作区相对路径组成的数组。"
                )),
            }
        }
        if !map
            .get("contextLines")
            .is_none_or(|value| value.as_u64().is_some_and(|value| value <= 20))
        {
            issues.push("contextLines 必须是 0 到 20 的整数。".into());
        }
        if !map.get("maxBytes").is_none_or(|value| {
            value
                .as_u64()
                .is_some_and(|value| (1..=MAX_BYTES).contains(&value))
        }) {
            issues.push(format!("maxBytes 必须是 1 到 {MAX_BYTES} 的整数。"));
        }
        if issues.is_empty() {
            Ok(())
        } else {
            Err(issues)
        }
    }

    async fn execute(&self, input: Value, context: ToolContext) -> ToolResult {
        // 目标项目位于工作区子目录时，从该子目录向上发现仓库根。
        let start = match input.get("path").and_then(Value::as_str) {
            Some(raw) => match resolve_existing(Path::new(raw), &context.cwd).await {
                Ok(path) => path,
                Err(error) => {
                    return ToolResult::failure(
                        "INVALID_GIT_PATH",
                        error.message,
                        context.started_at,
                        json!({"path":raw}),
                    );
                }
            },
            None => context.cwd.clone(),
        };
        let (workspace, root) = match repository_root_at(&context.cwd, &start).await {
            Ok(value) => value,
            Err(message) => {
                return ToolResult::failure(
                    "NOT_GIT_REPOSITORY",
                    message,
                    context.started_at,
                    json!({"start":start}),
                );
            }
        };
        let max_bytes = input
            .get("maxBytes")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_MAX_BYTES) as usize;
        let diff = match run_git(&workspace, &root, &base_args(&input), max_bytes).await {
            Ok(output) if output.status.success() => output,
            Ok(output) => {
                return ToolResult::failure(
                    "GIT_DIFF_ERROR",
                    String::from_utf8_lossy(&output.stderr).trim(),
                    context.started_at,
                    json!({}),
                );
            }
            Err(message) => {
                return ToolResult::failure(
                    "GIT_DIFF_ERROR",
                    message,
                    context.started_at,
                    json!({}),
                );
            }
        };
        let mut name_args = base_args(&input);
        name_args.retain(|arg| !arg.starts_with("--unified="));
        let separator = name_args
            .iter()
            .position(|arg| arg == "--")
            .unwrap_or(name_args.len());
        name_args.insert(separator, "--name-status".to_owned());
        name_args.insert(separator + 1, "-z".to_owned());
        let files = match run_git(&workspace, &root, &name_args, 2 * 1024 * 1024).await {
            Ok(output) if output.status.success() => parse_name_status(&output.stdout),
            _ => Vec::new(),
        };
        let bytes_returned = diff.stdout.len();
        ToolResult::success(
            json!({
                "scope": input.get("scope").and_then(Value::as_str).unwrap_or("worktree"),
                "diff": String::from_utf8_lossy(&diff.stdout),
                "files": files,
                "bytesReturned": bytes_returned,
                "truncated": diff.stdout_truncated,
            }),
            context.started_at,
            json!({"repositoryRoot":root}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use chrono::Utc;
    use tempfile::tempdir;
    use tokio_util::sync::CancellationToken;

    use crate::changes::NoopChangeLedger;
    use crate::permission::PermissionMode;

    fn context(cwd: &Path) -> ToolContext {
        ToolContext {
            session_id: uuid::Uuid::new_v4(),
            call_id: uuid::Uuid::new_v4(),
            cwd: cwd.to_path_buf(),
            permission_mode: PermissionMode::AutoSafe,
            cancellation: CancellationToken::new(),
            started_at: Utc::now(),
            change_ledger: Arc::new(NoopChangeLedger),
            progress: None,
            command_rules: Default::default(),
        }
    }

    fn run_git(dir: &Path, args: &[&str]) {
        let status = std::process::Command::new("git")
            .args(args)
            .current_dir(dir)
            .status()
            .unwrap();
        assert!(status.success(), "git {args:?} 失败");
    }

    /// 在工作区子目录建一个带未提交改动的嵌套仓库。
    async fn nested_repo() -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
        let parent = tempdir().unwrap();
        let workspace = parent.path().join("workspace");
        let project = workspace.join("projects").join("target");
        tokio::fs::create_dir_all(&project).await.unwrap();
        run_git(&project, &["init", "--quiet"]);
        tokio::fs::write(project.join("a.txt"), "hello\n")
            .await
            .unwrap();
        run_git(&project, &["add", "."]);
        run_git(
            &project,
            &[
                "-c",
                "user.email=t@t",
                "-c",
                "user.name=t",
                "commit",
                "--quiet",
                "-m",
                "init",
            ],
        );
        tokio::fs::write(project.join("a.txt"), "hello changed\n")
            .await
            .unwrap();
        (
            parent,
            tokio::fs::canonicalize(&workspace).await.unwrap(),
            tokio::fs::canonicalize(&project).await.unwrap(),
        )
    }

    #[tokio::test]
    async fn path参数支持工作区子目录内的嵌套仓库() {
        let (_parent, workspace, project) = nested_repo().await;
        let result = GitDiffTool
            .execute(
                json!({"path": project.to_str().unwrap()}),
                context(&workspace),
            )
            .await;
        assert!(result.error.is_none(), "应成功：{:?}", result.error);
        let output = result.output.unwrap();
        assert!(output["diff"].as_str().unwrap().contains("hello changed"));
        assert_eq!(output["files"][0]["path"], "a.txt");
    }

    #[tokio::test]
    async fn 不带path时仍从工作区根解析并在非仓库时报错() {
        let (_parent, workspace, _project) = nested_repo().await;
        let result = GitDiffTool.execute(json!({}), context(&workspace)).await;
        assert_eq!(result.error.unwrap().code, "NOT_GIT_REPOSITORY");
    }

    #[test]
    fn path参数校验拒绝非法取值() {
        let tool = GitDiffTool;
        assert!(tool.validate(&json!({"path": ""})).is_err());
        assert!(tool.validate(&json!({"path": 1})).is_err());
        assert!(tool.validate(&json!({"unknown": true})).is_err());
        assert!(tool.validate(&json!({"path": "projects/target"})).is_ok());
    }
}
