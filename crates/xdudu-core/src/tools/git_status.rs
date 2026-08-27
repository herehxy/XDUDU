//! `git_status`：使用固定 Git 参数返回结构化仓库状态。

use std::path::Path;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::{SideEffectKind, permission::PermissionLevel};

use super::git_common::{repository_root_at, run_git};
use super::path_policy::resolve_existing;
use super::{Tool, ToolContext, ToolDefinition, ToolResult, object, reject_unknown_fields};

const MAX_ENTRIES: usize = 10_000;

pub struct GitStatusTool;

fn parse_branch(bytes: &[u8]) -> Value {
    let mut oid = None;
    let mut head = None;
    let mut upstream = None;
    let mut ahead = 0_i64;
    let mut behind = 0_i64;
    for field in bytes.split(|byte| *byte == 0) {
        let line = String::from_utf8_lossy(field);
        if let Some(value) = line.strip_prefix("# branch.oid ") {
            oid = Some(value.to_owned());
        } else if let Some(value) = line.strip_prefix("# branch.head ") {
            head = Some(value.to_owned());
        } else if let Some(value) = line.strip_prefix("# branch.upstream ") {
            upstream = Some(value.to_owned());
        } else if let Some(value) = line.strip_prefix("# branch.ab ") {
            for part in value.split_ascii_whitespace() {
                if let Some(value) = part.strip_prefix('+') {
                    ahead = value.parse().unwrap_or(0);
                } else if let Some(value) = part.strip_prefix('-') {
                    behind = value.parse().unwrap_or(0);
                }
            }
        }
    }
    let detached = head.as_deref() == Some("(detached)");
    json!({
        "head": head,
        "oid": oid,
        "upstream": upstream,
        "ahead": ahead,
        "behind": behind,
        "detached": detached,
    })
}

fn entry(path: &str, original_path: Option<&str>, xy: &str, kind: &str) -> Value {
    let mut chars = xy.chars();
    let index_status = chars.next().unwrap_or('.');
    let worktree_status = chars.next().unwrap_or('.');
    json!({
        "path": path,
        "originalPath": original_path,
        "indexStatus": index_status.to_string(),
        "worktreeStatus": worktree_status.to_string(),
        "kind": kind,
    })
}

fn parse_entries(bytes: &[u8]) -> (Vec<Value>, bool) {
    let fields = bytes
        .split(|byte| *byte == 0)
        .filter(|field| !field.is_empty())
        .collect::<Vec<_>>();
    let mut entries = Vec::new();
    let mut index = 0;
    let mut truncated = false;
    while index < fields.len() {
        if entries.len() >= MAX_ENTRIES {
            truncated = true;
            break;
        }
        let record = String::from_utf8_lossy(fields[index]);
        if let Some(rest) = record.strip_prefix("1 ") {
            let parts = rest.splitn(9, ' ').collect::<Vec<_>>();
            if parts.len() == 9 {
                entries.push(entry(parts[8], None, parts[0], "ordinary"));
            }
        } else if let Some(rest) = record.strip_prefix("2 ") {
            let parts = rest.splitn(10, ' ').collect::<Vec<_>>();
            if parts.len() == 10 {
                let original = fields
                    .get(index + 1)
                    .map(|value| String::from_utf8_lossy(value));
                entries.push(entry(parts[9], original.as_deref(), parts[0], "renamed"));
                index += 1;
            }
        } else if let Some(path) = record.strip_prefix("? ") {
            entries.push(entry(path, None, "??", "untracked"));
        } else if let Some(path) = record.strip_prefix("! ") {
            entries.push(entry(path, None, "!!", "ignored"));
        } else if let Some(rest) = record.strip_prefix("u ") {
            let parts = rest.splitn(11, ' ').collect::<Vec<_>>();
            if parts.len() == 11 {
                entries.push(entry(parts[10], None, parts[0], "unmerged"));
            }
        }
        index += 1;
    }
    (entries, truncated)
}

#[async_trait]
impl Tool for GitStatusTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "git_status".into(),
            description:
                "返回 Git 分支、ahead/behind 以及暂存、修改、删除和未跟踪文件的结构化状态。默认检查工作区根；目标项目位于工作区子目录时，用 path 指定该子目录。"
                    .into(),
            input_schema: json!({
                "type":"object",
                "properties":{
                    "path":{"type":"string","minLength":1,"maxLength":4096}
                },
                "additionalProperties":false
            }),
            permission_level: PermissionLevel::ReadOnly,
            side_effect: SideEffectKind::None,
            default_timeout: Duration::from_secs(15),
        }
    }

    fn validate(&self, input: &Value) -> Result<(), Vec<String>> {
        let map = object(input)?;
        let mut issues = Vec::new();
        reject_unknown_fields(map, &["path"], &mut issues);
        if let Some(path) = map.get("path")
            && (!path.is_string()
                || path
                    .as_str()
                    .is_some_and(|value| value.is_empty() || value.len() > 4096))
        {
            issues.push("path 必须是 1 到 4096 字节的字符串。".into());
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
        let args = vec![
            "--no-optional-locks".to_owned(),
            "status".to_owned(),
            "--porcelain=v2".to_owned(),
            "--branch".to_owned(),
            "-z".to_owned(),
        ];
        let status = match run_git(&workspace, &root, &args, 4 * 1024 * 1024).await {
            Ok(output) if output.status.success() => output,
            Ok(output) => {
                return ToolResult::failure(
                    "GIT_STATUS_ERROR",
                    String::from_utf8_lossy(&output.stderr).trim(),
                    context.started_at,
                    json!({}),
                );
            }
            Err(message) => {
                return ToolResult::failure(
                    "GIT_STATUS_ERROR",
                    message,
                    context.started_at,
                    json!({}),
                );
            }
        };
        let (entries, count_truncated) = parse_entries(&status.stdout);
        let truncated = status.stdout_truncated || count_truncated;
        ToolResult::success(
            json!({
                "branch": parse_branch(&status.stdout),
                "clean": entries.is_empty(),
                "entries": entries,
                "truncated": truncated,
            }),
            context.started_at,
            json!({"repositoryRoot":root}),
        )
    }
}
