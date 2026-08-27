//! Git 专用工具共享的可执行文件、仓库边界和有界输出逻辑。

use std::{
    path::{Path, PathBuf},
    process::Stdio,
};

use tokio::{
    io::{AsyncReadExt, BufReader},
    process::Command,
};

pub struct GitOutput {
    pub status: std::process::ExitStatus,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub stdout_truncated: bool,
}

async fn safe_git_executable(workspace: &Path) -> Result<PathBuf, String> {
    let workspace = tokio::fs::canonicalize(workspace)
        .await
        .map_err(|error| format!("工作区无效：{error}"))?;
    let path = std::env::var_os("PATH").ok_or_else(|| "PATH 未设置。".to_owned())?;
    for directory in std::env::split_paths(&path) {
        if !directory.is_absolute() {
            continue;
        }
        let directory = match tokio::fs::canonicalize(directory).await {
            Ok(directory) => directory,
            Err(_) => continue,
        };
        if directory.starts_with(&workspace) {
            continue;
        }
        let candidate = directory.join(if cfg!(windows) { "git.exe" } else { "git" });
        let candidate = match tokio::fs::canonicalize(candidate).await {
            Ok(candidate) => candidate,
            Err(_) => continue,
        };
        let metadata = match tokio::fs::metadata(&candidate).await {
            Ok(metadata) => metadata,
            Err(_) => continue,
        };
        if metadata.is_file() {
            return Ok(candidate);
        }
    }
    Err("找不到工作区外可信的 Git 可执行文件。".into())
}

pub async fn run_git(
    workspace: &Path,
    cwd: &Path,
    args: &[String],
    max_stdout: usize,
) -> Result<GitOutput, String> {
    let executable = safe_git_executable(workspace).await?;
    let mut command = Command::new(executable);
    command
        .args(["-c", "core.fsmonitor=false"])
        .args(args)
        .current_dir(cwd)
        .env("GIT_OPTIONAL_LOCKS", "0")
        .env("GIT_PAGER", "cat")
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_COMMON_DIR")
        .env_remove("GIT_INDEX_FILE")
        .env_remove("GIT_OBJECT_DIRECTORY")
        .env_remove("GIT_ALTERNATE_OBJECT_DIRECTORIES")
        .env_remove("GIT_CEILING_DIRECTORIES")
        .env_remove("GIT_DISCOVERY_ACROSS_FILESYSTEM")
        .env_remove("GIT_CONFIG_SYSTEM")
        .env_remove("GIT_CONFIG_GLOBAL")
        .env_remove("GIT_CONFIG_PARAMETERS")
        .env_remove("GIT_CONFIG_COUNT")
        .env_remove("GIT_EXTERNAL_DIFF")
        .env_remove("GIT_DIFF_OPTS")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let mut child = command
        .spawn()
        .map_err(|error| format!("启动 Git 失败：{error}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "无法读取 Git stdout。".to_owned())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "无法读取 Git stderr。".to_owned())?;
    let stdout_task = tokio::spawn(async move {
        let mut reader = BufReader::new(stdout).take(max_stdout.saturating_add(1) as u64);
        let mut bytes = Vec::new();
        reader
            .read_to_end(&mut bytes)
            .await
            .map(|_| bytes)
            .map_err(|error| error.to_string())
    });
    let stderr_task = tokio::spawn(async move {
        let mut reader = BufReader::new(stderr).take(64 * 1024);
        let mut bytes = Vec::new();
        reader
            .read_to_end(&mut bytes)
            .await
            .map(|_| bytes)
            .map_err(|error| error.to_string())
    });
    let status = child
        .wait()
        .await
        .map_err(|error| format!("等待 Git 失败：{error}"))?;
    let mut stdout = stdout_task
        .await
        .map_err(|error| format!("读取 Git stdout 任务失败：{error}"))??;
    let stderr = stderr_task
        .await
        .map_err(|error| format!("读取 Git stderr 任务失败：{error}"))??;
    let stdout_truncated = stdout.len() > max_stdout;
    stdout.truncate(max_stdout);
    Ok(GitOutput {
        status,
        stdout,
        stderr,
        stdout_truncated,
    })
}

/// 从 `start` 目录向上发现 Git 仓库；工作区边界仍以 `workspace` 为准，
/// 仓库根和元数据目录都必须位于工作区内。用于目标项目位于工作区子目录、
/// 而工作区根本身不是 Git 仓库的场景。
pub async fn repository_root_at(
    workspace: &Path,
    start: &Path,
) -> Result<(PathBuf, PathBuf), String> {
    let workspace = tokio::fs::canonicalize(workspace)
        .await
        .map_err(|error| format!("工作区无效：{error}"))?;
    let start = tokio::fs::canonicalize(start)
        .await
        .map_err(|error| format!("起始目录无效：{error}"))?;
    if !start.starts_with(&workspace) {
        return Err("起始目录位于 XDUDU 工作区外，已拒绝访问。".into());
    }
    let args = vec![
        "--no-optional-locks".to_owned(),
        "rev-parse".to_owned(),
        "--show-toplevel".to_owned(),
    ];
    let output = run_git(&workspace, &start, &args, 16 * 1024).await?;
    if !output.status.success() {
        return Err(format!(
            "当前目录不是 Git 仓库：{}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let text =
        String::from_utf8(output.stdout).map_err(|_| "Git 仓库路径不是有效 UTF-8。".to_owned())?;
    let root = tokio::fs::canonicalize(text.trim())
        .await
        .map_err(|error| format!("无法解析 Git 仓库根目录：{error}"))?;
    if !root.starts_with(&workspace) {
        return Err("Git 仓库根目录位于 XDUDU 工作区外，已拒绝访问。".into());
    }
    let git_dir_args = vec![
        "--no-optional-locks".to_owned(),
        "rev-parse".to_owned(),
        "--absolute-git-dir".to_owned(),
    ];
    let git_dir_output = run_git(&workspace, &root, &git_dir_args, 16 * 1024).await?;
    if !git_dir_output.status.success() {
        return Err(format!(
            "无法读取 Git 元数据目录：{}",
            String::from_utf8_lossy(&git_dir_output.stderr).trim()
        ));
    }
    let git_dir_text = String::from_utf8(git_dir_output.stdout)
        .map_err(|_| "Git 元数据目录不是有效 UTF-8。".to_owned())?;
    let git_dir = tokio::fs::canonicalize(git_dir_text.trim())
        .await
        .map_err(|error| format!("无法解析 Git 元数据目录：{error}"))?;
    if !git_dir.starts_with(&workspace) {
        return Err("Git 元数据目录位于 XDUDU 工作区外，已拒绝访问。".into());
    }
    Ok((workspace, root))
}

pub fn safe_relative_path(value: &str) -> bool {
    let path = Path::new(value);
    !value.is_empty()
        && value.len() <= 4096
        && !path.is_absolute()
        && !path
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[tokio::test]
    async fn 拒绝元数据目录位于工作区外的仓库() {
        let parent = tempdir().unwrap();
        let workspace = parent.path().join("workspace");
        let git_dir = parent.path().join("external.git");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        let status = std::process::Command::new("git")
            .args([
                "init",
                "--quiet",
                "--separate-git-dir",
                git_dir.to_str().unwrap(),
                workspace.to_str().unwrap(),
            ])
            .status()
            .unwrap();
        assert!(status.success());
        let error = repository_root_at(&workspace, &workspace)
            .await
            .unwrap_err();
        assert!(error.contains("元数据目录位于 XDUDU 工作区外"));
    }

    #[tokio::test]
    async fn 从工作区子目录发现嵌套仓库() {
        let parent = tempdir().unwrap();
        let workspace = parent.path().join("workspace");
        let project = workspace.join("projects").join("target");
        tokio::fs::create_dir_all(&project).await.unwrap();
        let status = std::process::Command::new("git")
            .args(["init", "--quiet", project.to_str().unwrap()])
            .status()
            .unwrap();
        assert!(status.success());
        let workspace = tokio::fs::canonicalize(&workspace).await.unwrap();
        let project = tokio::fs::canonicalize(&project).await.unwrap();
        let (resolved_workspace, root) = repository_root_at(&workspace, &project)
            .await
            .expect("应发现嵌套仓库");
        assert_eq!(resolved_workspace, workspace);
        assert_eq!(root, project);
    }

    #[tokio::test]
    async fn 工作区根非仓库时从子目录出发会失败() {
        let parent = tempdir().unwrap();
        let workspace = parent.path().join("workspace");
        let project = workspace.join("projects").join("target");
        tokio::fs::create_dir_all(&project).await.unwrap();
        let error = repository_root_at(&workspace, &project).await.unwrap_err();
        assert!(error.contains("不是 Git 仓库"));
    }
}
