//! Skills 技能系统：`SKILL.md` 目录发现、frontmatter 校验与按需加载。
//!
//! 与 Claude Code / opencode 生态目录兼容。技能正文只作为系统提示词
//! 的一部分注入，影响模型工作方式；不改变任何权限、审批或配置边界。
//! 项目目录的技能属于不可信输入，与项目指令采用相同的信任模型。

use std::{
    env, fs,
    path::{Path, PathBuf},
};

use crate::error::{ErrorKind, XduduError, XduduResult};

/// 技能目录名的合法格式：小写字母/数字开头与结尾，中间可含连字符。
const NAME_PATTERN: &str = "^[a-z0-9]([a-z0-9-]{0,62}[a-z0-9])?$";
/// frontmatter 只读取前 512 字节，防止超大元数据注入。
const FRONTMATTER_SCAN_BYTES: usize = 512;
/// 技能正文大小上限。
const MAX_SKILL_BODY_BYTES: u64 = 64 * 1024;
/// description 长度上限。
const MAX_DESCRIPTION_CHARS: usize = 1024;

/// 单个技能：来自 `SKILL.md`，正文不含 frontmatter。
#[derive(Debug, Clone)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub body: String,
    /// 来源层级（如 `project/.xdudu/skills`、`user/.claude/skills`），用于展示。
    pub source_label: String,
    pub path: PathBuf,
}

/// 目录发现优先级（第一个命中者优先）。
/// 用户级路径以家目录为根：`~/.config/xdudu/skills`、`~/.claude/skills`、
/// `~/.config/opencode/skills`。
const DISCOVERY_LAYERS: [(&str, &str); 6] = [
    ("project", ".xdudu/skills"),
    ("project", ".claude/skills"),
    ("project", ".opencode/skills"),
    ("user", ".config/xdudu/skills"),
    ("user", ".claude/skills"),
    ("user", ".config/opencode/skills"),
];

/// 用户级技能发现的家目录根（`~`）。
pub fn user_skill_root() -> XduduResult<PathBuf> {
    let home = env::var_os("HOME").ok_or_else(|| {
        XduduError::new(ErrorKind::ConfigError, "无法确定用户主目录：HOME 未设置。")
    })?;
    Ok(PathBuf::from(home))
}

fn valid_name(name: &str) -> bool {
    let Ok(regex) = regex::Regex::new(NAME_PATTERN) else {
        return false;
    };
    regex.is_match(name)
}

/// 解析 `SKILL.md` 的 YAML frontmatter：返回 `(name, description, 结束字节偏移)`。
///
/// 只识别 `name` 与 `description` 两个必填字段；前 512 字节内必须出现
/// 闭合的 `---`。解析失败返回带原因的错误字符串。
fn parse_frontmatter(head: &str) -> Result<(String, String, usize), String> {
    let head = head.trim_start();
    let Some(rest) = head.strip_prefix("---\n") else {
        return Err("必须以 YAML frontmatter（---）开头。".into());
    };
    let end = rest
        .find("\n---")
        .ok_or_else(|| format!("frontmatter 未在 {FRONTMATTER_SCAN_BYTES} 字节内闭合。"))?;
    let body = &rest[..end];
    let mut name = None;
    let mut description = None;
    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some((key, value)) = trimmed.split_once(':') {
            let value = value.trim().trim_matches('"').trim_matches('\'').trim();
            match key.trim() {
                "name" => name = Some(value.to_owned()),
                "description" => description = Some(value.to_owned()),
                // license / metadata 等字段忽略，不参与校验。
                _ => {}
            }
        }
    }
    let name = name.ok_or_else(|| "frontmatter 缺少必填字段 name。".to_owned())?;
    if !valid_name(&name) {
        return Err(format!("name“{name}”不符合 {NAME_PATTERN} 格式。"));
    }
    let description = description
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "frontmatter 缺少必填字段 description。".to_owned())?;
    if description.chars().count() > MAX_DESCRIPTION_CHARS {
        return Err(format!("description 超过 {MAX_DESCRIPTION_CHARS} 字符。"));
    }
    // frontmatter 总字节数："---\n"（4）+ body 到 end（含结束换行前的全部）
    // + "\n---\n"（4）。end 指向 body 末尾的换行位置，故 offset = end + 9。
    let end_offset = end + 9;
    Ok((name, description, end_offset))
}

/// 从单个技能目录加载全部合法技能；无效项跳过并记录警告。
fn load_layer(
    scope_label: &str,
    layer_dir: PathBuf,
    skills: &mut Vec<Skill>,
    names: &mut std::collections::HashSet<String>,
    warnings: &mut Vec<String>,
) {
    let Ok(entries) = fs::read_dir(&layer_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let dir_path = entry.path();
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if !valid_name(&name) {
            warnings.push(format!("技能目录名“{name}”不符合命名规范，已跳过。"));
            continue;
        }
        if !dir_path.is_dir() {
            continue;
        }
        if names.contains(&name) {
            continue;
        }
        let skill_path = dir_path.join("SKILL.md");
        if skill_path.is_symlink() {
            warnings.push(format!("技能 {name} 的 SKILL.md 是符号链接，已跳过。"));
            continue;
        }
        let raw = match fs::read(&skill_path) {
            Ok(raw) => raw,
            Err(_) => continue,
        };
        if raw.len() as u64 > MAX_SKILL_BODY_BYTES {
            warnings.push(format!("技能 {name} 正文超过 64 KiB，已跳过。"));
            continue;
        }
        let head = String::from_utf8_lossy(&raw[..raw.len().min(FRONTMATTER_SCAN_BYTES)]);
        match parse_frontmatter(&head) {
            Ok((front_name, description, end_offset)) if front_name == name => {
                let body = raw
                    .get(end_offset..)
                    .map(String::from_utf8_lossy)
                    .unwrap_or_default()
                    .trim()
                    .to_owned();
                if body.is_empty() {
                    warnings.push(format!("技能 {name} 正文为空，已跳过。"));
                    continue;
                }
                skills.push(Skill {
                    name,
                    description,
                    body,
                    source_label: format!("{scope_label}/{}", layer_dir.display()),
                    path: skill_path,
                });
                names.insert(front_name);
            }
            Ok(_) => {
                warnings.push(format!(
                    "技能 {name} 的 frontmatter name 与目录名不一致，已跳过。"
                ));
            }
            Err(reason) => {
                warnings.push(format!("技能 {name} frontmatter 无效：{reason}"));
            }
        }
    }
}

/// 按优先级发现全部可用技能（同名取首个命中者），返回技能列表与警告。
pub fn discover_skills(cwd: &Path) -> (Vec<Skill>, Vec<String>) {
    discover_skills_with_user_root(cwd, user_skill_root().ok())
}

/// 可注入用户技能根的发现入口（测试使用，避免触碰全局环境变量）。
fn discover_skills_with_user_root(
    cwd: &Path,
    user_root: Option<PathBuf>,
) -> (Vec<Skill>, Vec<String>) {
    let mut skills = Vec::new();
    let mut names = std::collections::HashSet::new();
    let mut warnings = Vec::new();
    for (scope, layer) in DISCOVERY_LAYERS {
        let base = if scope == "project" {
            cwd.to_path_buf()
        } else {
            match &user_root {
                Some(root) => root.clone(),
                None => continue,
            }
        };
        load_layer(
            scope,
            base.join(layer),
            &mut skills,
            &mut names,
            &mut warnings,
        );
    }
    skills.sort_by(|left, right| left.name.cmp(&right.name));
    (skills, warnings)
}

/// 按名称查找技能（区分大小写）。
pub fn find_skill<'a>(skills: &'a [Skill], name: &str) -> Option<&'a Skill> {
    skills.iter().find(|skill| skill.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn frontmatter_解析与边界校验() {
        assert_eq!(
            parse_frontmatter("---\nname: git-release\ndescription: 发布流程\n---\n正文")
                .unwrap()
                .0,
            "git-release"
        );
        assert_eq!(
            parse_frontmatter("---\nname: git-release\ndescription: 发布流程\n---\n正文")
                .unwrap()
                .1,
            "发布流程"
        );
        // 结束偏移指向正文开始。
        let input = "---\nname: foo\ndescription: bar\n---\n正文 abc";
        let (_, _, offset) = parse_frontmatter(input).unwrap();
        assert_eq!(&input[offset..], "正文 abc");
        assert!(parse_frontmatter("无 frontmatter").is_err());
        assert!(parse_frontmatter("---\ndescription: 缺 name\n---\n").is_err());
        assert!(parse_frontmatter("---\nname: Bad_Name\ndescription: x\n---\n").is_err());
        assert!(parse_frontmatter("---\nname: ok\ndescription: \n---\n").is_err());
        assert!(
            parse_frontmatter(&format!(
                "---\nname: ok\ndescription: {}\n---\n",
                "x".repeat(1100)
            ))
            .is_err()
        );
    }

    #[test]
    fn 发现按优先级去重且非法技能跳过() {
        let root = tempdir().unwrap();
        let user_root = tempdir().unwrap();
        let write = |dir: &Path, name: &str, description: &str| {
            let skill_dir = dir.join(name);
            fs::create_dir_all(&skill_dir).unwrap();
            fs::write(
                skill_dir.join("SKILL.md"),
                format!("---\nname: {name}\ndescription: {description}\n---\n正文 {name}"),
            )
            .unwrap();
        };

        // 项目级 .xdudu/skills 与 .claude/skills 同名 → 前者优先。
        write(&root.path().join(".xdudu/skills"), "probe", "项目级");
        write(&root.path().join(".claude/skills"), "probe", "claude 级");
        write(
            &root.path().join(".opencode/skills"),
            "other",
            "opencode 级",
        );
        // 用户级。
        write(
            &user_root.path().join(".config/xdudu/skills"),
            "user-skill",
            "用户级",
        );
        // 非法：名字不匹配、frontmatter 无效。
        fs::create_dir_all(root.path().join(".xdudu/skills/Bad_Name")).unwrap();
        fs::write(
            root.path().join(".xdudu/skills/Bad_Name/SKILL.md"),
            "---\nname: Bad_Name\ndescription: x\n---\n",
        )
        .unwrap();
        fs::create_dir_all(root.path().join(".xdudu/skills/broken")).unwrap();
        fs::write(
            root.path().join(".xdudu/skills/broken/SKILL.md"),
            "没有 frontmatter",
        )
        .unwrap();

        let (skills, warnings) =
            discover_skills_with_user_root(root.path(), Some(user_root.path().to_path_buf()));
        let names: Vec<_> = skills.iter().map(|skill| skill.name.as_str()).collect();
        assert_eq!(names, vec!["other", "probe", "user-skill"]);
        let probe = find_skill(&skills, "probe").unwrap();
        assert!(probe.source_label.starts_with("project"));
        assert_eq!(probe.description, "项目级");
        assert_eq!(probe.body, "正文 probe");
        assert!(warnings.iter().any(|warning| warning.contains("Bad_Name")));
        assert!(warnings.iter().any(|warning| warning.contains("broken")));
    }

    #[test]
    fn 符号链接与超大正文被跳过() {
        let root = tempdir().unwrap();
        #[cfg(unix)]
        {
            fs::create_dir_all(root.path().join(".xdudu/skills/link-skill")).unwrap();
            std::os::unix::fs::symlink(
                "/etc/hosts",
                root.path().join(".xdudu/skills/link-skill/SKILL.md"),
            )
            .unwrap();
        }
        let big = root.path().join(".xdudu/skills/big-skill");
        fs::create_dir_all(&big).unwrap();
        fs::write(
            big.join("SKILL.md"),
            format!(
                "---\nname: big-skill\ndescription: big\n---\n{}",
                "x".repeat(70 * 1024)
            ),
        )
        .unwrap();
        // 注入 None 用户根：隔离本机 ~/.config/xdudu/skills 的真实技能。
        let (skills, warnings) = discover_skills_with_user_root(root.path(), None);
        assert!(skills.is_empty());
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("超过 64 KiB")),
            "{warnings:?}"
        );
    }
}
