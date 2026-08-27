use std::{
    fs,
    io::{Read, Write},
    net::TcpListener,
    process::{Command, Stdio},
    thread,
    time::Duration,
};

use serde_json::json;
use tempfile::tempdir;

fn xdudu() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_xdudu"));
    command.env("XDUDU_PROVIDER", "anthropic");
    command.env(
        "XDUDU_CONFIG_HOME",
        std::env::temp_dir().join(format!("xdudu-cli-tests-{}", std::process::id())),
    );
    command
}

fn session_list(cwd: &std::path::Path) -> String {
    let output = xdudu()
        .current_dir(cwd)
        .args(["session", "list"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

fn first_session_id(cwd: &std::path::Path) -> String {
    session_list(cwd)
        .lines()
        .skip(1)
        .find_map(|line| line.split_whitespace().next())
        .expect("应至少存在一个会话")
        .to_owned()
}

fn session_show(cwd: &std::path::Path, id: &str) -> serde_json::Value {
    let output = xdudu()
        .current_dir(cwd)
        .args(["session", "show", id])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn read_http_request(stream: &mut std::net::TcpStream) {
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut request = Vec::new();
    let mut buffer = [0_u8; 4096];
    let mut expected_length = None;
    loop {
        let read = stream.read(&mut buffer).unwrap();
        if read == 0 {
            break;
        }
        request.extend_from_slice(&buffer[..read]);
        if let Some(header_end) = request.windows(4).position(|part| part == b"\r\n\r\n") {
            let header_end = header_end + 4;
            if expected_length.is_none() {
                let headers = String::from_utf8_lossy(&request[..header_end]);
                expected_length = headers.lines().find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length:")?
                        .trim()
                        .parse::<usize>()
                        .ok()
                });
            }
            if request.len() >= header_end + expected_length.unwrap_or(0) {
                break;
            }
        }
    }
}

fn write_http_json(stream: &mut std::net::TcpStream, body: serde_json::Value) {
    let body = serde_json::to_string(&body).unwrap();
    write!(
        stream,
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        body.len(),
        body
    )
    .unwrap();
}

fn write_http_sse(stream: &mut std::net::TcpStream, body: &str) {
    write!(
        stream,
        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        body.len(),
        body
    )
    .unwrap();
}

fn two_turn_anthropic_server() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    thread::spawn(move || {
        let (mut first, _) = listener.accept().unwrap();
        read_http_request(&mut first);
        write_http_json(
            &mut first,
            json!({
                "content":[{"type":"tool_use","id":"call-1","name":"file_read","input":{"path":"fixture.txt"}}],
                "stop_reason":"tool_use",
                "usage":{"input_tokens":1,"output_tokens":1}
            }),
        );
        let (mut second, _) = listener.accept().unwrap();
        read_http_request(&mut second);
        write_http_json(
            &mut second,
            json!({
                "content":[{"type":"text","text":"Rust CLI E2E 完成"}],
                "stop_reason":"end_turn",
                "usage":{"input_tokens":2,"output_tokens":2}
            }),
        );
    });
    format!("http://{address}")
}

fn one_turn_anthropic_server(text: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        read_http_request(&mut stream);
        write_http_json(
            &mut stream,
            json!({
                "content":[{"type":"text","text":text}],
                "stop_reason":"end_turn",
                "usage":{"input_tokens":1,"output_tokens":1}
            }),
        );
    });
    format!("http://{address}")
}

fn plan_anthropic_server() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        read_http_request(&mut stream);
        write_http_json(
            &mut stream,
            json!({
                "content":[{
                    "type":"tool_use",
                    "id":"plan-1",
                    "name":"submit_plan",
                    "input":{
                        "steps":[{
                            "key":"verify",
                            "title":"验证实现",
                            "description":"运行相关质量检查",
                            "dependencies":[],
                            "completionCriteria":["检查全部通过"]
                        }]
                    }
                }],
                "stop_reason":"tool_use",
                "usage":{"input_tokens":4,"output_tokens":8}
            }),
        );
    });
    format!("http://{address}")
}

fn complete_plan_anthropic_server() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        read_http_request(&mut stream);
        write_http_json(
            &mut stream,
            json!({
                "content":[{
                    "type":"tool_use",
                    "id":"complete-1",
                    "name":"complete_step",
                    "input":{
                        "summary":"质量检查完成",
                        "evidence":[{
                            "criterionIndex":1,
                            "evidence":"模拟验收确认检查全部通过"
                        }]
                    }
                }],
                "stop_reason":"tool_use",
                "usage":{"input_tokens":3,"output_tokens":5}
            }),
        );
    });
    format!("http://{address}")
}

fn two_turn_anthropic_sse_server() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    thread::spawn(move || {
        let (mut first, _) = listener.accept().unwrap();
        read_http_request(&mut first);
        write_http_sse(
            &mut first,
            concat!(
                "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1}}}\n\n",
                "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call-1\",\"name\":\"file_read\",\"input\":{}}}\n\n",
                "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\":\\\"fixture.txt\\\"}\"}}\n\n",
                "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
                "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":1}}\n\n",
                "data: {\"type\":\"message_stop\"}\n\n",
            ),
        );
        let (mut second, _) = listener.accept().unwrap();
        read_http_request(&mut second);
        write_http_sse(
            &mut second,
            concat!(
                "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":2}}}\n\n",
                "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
                "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Rust CLI \"}}\n\n",
                "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"SSE 完成\"}}\n\n",
                "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
                "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n",
                "data: {\"type\":\"message_stop\"}\n\n",
            ),
        );
    });
    format!("http://{address}")
}

fn two_turn_anthropic_write_server() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    thread::spawn(move || {
        let (mut first, _) = listener.accept().unwrap();
        read_http_request(&mut first);
        write_http_json(
            &mut first,
            json!({
                "content":[{
                    "type":"tool_use",
                    "id":"write-1",
                    "name":"file_write",
                    "input":{"path":"created.txt","content":"approval test"}
                }],
                "stop_reason":"tool_use",
                "usage":{"input_tokens":1,"output_tokens":1}
            }),
        );
        let (mut second, _) = listener.accept().unwrap();
        read_http_request(&mut second);
        write_http_json(
            &mut second,
            json!({
                "content":[{"type":"text","text":"写入流程结束"}],
                "stop_reason":"end_turn",
                "usage":{"input_tokens":2,"output_tokens":2}
            }),
        );
    });
    format!("http://{address}")
}

#[test]
fn help_可以在没有_api_key_时运行() {
    let output = xdudu().arg("--help").output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("终端原生 AI 编程助手"));
    assert!(stdout.contains("--permission"));
    assert!(stdout.contains("--approval"));
    assert!(!stdout.contains("--tui"));
    assert!(stdout.contains("undo"));
    assert!(stdout.contains("approval"));
}

#[test]
fn approval_可以列出撤销和清除永久规则() {
    let dir = tempdir().unwrap();
    let config_home = dir.path().join("config-home");
    fs::create_dir_all(&config_home).unwrap();
    fs::write(
        config_home.join("approval-rules.json"),
        serde_json::to_vec_pretty(&json!({
            "schemaVersion": 1,
            "rules": [{
                "toolName": "web_fetch",
                "sideEffect": "network-access"
            }]
        }))
        .unwrap(),
    )
    .unwrap();

    let listed = xdudu()
        .current_dir(dir.path())
        .env("XDUDU_CONFIG_HOME", &config_home)
        .args(["approval", "list"])
        .output()
        .unwrap();
    assert!(listed.status.success());
    assert!(String::from_utf8_lossy(&listed.stdout).contains("web_fetch"));

    let revoked = xdudu()
        .current_dir(dir.path())
        .env("XDUDU_CONFIG_HOME", &config_home)
        .args(["approval", "revoke", "web_fetch"])
        .output()
        .unwrap();
    assert!(revoked.status.success());
    assert!(String::from_utf8_lossy(&revoked.stdout).contains("已撤销"));

    let cleared = xdudu()
        .current_dir(dir.path())
        .env("XDUDU_CONFIG_HOME", &config_home)
        .args(["approval", "clear"])
        .output()
        .unwrap();
    assert!(cleared.status.success());
    assert!(String::from_utf8_lossy(&cleared.stdout).contains("已清除 0 条"));
}

#[test]
fn 非法权限模式返回参数退出码() {
    let output = xdudu()
        .args(["--permission", "unknown", "test"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("非法权限模式"));
}

#[test]
fn 未知_provider_返回参数退出码() {
    let output = xdudu()
        .args(["--provider", "unknown", "test"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("不支持的 Provider"));
}

#[test]
fn 缺少_api_key_返回配置退出码和登录指引() {
    let output = xdudu()
        .env("XDUDU_PROVIDER", "anthropic")
        .env_remove("ANTHROPIC_API_KEY")
        .arg("test")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("ANTHROPIC_API_KEY"));
    assert!(stderr.contains("xdudu auth login anthropic"));
}

#[test]
fn config_命令无需_api_key_并展示来源() {
    let dir = tempdir().unwrap();
    let output = xdudu()
        .current_dir(dir.path())
        .env_remove("ANTHROPIC_API_KEY")
        .args(["config", "show"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("\"provider\""));
    assert!(stdout.contains("\"sources\""));
    assert!(!stdout.contains("API_KEY"));
}

#[test]
fn config_set_写入项目配置并可解释来源() {
    let dir = tempdir().unwrap();
    let set = xdudu()
        .current_dir(dir.path())
        .args(["config", "set", "agent.max_turns", "31"])
        .output()
        .unwrap();
    assert!(set.status.success());
    let explain = xdudu()
        .current_dir(dir.path())
        .args(["config", "explain", "agent.max_turns"])
        .output()
        .unwrap();
    assert!(explain.status.success());
    let stdout = String::from_utf8_lossy(&explain.stdout);
    assert!(stdout.contains("31"));
    assert!(stdout.contains("project"));
}

#[test]
fn 项目配置不能自动批准副作用() {
    let dir = tempdir().unwrap();
    let output = xdudu()
        .current_dir(dir.path())
        .args(["config", "set", "agent.approval", "always"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("只能把 agent.approval 收紧"));
}

#[test]
fn doctor_无需访问模型即可输出诊断() {
    let dir = tempdir().unwrap();
    let output = xdudu()
        .current_dir(dir.path())
        .env("ANTHROPIC_API_KEY", "test-key")
        .args(["doctor", "--json"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["ok"], true);
    assert!(report["checks"].as_array().unwrap().len() >= 4);
}

#[test]
fn rust_cli_真实进程完成工具循环并保存会话() {
    let dir = tempdir().unwrap();
    fs::write(dir.path().join("fixture.txt"), "fixture content").unwrap();
    let base_url = two_turn_anthropic_server();
    let output = xdudu()
        .current_dir(dir.path())
        .env("ANTHROPIC_API_KEY", "test-key")
        .env("ANTHROPIC_BASE_URL", base_url)
        .env("XDUDU_NO_STREAM", "true")
        .arg("读取 fixture.txt")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("Rust CLI E2E 完成"));

    assert!(dir.path().join(".xdudu/xdudu.db").exists());
    let id = first_session_id(dir.path());
    let session = session_show(dir.path(), &id);
    assert_eq!(session["status"], "completed");
    assert_eq!(session["toolCalls"][0]["toolName"], "file_read");
    assert_eq!(session["toolCalls"][0]["status"], "succeeded");
}

#[test]
fn session_可以列出显示并恢复会话() {
    let dir = tempdir().unwrap();
    let first = xdudu()
        .current_dir(dir.path())
        .env("ANTHROPIC_API_KEY", "test-key")
        .env(
            "ANTHROPIC_BASE_URL",
            one_turn_anthropic_server("第一轮完成"),
        )
        .args(["--no-stream", "创建会话"])
        .output()
        .unwrap();
    assert!(first.status.success());

    let id = first_session_id(dir.path());
    let before = session_show(dir.path(), &id);
    assert_eq!(before["messages"].as_array().unwrap().len(), 2);

    let resumed = xdudu()
        .current_dir(dir.path())
        .env("ANTHROPIC_API_KEY", "test-key")
        .env("ANTHROPIC_BASE_URL", one_turn_anthropic_server("恢复完成"))
        .args(["--no-stream", "session", "resume", &id, "继续执行"])
        .output()
        .unwrap();
    assert!(
        resumed.status.success(),
        "{}",
        String::from_utf8_lossy(&resumed.stderr)
    );
    assert!(String::from_utf8_lossy(&resumed.stdout).contains("恢复完成"));
    let after = session_show(dir.path(), &id);
    assert_eq!(after["messages"].as_array().unwrap().len(), 4);
}

#[test]
fn 非_tty_plan_生成后保持等待审批() {
    let dir = tempdir().unwrap();
    let mut child = xdudu()
        .current_dir(dir.path())
        .env("ANTHROPIC_API_KEY", "test-key")
        .env("ANTHROPIC_BASE_URL", plan_anthropic_server())
        .arg("--interactive")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all("/plan 完成 M7 验收\n".as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("等待审批"));
    let id = first_session_id(dir.path());
    let session = session_show(dir.path(), &id);
    assert_eq!(session["status"], "waiting_approval");
    assert_eq!(session["currentState"], "WAITING_APPROVAL");

    let listed = xdudu()
        .current_dir(dir.path())
        .env_remove("ANTHROPIC_API_KEY")
        .args(["plan", "list"])
        .output()
        .unwrap();
    assert!(
        listed.status.success(),
        "{}",
        String::from_utf8_lossy(&listed.stderr)
    );
    let plans: serde_json::Value = serde_json::from_slice(&listed.stdout).unwrap();
    assert_eq!(plans[0]["status"], "pending_approval");
    assert_eq!(plans[0]["schemaVersion"], 3);
}

#[test]
fn plan_cli_完成创建批准和执行闭环() {
    let dir = tempdir().unwrap();
    let created = xdudu()
        .current_dir(dir.path())
        .env("ANTHROPIC_API_KEY", "test-key")
        .env("ANTHROPIC_BASE_URL", plan_anthropic_server())
        .args(["plan", "create", "完成 CLI 计划闭环"])
        .output()
        .unwrap();
    assert!(
        created.status.success(),
        "{}",
        String::from_utf8_lossy(&created.stderr)
    );
    let created: serde_json::Value = serde_json::from_slice(&created.stdout).unwrap();
    let plan_id = created["plan"]["id"].as_str().unwrap();

    let approved = xdudu()
        .current_dir(dir.path())
        .env_remove("ANTHROPIC_API_KEY")
        .args(["plan", "approve", plan_id, "--reason", "E2E 批准"])
        .output()
        .unwrap();
    assert!(
        approved.status.success(),
        "{}",
        String::from_utf8_lossy(&approved.stderr)
    );

    let completed = xdudu()
        .current_dir(dir.path())
        .env("ANTHROPIC_API_KEY", "test-key")
        .env("ANTHROPIC_BASE_URL", complete_plan_anthropic_server())
        .args(["plan", "run", plan_id])
        .output()
        .unwrap();
    assert!(
        completed.status.success(),
        "{}",
        String::from_utf8_lossy(&completed.stderr)
    );
    let plan: serde_json::Value = serde_json::from_slice(&completed.stdout).unwrap();
    assert_eq!(plan["status"], "completed");
    assert_eq!(plan["steps"][0]["status"], "completed");
    assert_eq!(plan["steps"][0]["attempts"][0]["attempt"], 1);
    assert!(
        String::from_utf8_lossy(&completed.stderr).contains("证据 1： 模拟验收确认检查全部通过")
    );
}

#[test]
fn json_模式只输出可解析事件且没有横幅() {
    let dir = tempdir().unwrap();
    fs::write(dir.path().join("fixture.txt"), "fixture content").unwrap();
    let output = xdudu()
        .current_dir(dir.path())
        .env("ANTHROPIC_API_KEY", "test-key")
        .env("ANTHROPIC_BASE_URL", two_turn_anthropic_server())
        .args(["--json", "--no-stream", "读取 fixture.txt"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!stdout.contains("XDUDU v"));
    let events = stdout
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert!(events.iter().any(|event| event["type"] == "tool_started"));
    assert!(events.iter().any(|event| event["type"] == "run_completed"));
    assert!(!events.iter().any(|event| event["type"] == "debug_trace"));
}

#[test]
fn 高级模式输出脱敏结构化轨迹且不包含工具正文() {
    let dir = tempdir().unwrap();
    fs::write(dir.path().join("fixture.txt"), "fixture secret body").unwrap();
    let output = xdudu()
        .current_dir(dir.path())
        .env("ANTHROPIC_API_KEY", "test-key")
        .env("ANTHROPIC_BASE_URL", two_turn_anthropic_server())
        .args(["--json", "--debug-trace", "--no-stream", "读取 fixture.txt"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let traces = output
        .stdout
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .filter_map(|line| serde_json::from_slice::<serde_json::Value>(line).ok())
        .filter(|event| event["type"] == "debug_trace")
        .collect::<Vec<_>>();
    assert!(!traces.is_empty());
    assert!(
        traces
            .iter()
            .any(|event| event["phase"] == "provider_response")
    );
    let raw = serde_json::to_string(&traces).unwrap();
    assert!(!raw.contains("fixture secret body"));
    assert!(!raw.contains("toolOutput"));
    assert!(!raw.contains("reasoning_content"));
}

#[test]
fn no_stream_只在完成时打印最终文本() {
    let dir = tempdir().unwrap();
    fs::write(dir.path().join("fixture.txt"), "fixture content").unwrap();
    let output = xdudu()
        .current_dir(dir.path())
        .env("ANTHROPIC_API_KEY", "test-key")
        .env("ANTHROPIC_BASE_URL", two_turn_anthropic_server())
        .args(["--no-stream", "读取 fixture.txt"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout.matches("Rust CLI E2E 完成").count(), 1);
}

#[test]
fn 默认流式模式聚合工具参数并增量输出文本() {
    let dir = tempdir().unwrap();
    fs::write(dir.path().join("fixture.txt"), "fixture content").unwrap();
    let output = xdudu()
        .current_dir(dir.path())
        .env("ANTHROPIC_API_KEY", "test-key")
        .env("ANTHROPIC_BASE_URL", two_turn_anthropic_sse_server())
        .arg("读取 fixture.txt")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout.matches("Rust CLI SSE 完成").count(), 1);
}

#[test]
fn 非交互默认拒绝写入且显式批准可以执行() {
    let denied_dir = tempdir().unwrap();
    let denied = xdudu()
        .current_dir(denied_dir.path())
        .env("ANTHROPIC_API_KEY", "test-key")
        .env("ANTHROPIC_BASE_URL", two_turn_anthropic_write_server())
        .args(["--no-stream", "创建 created.txt"])
        .output()
        .unwrap();
    // 审批拒绝是用户/策略决策，不阻止任务正常收尾（工具仍记为 denied）。
    assert!(denied.status.success());
    assert!(!denied_dir.path().join("created.txt").exists());
    let denied_id = first_session_id(denied_dir.path());
    let denied_session = session_show(denied_dir.path(), &denied_id);
    assert_eq!(denied_session["status"], "completed");
    assert_eq!(denied_session["toolCalls"][0]["status"], "denied");
    assert_eq!(
        denied_session["toolCalls"][0]["approval"]["approved"],
        false
    );

    let allowed_dir = tempdir().unwrap();
    let allowed = xdudu()
        .current_dir(allowed_dir.path())
        .env("ANTHROPIC_API_KEY", "test-key")
        .env("ANTHROPIC_BASE_URL", two_turn_anthropic_write_server())
        .args(["--no-stream", "--approval", "always", "创建 created.txt"])
        .output()
        .unwrap();
    assert!(allowed.status.success());
    assert_eq!(
        fs::read_to_string(allowed_dir.path().join("created.txt")).unwrap(),
        "approval test"
    );
    assert_eq!(
        fs::read_dir(allowed_dir.path().join(".xdudu/changes/json"))
            .unwrap()
            .count(),
        1
    );
    let allowed_id = first_session_id(allowed_dir.path());
    let allowed_session = session_show(allowed_dir.path(), &allowed_id);
    assert_eq!(
        allowed_session["toolCalls"][0]["approval"]["scope"],
        "always"
    );
}

#[test]
fn 永久规则可在后续非交互会话批准同类工具() {
    let dir = tempdir().unwrap();
    let config_home = dir.path().join("config-home");
    fs::create_dir_all(&config_home).unwrap();
    fs::write(
        config_home.join("approval-rules.json"),
        serde_json::to_vec_pretty(&json!({
            "schemaVersion": 1,
            "rules": [{
                "toolName": "file_write",
                "sideEffect": "workspace-write"
            }]
        }))
        .unwrap(),
    )
    .unwrap();
    let output = xdudu()
        .current_dir(dir.path())
        .env("XDUDU_CONFIG_HOME", &config_home)
        .env("ANTHROPIC_API_KEY", "test-key")
        .env("ANTHROPIC_BASE_URL", two_turn_anthropic_write_server())
        .args(["--no-stream", "创建 created.txt"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        fs::read_to_string(dir.path().join("created.txt")).unwrap(),
        "approval test"
    );
    let session = session_show(dir.path(), &first_session_id(dir.path()));
    assert_eq!(session["toolCalls"][0]["approval"]["scope"], "always");
}

#[test]
fn undo_无需_api_key_即可撤销_agent_创建的文件() {
    let dir = tempdir().unwrap();
    let write = xdudu()
        .current_dir(dir.path())
        .env("ANTHROPIC_API_KEY", "test-key")
        .env("ANTHROPIC_BASE_URL", two_turn_anthropic_write_server())
        .args(["--no-stream", "--approval", "always", "创建 created.txt"])
        .output()
        .unwrap();
    assert!(write.status.success());
    assert!(dir.path().join("created.txt").exists());

    let undo = xdudu()
        .current_dir(dir.path())
        .env_remove("ANTHROPIC_API_KEY")
        .arg("undo")
        .output()
        .unwrap();
    assert!(
        undo.status.success(),
        "{}",
        String::from_utf8_lossy(&undo.stderr)
    );
    assert!(!dir.path().join("created.txt").exists());
    assert!(String::from_utf8_lossy(&undo.stdout).contains("已删除由 Agent 创建的文件"));
}

#[test]
fn undo_发现后续人工修改时拒绝覆盖() {
    let dir = tempdir().unwrap();
    let write = xdudu()
        .current_dir(dir.path())
        .env("ANTHROPIC_API_KEY", "test-key")
        .env("ANTHROPIC_BASE_URL", two_turn_anthropic_write_server())
        .args(["--no-stream", "--approval", "always", "创建 created.txt"])
        .output()
        .unwrap();
    assert!(write.status.success());
    fs::write(dir.path().join("created.txt"), "user edit").unwrap();

    let undo = xdudu()
        .current_dir(dir.path())
        .env_remove("ANTHROPIC_API_KEY")
        .arg("undo")
        .output()
        .unwrap();
    assert!(!undo.status.success());
    assert!(String::from_utf8_lossy(&undo.stderr).contains("又发生变化"));
    assert_eq!(
        fs::read_to_string(dir.path().join("created.txt")).unwrap(),
        "user edit"
    );
}

#[test]
fn mcp_cli_管理_http_配置且拒绝远程明文_http() {
    let config_home = tempdir().unwrap();
    let add = xdudu()
        .env("XDUDU_CONFIG_HOME", config_home.path())
        .args([
            "mcp",
            "add-http",
            "team",
            "https://mcp.example.com/mcp",
            "--auth",
        ])
        .output()
        .unwrap();
    assert!(
        add.status.success(),
        "{}",
        String::from_utf8_lossy(&add.stderr)
    );
    let list = xdudu()
        .env("XDUDU_CONFIG_HOME", config_home.path())
        .args(["mcp", "list"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&list.stdout);
    assert!(stdout.contains("team\tenabled\tstreamable-http"));
    let config = fs::read_to_string(config_home.path().join("mcp.toml")).unwrap();
    assert!(!config.contains("Bearer"));

    let insecure = xdudu()
        .env("XDUDU_CONFIG_HOME", config_home.path())
        .args(["mcp", "add-http", "bad", "http://example.com/mcp"])
        .output()
        .unwrap();
    assert!(!insecure.status.success());
    assert!(String::from_utf8_lossy(&insecure.stderr).contains("只允许 HTTPS"));
}

#[test]
fn plugin_cli_拒绝可执行入口并列出声明式插件() {
    let config_home = tempdir().unwrap();
    let plugins = config_home.path().join("plugins");
    fs::create_dir_all(&plugins).unwrap();
    fs::write(
        plugins.join("team.toml"),
        "schemaVersion=1\nid='team'\nname='Team'\nversion='1.0.0'\nenabled=false\n",
    )
    .unwrap();
    let list = xdudu()
        .env("XDUDU_CONFIG_HOME", config_home.path())
        .args(["plugin", "list"])
        .output()
        .unwrap();
    assert!(list.status.success());
    assert!(String::from_utf8_lossy(&list.stdout).contains("team\tdisabled"));

    fs::write(
        plugins.join("bad.toml"),
        "schemaVersion=1\nid='bad'\nname='Bad'\nversion='1'\nentry='payload.so'\n",
    )
    .unwrap();
    let rejected = xdudu()
        .env("XDUDU_CONFIG_HOME", config_home.path())
        .args(["plugin", "list"])
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    let error = String::from_utf8_lossy(&rejected.stderr);
    assert!(error.contains("插件清单") && error.contains("entry"));
}
