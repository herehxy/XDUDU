//! 正常终端上的差量活动区渲染。
//!
//! XDUDU 不进入 alternate screen：已完成的消息交给终端原生滚动缓冲，
//! 只有底部的 Composer、工具进度和状态栏需要重绘。本模块借鉴 Codex
//! TUI 的 double-buffer 思路，但保持实现独立：每一行先生成完整帧，
//! 再只写入发生变化的行，避免长回答过程中清屏和闪烁。

use std::io::{self, Write};

use crossterm::{
    cursor::MoveTo,
    queue,
    terminal::{Clear, ClearType},
};

#[derive(Debug, Default)]
pub(crate) struct InlineActivityDiff {
    previous_top: Option<u16>,
    previous_rows: Vec<String>,
    previous_width: u16,
}

impl InlineActivityDiff {
    pub(crate) fn reset(&mut self, out: &mut impl Write) -> io::Result<()> {
        if let Some(top) = self.previous_top {
            for offset in 0..self.previous_rows.len() {
                queue!(
                    out,
                    MoveTo(0, top.saturating_add(offset as u16)),
                    Clear(ClearType::CurrentLine)
                )?;
            }
        }
        self.previous_top = None;
        self.previous_rows.clear();
        self.previous_width = 0;
        Ok(())
    }

    pub(crate) fn render(
        &mut self,
        out: &mut impl Write,
        top: u16,
        rows: &[String],
        width: u16,
    ) -> io::Result<()> {
        let same_origin = self.previous_top == Some(top) && self.previous_width == width;
        let common = if same_origin {
            self.previous_rows.len().min(rows.len())
        } else {
            if let Some(previous_top) = self.previous_top {
                for offset in 0..self.previous_rows.len() {
                    queue!(
                        out,
                        MoveTo(0, previous_top.saturating_add(offset as u16)),
                        Clear(ClearType::CurrentLine)
                    )?;
                }
            }
            0
        };
        for (index, row) in rows.iter().enumerate() {
            if index >= common || self.previous_rows[index] != *row {
                queue!(
                    out,
                    MoveTo(0, top.saturating_add(index as u16)),
                    Clear(ClearType::CurrentLine),
                    crossterm::style::Print(row)
                )?;
            }
        }
        for index in rows.len()..self.previous_rows.len() {
            queue!(
                out,
                MoveTo(0, top.saturating_add(index as u16)),
                Clear(ClearType::CurrentLine)
            )?;
        }
        self.previous_top = Some(top);
        self.previous_rows.clear();
        self.previous_rows.extend_from_slice(rows);
        self.previous_width = width;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn snapshot(&self) -> (&[String], Option<u16>) {
        (&self.previous_rows, self.previous_top)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 首帧绘制后相同帧不重复写入() {
        let mut diff = InlineActivityDiff::default();
        let mut out = Vec::new();
        diff.render(&mut out, 10, &["状态".into(), "输入".into()], 80)
            .unwrap();
        let first = out.len();
        diff.render(&mut out, 10, &["状态".into(), "输入".into()], 80)
            .unwrap();
        assert_eq!(out.len(), first);
    }

    #[test]
    fn 行变化和缩短会清理残影() {
        let mut diff = InlineActivityDiff::default();
        let mut out = Vec::new();
        diff.render(&mut out, 4, &["一".into(), "二".into(), "三".into()], 80)
            .unwrap();
        out.clear();
        diff.render(&mut out, 4, &["一".into()], 80).unwrap();
        assert!(!out.is_empty());
        assert_eq!(diff.snapshot().0, &["一".to_owned()]);
    }

    #[test]
    fn 终端调整大小后完整重绘新位置() {
        let mut diff = InlineActivityDiff::default();
        let mut out = Vec::new();
        diff.render(&mut out, 4, &["内容".into()], 80).unwrap();
        out.clear();
        diff.render(&mut out, 8, &["内容".into()], 120).unwrap();
        assert!(!out.is_empty());
        assert_eq!(diff.snapshot().1, Some(8));
    }
}
