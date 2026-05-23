//! Fast byte-offset to line/snippet mapping for the experimental Rust core.
//!
//! `LineIndex` stores line-start byte offsets for a single file buffer. It lets
//! search code convert byte offsets to 1-based line numbers and snippets without
//! rescanning all earlier bytes for every match.

/// Byte-offset line index for a single file buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineIndex {
    line_starts: Vec<usize>,
    len: usize,
}

impl LineIndex {
    /// Build a line index from file bytes.
    pub fn new(bytes: &[u8]) -> Self {
        let mut line_starts = vec![0];
        for (index, byte) in bytes.iter().enumerate() {
            if *byte == b'\n' && index + 1 < bytes.len() {
                line_starts.push(index + 1);
            }
        }

        Self {
            line_starts,
            len: bytes.len(),
        }
    }

    /// Number of lines represented by this index.
    pub fn line_count(&self) -> usize {
        self.line_starts.len()
    }

    /// Return the 1-based line number containing `byte_offset`.
    pub fn line_number(&self, byte_offset: usize) -> usize {
        let offset = byte_offset.min(self.len);
        match self.line_starts.binary_search(&offset) {
            Ok(index) => index + 1,
            Err(index) => index.max(1),
        }
    }

    /// Return the byte range for the 1-based `line_number`.
    pub fn line_range(&self, bytes: &[u8], line_number: usize) -> Option<(usize, usize)> {
        if line_number == 0 || line_number > self.line_starts.len() {
            return None;
        }

        let start = self.line_starts[line_number - 1];
        let end = self
            .line_starts
            .get(line_number)
            .map(|next_start| next_start.saturating_sub(1))
            .unwrap_or_else(|| bytes.len())
            .min(bytes.len());
        let end = trim_line_ending(bytes, start, end);
        Some((start, end))
    }

    /// Return a trimmed one-line snippet for `byte_offset`.
    pub fn snippet_at(&self, bytes: &[u8], byte_offset: usize, max_chars: usize) -> String {
        let line = self.line_number(byte_offset);
        let Some((start, end)) = self.line_range(bytes, line) else {
            return String::new();
        };

        let snippet = String::from_utf8_lossy(&bytes[start..end]);
        let trimmed = snippet.trim();
        if trimmed.chars().count() <= max_chars {
            return trimmed.to_string();
        }

        trimmed.chars().take(max_chars).collect()
    }
}

fn trim_line_ending(bytes: &[u8], start: usize, mut end: usize) -> usize {
    while end > start && matches!(bytes[end - 1], b'\n' | b'\r') {
        end -= 1;
    }
    end
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_index_maps_offsets_to_one_based_lines() {
        let bytes = b"alpha\nbeta\ngamma";
        let index = LineIndex::new(bytes);

        assert_eq!(index.line_count(), 3);
        assert_eq!(index.line_number(0), 1);
        assert_eq!(index.line_number(5), 1);
        assert_eq!(index.line_number(6), 2);
        assert_eq!(index.line_number(10), 2);
        assert_eq!(index.line_number(11), 3);
        assert_eq!(index.line_number(999), 3);
    }

    #[test]
    fn line_range_trims_lf_and_crlf() {
        let bytes = b"alpha\r\nbeta\ngamma";
        let index = LineIndex::new(bytes);

        assert_eq!(index.line_range(bytes, 1), Some((0, 5)));
        assert_eq!(index.line_range(bytes, 2), Some((7, 11)));
        assert_eq!(index.line_range(bytes, 3), Some((12, 17)));
        assert_eq!(index.line_range(bytes, 0), None);
        assert_eq!(index.line_range(bytes, 4), None);
    }

    #[test]
    fn snippet_at_returns_trimmed_and_limited_text() {
        let bytes = b"first\n    second line has words\nthird";
        let index = LineIndex::new(bytes);

        assert_eq!(index.snippet_at(bytes, 10, 100), "second line has words");
        assert_eq!(index.snippet_at(bytes, 10, 6), "second");
    }

    #[test]
    fn empty_file_has_one_empty_line() {
        let bytes = b"";
        let index = LineIndex::new(bytes);

        assert_eq!(index.line_count(), 1);
        assert_eq!(index.line_number(0), 1);
        assert_eq!(index.line_range(bytes, 1), Some((0, 0)));
    }
}
