//! Experimental newline-delimited JSON-RPC stdio loop.
//!
//! This binary is intentionally minimal and additive. The Python MCP server
//! remains authoritative while Rust MCP compatibility is developed.

use std::io::{self, BufRead, Write};

fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = line?;
        if let Some(response) = gpu_search_mcp::handle_scaffold_json_rpc_line(&line) {
            writeln!(stdout, "{response}")?;
            stdout.flush()?;
        }
    }

    Ok(())
}
