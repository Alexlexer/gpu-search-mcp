use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;

use gpu_search_http::{AppState, app_with_state};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = parse_directory_arg();
    let state = if let Some(directory) = directory {
        eprintln!(
            "experimental gpu-search Rust HTTP indexing {}",
            directory.display()
        );
        AppState::from_directory(directory)?
    } else {
        AppState::empty()
    };

    let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8766);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("experimental gpu-search Rust HTTP server listening on http://{addr}");
    axum::serve(listener, app_with_state(state)).await?;
    Ok(())
}

fn parse_directory_arg() -> Option<PathBuf> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--directory" || arg == "--repo" {
            return args.next().map(PathBuf::from);
        }
    }
    None
}
