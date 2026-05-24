use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use gpu_search_http::app;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8766);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("experimental gpu-search Rust HTTP server listening on http://{addr}");
    axum::serve(listener, app()).await?;
    Ok(())
}
