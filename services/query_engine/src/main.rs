// This service accepts natural-language queries, performs multimodal vector search over indexed video data, and returns
// the most relevant timestamp.

mod models;
mod db;
mod api;

use anyhow::Ok;
use axum::{routing::post, Router};
use tokio::net::TcpListener;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Starting query engine...");

    // Initialize DB
    let state = db::init().await?;

    // Setup router
    let app = Router::new()
        .route("/search", post(api::search_handler))
        .with_state(state);

    // Start server
    let listener = TcpListener::bind("0.0.0.0:8080").await?;
    println!("Query Engine listening on http://0.0.0.0:8080");
    axum::serve(listener, app).await?;

    Ok(())
}