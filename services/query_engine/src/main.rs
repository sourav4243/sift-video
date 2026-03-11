// This service accepts natural-language queries, performs multimodal vector search over indexed video data, and returns
// the most relevant timestamp.

mod models;
mod db;
mod api;

use anyhow::Ok;
use axum::{routing::post, Router};
use tokio::net::TcpListener;
use tracing::info;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
    .with_env_filter(
        tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| "query_engine=info".into())
    )
    .with_timer(tracing_subscriber::fmt::time::ChronoLocal::new("%H:%M:%S".to_string()))
    .init();

    info!("Starting query engine...");

    // Initialize DB
    let state = db::init().await?;

    // Setup router
    let app = Router::new()
        .route("/search", post(api::search_handler))
        .route("/index", post(api::index_handler))
        .with_state(state)
        .into_make_service();

    // Start server
    let listener = TcpListener::bind("0.0.0.0:8080").await?;
    info!("Query Engine listening on http://0.0.0.0:8080");
    axum::serve(listener, app).await?;

    Ok(())
}