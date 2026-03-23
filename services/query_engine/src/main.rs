// This service accepts natural-language queries, performs multimodal vector search over indexed video data, and returns
// the most relevant timestamp.

mod models;
mod db;
mod api;

use std::{sync::{atomic::{AtomicBool, Ordering}, Arc}, time::Duration};
use axum::{Router, extract::DefaultBodyLimit, routing::{get, post}};
use tokio::net::TcpListener;
use tracing::{info, warn};
use tower_http::services::{ServeDir, ServeFile};

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
    let state_clone = state.clone();
    let pipeline_running = Arc::new(AtomicBool::new(false));

    tokio::fs::remove_file("/output/.trigger_ingest").await.ok();
    tokio::fs::remove_file("/output/.trigger_embed").await.ok();
    info!("Cleaned up stale trigger files");

    match db::check_and_trigger_pipeline(&state_clone, Arc::clone(&pipeline_running)).await {
        Ok(true) => info!("New videos detected on startup, pipeline triggered"),
        Ok(false) => info!("All videos already indexed, skipping startup pipeline"),
        Err(e) => warn!("Startup pipeline check failed: {}", e),
    }

    info!("Ingesting embeddings from disk...");
    if let Err(e) = db::ingest_from_disk(&state_clone).await {
        warn!("Initial ingest failed: {}, retrying in 30 seconds...", e);
    } else {
        info!("Initial ingest complete");
    }


    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(30)).await;

            // skip cycle if pipeline is already running
            if pipeline_running.load(Ordering::SeqCst) {
                info!("Pipeline already running, skipping cycle");
                continue;
            }

            match db::check_and_trigger_pipeline(&state_clone, Arc::clone(&pipeline_running)).await {
                Ok(true) => info!("Pipeline triggered, ingest will run after completion"),
                Ok(false) => {},
                Err(e) => {
                    warn!("Pipeline check failed: {}", e);
                    pipeline_running.store(false, Ordering::SeqCst);
                }
            }
        }
    });

    // Setup router
    let app = Router::new()
        .route("/search", post(api::search_handler))
        .route("/index", post(api::index_handler))
        .route("/upload", post(api::upload_handler))
        .route("/videos/list", get(api::video_list_handler))
        .route("/videos/*filename", get(api::video_handler))
        .layer(DefaultBodyLimit::max(1024 * 1024 * 1024))
        .fallback_service(ServeDir::new("/app/static")
            .fallback(ServeFile::new("/app/static/index.html")))
        .with_state(state);

    // Start server
    let listener = TcpListener::bind("0.0.0.0:8080").await?;
    info!("Query Engine listening on http://0.0.0.0:8080");
    axum::serve(listener, app).await?;

    Ok(())
}