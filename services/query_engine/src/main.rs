// This service accepts natural-language queries, performs multimodal vector search over indexed video data, and returns
// the most relevant timestamp.

use axum::{
    routing::post,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;

#[derive(Debug, Deserialize)]
struct SearchRequest {
    query: String,
}

#[derive(Debug, Serialize)]
struct SearchResponse {
    video_id: String,
    timestamp: f64,
    score: f64,
}

async fn search_handler(Json(payload): Json<SearchRequest>) -> Json<SearchResponse> {
    if payload.query.trim().is_empty() {
        return Json(SearchResponse {
            video_id: "".to_string(),
            timestamp: 0.0,
            score: 0.0
        });
    }

    // placeholder response
    Json(SearchResponse {
        video_id: "dummy_video.mp4".to_string(),
        timestamp: 42.0,
        score: 0.0
    })
}

#[tokio::main]
async fn main() {
    println!("Query Engine service started");

    let app = Router::new()
        .route("/search", post(search_handler));

    let listener = TcpListener::bind("0.0.0.0:8080")
        .await
        .expect("failed to bind port");

    println!("Listening on http://0.0.0.0:8080");

    axum::serve(listener, app)
        .await
        .expect("server failed");
}