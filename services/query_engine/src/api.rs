use std::sync::Arc;
use crate::models::{SearchRequest, SearchResponse};
use axum::{Json, extract::State};
use crate::db::AppState;

pub async fn search_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<SearchRequest>
) -> Json<SearchResponse> {
    
    println!("Received query: {}", payload.query);

    todo!("Call db::search_multimodal(state, query");
}