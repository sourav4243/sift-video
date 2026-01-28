use std::sync::Arc;
use axum::{Json, extract::State};

use crate::models::{SearchRequest, SearchResponse};
use crate::db::{AppState, search_multimodal};

pub async fn search_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<SearchRequest>
) -> Json<SearchResponse> {
    
    if payload.query.trim().is_empty() {
        return Json(SearchResponse { results: vec![] });
    }
    
    println!("Received query: {}", payload.query);

    match search_multimodal(&state.qdrant, payload.query).await {
        Ok(results) => Json(SearchResponse { results }),
        Err(e) => {
            eprintln!("Search failed: {:?}", e);
            return Json(SearchResponse { results: vec![] });
        }
    }
}