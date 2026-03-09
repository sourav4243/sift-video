use std::sync::Arc;
use axum::{Json, extract::State, http::StatusCode};

use crate::models::{SearchRequest, SearchResponse, IndexRequest, IndexResponse};
use crate::db::{AppState, search_multimodal, upsert_embedding};

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

pub async fn index_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<IndexRequest>
) -> (StatusCode, Json<IndexResponse>) {
    
    let id = payload.id.clone();

    match upsert_embedding(&state.qdrant, payload).await {
        Ok(collection) => {
            let resp = IndexResponse { success: true, id, collection };
            (StatusCode::OK, Json(resp))
        }
        Err(e) => {
            eprintln!("Index failed for id={}: {:?}", id, e);
            let resp = IndexResponse { success: false, id, collection: String::new() };
            (StatusCode::INTERNAL_SERVER_ERROR, Json(resp))
        }
    }
}