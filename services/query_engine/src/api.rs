use std::sync::Arc;
use axum::{Json, extract::State, http::StatusCode, response::IntoResponse};
use tracing::{info, error};

use crate::models::{SearchRequest, SearchResponse, IndexRequest, IndexResponse};
use crate::db::{AppState, search_multimodal, upsert_embedding};

pub async fn search_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<SearchRequest>
) -> impl IntoResponse {
    
    if payload.query.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, Json(SearchResponse { results: vec![] })).into_response();
    }
    
    info!("Received query: {}", payload.query);

    match search_multimodal(&state.qdrant, payload.query).await {
        Ok(results) => {
            (StatusCode::OK, Json(SearchResponse { results })).into_response()
        }
        Err(e) => {
            error!("Search failed: {:?}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(SearchResponse { results: vec![] })).into_response()
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
            error!("Index failed for id={}: {:?}", id, e);
            let resp = IndexResponse { success: false, id, collection: String::new() };
            (StatusCode::INTERNAL_SERVER_ERROR, Json(resp))
        }
    }
}