use std::sync::Arc;

use axum::{Json, extract::State};

use crate::models::{SearchRequest, SearchResponse};

pub async fn search_handler(State(state): State<Arc<()>>, Json(payload): Json<SearchRequest>) -> Json<SearchResponse> {
    todo!();
}