use serde::{Serialize, Deserialize};

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
}

#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub video_id: String,
    pub video_name: String,
    pub timestamp: f64,
    pub score: f64,
    pub match_type: String,
    pub match_context: String,
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
}