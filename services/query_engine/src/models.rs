use serde::{Serialize, Deserialize};

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub video_id: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
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

// Indexing

#[derive(Debug, Deserialize)]
pub struct EmbeddingMetadata {
    pub video_id: String,
    pub video_name: String,
    pub start_time: f64,
    pub end_time: f64,
    pub modality: String,  // "audio" | "visual"
    pub text_content: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct IndexRequest {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: EmbeddingMetadata,
}

#[derive(Debug, Serialize)]
pub struct IndexResponse {
    pub success: bool,
    pub id: String,
    pub collection: String,
}

#[derive(Debug, Serialize)]
pub struct VideoInfo {
    pub video_id: String,
    pub video_name: String,
    pub audio_segments: u64,
    pub visual_frames: u64,
}

#[derive(Debug, Serialize)]
pub struct VideoListResponse {
    pub videos: Vec<VideoInfo>,
}