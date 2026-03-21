use std::sync::Arc;
use axum::{Json, extract::State, http::{StatusCode, header, HeaderMap}, response::{IntoResponse, Response}};
use tokio::{fs::File, io::AsyncReadExt};
use tokio_util::io::ReaderStream;
use tracing::{info, error};

use crate::models::{SearchRequest, SearchResponse, IndexRequest, IndexResponse};
use crate::db::{AppState, search_multimodal, upsert_embedding};

pub async fn search_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<SearchRequest>
) -> Response {
    
    if payload.query.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, Json(SearchResponse { results: vec![] })).into_response();
    }
    
    info!("Received query: {}", payload.query);

    match search_multimodal(&state, payload.query).await {
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

pub async fn video_handler(
    axum::extract::Path(filename): axum::extract::Path<String>,
    headers: HeaderMap,
) -> Response {
    let filename = filename.trim_start_matches('/');
    let path = format!("/videos/{}", filename);

    let mime = match std::path::Path::new(filename)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase()
        .as_str()
    {
        "mp4" => "video/mp4",
        "webm" => "video/webm",
        "mkv"  => "video/x-matroska",
        "mov"  => "video/quicktime",
        "avi"  => "video/x-msvideo",
        "ogv"  => "video/ogg",
        _      => "application/octet-stream",
    };

    let file = match File::open(&path).await {
        Ok(f) => f,
        Err(e) => {
            error!("Video not found: {} - {}", path, e);
            return (StatusCode::NOT_FOUND, "not found").into_response();
        }
    };

    let file_size = match file.metadata().await {
        Ok(m) => m.len(),
        Err(e) => {
            error!("Could not read metadata for {}: {}", path, e);
            return (StatusCode::INTERNAL_SERVER_ERROR, "metadata error").into_response();
        }
    };

    if let Some(range_header) = headers.get(header::RANGE) {
        let range_str = match range_header.to_str() {
            Ok(s) => s,
            Err(_) => return (StatusCode::BAD_REQUEST, "invalid range").into_response(),
        };

        let range = range_str.trim_start_matches("bytes=");
        let parts: Vec<&str> = range.split("-").collect();

        let start: u64 = parts[0].parse().unwrap_or(0);
        let end: u64 = if parts.len() > 1 && !parts[1].is_empty() {
            parts[1].parse().unwrap_or(file_size - 1)
        } else {
            file_size - 1
        };

        let end = end.min(file_size - 1);
        let chunk = end - start + 1;

        use tokio::io::AsyncSeekExt;
        let mut file = file;
        if let Err(e) = file.seek(std::io::SeekFrom::Start(start)).await {
            error!("Seek failed: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, "seek error").into_response();
        }

        let stream = ReaderStream::new(file.take(chunk));
        let body = axum::body::Body::from_stream(stream);

        return Response::builder()
            .status(StatusCode::PARTIAL_CONTENT)
            .header(header::CONTENT_TYPE, mime)
            .header(header::CONTENT_LENGTH, chunk)
            .header(header::CONTENT_RANGE, format!("bytes {}-{}/{}", start, end, file_size))
            .header(header::ACCEPT_RANGES, "bytes")
            .body(body)
            .unwrap();
    }

    let stream = ReaderStream::new(file);
    let body = axum::body::Body::from_stream(stream);

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, mime)
        .header(header::CONTENT_LENGTH, file_size)
        .header(header::ACCEPT_RANGES, "bytes")
        .body(body)
        .unwrap()
}