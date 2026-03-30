use std::sync::Arc;
use axum::{Json, extract::{Path, State}, http::{HeaderMap, StatusCode, header}, response::{IntoResponse, Response}};
use tokio::{fs::File, io::AsyncReadExt};
use tokio_util::io::ReaderStream;
use tracing::{info, error, debug};

use crate::{db, models::{IndexRequest, IndexResponse, SearchRequest, SearchResponse, VideoListResponse}};
use crate::db::{AppState, search_multimodal, upsert_embedding};

#[derive(Debug, serde::Deserialize)]
pub struct DownloadRequest {
    pub url: String,
}

pub async fn download_handler(
    Json(payload): Json<DownloadRequest>,
) -> Response {
    let url = payload.url.trim().to_string();

    if url.is_empty() {
        return (StatusCode::BAD_REQUEST, "url is required").into_response();
    }

    info!("yt-dlp download requested: {}", url);

    let meta_output = match tokio::process::Command::new("yt-dlp")
        .args(["--no-playlist", "--dump-json", &url])
        .output()
        .await
    {
        Ok(o) => o,
        Err(e) => {
            error!("Failed to fetch metadata: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, "metadata fetch failed").into_response();
        }
    };

    if !meta_output.status.success() {
        let err = String::from_utf8_lossy(&meta_output.stderr);
        return (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response();
    }

    // parse JSON
    let json: serde_json::Value = match serde_json::from_slice(&meta_output.stdout) {
        Ok(j) => j,
        Err(e) => {
            error!("JSON parse failed: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, "invalid metadata").into_response();
        }
    };

    let title = json["title"].as_str().unwrap_or("unknown").to_string();
    let duration = json["duration"].as_f64().unwrap_or(0.0);
    let thumbnail = json["thumbnail"].as_str().unwrap_or("").to_string();

    // await download
    let download = tokio::process::Command::new("yt-dlp")
        .args([
            "--no-playlist",
            "-o", "/videos/%(title)s.%(ext)s",
            "--remux-video", "mp4",
            &url,
        ])
        .output()
        .await;

    match download {
        Err(e) => {
            error!("yt-dlp failed to run: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, "yt-dlp failed").into_response();
        }
        Ok(out) if !out.status.success() => {
            let err = String::from_utf8_lossy(&out.stderr);
            error!("yt-dlp exited with error: {}", err);
            return (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response();
        }
        _ => {}
    }

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "title": title,
            "duration": duration,
            "thumbnail": thumbnail
        }))
    ).into_response()
}

pub async fn delete_video_handler(
    State(state): State<Arc<AppState>>,
    Path(filename): Path<String>,
) -> Response {
    let filename = filename.trim_start_matches('/');
    let path = format!("/videos/{}", filename);
    let video_id = std::path::Path::new(&filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();

    info!("Starting full cleanup for video: {}", filename);

    // delete original video file
    if let Err(e) = tokio::fs::remove_file(&path).await {
        error!("Failed to delete file {}: {}", path, e);
    }

    // delete qdrant embeddings
    if let Err(e) = db::delete_video_embeddings(&state.qdrant, &video_id).await {
        error!("Failed to delete embeddings for {}: {}", video_id, e);
    }

    // scrub frames directory
    let frames_dir = format!("/output/frames/{}", video_id);
    if let Err(e) = tokio::fs::remove_dir_all(&frames_dir).await {
        debug!("Skipped frames dir {} (may not exist): {}", frames_dir, e);
    }

    // scrub embeddings directory
    let embeddings_dir = format!("/output/embeddings/{}", video_id);
    if let Err(e) = tokio::fs::remove_dir_all(&embeddings_dir).await {
        debug!("Skipped embeddings dir {} (may not exist): {}", embeddings_dir, e);
    }

    // scrub wav/srt/txt/vtt files from /output
    for ext in &["wav", "srt", "txt", "vtt"] {
        let file_path = format!("/output/{}.{}", video_id, ext);
        if let Err(e) = tokio::fs::remove_file(&file_path).await {
            debug!("Skipped {} (may not exist): {}", file_path, e);
        }
    }

    let marker = format!("/output/.indexed_{}", video_id);
    if let Err(e) = tokio::fs::remove_file(&marker).await {
        debug!("No marker file to delete for {}: {}", video_id, e);
    }

    // scrub transcripts.json
    let transcripts_path = "/output/transcripts.json";
    if let Ok(contents) = tokio::fs::read_to_string(transcripts_path).await {
        if let Ok(mut json) = serde_json::from_str::<serde_json::Value>(&contents) {
            let mut modified = false;

            if let Some(array) = json.as_array_mut() {
                let initial_len = array.len();

                info!("Before delete: {} segments in transcripts.json", initial_len);
                info!("Trying to match: filename='{}' video_id='{}'", filename, video_id);

                // log first few entries to see what's actually stored
                for item in array.iter().take(5) {
                    info!(
                        "Sample entry — video_name={:?} video_id={:?}",
                        item.get("video_name").and_then(|v| v.as_str()),
                        item.get("video_id").and_then(|v| v.as_str())
                    );
                }

                array.retain(|item| {
                    let name_match = item
                        .get("video_name")
                        .and_then(|v| v.as_str())
                        .map(|n| n == filename)
                        .unwrap_or(false);

                    let id_match = item
                        .get("video_id")
                        .and_then(|v| v.as_str())
                        .map(|id| id == video_id)
                        .unwrap_or(false);

                    if name_match || id_match {
                        info!(
                            "Dropping segment: video_name={:?} video_id={:?} name_match={} id_match={}",
                            item.get("video_name").and_then(|v| v.as_str()),
                            item.get("video_id").and_then(|v| v.as_str()),
                            name_match,
                            id_match
                        );
                    }

                    !name_match && !id_match
                });

                info!(
                    "After delete: {} segments remain (removed {})",
                    array.len(),
                    initial_len - array.len()
                );

                modified = array.len() != initial_len;
            }

            if modified {
                if let Ok(new_contents) = serde_json::to_string_pretty(&json) {
                    let _ = tokio::fs::write(transcripts_path, new_contents).await;
                    info!("Removed {} from transcripts.json", video_id);
                }
            } else {
                info!("No segments matched for deletion in transcripts.json");
            }
        }
    }

    info!("Cleanup completed for video: {}", filename);
    StatusCode::OK.into_response()
}

pub async fn search_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<SearchRequest>
) -> Response {
    
    if payload.query.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, Json(SearchResponse { results: vec![] })).into_response();
    }
    
    info!("Received query: {}", payload.query);

    match search_multimodal(&state, payload.query, payload.video_id, payload.match_type).await {
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

pub async fn video_list_handler(
    State(state): State<Arc<AppState>>,
) -> Response {
    match db::list_indexed_videos(&state).await {
        Ok(videos) => {
            (StatusCode::OK, Json(VideoListResponse { videos })).into_response()
        }
        Err(e) => {
            error!("Failed to list videos: {:?}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, "error").into_response()
        }
    }
}

pub async fn upload_handler(
    mut multipart: axum::extract::Multipart,
) -> Response {
    while let Some(field) = multipart.next_field().await.unwrap_or(None) {
        let filename = match field.file_name() {
            Some(f) => f.to_string(),
            None => continue,
        };

        let allowed = ["mp4", "webm", "mkv", "mov", "avi", "ogv"];
        let ext = std::path::Path::new(&filename)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        if !allowed.contains(&ext.as_str()) {
            return (StatusCode::BAD_REQUEST, "unsupported file type").into_response();
        }

        let path = format!("/videos/{}", filename);
        let data = match field.bytes().await {
            Ok(b) => b,
            Err(e) => {
                error!("Failed to read upload: {}", e);
                return (StatusCode::INTERNAL_SERVER_ERROR, "read error").into_response();
            }
        };

        if let Err(e) = tokio::fs::write(&path, data).await {
            error!("Failed to write video: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, "write error").into_response();
        }

        info!("Uploaded: {}", filename);
    }

    StatusCode::OK.into_response()
}

pub async fn health_handler(
    State(state): State<Arc<AppState>>,
) -> Response {
    let qdrant_ok = state.qdrant.health_check().await.is_ok();
    let pipeline_running = state.pipeline_running.load(std::sync::atomic::Ordering::SeqCst);

    let status = if qdrant_ok { "ok" } else { "degraded" };
    let code = if qdrant_ok { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE };

    (code, Json(serde_json::json!({
        "status": status,
        "qdrant": qdrant_ok,
        "pipeline_running": pipeline_running,
    }))).into_response()
}