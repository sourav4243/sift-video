use anyhow::Result;
use std::{collections::{HashMap, HashSet}, fs, sync::{Arc, atomic::{AtomicBool, Ordering}}, time::Duration};
use tokio::time::sleep;
use qdrant_client::{Qdrant, qdrant::{Condition, CountPointsBuilder, CreateCollectionBuilder, Distance, Filter, PointStruct, SearchPointsBuilder, UpsertPointsBuilder, Value, VectorParamsBuilder, value::Kind}};
use tracing::{error, info, info_span, warn};
use uuid::Uuid;
use open_clip_inference::TextEmbedder;

use crate::models::{EmbeddingMetadata, IndexRequest, SearchResult};

const VECTOR_SIZE: u64 = 768;
const AUDIO_COLLECTION: &str = "audio_segments";
const VISUAL_COLLECTION: &str = "visual_frames";
const AUDIO_WEIGHT: f64 = 0.3;
const VISUAL_WEIGHT: f64 = 0.7;

#[derive(Clone)]
pub struct AppState {
    pub qdrant: Arc<Qdrant>,
    pub embedder: Arc<TextEmbedder>,
}

pub async fn init() -> Result<Arc<AppState>> {
    let qdrant_url = std::env::var("QDRANT_URL")
        .unwrap_or_else(|_| "http://localhost:6334".to_string());

    info!("Connecting to Qdrant at {}", qdrant_url);

    // retry loop
    let mut retries = 5;
    let client = loop {
        match Qdrant::from_url(&qdrant_url).build() {
            Ok(c) => {
                match c.health_check().await {
                    Ok(_) => {
                        info!("Successfully connected to Qdrant!");
                        break c;
                    },
                    Err(e) => {
                        warn!("Qdrant not ready yet: {}", e);
                    }
                }
            },
            Err(e) => error!("Failed to build client: {}", e),
        }

        if retries == 0 {
            anyhow::bail!("Could not connect to Qdrant after multiple retries");
        }

        println!("Waiting for Qdrant to start... ({} retries left)", retries);
        sleep(Duration::from_secs(3)).await;
        retries -= 1;
    };

    ensure_collection(&client, AUDIO_COLLECTION).await?;
    ensure_collection(&client, VISUAL_COLLECTION).await?;

    // load CLIP model
    info!("Loading CLIP text embedder...");
    let embedder = TextEmbedder::from_hf("RuteNL/MobileCLIP2-S3-OpenCLIP-ONNX").build().await?;
    info!("CLIP text embedder loaded");

    Ok(Arc::new(AppState {
        qdrant: Arc::new(client),
        embedder: Arc::new(embedder),
    }))
}

async fn ensure_collection(client: &Qdrant, name: &str) -> Result<()> {
    if !client.collection_exists(name).await? {
        info!("Creating collection: {}", name);

        client.create_collection(
            CreateCollectionBuilder::new(name)
                .vectors_config(VectorParamsBuilder::new(VECTOR_SIZE, Distance::Cosine))
        ).await?;
    }
    Ok(())
}

pub async fn embed_query(state: &Arc<AppState>, query: &str) -> Result<Vec<f32>> {
    let state = state.clone();
    let query = query.to_string();
    let row = tokio::task::spawn_blocking(move || {
        let vector = state.embedder.embed_texts(&[query.as_str()])?;
        Ok::<Vec<f32>, anyhow::Error>(vector.row(0).to_vec())
    }).await??;
    info!(dims = row.len(), "Query embedded");
    Ok(row)
}

pub async fn upsert_embedding(client: &Qdrant, req: IndexRequest) -> Result<String> {
    let collection = match req.metadata.modality.as_str() {
        "audio" => AUDIO_COLLECTION,
        "visual" => VISUAL_COLLECTION,
        other => anyhow::bail!("Unknown modality: '{}'. Expected 'audio' or 'visual'.", other),
    };

    // build the payload
    let mut payload: HashMap<String, Value> = HashMap::new();

    payload.insert("video_id".to_string(), string_val(&req.metadata.video_id));
    payload.insert("video_name".to_string(), string_val(&req.metadata.video_name));
    payload.insert("start_time".to_string(), double_val(req.metadata.start_time));
    payload.insert("end_time".to_string(), double_val(req.metadata.end_time));
    payload.insert("modality".to_string(), string_val(&req.metadata.modality));

    // text_content is audio-only but store it if present
    if let Some(ref text) = req.metadata.text_content {
        payload.insert("text_content".to_string(), string_val(text));
    }

    let point = PointStruct::new(req.id.clone(), req.vector, payload);

    client.upsert_points(
        UpsertPointsBuilder::new(collection, vec![point])
    ).await?;

    info!("Indexed [{}] id={} video={} t={:.2}s",
    collection, req.id, req.metadata.video_id, req.metadata.start_time);

    Ok(collection.to_string())
}

pub async fn search_multimodal(state: &Arc<AppState>, query: String) -> Result<Vec<SearchResult>> {
    info_span!("search_multimodal").in_scope(|| {
        info!(query = query.as_str(), "Starting multimodal search");
    });

    let query_vector = embed_query(state, &query).await?;

    // Search Audio
    let audio_future = state.qdrant.search_points(
        SearchPointsBuilder::new(AUDIO_COLLECTION, query_vector.clone(), 5)
            .with_payload(true)
    );

    // Search Visual
    let visual_future = state.qdrant.search_points(
        SearchPointsBuilder::new(VISUAL_COLLECTION, query_vector, 5)
            .with_payload(true)
    );

    // Run parallel
    let (audio_results, visual_results) = tokio::join!(audio_future, visual_future);

    let mut audio_hits: Vec<SearchResult> = Vec::new();
    let mut visual_hits: Vec<SearchResult> = Vec::new();

    // Process audio (weight)
    if let Ok(response) = audio_results {
        for point in response.result {
            let payload = point.payload;
            audio_hits.push(SearchResult {
                video_id: get_str(&payload, "video_id"),
                video_name: get_str(&payload, "video_name"),
                timestamp: get_f64(&payload, "start_time"),
                score: point.score as f64 * AUDIO_WEIGHT,
                match_type: "audio".to_string(),
                match_context: get_str(&payload, "text_content"),
            });
        }
    }

    // Process Visual (weight)
    if let Ok(response) = visual_results {
        for point in response.result {
            let payload = point.payload;
            visual_hits.push(SearchResult {
                video_id: get_str(&payload, "video_id"),
                video_name: get_str(&payload, "video_name"),
                timestamp: get_f64(&payload, "start_time"),
                score: point.score as f64 * VISUAL_WEIGHT,
                match_type: "visual".to_string(),
                match_context: "Visual Frame".to_string(),
            });
        }
    }

    const MERGE_WINDOW: f64 = 2.0;

    let mut final_results: Vec<SearchResult> = Vec::new();
    let mut matched_visual: Vec<bool> = vec![false; visual_hits.len()];

    for audio in audio_hits {
        let mut merged_score = audio.score;

        for (i, visual) in visual_hits.iter().enumerate() {
            if visual.video_id == audio.video_id
                && (visual.timestamp - audio.timestamp).abs() <= MERGE_WINDOW
            {
                merged_score += visual.score;
                matched_visual[i] = true;
            }
        }

        final_results.push(SearchResult {
            score: merged_score,
            match_type: if merged_score > audio.score {
                "audio+visual".to_string()
            } else {
                "audio".to_string()
            },
            ..audio.clone()
        });
    }
    
    for (i, visual) in visual_hits.into_iter().enumerate() {
        if !matched_visual[i] {
            final_results.push(visual);
        }
    }

    final_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    Ok(final_results)
}

pub async fn ingest_from_disk(state: &Arc<AppState>) -> Result<()> {
    let output_dir = std::env::var("OUTPUT_DIR")
        .unwrap_or_else(|_| "/output".to_string());

    ingest_visual(state, &output_dir).await?;
    ingest_audio(state, &output_dir).await?;

    Ok(())
}

async fn ingest_visual(state: &Arc<AppState>, output_dir: &str) -> Result<()> {
    let embed_root = std::path::Path::new(output_dir).join("embeddings");

    if !embed_root.exists() {
        warn!("No embeddings directory found at {:?}, skipping visual ingest", embed_root);
        return Ok(());
    }

    for video_dir in fs::read_dir(&embed_root)? {
        let video_dir = video_dir?;
        if !video_dir.file_type()?.is_dir() { continue; }

        let video_name = video_dir.file_name().to_string_lossy().to_string();
        let video_id = video_name.clone();

        if is_video_indexed(&state.qdrant, VISUAL_COLLECTION, &video_id).await {
            info!("Skipping already-indexed frames for {}", video_id);
            continue;
        }

        let actual_filename = fs::read_dir("/videos")
            .ok()
            .and_then(|entries| {
                entries.filter_map(|e| e.ok()).find_map(|e| {
                    let name = e.file_name().to_string_lossy().to_string();
                    let stem = std::path::Path::new(&name)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("")
                        .to_string();
                    if stem == video_name { Some(name) } else { None }
                })
            })
            .unwrap_or_else(|| {
                warn!("Could not find video file for stem '{}', failling back to .mp4", video_name);
                format!("{}.mp4", video_name)
            });

        for entry in fs::read_dir(video_dir.path())? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|e| e.to_str()) != Some("bin") { continue; }
            
            // derive timestamp from filename: frame_0001 -> index 1 -> 1 * 2.0s
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            let frame_index = stem
                .trim_start_matches("frame_")
                .parse::<f64>()
                .unwrap_or(0.0);
            let timestamp = (frame_index - 1.0) * 2.0;

            // read raw f32s
            let bytes = fs::read(&path)?;
            let vector: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();

            if vector.len() != VECTOR_SIZE as usize {
                warn!("Skipping {:?}: expected {} floats, got {}", path, VECTOR_SIZE, vector.len());
                continue;
            }

            let req = IndexRequest {
                id: Uuid::new_v4().to_string(),
                vector,
                metadata: EmbeddingMetadata {
                    video_id: video_id.clone(),
                    video_name: actual_filename.clone(),
                    start_time: timestamp,
                    end_time: timestamp + 2.0,
                    modality: "visual".to_string(),
                    text_content: None,
                },
            };

            upsert_embedding(&state.qdrant, req).await?;
        }
    }

    Ok(())
}

async fn ingest_audio(state: &Arc<AppState>, output_dir: &str) -> Result<()> {
    let transcripts_path = std::path::Path::new(output_dir).join("transcripts.json");

    if !transcripts_path.exists() {
        warn!("No transcripts.json found at {:?}, skipping audio ingest", transcripts_path);
        return Ok(());
    }

    let content = fs::read_to_string(&transcripts_path)?;

    let segments: Vec<serde_json::Value> = serde_json::from_str(&content)?;

    info!("Ingesting {} transcript segments", segments.len());

    let mut skip_videos: HashSet<String> = HashSet::new();
    let mut seen: HashSet<String> = HashSet::new();

    for seg in &segments {
        let video_id = seg["video_id"].as_str().unwrap_or("unknown").to_string();
        if seen.insert(video_id.clone()) {
            if is_video_indexed(&state.qdrant, AUDIO_COLLECTION, &video_id).await {
                skip_videos.insert(video_id);
            }
        }
    }

    for seg in segments {
        let video_id = seg["video_id"].as_str().unwrap_or("unknown").to_string();

        if skip_videos.contains(&video_id) {
            continue;
        }

        let video_name = seg["video_name"].as_str().unwrap_or("unknown").to_string();
        let segment_id = seg["segment_id"].as_str().unwrap_or("").to_string();
        let text = seg["text"].as_str().unwrap_or("").to_string();
        let start_time = seg["start_time"].as_f64().unwrap_or(0.0);
        let end_time = seg["end_time"].as_f64().unwrap_or(0.0);


        if text.is_empty() {
            warn!("Skipping empty segment {}", segment_id);
            continue;
        }

        let vector = embed_query(state, &text).await?;

        let req = IndexRequest {
            id: Uuid::new_v4().to_string(),
            vector,
            metadata: EmbeddingMetadata {
                video_id,
                video_name,
                start_time,
                end_time,
                modality: "audio".to_string(),
                text_content: Some(text),
            },
        };

        upsert_embedding(&state.qdrant, req).await?;
    }

    Ok(())
}

async fn is_video_indexed(client: &Qdrant, collection: &str, video_id: &str) -> bool {
    let filter = Filter::must([
        Condition::matches("video_id", video_id.to_string())
    ]);

    let count = client.count(
        CountPointsBuilder::new(collection)
            .filter(filter)
    ).await;

    matches!(count, Ok(r)if r.result.unwrap_or_default().count > 0)
}

pub async fn check_and_trigger_pipeline(state: &Arc<AppState>, pipeline_running: Arc<AtomicBool>) -> Result<()> {
    let videos_dir = "/videos";

    let entries = match fs::read_dir(videos_dir) {
        Ok(e) => e,
        Err(_) => return Ok(()),
    };

    let video_extensions = ["mp4", "webm", "mkv", "mov", "avi", "ogv"];

    for entry in entries.filter_map(|e| e.ok()) {
        let name = entry.file_name().to_string_lossy().to_string();
        let ext = std::path::Path::new(&name)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        if !video_extensions.contains(&ext.as_str()) { continue; }

        let stem = std::path::Path::new(&name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();

        let in_audio = is_video_indexed(&state.qdrant, AUDIO_COLLECTION, &stem).await;
        let in_visual= is_video_indexed(&state.qdrant, VISUAL_COLLECTION, &stem).await;

        if !in_audio || !in_visual {
            info!("New video detected: {} - triggering ingestion pipeline", name);

            pipeline_running.store(true, Ordering::SeqCst);

            let flag = Arc::clone(&pipeline_running);
            let state_spawn = Arc::clone(state);

            tokio::spawn(async move {
                trigger_pipeline().await;
                if let Err(e) = ingest_from_disk(&state_spawn).await {
                    warn!("Post-pipeline ingest faield: {}", e);
                }
                flag.store(false, Ordering::SeqCst);
                info!("Pipeline complete, flag cleared");
            });
            break;
        }
    }
    Ok(())
}

async fn trigger_pipeline() {
    use tokio::process::Command;

    for service in ["ingestion", "embedding"] {
        info!("Running {}...", service);
        let status = Command::new("docker-compose")
            .args(["-f", "/app/docker-compose.yml", "run", "--rm", service])
            .current_dir("/app")
            .status()
            .await;

        match status {
            Ok(s) if s.success() => info!("{} complete", service),
            Ok(s) => warn!("{} exited with status: {}", service, s),
            Err(e) => warn!("Failed to run {}: {}", service, e),
        }
    }

}

// Helpers

fn string_val(s: &str) -> Value {
    Value { kind: Some(Kind::StringValue(s.to_string())) }
}

fn double_val(n: f64) -> Value {
    Value { kind: Some(Kind::DoubleValue(n)) }
}

fn get_str(payload: &HashMap<String, Value>, key: &str) -> String {
    match payload.get(key) {
        Some(Value { kind: Some(Kind::StringValue(s)) }) => s.clone(),
        _ => "unknown".to_string(),
    }
}

fn get_f64(payload: &HashMap<String, Value>, key: &str) -> f64 {
    match payload.get(key) {
        Some(Value { kind: Some(Kind::DoubleValue(n)) }) => *n,
        Some(Value { kind: Some(Kind::IntegerValue(n)) }) => *n as f64,
        _ => 0.0,
    }
}