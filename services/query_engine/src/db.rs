use anyhow::Result;
use std::{sync::{Arc, Mutex}, collections::HashMap, time::Duration};
use tokio::time::sleep;
use qdrant_client::{Qdrant, qdrant::{CreateCollectionBuilder, Distance, VectorParamsBuilder, SearchPointsBuilder, Value, value::Kind, UpsertPointsBuilder, PointStruct}};
use tracing::{error, info, info_span, warn};
use ort::{inputs, session::Session}; 
use tokenizers::Tokenizer;

use crate::models::{IndexRequest, SearchResult};

const VECTOR_SIZE: u64 = 512;
const AUDIO_COLLECTION: &str = "audio_segments";
const VISUAL_COLLECTION: &str = "visual_frames";

const CLIP_MAX_TOKENS: usize = 77;

pub struct AppState {
    pub qdrant: Arc<Qdrant>,
    pub clip_session: Arc<Mutex<Session>>,
    pub tokenizer: Arc<Tokenizer>,
}

pub async fn init() -> Result<Arc<AppState>> {
    let qdrant_url = std::env::var("QDRANT_URL")
        .unwrap_or_else(|_| "http://localhost:6334".to_string());

    let model_path = std::env::var("CLIP_MODEL_PATH")
        .unwrap_or_else(|_| "models/model.onnx".to_string());

    let tokenizer_path = std::env::var("CLIP_TOKENIZER_PATH")
        .unwrap_or_else(|_| "models/tokenizer.json".to_string());

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
    info!("Loading CLIP model from {}", model_path);
    let clip_session = Session::builder()?.commit_from_file(&model_path)?;

    // load tokenizer
    info!("Loading tokenizer from {}", tokenizer_path);
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    info!("CLIP model and tokenizer loaded");

    Ok(Arc::new(AppState {
        qdrant: Arc::new(client),
        clip_session: Arc::new(Mutex::new(clip_session)),
        tokenizer: Arc::new(tokenizer),
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

pub fn embed_query(state: &AppState, query: &str) -> Result<Vec<f32>> {
    // Tokenize
    let encoding = state.tokenizer.encode(query, true)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

    // pad/truncate to CLIP_MAX_TOKENS
    let mut input_ids: Vec<i64> = encoding.get_ids()
        .iter().map(|&x| x as i64).collect();
    let mut attention_mask: Vec<i64> = encoding.get_attention_mask()
        .iter().map(|&x| x as i64).collect();

    input_ids.truncate(CLIP_MAX_TOKENS);
    attention_mask.truncate(CLIP_MAX_TOKENS);

    while input_ids.len() < CLIP_MAX_TOKENS {
        input_ids.push(0);
        attention_mask.push(0);
    }

    let input_ids_arr = ndarray::Array2::from_shape_vec(
        (1, CLIP_MAX_TOKENS), input_ids
    )?.into_dyn();
    let attention_mask_arr = ndarray::Array2::from_shape_vec(
        (1, CLIP_MAX_TOKENS), attention_mask
    )?.into_dyn();

    let pixel_values_arr = ndarray::Array4::<f32>::zeros((1, 3, 224, 224)).into_dyn();

    let input_ids_val = ort::value::Value::from_array(input_ids_arr)?;
    let attn_mask_val = ort::value::Value::from_array(attention_mask_arr)?;
    let pixel_values_val = ort::value::Value::from_array(pixel_values_arr)?;

    let mut session = state.clip_session.lock()
        .map_err(|e| anyhow::anyhow!("Session mutex poisoned: {}", e))?;

    let outputs = session.run(inputs![
        "input_ids" => input_ids_val,
        "attention_mask" => attn_mask_val,
        "pixel_values" => pixel_values_val
    ])?;

    let (_shape, data) = outputs["text_embeds"]
        .try_extract_tensor::<f32>()?;

    let vector: Vec<f32> = data.iter().cloned().collect();

    info!(dims = vector.len(), "Query embedded");

    Ok(vector)
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

pub async fn search_multimodal(state: &AppState, query: String) -> Result<Vec<SearchResult>> {
    let _span = info_span!("search_multimodal").entered();

    let query_vector = embed_query(state, &query)?;

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

    let mut final_results = Vec::new();

    // Process audio (weight = 0.6)
    if let Ok(response) = audio_results {
        for point in response.result {
            let payload = point.payload;
            final_results.push(SearchResult {
                video_id: get_str(&payload, "video_id"),
                video_name: get_str(&payload, "video_name"),
                timestamp: get_f64(&payload, "start_time"),
                score: point.score as f64 * 0.6,
                match_type: "audio".to_string(),
                match_context: get_str(&payload, "text_content"),
            });
        }
    }

    // Process Visual (Weight = 0.4)
    if let Ok(response) = visual_results {
        for point in response.result {
            let payload = point.payload;
            final_results.push(SearchResult {
                video_id: get_str(&payload, "video_id"),
                video_name: get_str(&payload, "video_name"),
                timestamp: get_f64(&payload, "start_time"),
                score: point.score as f64 * 0.4,
                match_type: "visual".to_string(),
                match_context: "Visual Frame".to_string(),
            });
        }
    }

    final_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    Ok(final_results)
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