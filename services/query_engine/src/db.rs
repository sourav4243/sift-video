use anyhow::Result;
use std::sync::Arc;
use std::collections::HashMap;
use std::time::Duration;
use tokio::time::sleep;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{CreateCollectionBuilder, Distance, VectorParamsBuilder, SearchPointsBuilder, Value, value::Kind};

use crate::models::SearchResult;

const VECTOR_SIZE: u64 = 512;
const AUDIO_COLLECTION: &str = "audio_segments";
const VISUAL_COLLECTION: &str = "visual_frames";

pub struct AppState {
    pub qdrant: Qdrant,
}

pub async fn init() -> Result<Arc<AppState>> {
    let qdrant_url = std::env::var("QDRANT_URL")
        .unwrap_or_else(|_| "http://localhost:6334".to_string());

    println!("Connecting to Qdrant at {}", qdrant_url);

    // retry loop
    let mut retries = 5;
    let client = loop {
        match Qdrant::from_url(&qdrant_url).build() {
            Ok(c) => {
                match c.health_check().await {
                    Ok(_) => {
                        println!("Successfully connected to Qdrant!");
                        break c;
                    },
                    Err(e) => {
                        println!("Qdrant not ready yet: {}", e);
                    }
                }
            },
            Err(e) => println!("Failed to build client: {}", e),
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

    Ok(Arc::new(AppState { qdrant: client }))
}

async fn ensure_collection(client: &Qdrant, name: &str) -> Result<()> {
    if !client.collection_exists(name).await? {
        println!("Creating collection: {}", name);

        client.create_collection(
            CreateCollectionBuilder::new(name)
                .vectors_config(VectorParamsBuilder::new(VECTOR_SIZE, Distance::Cosine))
        ).await?;
    }
    Ok(())
}

pub async fn search_multimodal(client: &Qdrant, _query: String) -> Result<Vec<SearchResult>> {
    // TODO!
    // Vectorize: Run CLIP model here
    let dummy_vector = vec![0.1; VECTOR_SIZE as usize];

    // Search Audio
    let audio_future = client.search_points(
        SearchPointsBuilder::new(AUDIO_COLLECTION, dummy_vector.clone(), 5)
            .with_payload(true)
    );

    // Search Visual
    let visual_future = client.search_points(
        SearchPointsBuilder::new(VISUAL_COLLECTION, dummy_vector, 5)
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