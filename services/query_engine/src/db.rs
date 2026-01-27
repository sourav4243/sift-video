use anyhow::Result;
use std::sync::Arc;
use qdrant_client::Qdrant;

pub struct AppState {
    pub qdrant: Qdrant,
}

pub async fn init() -> Result<Arc<AppState>> {
    let qdrant_url = std::env::var("QDRANT_URL")
        .unwrap_or_else(|_| "https://localhost:6334".to_string());

    println!("Connecting to Qdrant at {}", qdrant_url);

    let client = Qdrant::from_url(&qdrant_url)
        .build()?;

    // todo!("Check if collections exist and create them if not");

    Ok(Arc::new(AppState { qdrant: client }))
}