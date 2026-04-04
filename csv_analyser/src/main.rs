use csv::ReaderBuilder;
use serde::Deserialize;

const DEV_UTTERANCE_COUNT: f64 = 140950.0;

#[derive(Debug, Deserialize)]
struct Record {
    run_num: u32,
    r#type: u32,
    time: f64,
    max_memory: f64,
    if_best: u32,
}

fn main() {
    // fill in the CSV file path
    let file_path = String::from("");

    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(&file_path)
        .expect("Failed to open CSV file");

    let records: Vec<Record> = reader
        .deserialize()
        .map(|r| r.expect("Failed to parse CSV record"))
        .collect();

    // Collect Dev max_memory values for average GPU memory usage
    let dev_memories: Vec<f64> = records
        .iter()
        .filter(|r| r.r#type == 2)
        .map(|r| r.max_memory)
        .collect();
    let avg_dev_memory = dev_memories.iter().sum::<f64>() / dev_memories.len() as f64;

    // Collect Train times for average time per training epoch
    let train_times: Vec<f64> = records
        .iter()
        .filter(|r| r.r#type == 1)
        .map(|r| r.time)
        .collect();
    let avg_train_time = train_times.iter().sum::<f64>() / train_times.len() as f64;

    // Find the last occurrence of if_best=1, sum all times up to and including it
    let last_best_idx = records
        .iter()
        .rposition(|r| r.if_best == 1)
        .expect("No best model found (if_best=1)");
    let time_to_best_secs: f64 = records[..=last_best_idx].iter().map(|r| r.time).sum();
    let time_to_best_mins = time_to_best_secs / 60.0;

    // Collect Dev times for latency per utterance
    let dev_times: Vec<f64> = records
        .iter()
        .filter(|r| r.r#type == 2)
        .map(|r| r.time)
        .collect();
    let avg_dev_time_ms = (dev_times.iter().sum::<f64>() / dev_times.len() as f64) * 1000.0;
    let latency_per_utterance = avg_dev_time_ms / DEV_UTTERANCE_COUNT;

    println!("Memory Usage: {:.1} GB", avg_dev_memory);
    println!("Time per Training Epoch: {:.1} s", avg_train_time);
    println!("Time to Best Model: {:.2} min", time_to_best_mins);
    println!("Latency per Utterance: {:.4} ms/utterance", latency_per_utterance);
}
