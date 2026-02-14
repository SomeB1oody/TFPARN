use std::fs::{File};
use std::io::{BufRead, BufReader, Write};

fn main() -> std::io::Result<()> {
    // Define input file paths
    let asv5_train = "/Volumes/Stan_8TB/Porgramming/Data/ASVspoof5_Dataset/ASVspoof5_protocols/ASVspoof5.train.tsv";
    let asv2019_la_train = "/Volumes/Stan_8TB/Porgramming/Data/AVSspoof_2019_Dataset/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt";
    let asv2019_la_eval = "/Volumes/Stan_8TB/Porgramming/Data/AVSspoof_2019_Dataset/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt";
    let asv2021_la_eval = "/Volumes/Stan_8TB/Porgramming/Data/ASVspoof_2021_Dataset/LA-keys-full/keys/LA/ASV/trial_metadata.txt";

    // Read ASV5 protocol
    let asv5_lines = read_protocol_file(asv5_train)?;

    // Convert 2019LA train protocol
    let asv2019_train_converted = convert_2019_to_asv5(asv2019_la_train)?;

    // Convert 2019LA eval protocol
    let asv2019_eval_converted = convert_2019_to_asv5(asv2019_la_eval)?;

    // Convert 2021LA eval protocol
    let asv2021_eval_converted = convert_2021_to_asv5(asv2021_la_eval)?;

    // Generate first file: ASV5+2019LA_Train.txt
    let mut output1 = File::create("ASV5+2019LA_Train.txt")?;
    for line in &asv5_lines {
        writeln!(output1, "{}", line)?;
    }
    for line in &asv2019_train_converted {
        writeln!(output1, "{}", line)?;
    }
    println!("Generated file: ASV5+2019LA_Train.txt");

    // Generate second file: ASV5+2019LA_Train+2021LA_Eval.txt
    let mut output2 = File::create("ASV5+2019LA_Train+2021LA_Eval.txt")?;
    for line in &asv5_lines {
        writeln!(output2, "{}", line)?;
    }
    for line in &asv2019_train_converted {
        writeln!(output2, "{}", line)?;
    }
    for line in &asv2021_eval_converted {
        writeln!(output2, "{}", line)?;
    }
    println!("Generated file: ASV5+2019LA_Train+2021LA_Eval.txt");

    // Generate third file: ASV5+2019LA_Train+2021LA_Eval+2019LA_Eval.txt
    let mut output3 = File::create("ASV5+2019LA_Train+2021LA_Eval+2019LA_Eval.txt")?;
    for line in &asv5_lines {
        writeln!(output3, "{}", line)?;
    }
    for line in &asv2019_train_converted {
        writeln!(output3, "{}", line)?;
    }
    for line in &asv2021_eval_converted {
        writeln!(output3, "{}", line)?;
    }
    for line in &asv2019_eval_converted {
        writeln!(output3, "{}", line)?;
    }
    println!("Generated file: ASV5+2019LA_Train+2021LA_Eval+2019LA_Eval.txt");

    println!("All files generated successfully!");
    Ok(())
}

// Read protocol file (generic)
fn read_protocol_file(path: &str) -> std::io::Result<Vec<String>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if !line.trim().is_empty() {
            lines.push(line);
        }
    }

    Ok(lines)
}

// Convert ASVspoof 2019 format to ASV5 format
// 2019 format: SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY
// ASV5 format: SPEAKER_ID FLAC_FILE_NAME SPEAKER_GENDER CODEC CODEC_Q CODEC_SEED ATTACK_TAG ATTACK_LABEL KEY TMP
fn convert_2019_to_asv5(path: &str) -> std::io::Result<Vec<String>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut converted_lines = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 5 {
            eprintln!("Warning: Skipping incorrectly formatted line: {}", line);
            continue;
        }

        let speaker_id = parts[0];           // SPEAKER_ID
        let audio_file_name = parts[1];      // FLAC_FILE_NAME
        let speaker_gender = "-";            // No such information
        let codec = "-";                     // No such information
        let codec_q = "-";                   // No such information
        let codec_seed = "-";                // No such information
        let attack_tag = "-";                // No such information
        let attack_label = parts[3];         // ATTACK_LABEL (SYSTEM_ID)
        let key = parts[4];                  // KEY
        let tmp = "-";                       // Reserved column

        let converted = format!(
            "{} {} {} {} {} {} {} {} {} {}",
            speaker_id, audio_file_name, speaker_gender, codec, codec_q,
            codec_seed, attack_tag, attack_label, key, tmp
        );
        converted_lines.push(converted);
    }

    Ok(converted_lines)
}

// Convert ASVspoof 2021 format to ASV5 format
// 2021 format: SPEAKER_ID AUDIO_FILE_NAME CODEC TRANSMISSION ATTACK_LABEL KEY SOURCE
// ASV5 format: SPEAKER_ID FLAC_FILE_NAME SPEAKER_GENDER CODEC CODEC_Q CODEC_SEED ATTACK_TAG ATTACK_LABEL KEY TMP
fn convert_2021_to_asv5(path: &str) -> std::io::Result<Vec<String>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut converted_lines = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 6 {
            eprintln!("Warning: Skipping incorrectly formatted line: {}", line);
            continue;
        }

        let speaker_id = parts[0];           // SPEAKER_ID

        // Clean filename: remove LA2021- prefix, remove - and suffix after LA_E_xxxxxxx
        let mut cleaned_name = parts[1].to_string();
        if cleaned_name.starts_with("LA2021-") {
            cleaned_name = cleaned_name.trim_start_matches("LA2021-").to_string();
        }
        // Find 7-digit number after LA_E_, remove - and suffix after it
        if let Some(pos) = cleaned_name.find("LA_E_") {
            if let Some(end_pos) = cleaned_name[pos+5..].find(|c: char| !c.is_numeric()) {
                let numeric_part = &cleaned_name[pos+5..pos+5+end_pos];
                if numeric_part.len() == 7 {
                    cleaned_name = cleaned_name[..pos+5+7].to_string();
                }
            }
        }
        let audio_file_name = cleaned_name.as_str();  // FLAC_FILE_NAME

        let speaker_gender = "-";            // No such information
        let codec = parts[2];                // CODEC
        let codec_q = "-";                   // No such information
        let codec_seed = "-";                // No such information
        let attack_tag = "-";                // No such information

        // 2021LA format special: bonafide in column 5 (index 4), spoof in column 6 (index 5)
        let (attack_label, key) = if parts[4] == "bonafide" {
            ("-", parts[4])                  // bonafide: no attack type, key in column 5
        } else {
            (parts[4], parts[5])             // spoof: attack type in column 5, key in column 6
        };

        let tmp = "-";                       // Reserved column

        let converted = format!(
            "{} {} {} {} {} {} {} {} {} {}",
            speaker_id, audio_file_name, speaker_gender, codec, codec_q,
            codec_seed, attack_tag, attack_label, key, tmp
        );
        converted_lines.push(converted);
    }

    Ok(converted_lines)
}
