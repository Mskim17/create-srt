use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};
use std::process::{Command, Stdio};
use std::io::{Read, BufReader, Write};
use std::fs::File;
use std::path::Path;
use hound::{WavWriter, WavSpec, SampleFormat, WavReader};
use rfd::FileDialog;

/// SRT 시간 포맷 변환 함수 (Whisper 10ms 단위를 ms로 변환)
fn format_srt_time(whisper_time: i64) -> String {
    let milliseconds = whisper_time * 10;
    let seconds = milliseconds / 1000;
    let ms = milliseconds % 1000;
    let minutes = seconds / 60;
    let hours = minutes / 60; 

    format!(
        "{:02}:{:02}:{:02},{:03}",
        hours,
        minutes % 60,
        seconds % 60,
        ms
    )
}

fn main() -> anyhow::Result<()> {
    // 0. 사용자로부터 파일 선택 받기
    println!("📂 처리할 영상 파일을 선택해주세요...");
    let file_path = FileDialog::new()
        .add_filter("Video Files", &["mp4", "mkv", "avi", "mov"])
        .add_filter("Audio Files", &["wav", "mp3", "m4a"])
        .set_directory(".") // 현재 폴더에서 시작
        .pick_file();

    // 사용자가 취소를 눌렀을 경우 처리
    let input_file = match file_path {
        Some(path) => path,
        None => {
            println!("❌ 파일 선택이 취소되었습니다. 프로그램을 종료합니다.");
            return Ok(());
        }
    };

    let input_path_str = input_file.to_str().unwrap();
    println!("✅ 선택된 파일: {}", input_path_str);

    // --- 설정 변수 ---
    let output_wav = "temp_audio.wav";                    // 중간 오디오 파일
    let model_path = "./ggml-kotoba-whisper-v2.0-q5_0.bin";                 // 모델 파일
    let srt_output = format!("{}.srt", input_file.file_stem().unwrap().to_str().unwrap());

    // 1. 오디오 추출 단계
    println!("🚀 [1/4] 오디오 추출 시작 (FFmpeg)...");
    let spec = WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut child = Command::new("ffmpeg")
        .args([
            "-i", input_path_str,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-f", "s16le",
            "pipe:1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;

    let stdout = child.stdout.take().ok_or_else(|| anyhow::anyhow!("STDOUT 오픈 실패"))?;
    let mut reader = BufReader::new(stdout);
    let mut writer = WavWriter::create(output_wav, spec)?;
    let mut buffer = [0u8; 2];

    while reader.read_exact(&mut buffer).is_ok() {
        let sample = i16::from_le_bytes(buffer);
        writer.write_sample(sample)?;
    }
    child.wait()?;
    writer.finalize()?;
    println!("✅ 오디오 추출 완료.");

    // 2. Whisper 모델 초기화
    if !Path::new(model_path).exists() {
        return Err(anyhow::anyhow!("모델 파일이 없습니다! {}을 확인하세요.", model_path));
    }
    println!("🚀 [2/4] Whisper 모델 로드 중...");
    let ctx = WhisperContext::new_with_params(model_path, WhisperContextParameters::default())?;

    // 3. 오디오 데이터를 f32 Vec으로 로드
    println!("🎵 [3/4] 오디오 데이터 변환 중...");
    let mut wav_reader = WavReader::open(output_wav)?;
    let audio_data: Vec<f32> = wav_reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();

    // 4. 음성 인식 및 자막 생성
    println!("🤖 [4/4] 일본어 음성 인식 및 자막 생성 시작...");
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some("ja"));
    params.set_print_special(false);
    params.set_print_progress(true);
    params.set_print_timestamps(true);

    let mut state = ctx.create_state()?;
    state.full(params, &audio_data).expect("추론 실패");

    let num_segments = state.full_n_segments()?;
    let mut srt_content = String::new();

    for i in 0..num_segments {
        let text = state.full_get_segment_text(i)?;
        let t0 = state.full_get_segment_t0(i)?;
        let t1 = state.full_get_segment_t1(i)?;

        let srt_segment = format!(
            "{}\n{} --> {}\n{}\n\n",
            i + 1,
            format_srt_time(t0),
            format_srt_time(t1),
            text.trim()
        );
        srt_content.push_str(&srt_segment);
    }

    // 결과 저장
    let mut file = File::create(&srt_output)?;
    file.write_all(srt_content.as_bytes())?;

    println!("\n✨ 모든 작업이 완료되었습니다!");
    println!("📄 생성된 자막: {}", &srt_output);

    // (옵션) 임시 WAV 파일 삭제를 원하시면 아래 주석을 해제하세요.
    std::fs::remove_file(output_wav)?;

    Ok(())
}