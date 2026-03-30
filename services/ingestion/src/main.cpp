#include <iostream>
#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <thread>
#include <chrono>
#include <future>
#include <set>

namespace fs = std::filesystem;

std::string json_escape(const std::string &input){
    std::string out;
    out.reserve(input.size());

    for(char c: input){
        switch(c){
            case '\"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c;
        }
    }
    return out;
}

double to_seconds(const std::string &t){
    // HH:MM:SS,mmm
    int h, m, s, ms;
    sscanf(t.c_str(), "%d:%d:%d,%d", &h, &m, &s, &ms);
    return h * 3600 + m * 60 + s + ms/1000.0;
}

struct Segment {
    std::string video_id;
    std::string video_name;
    std::string segment_id;
    std::string text;
    double start_time;
    double end_time;
};


void write_frames_metadata(const fs::path frames_dir, double interval_sec){
    std::vector<std::string> frames;

    for(const auto& entry : fs::directory_iterator(frames_dir)){
        if(entry.is_regular_file() && entry.path().extension() == ".jpg"){
            frames.push_back(entry.path().filename().string());
        }
    }

    std::sort(frames.begin(), frames.end());

    std::ofstream out(frames_dir / "frames.json");
    out << "[\n";

    for(size_t i=0; i<frames.size(); i++){
        double timestamp = i * interval_sec;

        out << "  {\n";
        out << "    \"frame_id\": \"" << frames[i] << "\",\n";
        out << "    \"timestamp\": " << timestamp << "\n";
        out << "  }";

        if(i+1 < frames.size()) out << ",";
        out << "\n";
    }

    out << "]\n";
}

bool already_processed(const std::string& video_id){
    // a video is already processed if it is already in Qdrant
    // we use a simple marker file to track this 
    return fs::exists("/output/.indexed_" + video_id);
}


// Load existing segments from transcripts.json, excluding a set of video_ids that we are about to reprocess
std::vector<Segment> load_existing_segments(const std::set<std::string>& exclude_ids){
    std::vector<Segment> existing;
    fs::path path = "/output/transcripts.json";
    if(!fs::exists(path)) return existing;

    std::ifstream f(path);
    if(!f.is_open()) return existing;

    // line-by-line parser: reads objects between { } and extracts fields
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    auto extract_str = [&](const std::string& obj, const std::string& key) -> std::string {
        std::string search = "\"" + key + "\": \"";
        size_t k = obj.find(search);
        if(k == std::string::npos) return "";
        k += search.size();
        size_t ke = obj.find("\"", k);
        return ke == std::string::npos ? "" : obj.substr(k, ke - k);
    };

    auto extract_num = [&](const std::string& obj, const std::string& key) -> double {
        std::string search = "\"" + key + "\": ";
        size_t k = obj.find(search);
        if(k == std::string::npos) return 0.0;
        k += search.size();
        size_t ke = obj.find_first_of(",\n}", k);
        try { return std::stod(obj.substr(k, ke - k)); }
        catch (...) { return 0.0; }
    };

    size_t pos = 0;
    while((pos = content.find("  {", pos)) != std::string::npos){
        size_t end = content.find("\n  }", pos);
        if(end == std::string::npos) break;
        std::string obj = content.substr(pos, end - pos + 4);

        std::string video_id = extract_str(obj, "video_id");
        if(!video_id.empty() && exclude_ids.find(video_id) == exclude_ids.end()){
            Segment seg;
            seg.video_id   = video_id;
            seg.video_name = extract_str(obj, "video_name");
            seg.segment_id = extract_str(obj, "segment_id");
            seg.text       = extract_str(obj, "text");
            seg.start_time = extract_num(obj, "start_time");
            seg.end_time   = extract_num(obj, "end_time");
            if(!seg.video_name.empty()) existing.push_back(seg);
        }
        pos = end + 4;
    }
    return existing;
}

void write_transcripts(const std::vector<Segment>& segments){
    std::ofstream out("/output/transcripts.json");
    out << "[\n";
    for(size_t i=0; i<segments.size(); ++i){
        const auto& s = segments[i];
        out << "  {\n";
        out << "    \"video_id\": \"" << s.video_id << "\",\n";
        out << "    \"video_name\": \"" << s.video_name << "\",\n";
        out << "    \"segment_id\": \"" << s.segment_id << "\",\n";
        out << "    \"text\": \"" << json_escape(s.text) << "\",\n";
        out << "    \"start_time\": " << s.start_time << ",\n";
        out << "    \"end_time\": " << s.end_time << "\n";
        out << "  }" << (i + 1 <segments.size() ? "," : "") << "\n";
    }
    out << "]\n";
}

void cleanup_audio_files(const fs::path& output_dir, const std::string& video_stem){
    for(const auto& ext: {"wav", "srt", "txt", "vtt"}){
        fs::path p = output_dir / (video_stem + "." + ext);
        if(fs::exists(p)){
            fs::remove(p);
            std::cout << "[Cleanup] Deleted: " << p << std::endl;
        }
    }
}

void process_videos(){
    fs::path videos_dir = "/videos";
    fs::path output_dir = "/output";

    if(!fs::exists(videos_dir)){
        std::cerr << "Error: /videos directory not found" << std::endl;
        return;
    }
    
    // determine which videos need processing
    std::set<std::string> videos_to_process;
    for(const auto& entry: fs::directory_iterator(videos_dir)){
        if(!entry.is_regular_file()) continue;
        std::string video_stem = entry.path().stem().string();
        if(!already_processed(video_stem)){
            videos_to_process.insert(video_stem);
        }else{
            // skip already processed videos
            std::cout << "[Skip] Already processed: " << video_stem << std::endl;
        }
    }

    if(videos_to_process.empty()){
        std::cout << "[Ingestion] All videos already processed\n";
        return;
    }

    // load existing segments for videos we're not processing
    std::vector<Segment> all_segments = load_existing_segments(videos_to_process);
    std::cout << "[Ingestion] Loaded " << all_segments.size() << " existing segments from other videos\n";

    for(const auto& entry: fs::directory_iterator(videos_dir)){
        if(!entry.is_regular_file()) continue;

        std::string video_path = entry.path().string();
        std::string video_stem = entry.path().stem().string();
        std::string video_name = entry.path().filename().string();

        if(videos_to_process.find(video_stem) == videos_to_process.end()) continue;

        // Video to Frames
        fs::path frames_dir = output_dir / "frames" / video_stem;
        fs::create_directories(frames_dir);
        std::string frames_pattern = (frames_dir / "frame_%04d.jpg").string();
        
        // build FFmped commands
        #ifdef USE_CUDA
            std::string ffmpeg_frames_cmd = "ffmpeg -hwaccel cuda -y -i \"" + video_path + "\" -vf \"fps=1/1\" \"" + frames_pattern + "\" > /dev/null 2>&1";
        #else
            std::string ffmpeg_frames_cmd = "ffmpeg -y -i \"" + video_path + "\" -vf \"fps=1/1\" \"" + frames_pattern + "\" > /dev/null 2>&1";
        #endif

        // Video to Audio
        std::string audio_path = (output_dir / (video_stem + ".wav")).string();
        std::string ffmpeg_audio_cmd = "ffmpeg -y -i \"" + video_path + "\" -ar 16000 -ac 1 \"" + audio_path + "\" > /dev/null 2>&1";

        std::cout << "[FFmpeg] Extracting frames and audio: " << video_name << std::endl;

        auto frames_future = std::async(std::launch::async, [&](){ return system(ffmpeg_frames_cmd.c_str()); });
        auto audio_future = std::async(std::launch::async, [&](){ return system(ffmpeg_audio_cmd.c_str()); });

        int frames_result = frames_future.get();
        int audio_result = audio_future.get();

        if(frames_result != 0 || audio_result != 0){
            std::cerr << "FFmpeg failed for " << video_path << std::endl;
            continue;
        }

        write_frames_metadata(frames_dir, 1.0);
        
        // Audio to srt
        std::string srt_prefix = (output_dir / video_stem).string();
        std::string whisper_cmd = 
        "/app/external/whisper/build/bin/whisper-cli "
        "-m /app/external/whisper/models/ggml-small.en.bin "
        "-t 4 "
        "-f \"" + audio_path + "\" "
        "-osrt -otxt -ovtt "
        "-of \"" + srt_prefix + "\"";
        
        std::cout << "[Whisper] Transcribing: " << audio_path << std::endl;
        if(system(whisper_cmd.c_str())!=0){
            std::cerr << "Whisper failed for " << audio_path << std::endl;
            continue;
        }
        
        
        // srt to transcript
        std::ifstream srt_file(srt_prefix + ".srt");
        if(!srt_file.is_open()){
            std::cerr << "Failed to open SRT for " << video_name << std::endl;
            continue;
        }
        
        std::string line;
        while(std::getline(srt_file, line)){
            if(line.empty()) continue;
            std::string index = line;   // subtitle index
            std::getline(srt_file, line);   // timestamp line
            if(line.size() < 29) continue;
            std::string start = line.substr(0, 12);
            std::string end = line.substr(17, 12);
    
            double start_sec = to_seconds(start);
            double end_sec = to_seconds(end);
    
            std::string text, temp;
            while(std::getline(srt_file, temp) && !temp.empty()){
                if(!text.empty()) text += " ";
                text += temp;
            }
    
            Segment seg;
            seg.video_id = video_stem;
            seg.video_name = video_name;
            seg.segment_id = video_stem + "_" + index;
            seg.text = text;
            seg.start_time = start_sec;
            seg.end_time = end_sec;
    
            all_segments.push_back(seg);
        }
        srt_file.close();

        // delete wav/srt/txt/vtt - no longer needed after transcript is built
        cleanup_audio_files(output_dir, video_stem);
        std::cout << "[Ingestion] Cleaned up audio files for: " << video_stem << std::endl;

        // mark as processed so we skip on next trigger
        std::ofstream marker("/output/.indexed_" + video_stem);
        marker << "1";
    }

    write_transcripts(all_segments);
    std::cout << "[Ingestion] transcripts.json written (" << all_segments.size() << " segments)\n";
}


int main() {
    std::cout << "[Ingestion] Waiting for trigger...\n";

    const std::string trigger = "/output/.trigger_ingest";

    while(true){
        if(!fs::exists(trigger)){
            std::this_thread::sleep_for(std::chrono::seconds(5));
            continue;
        }

        std::cout << "[Ingestion] Trigger detected, processing...\n";
        process_videos();

        fs::remove(trigger);
        std::cout << "[Ingestion] Done, trigger cleared\n";
    }
    return 0;
}