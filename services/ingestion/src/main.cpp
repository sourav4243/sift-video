#include <iostream>
#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdio>

namespace fs = std::filesystem;

std:: string json_escape(const std::string &input){
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

int main() {
    fs::path videos_dir = "/videos";
    fs::path output_dir = "/output";

    if(!fs::exists(videos_dir)){
        std::cerr << "Error: /videos directory not found" << std::endl;
        return 1;
    }
    
    std::vector<Segment>all_segments;
    
    for(const auto& entry: fs::directory_iterator(videos_dir)){
        if(!entry.is_regular_file()) continue;

        std::string video_path = entry.path().string();
        std::string video_stem = entry.path().stem().string();      // video
        std::string video_name = entry.path().filename().string();  // video.mp4

        // Video to Audio
        std::string audio_path = (output_dir / (video_stem + ".wav")).string();
        
        std::string ffmpeg_cmd = "ffmpeg -y -i \"" + video_path + "\" -ar 16000 -ac 1 \"" + audio_path + "\"";

        std::cout << "[FFmpeg] Extracting audio: " << audio_path << std::endl;

        if(system(ffmpeg_cmd.c_str())!=0){
            std::cerr << "FFmpeg failed for " << video_path << std::endl;
            continue;
        }
      
        // Audio to srt
        std::string srt_prefix = (output_dir / video_stem).string();
        
        std::string whisper_cmd = 
        "/app/external/whisper/build/bin/whisper-cli "
        "-m /app/external/whisper/models/ggml-small.en.bin "
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
    }



    // write transcript.json
    std::ofstream out("/output/transcripts.json");
    out << "[\n";

    for(size_t i=0; i<all_segments.size(); ++i){
        const auto& s = all_segments[i];
        out << "  {\n";
        out << "    \"video_id\": \"" << s.video_id << "\",\n";
        out << "    \"video_name\": \"" << s.video_name << "\",\n";
        out << "    \"segment_id\": \"" << s.segment_id << "\",\n";
        out << "    \"text\": \"" << json_escape(s.text) << "\",\n";
        out << "    \"start_time\": " << s.start_time << ",\n";
        out << "    \"end_time\": " << s.end_time << "\n";
        out << "  }" << (i + 1 <all_segments.size()? ",":"") << "\n";
    }

    out << "]\n";

    std::cout << "transcript.json generated" << std::endl;
    return 0;
}