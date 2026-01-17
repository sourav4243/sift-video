#include <iostream>
#include <filesystem>
#include <cstdlib>
#include <fstream>

namespace fs = std::filesystem;

int main() {
    std::cout << "Ingestion service started" << std::endl;

    fs::path videos_dir = "/videos";
    fs::path output_dir = "/output";

    if(!fs::exists(videos_dir)){
        std::cerr << "Error: /videos directory not found" << std::endl;
        return 1;
    }

    for(const auto& entry: fs::directory_iterator(videos_dir)){
        if(!entry.is_regular_file()) continue;

        std::string video_path = entry.path().string();
        std::string video_name = entry.path().stem().string();

        std::string audio_out = (output_dir / (video_name + ".wav")).string();
        
        std::string cmd = "ffmpeg -y -i \"" + video_path + "\" -ar 16000 -ac 1 \"" + audio_out + "\"";

        std::cout << "Extracting audio: " << audio_out << std::endl;

        int ret = system(cmd.c_str());
        if(ret!=0){
            std::cerr << "FFmpeg failed for " << video_path << std::endl;
        }
    }
    return 0;
}