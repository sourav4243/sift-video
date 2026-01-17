#include <iostream>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

int main() {
    std::cout << "Ingestion service started" << std::endl;

    fs::path directoryPath = "/videos";

    if(!fs::exists(directoryPath)){
        std::cerr << "Error: /videos directory not found" << std::endl;
        return 1;
    }

    for(const auto& entry: fs::directory_iterator(directoryPath)){
        if(entry.is_regular_file()){
            std:: cout << "Found files: " << entry.path().filename() << std::endl;
        }
    }
    return 0;
}