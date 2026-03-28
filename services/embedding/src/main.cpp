#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <fstream>
#include <filesystem>
#include <unistd.h>
#include <thread>
#include <chrono>

namespace fs = std::filesystem;

std::vector<float> preprocess(const std::string& path){
    cv::Mat img = cv::imread(path);

    if(img.empty()) throw std::runtime_error("Failed to load image");

    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::resize(img, img, cv::Size(256, 256));
    img.convertTo(img, CV_32F, 1.0/255.0);
    
    std::vector<float> tensor(3*256*256);

    for(int y=0; y<256; y++){
        for(int x=0; x<256; x++){
            cv::Vec3f p = img.at<cv::Vec3f>(y, x);
            for(int c=0; c<3; c++){
                tensor[c*256*256 + y*256 + x] = p[c];
            }
        }
    }

    return tensor;
}

void save_embedding(const std::string& path, float* data, size_t size){
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<char*>(data), size* sizeof(float));
}

void process_embeddingss(Ort::Session& session, Ort::MemoryInfo& mem){
    std::vector<int64_t> image_shape = {1, 3, 256, 256};
    
    const char* input_names[] = {"pixel_values"};
    const char* output_names[] = {"image_embeddings"};
    
    fs::path frames_root = "/output/frames";
    fs::path embed_root = "/output/embeddings";

    for(const auto& video_dir: fs::directory_iterator(frames_root)){
        if(!video_dir.is_directory()) continue;

        std::string video_name = video_dir.path().filename().string();

        fs::path out_dir = embed_root / video_name;
        fs::create_directories(out_dir);

        for(const auto& frame: fs::directory_iterator(video_dir.path())){
            if(frame.path().extension() != ".jpg") continue;

            std::string frame_path = frame.path().string();
            std::string frame_name = frame.path().stem().string();
            std::string out_path = (out_dir / (frame_name + ".bin")).string();
            
            // skip already embedded videos
            if(fs::exists(out_path)) continue;

            std::vector<float> image_tensor = preprocess(frame_path);

            Ort::Value pixel_values = Ort::Value::CreateTensor<float>(
                mem,
                image_tensor.data(),
                image_tensor.size(),
                image_shape.data(),
                image_shape.size()
            );
        
            Ort::Value inputs[] = { std::move(pixel_values) };

            auto outputs = session.Run(
                Ort::RunOptions{nullptr},
                input_names, inputs, 1,
                output_names, 1
            );

            float* embed = outputs[0].GetTensorMutableData<float>();
            size_t embed_size = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();


            save_embedding(out_path, embed, embed_size);

            std::cout << "[Embedding] Embedded: " << frame_name << std::endl;
        }
    }
    return;
}

int main(){
    const std::string trigger = "/output/.trigger_embed";
    const std::string model_path = "models/visual.onnx";

    // env is cheap (memory-wise): just a logging/thread context, can stay alive
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "clip");

    std::cout << "[Embedding] Waiting for trigger...\n";

    while(true){
        if(!fs::exists(trigger)){
            std::this_thread::sleep_for(std::chrono::seconds(5));
            continue;
        }

        std::cout << "[Embedding] Trigger detected, loading model...\n";

        // session of model created here - memory allocated only if model is required (lazy loading)
        {
            Ort::SessionOptions opts;

            // reduce internal thread pool of ONNX runtime, to save some RAM
            opts.SetIntraOpNumThreads(2);
            opts.SetInterOpNumThreads(1);

            // GPU offload when compiled with -DUSE_CUDA=1
            #ifdef USE_CUDA
            {
                OrtCUDAProviderOptions cuda_opts{};
                cuda_opts.device_id = 0;
                cuda_opts.arena_extend_strategy = 0;
                cuda_opts.gpu_mem_limit = 2ULL * 1024 * 1024 * 1024;    // 2GB VRAM cap
                opts.AppendExecutionProvider_CUDA(cuda_opts);
                std::cout << "[Embedding] CUDA execution provider enabled\n";
            }
            #endif

            Ort::Session session(env, model_path.c_str(), opts);
            Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            
            std::cout << "[Embedding] Model loaded, processing...\n";
            process_embeddingss(session, mem);

            // session and mem out of scope here: destructor releases ~450MB of model memory
        }

        fs::remove(trigger);

        std::cout << "[Embedding] Done, model unloaded, trigger cleared\n";
    }
    
    return 0;
}