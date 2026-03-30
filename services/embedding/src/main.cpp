#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#include <fstream>
#include <filesystem>
#include <unistd.h>
#include <sys/wait.h>
#include <thread>
#include <chrono>
#include <future>

namespace fs = std::filesystem;

void save_embedding(const std::string& path, float* data, size_t size){
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<char*>(data), size* sizeof(float));
}

void process_embeddings(Ort::Session& session, Ort::MemoryInfo& mem){
    #ifdef USE_CUDA
        const int BATCH_SIZE = 32;
    #else
        const int BATCH_SIZE = 1;
    #endif
    
    const char* input_names[] = {"pixel_values"};
    const char* output_names[] = {"image_embeddings"};
    
    fs::path frames_root = "/output/frames";
    fs::path embed_root = "/output/embeddings";

    for(const auto& video_dir: fs::directory_iterator(frames_root)){
        if(!video_dir.is_directory()) continue;

        std::string video_name = video_dir.path().filename().string();
        fs::path out_dir = embed_root / video_name;
        fs::create_directories(out_dir);

        std::vector<std::string> batch_paths;
        std::vector<std::string> batch_names;

        for(const auto& frame: fs::directory_iterator(video_dir.path())){
            if(frame.path().extension() != ".jpg") continue;
            std::string frame_name = frame.path().stem().string();
            std::string out_path = (out_dir / (frame_name + ".bin")).string();
            if(fs::exists(out_path)) continue;  // skip already embedded frames
            batch_paths.push_back(frame.path().string());
            batch_names.push_back(frame_name);
        }
            
        for(size_t i = 0; i < batch_paths.size(); i += BATCH_SIZE){
            size_t current_batch_size = std::min((size_t)BATCH_SIZE, batch_paths.size() - i);
            
            struct ThreadResult {
                bool valid = false;
                cv::Mat img;
                std::string name;
            };

            std::vector<ThreadResult> thread_results(current_batch_size);
            std::vector<std::future<void>> futures;

            for(size_t j = 0; j < current_batch_size; j++){
                futures.push_back(std::async(std::launch::async,
                [this_path = batch_paths[i + j], this_name = batch_names[i + j], &thread_results, j](){
                    cv::Mat img = cv::imread(this_path);
                    if(!img.empty()){
                        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
                        thread_results[j] = {true, img, this_name};
                    }
                }));
            }

            for(auto& f: futures) f.get();

            std::vector<cv::Mat> imgs;
            std::vector<std::string> valid_names;
            for(const auto& res: thread_results){
                if(res.valid){
                    imgs.push_back(res.img);
                    valid_names.push_back(res.name);
                }
            }

            size_t actual_batch_size = imgs.size();
            if(actual_batch_size == 0) continue;

            cv::Mat blob = cv::dnn::blobFromImages(imgs, 1.0/255.0, cv::Size(256, 256), cv::Scalar(), false, false);

            std::vector<float> image_tensor;
            if(blob.isContinuous()){
                image_tensor.assign((float*)blob.datastart, (float*)blob.dataend);
            }

            std::vector<int64_t> image_shape = {(int64_t)actual_batch_size, 3, 256, 256};

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
            size_t single_embed_size = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount() / actual_batch_size;

            for(size_t j = 0; j < actual_batch_size; j++){
                std::string out_path = (out_dir / (valid_names[j] + ".bin")).string();
                save_embedding(out_path, embed + (j * single_embed_size), single_embed_size);
                std::cout << "[Embedding] Embedded: " << valid_names[j] << std::endl;

            }
        }

        // delete frames for this video after embedding - no longer needed
        std::error_code ec;
        fs::remove_all(video_dir.path(), ec);
        if(!ec){
            std::cout << "[Embedding] Deleted frames for: " << video_name << std::endl;
        }
    }
    return;
}

void run_embedding_worker(){
    const std::string model_path = "models/visual.onnx";

    cv::setNumThreads(1);
    
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "clip");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(2);
    opts.SetInterOpNumThreads(1);
    
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
    process_embeddings(session, mem);
    std::cout << "[Embedding] Done\n";
    // child exits here - OS reclaims ALL memory including ONNX arena
}

int main(){
    const std::string trigger = "/output/.trigger_embed";

    std::cout << "[Embedding] Waiting for trigger...\n";

    while(true){
        if(!fs::exists(trigger)){
            std::this_thread::sleep_for(std::chrono::seconds(5));
            continue;
        }

        std::cout << "[Embedding] Trigger detected, forking worker...\n";

        pid_t pid = fork();

        if(pid == 0){
            // child process does all the work and then exits
            // when it exits, OS reclaims everything including ONNX arena memory
            run_embedding_worker();
            fs::remove(trigger);
            exit(0);
        }else if(pid > 0){
            // parent process just waits for child to finish
            int status;
            waitpid(pid, &status, 0);

            if(WIFEXITED(status) && WEXITSTATUS(status) == 0){
                std::cout << "[Embedding] Worker finished, memory fully released\n";
            } else {
                std::cerr << "[Embedding] Worker failed with status " << status << "\n";
                // remove trigger anyway so pipeline doesn't get stuck
                fs::remove(trigger);
            }
        }else{
            std::cerr << "[Embedding] fork() failed, falling back to in-process execution\n";
            // fallback: run in-process if fork fails (shouldn't happen in Docker)
            run_embedding_worker();
            fs::remove(trigger);
        } 
    }
    
    return 0;
}