#include "utils.h"
#include "dispatcher.h"
#include <string.h>

bool file_exists(const char* filename) {
    struct stat buffer;
    return (stat(filename, &buffer) == 0);
}

void load_binary_file(const char* filename, float* data, size_t size) {
    FILE* file = fopen(filename, "rb");
    if (!file) ERR("fopen");

    size_t read = fread(data, sizeof(float), size, file);
    fclose(file);

    if (read != size) ERR("fread");
}


void save_binary_file(const char* filename, const float* data, size_t size) {
    FILE* file = fopen(filename, "wb");
    if (!file) ERR("fopen");

    size_t written = fwrite(data, sizeof(float), size, file);
    fclose(file);

    if (written != size) ERR("fwrite");
}


void parse_config_string(const char* config_str, int* batch_size, int* num_heads, int* seq_len, int* head_dim) {
    const char* folder_name = strrchr(config_str, '/');
    if (folder_name) {
        folder_name++; // skip the last '/'
    } else {
        folder_name = config_str; // no path separator, use the whole string
    }
    
    int parsed = sscanf(folder_name, "B%d_H%d_S%d_D%d", batch_size, num_heads, seq_len, head_dim);
    if (parsed != 4) ERR("sscanf");
}



void parse_args(int argc, char** argv, ComputeDataType* compute_data_type, ComputeType* compute_method, ModeType* mode, char** data_path)
{
    if (argc < 5) {
        usage(argv[0]);
    }

    // Parse compute method
    if (strcmp(argv[1], "fa2") == 0)
    {
        *compute_method = ComputeType::FlashAttention2;
    }
    else if (strcmp(argv[1], "fa1") == 0)
    {
        *compute_method = ComputeType::FlashAttention1;
    }
    else if (strcmp(argv[1], "naive") == 0)
    {
        *compute_method = ComputeType::Naive;
    }
    else usage(argv[0]);

    // Parse mode
    if (strcmp(argv[2], "forward") == 0)
    {
        *mode = ModeType::Forward;
    }
    else if (strcmp(argv[2], "backward") == 0)
    {
        *mode = ModeType::Backward;
    }
    else if (strcmp(argv[2], "forward_backward") == 0)
    {
        *mode = ModeType::ForwardBackward;
    }
    else usage(argv[0]);

    // Parse compute data type
    if (strcmp(argv[3], "fp16") == 0)
    {
        *compute_data_type = ComputeDataType::FP16;
    }
    else if (strcmp(argv[3], "fp32") == 0)
    {
        *compute_data_type = ComputeDataType::FP32;
    }
    else usage(argv[0]);

    *data_path = argv[4];
}

