#include <torch/torch.h>
#include <math.h>
#include <iostream>
#include <cmath>
#include <limits>
#include <chrono>
#include <ctime>
#include <random>
#include "scheduler.h"

using namespace torch::indexing;

struct PositionalEncodingImpl : torch::nn::Module{
    PositionalEncodingImpl(){

    }
    PositionalEncodingImpl(int64_t d_model, int64_t max_len=5000){
        pe = torch::zeros({max_len, d_model});
        position = torch::arange(0, max_len,
            torch::TensorOptions(torch::kFloat32).requires_grad(false));
        position = position.unsqueeze(1);
        torch::Tensor temp = torch::arange(0, d_model, 2, torch::TensorOptions(torch::kFloat32).requires_grad(false));
        div_term = torch::exp(temp * (std::log(10000.0) / d_model));


        pe.index_put_({Slice(), Slice(0, None, 2)}, torch::sin(position * div_term));
        pe.index_put_({Slice(), Slice(1, None, 2)}, torch::cos(position * div_term));


        
        pe = pe.unsqueeze(0).transpose(0, 1);
        register_parameter("pe", pe);
        register_parameter("position", position);
        register_parameter("div_term", div_term);
        register_buffer("pe", pe);
    }

    torch::Tensor forward(torch::Tensor x){
        x = x + pe.index({Slice(0, x.size(0)), Slice()});
        return x;
    }

    torch::Tensor pe;
    torch::Tensor position;
    torch::Tensor div_term;
};

TORCH_MODULE(PositionalEncoding);

struct TransformerModel : torch::nn::Module{
    TransformerModel(int64_t feature_size = 250, int64_t nlayers = 1, float dropout_p=0.1){
        pos_encoder = PositionalEncoding(feature_size);
        torch::nn::TransformerEncoderLayerOptions elOptions = 
            torch::nn::TransformerEncoderLayerOptions(feature_size, 10);
        torch::nn::TransformerEncoderLayer encoder_layers = torch::nn::TransformerEncoderLayer(
            elOptions.dropout(dropout_p));
        torch::nn::TransformerEncoderOptions enOptions = torch::nn::TransformerEncoderOptions(encoder_layers, nlayers);
        transformer_encoder = torch::nn::TransformerEncoder(enOptions);
        decoder = torch::nn::Linear(feature_size, 1);
        register_module("pos_encoder", pos_encoder);
        register_module("transformer_encoder", transformer_encoder);
        register_module("decoder", decoder);
    }

    void init_weights(){
        float initrange = 0.1;
        decoder->bias.data().zero_();
        decoder->weight.data().uniform_(-initrange, initrange);
    }

    torch::Tensor _generate_square_subsequent_mask(int sz){
        torch::Tensor mask = (torch::triu(torch::ones({sz, sz})) == 1).transpose(0, 1).to(torch::kFloat32);
        mask = mask.masked_fill(mask == 0, -std::numeric_limits<float>::infinity()).masked_fill(mask == 1, 0.f);
        return mask;
    }

    torch::Tensor forward(torch::Tensor src){
        if (false == is_mask_generated){
            torch::Tensor mask = _generate_square_subsequent_mask(src.size(0)).to(src.device());
            src_mask = mask;
            is_mask_generated = true;
        }

        src = pos_encoder(src);
        torch::Tensor output = transformer_encoder(src, src_mask);
        output = decoder(output);
        return output;
    }

    torch::Tensor src_mask;
    bool is_mask_generated = false;
    PositionalEncoding pos_encoder;
    torch::nn::TransformerEncoder transformer_encoder = nullptr;
    torch::nn::Linear decoder = nullptr;
    int64_t ninp;
};

torch::Tensor create_inout_sequences(torch::Tensor input_data, int64_t tw, int64_t output_window = 1){
    torch::Tensor temp = torch::empty({input_data.size(0) - tw, 2, tw}, torch::TensorOptions(torch::kFloat32));
    auto len = input_data.numel();
    auto max_counter = len - tw;
    int64_t k = 0;
    for (auto i = 0; i < max_counter; i++){
        torch::Tensor train_seq = input_data.index({Slice(i, i + tw)});
        temp[i][0] = input_data.index({Slice(i, i + tw)});
        temp[i][1] = input_data.index({Slice(i + output_window, i + tw + output_window)});

    }

    return temp;
}

std::tuple<torch::Tensor, torch::Tensor> get_data(int64_t output_window = 1){
    //construct a little toy dataset
    auto time = torch::arange(0, 400, 0.1);
    auto amplitude = torch::sin(time) + torch::sin(time * 0.05) + torch::sin(time * 0.12);// + dist(mt);

    
    //from sklearn.preprocessing import MinMaxScaler

    
    //looks like normalizing input values curtial for the model
    //scaler = MinMaxScaler(feature_range=(-1, 1)) 
    //amplitude = scaler.fit_transform(series.to_numpy().reshape(-1, 1)).reshape(-1)
    //amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)
    
    
    auto samples = 2600;

    auto train_data = amplitude.index({Slice(None, samples)});
    auto test_data = amplitude.index({Slice(samples, None)});

    //convert our train data into a pytorch train tensor
    auto input_window = 100;

    auto train_sequence = create_inout_sequences(train_data,input_window);
    train_sequence = train_sequence.index({Slice(None,-output_window)});
    
    auto test_sequence = create_inout_sequences(test_data,input_window);
    test_sequence = test_sequence.index({Slice(None,-output_window)});

    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);

    return std::make_tuple(train_sequence.to(device),test_sequence.to(device));
}

std::tuple<torch::Tensor, torch::Tensor> get_batch(torch::Tensor source, int64_t i, int64_t batch_size, int64_t input_window = 100){
    auto seq_len = std::min(batch_size, source.size(0) - i);
    
    auto data = source.index({Slice(i, i + seq_len)});
    auto input = data.index({Slice(), 0, Slice()});
    auto target = data.index({Slice(), 1, Slice()});
    auto temp = input.numel()/100;
    if (temp > 10)
        temp = 10;
    input = torch::reshape(input, {100, temp, 1});
    target = torch::reshape(target, {100, temp, 1});
    return std::make_tuple(input, target);
}


void train(TransformerModel model, torch::Tensor train_data, int64_t num_epochs = 100){
    model.train();
    auto total_loss = 0.0;
    auto start_time = std::chrono::system_clock::now();
    auto batch_size = 10;
    auto batch = 0;

    torch::nn::MSELoss criterion;


    auto learning_rate = 0.005;
    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(learning_rate));
    scheduler::StepLR<decltype(optimizer)> scheduler(optimizer, 1.0, 0.95);

    for(int64_t i = 0; i <= num_epochs; i++){
        auto start_time = std::chrono::system_clock::now();
        std::cout<<"Epoch "<<i<<std::endl;
        batch = 0;
        for (int64_t j = 0; j < train_data.size(0); j = j + batch_size, batch++){
            auto data = get_batch(train_data, j, batch_size);
            optimizer.zero_grad();
            auto output = model.forward(std::get<0>(data));
            
            auto loss = criterion(output, std::get<1>(data));
            loss.backward();
            torch::nn::utils::clip_grad_norm_(model.parameters(), 0.7);
            optimizer.step();
            total_loss += loss.item<double>();
            auto log_interval = int(train_data.size(0)) / (batch_size * 5);
            if (batch != 0 && 0 == batch % log_interval){
                auto curr_loss = total_loss / log_interval;
                auto elapsed = std::chrono::system_clock::now() - start_time;
                std::cout<<"|epoch "<<i<<" | "<<batch<<"/"<<train_data.size(0)/batch_size;
                std::cout<<" batches | "<<(elapsed.count() * 10)<<" ms | loss"<<curr_loss<<std::endl;;
                total_loss = 0;
                start_time = std::chrono::system_clock::now();
            }
        }

        scheduler.step();
    }

    return;
}

void evaluate(TransformerModel model, torch::Tensor eval_data){
    model.eval();
    auto batch_size = 10;
    auto total_loss = 0.0;
    torch::nn::MSELoss criterion;

    std::cout<<"Evaluating:";
    for (int64_t j = 0; j < eval_data.size(0); j = j + batch_size){
            auto data = get_batch(eval_data, j, batch_size);
            auto output = model.forward(std::get<0>(data));
            auto loss = criterion(output, std::get<1>(data));
            total_loss += loss.item<double>();
    }

    std::cout<<"Evaluation Loss: "<<total_loss<<std::endl;
    return;
}

int main(){
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);

    auto model = TransformerModel();
    model.to(device);

    auto data = get_data();
    train(model, std::get<0>(data));
    evaluate(model, std::get<1>(data));

    return 0;
    
}

