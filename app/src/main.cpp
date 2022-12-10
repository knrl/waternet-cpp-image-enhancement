#include <iostream>

#include "net.hpp"
#include "underwaterdataset.hpp"

void train_test() {
    UnderWaterDataset::DatasetStruct train_data_struct {
        "/home/.../dataset/train",
        100, /* size */
        128,  /* H */
        128,  /* W */
        3};  /* C */

    UnderWaterDataset::DatasetStruct test_data_struct {
        "/home/.../dataset/test",
        43, /* size */
        128, /* H */
        128, /* W */
        3}; /* C */

    auto train_dataset = UnderWaterDataset(UnderWaterDataset::Mode::kTrain, train_data_struct)
        .map(torch::data::transforms::Stack<>());
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), train_data_struct.dataset_size);

    auto test_dataset = UnderWaterDataset(UnderWaterDataset::Mode::kTest, test_data_struct)
        .map(torch::data::transforms::Stack<>());
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), test_data_struct.dataset_size);

    torch::Device device = torch::Device(torch::kCPU);

    WaterNet model(true /* bias */, 2 /* padding */, 1 /* dilation */);
    model->to(device);
    model->eval();

    constexpr double learning_rate  = 0.001;
    constexpr double gamma          = 0.1;
    constexpr uint32_t step_size    = 10000;
    constexpr uint32_t num_of_epoch = 10;
    const std::string vgg19_pretrained_path{"/home/.../vgg19_pretrained.pt"};

    torch::jit::script::Module vgg_model;
    try {
        vgg_model = torch::jit::load(vgg19_pretrained_path);
    } catch(const torch::Error& e) {
        std::cerr << e.what();
        std::cout << "An error has occured while model loading." << std::endl;
        exit(EXIT_FAILURE);
    }
    vgg_model.to(device);
    vgg_model.eval();

    torch::optim::Adam optimizer(model->parameters(),
        torch::optim::AdamOptions(learning_rate).betas(std::make_tuple(0.5, 0.999)));
    torch::optim::StepLR scheduler(optimizer, step_size, gamma);

    auto images    = torch::empty({1, train_data_struct.C, train_data_struct.H, train_data_struct.W}, torch::kFloat);
    auto images_wb = torch::empty({1, train_data_struct.C, train_data_struct.H, train_data_struct.W}, torch::kFloat);
    auto images_he = torch::empty({1, train_data_struct.C, train_data_struct.H, train_data_struct.W}, torch::kFloat);
    auto images_gc = torch::empty({1, train_data_struct.C, train_data_struct.H, train_data_struct.W}, torch::kFloat);

    for (size_t epoch=0;epoch<num_of_epoch;++epoch) {
        std::map<std::string, double> epoch_metrics{
            {"loss", 0.0},{"perceptual_loss", 0.0},{"mse_loss", 0.0}};

        for (auto& batch : *train_loader) {
        auto input  = batch.data.to(torch::kCPU);

        for (uint32_t num_of_item=0;num_of_item<train_data_struct.dataset_size;++num_of_item) {
            auto target = batch.target.to(torch::kCPU);
            images[0]    = input[0][num_of_item];
            images_wb[0] = input[1][num_of_item];
            images_he[0] = input[2][num_of_item];
            images_gc[0] = input[3][num_of_item];

            torch::Tensor out = model->forward(images, images_wb, images_he, images_gc);

            // normalization
            torch::Tensor normalized_out_x = torch::data::transforms::Normalize<>(
                {0.485, 0.456, 0.406}, {0.229, 0.224, 0.225})(out);
            torch::Tensor normalized_out_y = torch::data::transforms::Normalize<>(
                {0.485, 0.456, 0.406}, {0.229, 0.224, 0.225})(target);

            torch::Tensor vgg_out_x = vgg_model.forward({normalized_out_x}).toTensor();
            torch::Tensor vgg_out_y = vgg_model.forward({normalized_out_y}).toTensor();

            // metric calculation and tensor operations
            torch::Tensor perceptual_dist = torch::square(
                torch::mul(torch::sub(vgg_out_x, vgg_out_y), 255.0));
            torch::Tensor perceptual_loss = torch::mean(perceptual_dist);
            torch::Tensor mse_loss = torch::mean(torch::square(torch::mul(torch::sub(out, target), 255.0)));
            torch::Tensor loss_var = torch::add(torch::mul(perceptual_loss, 0.05), mse_loss);

            // optimization phase
            optimizer.zero_grad();
            loss_var.backward();
            optimizer.step();
            scheduler.step();

            // and other loss and metric things
            epoch_metrics["loss"]            += loss_var.item<double>();
            epoch_metrics["perceptual_loss"] += perceptual_loss.item<double>();
            epoch_metrics["mse_loss"]        += mse_loss.item<double>();
        }}

        std::cout << "[Epoch: "           << epoch + 1  
                  << "] loss: "           << epoch_metrics["loss"] / train_data_struct.dataset_size
                  << " Perceptual loss: " << epoch_metrics["peceptual_loss"] / train_data_struct.dataset_size
                  << " MSE loss: "        << epoch_metrics["mse_loss"] / train_data_struct.dataset_size
                  << std::endl; 

        /*** Val/Test ***/
        {
            torch::NoGradGuard no_grad;
            std::map<std::string, double> test_metrics{
                {"loss", 0.0},{"perceptual_loss", 0.0},{"mse_loss", 0.0}};

            for (auto& test_batch : *test_loader) {
            auto input  = test_batch.data.to(torch::kCPU);

            for (uint32_t num_of_item=0;num_of_item<test_data_struct.dataset_size;++num_of_item) {
                auto target  = test_batch.target.to(torch::kCPU);
                images[0]    = input[0][num_of_item];
                images_wb[0] = input[1][num_of_item];
                images_he[0] = input[2][num_of_item];
                images_gc[0] = input[3][num_of_item];

                torch::Tensor out = model->forward(images, images_wb, images_he, images_gc);

                // normalization
                torch::Tensor normalized_out_x = torch::data::transforms::Normalize<>(
                    {0.485, 0.456, 0.406}, {0.229, 0.224, 0.225})(out);
                torch::Tensor normalized_out_y = torch::data::transforms::Normalize<>(
                    {0.485, 0.456, 0.406}, {0.229, 0.224, 0.225})(target);

                torch::Tensor vgg_out_x = vgg_model.forward({normalized_out_x}).toTensor();
                torch::Tensor vgg_out_y = vgg_model.forward({normalized_out_y}).toTensor();

                // metric calculation and tensor operations
                torch::Tensor perceptual_dist = torch::square(
                    torch::mul(torch::sub(vgg_out_x, vgg_out_y), 255.0));
                torch::Tensor perceptual_loss = torch::mean(perceptual_dist);
                torch::Tensor mse_loss = torch::mean(torch::square(torch::mul(torch::sub(out, target), 255.0)));
                torch::Tensor loss_var = torch::add(torch::mul(perceptual_loss, 0.05), mse_loss);

                // and other loss and metric things
                test_metrics["loss"]            += loss_var.item<double>();
                test_metrics["perceptual_loss"] += perceptual_loss.item<double>();
                test_metrics["mse_loss"]        += mse_loss.item<double>();
            }}

            // print the test metrics
            std::cout << "[Test    ] loss: "  << test_metrics["loss"] / test_data_struct.dataset_size
                      << " Perceptual loss: " << test_metrics["peceptual_loss"] / test_data_struct.dataset_size
                      << " MSE loss: "        << test_metrics["mse_loss"] / test_data_struct.dataset_size
                      << std::endl; 
        }
    }
}

int main(int argc, const char* argv[]) {
    train_test();
    return EXIT_SUCCESS;
}
