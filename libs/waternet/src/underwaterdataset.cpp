#include "underwaterdataset.hpp"

torch::Tensor UnderWaterDataset::matToTorchTensor(cv::Mat img) {
    torch::Device device = torch::Device(torch::kCPU);
    torch::Tensor tensor_image = torch::from_blob(img.data, {1, dataSt.H, dataSt.W, dataSt.C}, torch::kByte);
    tensor_image = tensor_image.to(device);
    tensor_image = tensor_image.permute({0, 3, 1, 2});
    tensor_image = tensor_image.to(torch::kFloat);
    tensor_image = tensor_image.div(255.0);
    return tensor_image;
}

std::vector<torch::Tensor> UnderWaterDataset::readAnImageReturnTensorVec(std::string imagePath) {
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
    if(img.empty()) {
        std::cout << "Could not read the image: " << imagePath << std::endl;
        return std::vector<torch::Tensor>();
    }
    cv::resize(img, img, cv::Size(dataSt.W, dataSt.H), 0, 0, cv::INTER_CUBIC); // INTER_LINEAR

    preprocess preObj;
    cv::Mat histEqIMg;
    cv::Mat gamCorIMg;
    cv::Mat whiteBalanceTranIMg;
    preObj.gamma_correction(       img /* source */, gamCorIMg           /* destination */, 1.0);
    preObj.histeq(                 img /* source */, histEqIMg           /* destination */);
    preObj.white_balance_transform(img /* source */, whiteBalanceTranIMg /* destination */);

    std::vector<torch::Tensor> modelInput;
    modelInput.push_back(matToTorchTensor(img));                    // [0] tensor_image
    modelInput.push_back(matToTorchTensor(whiteBalanceTranIMg));    // [1] tensor_wb_image
    modelInput.push_back(matToTorchTensor(histEqIMg));              // [2] tensor_he_image
    modelInput.push_back(matToTorchTensor(gamCorIMg));              // [3] tensor_gc_image

    return modelInput;
}

std::pair<std::vector<torch::Tensor>, torch::Tensor> UnderWaterDataset::read_data(bool train) {
    auto images    = torch::empty({dataSt.dataset_size, dataSt.C, dataSt.H, dataSt.W}, torch::kFloat);
    auto images_wb = torch::empty({dataSt.dataset_size, dataSt.C, dataSt.H, dataSt.W}, torch::kFloat);
    auto images_he = torch::empty({dataSt.dataset_size, dataSt.C, dataSt.H, dataSt.W}, torch::kFloat);
    auto images_gc = torch::empty({dataSt.dataset_size, dataSt.C, dataSt.H, dataSt.W}, torch::kFloat);
    auto targets   = torch::empty({dataSt.dataset_size, dataSt.C, dataSt.H, dataSt.W}, torch::kFloat);

    // root = "<absolute-path>/dataset/"
    std::string data_file  {dataSt.dataset_path + "/data"};
    std::string target_file{dataSt.dataset_path + "/target"};

    uint32_t i{0};
    std::string ext{".png"};
    cv::Mat target_img;
    torch::Tensor target_tensor;
    std::vector<torch::Tensor> img_tensor_vec;
    for (const auto& p : fs::directory_iterator(data_file)) {
        if (p.path().extension() == ext) {
            std::string s{p.path()};

            img_tensor_vec = readAnImageReturnTensorVec(s);
            if (img_tensor_vec.empty()) 
                continue;

            target_img = cv::imread(target_file + "/" + s.substr(47, s.find(ext)), cv::IMREAD_COLOR);
            if(target_img.empty()) 
                continue;

            cv::resize(target_img, target_img, cv::Size(dataSt.W, dataSt.H), 0, 0, cv::INTER_CUBIC);
            target_tensor = matToTorchTensor(target_img);
            targets[i]    = target_tensor[0];

            images[i]    = img_tensor_vec[0][0];
            images_wb[i] = img_tensor_vec[1][0];
            images_he[i] = img_tensor_vec[2][0];
            images_gc[i] = img_tensor_vec[3][0];

            ++i;
            if (i == dataSt.dataset_size)
                break;
        }
    }

    // static_cast<std::pair<std::vector<torch::Tensor>, torch::Tensor>>
    return {std::vector<torch::Tensor>{images, images_wb, images_he, images_gc}, targets};
}

UnderWaterDataset::UnderWaterDataset(Mode mode, const UnderWaterDataset::DatasetStruct& dataSt) 
    : mode_(mode), dataSt(dataSt) {
    auto data = read_data(mode == Mode::kTrain);
    images_   = std::move(data.first);
    targets_  = std::move(data.second);
}

torch::data::Example<> UnderWaterDataset::get(size_t index) { return {images_[index], targets_[index]}; }

torch::optional<size_t> UnderWaterDataset::size() const { return images_.size(); } // images_[0].sizes()[0];

bool UnderWaterDataset::is_train() const noexcept { return mode_ == Mode::kTrain; }

const std::vector<torch::Tensor>& UnderWaterDataset::images() const { return images_; }

const torch::Tensor& UnderWaterDataset::targets() const { return targets_; }