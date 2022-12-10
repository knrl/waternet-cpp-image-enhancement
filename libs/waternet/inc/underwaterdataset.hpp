#include <filesystem>
#include <vector>
#include <string>

#include <torch/data/example.h>
#include <torch/data/datasets/base.h>
#include <torch/types.h>
#include <torch/script.h>
#include <torch/torch.h>

#include "preprocess.hpp"

namespace fs = std::filesystem;

class UnderWaterDataset : public torch::data::datasets::Dataset<UnderWaterDataset> {
public:
   typedef struct {
      std::string dataset_path;
      uint32_t dataset_size;
      uint32_t H;
      uint32_t W;
      uint32_t C;
   } DatasetStruct;

   // The mode in which the dataset is loaded
   enum Mode { kTrain, kTest };

   explicit UnderWaterDataset(Mode mode, const DatasetStruct& dataSt);

   std::pair<std::vector<torch::Tensor>, torch::Tensor> read_data(bool train);

   std::vector<torch::Tensor> readAnImageReturnTensorVec(std::string imagePath);

   torch::Tensor matToTorchTensor(cv::Mat img);

   // Returns the `Example` at the given `index`.
   torch::data::Example<> get(size_t index) override;

   // Returns the size of the dataset.
   torch::optional<size_t> size() const override;

   // Returns true if this is the training subset of UnderWaterDataset.
   bool is_train() const noexcept;

   // Returns all images stacked into a single tensor.
   const std::vector<torch::Tensor>& images() const;

   // Returns all targets stacked into a single tensor.
   const torch::Tensor& targets() const;

 private:
   std::vector<torch::Tensor> images_;
   torch::Tensor targets_;
   Mode mode_;
   DatasetStruct dataSt;
};