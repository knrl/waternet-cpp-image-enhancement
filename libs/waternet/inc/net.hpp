#include <torch/torch.h>
#include <vector>

class ConfidenceMapGeneratorImpl : public torch::nn::Module {
public:
    ConfidenceMapGeneratorImpl(bool bias, uint_fast8_t pad, uint_fast8_t dilation) 
    : conv1(torch::nn::Conv2dOptions(12, 128, 7)  /* (in_channels, out_channels, kernel_size)*/
        .bias(bias)
        .padding(pad)
        .dilation(dilation)
      ),
    conv2(torch::nn::Conv2dOptions(128, 128, 5)  
        .bias(bias)
        .padding(pad)
        .dilation(dilation)
      ),
    conv3(torch::nn::Conv2dOptions(128, 128, 3)  
        .bias(bias)
        .padding(pad)
        .dilation(dilation)
      ),
    conv4(torch::nn::Conv2dOptions(128, 64, 1)  
        .bias(bias)
        .padding(pad)
        .dilation(dilation)
      ),
    conv5(torch::nn::Conv2dOptions(64, 64, 7)  
        .bias(bias)
        .padding(pad)
        .dilation(dilation)
      ),
    conv6(torch::nn::Conv2dOptions(64, 64, 5)  
        .bias(bias)
        .padding(1)
        .dilation(dilation)
      ),
    conv7(torch::nn::Conv2dOptions(64, 64, 3)  
        .bias(bias)
        .padding(1)
        .dilation(dilation)
      ),
    conv8(torch::nn::Conv2dOptions(64, 3, 3)  
        .bias(bias)
        .padding(1)
        .dilation(dilation))
    {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("conv4", conv4);
        register_module("conv5", conv5);
        register_module("conv6", conv6);
        register_module("conv7", conv7);
        register_module("conv8", conv8);
    }

    std::vector<torch::Tensor> forward(torch::Tensor x, torch::Tensor wb, torch::Tensor he, torch::Tensor gc) {
        x = torch::cat({x, wb, he, gc}, 1); /* ({tensors}, dimension)*/
        x = torch::relu(conv1->forward(x));  // nn::ReLU relu1; relu1(nn::ReLUOptions(...).); x = relu1(conv1(x));
        x = torch::relu(conv2->forward(x));
        x = torch::relu(conv3->forward(x));
        x = torch::relu(conv4->forward(x));
        x = torch::relu(conv5->forward(x));
        x = torch::relu(conv6->forward(x));
        x = torch::relu(conv7->forward(x));
        x = torch::sigmoid(conv8->forward(x));
        auto splits = x.chunk(3, 1);    /* (patches, dimension)*/
        return splits;
        // tempTensor[0] = splits[0];
        // tempTensor[1] = splits[1];
        // tempTensor[2] = splits[2];
    }

private:
    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::Conv2d conv3;
    torch::nn::Conv2d conv4;
    torch::nn::Conv2d conv5;
    torch::nn::Conv2d conv6;
    torch::nn::Conv2d conv7;
    torch::nn::Conv2d conv8;
}; TORCH_MODULE(ConfidenceMapGenerator);


class RefinerImpl : public torch::nn::Module {
public:
    RefinerImpl(bool bias, uint_fast8_t pad, uint_fast8_t dilation)
    : conv1(torch::nn::Conv2dOptions(6, 32, 7)  
        .bias(bias)
        .padding(pad)
        .dilation(dilation)
      ),
    conv2(torch::nn::Conv2dOptions(32, 32, 5)  
        .bias(bias)
        .padding(pad)
        .dilation(dilation)
      ),
    conv3(torch::nn::Conv2dOptions(32, 3, 3)  
        .bias(bias)
        .padding(pad)
        .dilation(dilation))
    {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor xbar) {
        x = torch::cat({x, xbar}, 1);     /* ({tensors}, dimension)*/
        x = torch::relu(conv1->forward(x));
        x = torch::relu(conv2->forward(x));
        x = torch::relu(conv3->forward(x));
        return x;
    }

private:
    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::Conv2d conv3;
}; TORCH_MODULE(Refiner);


class WaterNetImpl : public torch::nn::Module {
public:
    WaterNetImpl(bool bias, uint_fast8_t pad, uint_fast8_t dilation)
    : wb_refiner(Refiner(bias, pad, dilation)),
      he_refiner(Refiner(bias, pad, dilation)),
      gc_refiner(Refiner(bias, pad, dilation)),
      cmg(ConfidenceMapGenerator(bias, pad, dilation))
    {
        register_module("wb_refiner", wb_refiner);
        register_module("he_refiner", he_refiner);
        register_module("gc_refiner", gc_refiner);
        register_module("cmg", cmg);
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor wb, torch::Tensor he, torch::Tensor gc) {
        torch::Tensor x_0 = x.clone();
        torch::Tensor x_1 = x.clone();
        torch::Tensor x_2 = x.clone();

        std::vector<torch::Tensor> temp = cmg->forward(x, wb, he, gc);
        torch::Tensor refined_wb = wb_refiner->forward(x_0, wb);
        torch::Tensor refined_he = he_refiner->forward(x_1, he);
        torch::Tensor refined_gc = gc_refiner->forward(x_2, gc);

        torch::Tensor result = torch::mul(refined_wb, temp[0]) + 
                               torch::mul(refined_he, temp[1]) +
                               torch::mul(refined_gc, temp[2]);
        return result;
    }

private:
    Refiner wb_refiner;
    Refiner he_refiner;
    Refiner gc_refiner;
    ConfidenceMapGenerator cmg;
}; TORCH_MODULE(WaterNet);