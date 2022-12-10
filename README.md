# waternet-cpp-image-enhancement

A Libtorch implementation of an image enhancement deep learning algorithm (<a href="https://github.com/tnwei/waternet">waternet</a>) with C++. 

### Requirements
1. LibTorch v1.12.1
2. OpenCV v4.0.0

### To compile
1. cmake 3.25 +
2. gcc 9.4 +
3. Don't forget to indicate libtorch path ``` -DCMAKE_PREFIX_PATH="_libtorch_path" ```

### Citation
The model that used in this project.

```
@article{li2019underwater,
  title={An underwater image enhancement benchmark dataset and beyond},
  author={Li, Chongyi and Guo, Chunle and Ren, Wenqi and Cong, Runmin and Hou, Junhui and Kwong, Sam and Tao, Dacheng},
  journal={IEEE Transactions on Image Processing},
  volume={29},
  pages={4376--4389},
  year={2019},
  publisher={IEEE}
}
```
