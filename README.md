# Encoder-Decoder Image Captioning using PyTorch
---
This project aims to generate captions for images using an encoder-decoder architecture implemented in PyTorch. The model is trained on the Flickr8k dataset, a widely used benchmark dataset for image captioning tasks. The implementation provides a user-friendly interface deployed on Streamlit Cloud for generating captions for custom images.

https://github.com/aryan-garg0729/Image-Captioning-Pytorch/assets/155893692/9fb22f42-c370-4a07-9621-be5be2bcb544

## Installation

To run the project locally, follow these steps:

1. Clone this repository:

    ```
    git clone https://github.com/aryan-garg0729/Image-Captioning-Pytorch.git
    ```

2. Install the required dependencies:

    ```
    pip install -r requirements.txt
    ```

## Usage

1. Navigate to the project directory:

    ```
    cd Image-Captioning-Pytorch
    ```

2. Run the Streamlit app:

    ```
    streamlit run app.py
    ```

3. Upload an image for which you want to generate a caption.

## Model Details

The model architecture follows the standard encoder-decoder framework for image captioning:

- **Encoder**: ResNet-50 pretrained on ImageNet is used to extract features from the input image. The output is a feature vector representing the image.
  
- **Decoder**: A recurrent neural network (RNN) with LSTM cells is employed to generate captions based on the image features. The decoder predicts the next word in the caption conditioned on the previously generated words and the image features.

The model is trained using the Flickr8k dataset, consisting of 8,000 images each paired with five captions. The training objective is to minimize the cross-entropy loss between the predicted captions and the ground truth captions.

## Cherry-picked Results
Here are some cherry-picked results showcasing the caption generation capabilities of the model:

| Image | Generated Caption |
| --- | --- |
| <img src="https://github.com/aryan-garg0729/Image-Captioning-Pytorch/assets/155893692/c7e34daa-4d3f-423e-9d6c-084d59152d46" width="200" height="200"> | a black dog running through the water |
| <img src="https://github.com/aryan-garg0729/Image-Captioning-Pytorch/assets/155893692/814979bf-a478-4436-8dea-ae76b5a5fa86" width="200" height="200"> | two people are riding bicycles along a rocky path |
| <img src="https://github.com/aryan-garg0729/Image-Captioning-Pytorch/assets/155893692/d5132d1a-70aa-4f69-876f-c6f341767679" width="200" height="200"> | a man is standing on a snowy mountain with a mountain in the background |
| <img src="https://github.com/aryan-garg0729/Image-Captioning-Pytorch/assets/155893692/25731157-10d9-48e2-b2d3-30decc95e725" width="200" height="200"> | a boy is jumping off of a water fountain |
   
Feel free to explore the project and generate captions for your own images

## Acknowledgements

---

- The implementation of the encoder-decoder architecture is based on the PyTorch tutorials and the Flickr8k dataset is obtained from the official website.
- Special thanks to Streamlit for providing an easy-to-use platform for deploying machine learning applications.
  
## License

This project is licensed under the MIT License
