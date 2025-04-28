# ✨ DigitGAN - Handwritten Digit Generation with PyTorch

![PyTorch](https://img.shields.io/badge/Built%20With-PyTorch-red?style=for-the-badge&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-App-orange?style=for-the-badge&logo=streamlit)

---

## 🚀 Launch the App

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://digitgan.streamlit.app/)

---

## 📚 Project Overview

**DigitGAN** is a simple but powerful Generative Adversarial Network (GAN) built to create realistic handwritten digits.  
Trained on MNIST-inspired data using **PyTorch**, it demonstrates the core principles of GANs: generating new, unseen images purely from random noise.

This project includes:
- A **pre-trained generator** (`generator.pth`)
- An interactive **Streamlit web app** for easy digit generation
- A clean **PyTorch notebook** showing the training pipeline
- Full source code and deployment-ready setup

No need to retrain — just run and generate!

---

## 🎯 Features

- ✅ Generate 28x28 grayscale handwritten digits on demand
- ✅ Pre-trained generator model included
- ✅ Simple and intuitive Streamlit app interface
- ✅ Lightweight and fast (can run on CPU)
- ✅ Fully open-source under MIT License

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ryhan9834/DigitGAN-Handwritten-Digit-Generation-with-PyTorch.git
   cd DigitGAN-Handwritten-Digit-Generation-with-PyTorch

2. **Run the Streamlit App:**

    To run the Streamlit app and start generating digits, use:

    ```bash
    streamlit run streamlit_app.py

## 🧠 Model Details

The Generator is a fully connected neural network composed of:

- **Linear layers**: For mapping the noise vector to the image space.
- **Batch Normalization**: To stabilize training by normalizing activations.
- **ReLU activations**: For introducing non-linearity.
- **Sigmoid activation**: For outputting values between 0 and 1 (to match pixel intensities).

The Generator takes a random noise vector (`z_dim=64`) and generates a 28x28 flattened image. It is trained with adversarial loss to produce realistic handwritten digits closely resembling those from the MNIST dataset.

## 📂 Project Structure

```bash
DigitGAN/
├── digitgan.ipynb         # Training and Model Building Notebook
├── generator.pth          # Pre-trained Generator Weights
├── streamlit_app.py       # Streamlit App for Generating Digits
├── requirements.txt       # List of Required Python Libraries
├── README.md              # Project Documentation
└── .gitignore             # Files and folders to ignore in Git
```

## 📋 Requirements

- **torch**: PyTorch framework for building the GAN
- **torchvision**: For image transformations and datasets
- **streamlit**: For building the interactive web app
- **matplotlib**: For visualizing images during training
- **numpy**: For handling arrays and computations

You can install the required libraries via:

```bash
pip install -r requirements.txt
```

## 🌐 Deployment

The app can be deployed easily using Streamlit Cloud, allowing users to interact with the digit generator online without local setup. Deployment steps are included inside the README for easy setup!

## 📜 License

This project is licensed under the MIT License. Feel free to use, modify, and distribute it with proper attribution.

## 🙌 Acknowledgements

- **PyTorch**: The framework used to build and train the GAN.
- **Streamlit**: The tool used to create the interactive app.
- **MNIST Dataset**: The dataset used for training, provided by Yann LeCun et al.

## 🧑‍💻 Author

Made with ❤️ by Ryhan9834
