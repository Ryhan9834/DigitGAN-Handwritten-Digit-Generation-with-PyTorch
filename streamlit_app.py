import streamlit as st
import torch
import matplotlib.pyplot as plt

# --- Define the model again ---
def get_generator_block(input_dim, output_dim):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, output_dim),
        torch.nn.BatchNorm1d(output_dim),
        torch.nn.ReLU(inplace=True)
    )

class Generator(torch.nn.Module):
    def __init__(self, z_dim=64, im_dim=28*28, hidden_dim=128):
        super().__init__()
        self.gen = torch.nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            torch.nn.Linear(hidden_dim * 8, im_dim),
            torch.nn.Sigmoid()
        )
    def forward(self, noise):
        return self.gen(noise)

# --- Load the generator ---
gen = Generator()
gen.load_state_dict(torch.load("generator.pth", map_location=torch.device('cpu')))
gen.eval()

# --- Streamlit app ---
st.title("Handwritten Digit Generator ✍️✨")

num_images = st.slider("How many digits to generate?", 1, 30, 9)

if st.button("Generate!"):
    noise = torch.randn(num_images, 64)
    fake_images = gen(noise).view(-1, 1, 28, 28)

    cols = st.columns(3)
    for idx, img in enumerate(fake_images):
        col = cols[idx % 3]
        with col:
            st.image(img.detach().numpy().squeeze(), width=100, caption=f"Digit {idx+1}")
