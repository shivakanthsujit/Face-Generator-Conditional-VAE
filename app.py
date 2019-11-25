import streamlit as st

import torch
import torchvision
from torchvision import transforms
from model import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

project_dir = ""
model_dir = project_dir + "models/"

vae = VAE().to(device)
vae.load_state_dict(
    torch.load(model_dir + "Conditional-VAE.pt", map_location=torch.device(device))
)


def generate(z, label):
    vae.eval()
    with torch.no_grad():
        z = z.to(device)
        label = label.to(device)
        z1 = vae.fc3(z)
        z1 = torch.cat((z1, label))
        x = vae.fc4(z1)
        x = x.reshape(1, -1)
        x = vae.decoder(x)
        x = transforms.ToPILImage()(x.squeeze()).convert("RGB")
        x = transforms.Resize(250)(x)
        st.title("Generated Image")
        st.image(x)
        # gen_image.image(x)


latent_vector = lambda x: torch.randn(128).mul(x)


@st.cache()
def generate_z():
    z = latent_vector(1.0)
    return z


"""
# Face Generator
#### Created using a Conditional VAE trained on the CelebA dataset

Drag the sliders and the model will generate a face from those values. Additional attributes can be enabled from the sidebar\n
The code for this project can be found [here](https://github.com/shivakanthsujit/Face-Generator-Conditional-VAE)
"""
st.sidebar.markdown(
    "There are additional atrributes available that can be changed. These are turned off by default since they aren't very apparent as you change them."
)
add_attr = st.sidebar.checkbox("Display additional attributes?",)

black = st.slider("Black Hair", -1.0, 1.0, 1.0, 0.01, key="black")
blond = st.slider("Blond Hair", -1.0, 1.0, -1.0, 0.01, key="blond")
male = st.slider("Male", -1.0, 1.0, 1.0, 0.01, key="male")
smiling = st.slider("Smiling", -1.0, 1.0, 1.0, 0.01, key="smiling")

if add_attr:
    brown = st.slider("Brown Hair", -1.0, 1.0, -1.0, 0.01, key="brown")
    straight_hair = st.slider(
        "Straight_Hair", -1.0, 1.0, 1.0, 0.01, key="straight_hair"
    )
    wavy_hair = st.slider("Straight_Hair", -1.0, 1.0, 1.0, 0.01, key="wavy_hair")
    no_beard = st.slider("No_Beard", -1.0, 1.0, 1.0, 0.01, key="np_beard")
    young = st.slider("Young", -1.0, 1.0, 1.0, 0.01, key="young")
else:
    brown = -1.0
    straight_hair = 1.0
    wavy_hair = -1.0
    no_beard = 1.0
    young = 1.0

values = {
    "Black_Hair": black,
    "Blond_Hair": blond,
    "Brown_Hair": brown,
    "Male": male,
    "No_Beard": no_beard,
    "Smiling": smiling,
    "Straight_Hair": straight_hair,
    "Wavy_Hair": wavy_hair,
    "Young": young,
}
label = torch.FloatTensor(list(map(float, values.values())))
z = generate_z()

# st.title("Generated Image")
# gen_image = st.empty()


if st.button("Randomise Face"):
    z = latent_vector(1.0)
    generate(z, label)
else:
    generate(z, label)
