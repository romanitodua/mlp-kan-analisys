from models.utils import load_images
from tests.compare import model_analysis

# load images and corresponding labels
images, labels = load_images(5, "geometric_figures")

# define arguments for analysis
epochs = 50
hidden_layers_mlp = [20]
hidden_layers_kan = [64, 32, 16]
learning_rate = 0.001
# eye is set to 5 by default, output layer
model_analysis(images, labels, epochs, hidden_layers_mlp, hidden_layers_kan, learning_rate)
