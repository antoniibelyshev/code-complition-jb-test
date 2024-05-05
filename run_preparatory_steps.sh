echo "Installing necessary packages"
pip install transformers -U
pip install nltk
pip install rouge
pip install kopyt
pip install wandb

echo "Cloning kotlin repo..."
git clone https://github.com/JetBrains/kotlin.git

echo "Cloning codexglue method generation dataset"
git clone git clone https://huggingface.co/datasets/microsoft/codexglue_method_generation
