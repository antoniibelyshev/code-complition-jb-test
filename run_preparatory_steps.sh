echo "Installing necessary packages"
pip install requirements.txt

echo "Cloning kotlin repo..."
git clone https://github.com/JetBrains/kotlin.git

echo "Extracting kotlin code..."
python utils/extract_kotlin_code.py

echo "Cloning codexglue method generation dataset"
git clone git clone https://huggingface.co/datasets/microsoft/codexglue_method_generation

echo "Preprocessing codexglue method generation dataset"
python utils/preprocess_codexglue_test_dataset.py
