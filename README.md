# translation_model

## Setup Instructions

### Environment Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Sunny-Soni00/translation_model.git
   cd translation_model
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Data Preprocessing
1. Load the dataset:
   - Ensure you have the dataset file in the repository directory.
   - The dataset should have two columns: 'english' and 'hinglish'.

2. Preprocess the data:
   - Convert all text to lowercase.
   - Remove rows where the sentence is malformed or swapped (e.g., English appears in Hinglish column).
   - Remove rows with spelling issues like 'cle__(ar)' or other invalid tokens.
   - Drop duplicate rows.
   - (Optional) Trim sentences longer than a defined threshold based on 95th percentile.

### Tokenization
1. Use subword-based tokenization (prefer SentencePiece or WordPiece if supported).
2. Train separate tokenizers for English and Hinglish with vocabulary size ~16000.

### Model Training
1. Define the model architecture:
   - Use Embedding → Encoder LSTM → Decoder LSTM → Dense Softmax.
   - Apply teacher forcing during training.
   - Use categorical crossentropy loss and reduce it as much as possible.
   - Add EarlyStopping callback based on validation loss.

2. Train the model:
   - Use the preprocessed and tokenized dataset.
   - Save the trained model and tokenizers.

### Evaluation
1. Use beam search decoding for prediction.
2. Evaluate output using BLEU score on test set.

### Example Commands
1. To preprocess the data:
   ```bash
   python preprocess_data.py --input_file dataset.csv --output_file cleaned_dataset.csv
   ```

2. To train the model:
   ```bash
   python train_translation_model.py --data_file cleaned_dataset.csv
   ```

3. To evaluate the model:
   ```bash
   python evaluate_model.py --model_file trained_model.h5 --tokenizer_file tokenizer.model
   ```

### Expected Outputs
1. Preprocessing:
   - Cleaned dataset saved as `cleaned_dataset.csv`.

2. Training:
   - Trained model saved as `trained_model.h5`.
   - Tokenizers saved as `tokenizer.model`.

3. Evaluation:
   - BLEU score printed in the console.
