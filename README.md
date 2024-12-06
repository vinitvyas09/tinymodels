# Tiny Emotion Classification Models

This project demonstrates training and evaluation of two extremely compact neural network models for emotion (sentiment) classification on a Twitter dataset. Both models perform binary classification (positive vs. negative), but differ significantly in size:

- **tinymodel-emotion-classification-10kB.py:** A minimal model constrained to under 10kB. Achieved by using a small embedding dimension, a reduced vocabulary (e.g., 500 words), and short input sequences.
- **tinymodel-emotion-classification-250kB.py:** A larger model (~250kB), with a bigger vocabulary and embedding dimension, offering potentially higher accuracy at the cost of increased model size.

## Key Steps

1. **Data Acquisition:**  
   The scripts automatically download and parse a Twitter sentiment dataset. Each tweet is labeled as `0` (negative) or `1` (positive).

2. **Preprocessing:**  
   Text is tokenized, truncated or padded to a fixed length, and converted into integer sequences using a limited vocabulary.

3. **Model Architecture:**  
   Each model uses a simple embedding layer followed by a linear layer with a sigmoid activation. The small model uses fewer parameters to stay under 10kB.

4. **Training and Evaluation:**  
   Before training, accuracy is checked on the test set. The model is then trained for a few epochs, and accuracy is measured again to confirm improvements. Both scripts allow toggling verbosity and adjusting the number of epochs.

5. **Profiling and Saving:**  
   The code includes profiling to measure runtime performance and saves the trained model to disk, reporting its final size.

## Usage

### Requirements
``` bash
pip install -r requirements.txt
```
### Train and evaluate the models

```bash
# Minimal model
python3 tinymodel-emotion-classification-10kB.py

# Larger model
python3 tinymodel-emotion-classification-250kB.py
```
Adjust model parameters (e.g., NUM_EPOCHS, ENABLE_LOGS) at the top of each script. The final output includes initial accuracy, final accuracy, total training time, and model file size.

This project exemplifies how limiting model parameters can drastically reduce size, enabling deployment on devices with strict memory constraints, while still performing basic emotion classification.