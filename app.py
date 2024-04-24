from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import tensorflow as tf

app = Flask(__name__)

# Load the saved model without compiling it
model_path = r"C:\Users\Lenovo\Desktop\Mercedes-Benz\chatbot_model.h5"
model = load_model(model_path, compile=False)

# Load your tokenizer and set your max_sequence_len here
tokenizer_path = r"C:\Users\Lenovo\Desktop\Mercedes-Benz\tokenizer.pkl"  # Example path to tokenizer file
max_sequence_len = 100  # Example value for max_sequence_len

# Load the tokenizer
with open(tokenizer_path, 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Define your optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

# Define weight decay rate
weight_decay = 0.01

# Define the loss function with weight decay
@tf.function
def compute_loss(model, x, y):
    logits = model(x, training=True)
    loss = tf.keras.losses.categorical_crossentropy(y_true=y, y_pred=logits, from_logits=True)
    # Apply weight decay
    for var in model.trainable_variables:
        if "kernel" in var.name:
            loss += weight_decay * tf.nn.l2_loss(var)
    return loss

# Compile your model with the optimizer
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['message']
    
    # Preprocess user input
    input_sequence = tokenizer.texts_to_sequences([user_input])
    input_sequence = pad_sequences(input_sequence, maxlen=max_sequence_len-1, padding='pre')
    
    # Generate response
    predicted_probs = model.predict(input_sequence, verbose=0)[0]
    predicted_index = np.argmax(predicted_probs)
    predicted_word = tokenizer.index_word[predicted_index]
    
    return jsonify({'response': predicted_word})

if __name__ == '__main__':
    app.run(debug=True)
