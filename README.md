# CS6910_Assignment_3
The recurrent neural network based encoder-decoder (seq_to_seq) has been implemented using tensorflow Keras.

The function fit_model contains the encoder-decoder pipeline with input embedding (input embedding, encoder layers, decoder layers, back_prop, update_parameters),
which is used for computing the training and the validation error along with validation accuracy.

After defining the model, the training of the model can be done using the fit_model function.

The sweep configuration that we used for hyperparameter tuning are:


sweep_config = {
    'method': 'bayes', 
    'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'   
    },
    'parameters': {

        'dropout': {
            'values': [0.0, 0.1, 0.2]
        },
        'learning_rate': {
            'values': [1e-3, 1e-4]
        },
        'batch_size': {
            'values': [32,64, 128]
        },
        'ip_emb': {
            'values': [32, 64, 128, 256]
        },
        'num_enc': {
            'values': [1, 2, 3]
        },
        'num_dec': {
            'values': [1, 2, 3]
        },
        'hidden_layer':{
            'values': [32, 64, 128]
        },
        'cell': {
            'values': ['RNN', 'GRU', 'LSTM']
        },
        'dec_search': {
            'values': ['beam_search', 'greedy']
        },
        'beam_width':{
            'values': [3,5]
        }
    }
}

Followed by implementation of the RNN without attention mechanism and with attention mechanism.

The function attention_plot plots the attention heatmaps for multiple inputs.

Dataset: Dakshina Dataset, using

"!wget https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar
!tar -xf dakshina_dataset_v1.0.tar"

command, directly Dakshina dataset can be downloaded.

The transliteration task is carried for Hindi lexicons from Google's Dakshina dataset.

The Hindi lexicons contains 44204 training words, 4358 validation words and 4502 testing words.

Tokenization of the characters in the dataset are performed using numpy.

References:

https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/

https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/

https://machinelearningmastery.com/use-dropout-lstm-networks-time-series-forecasting/

https://towardsdatascience.com/visualising-lstm-activations-in-keras-b50206da96ff

https://medium.com/datalogue/attention-in-keras-1892773a4f22
