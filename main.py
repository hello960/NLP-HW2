from sre_parse import Tokenizer
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
from utilities import Utilities
from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from TrainingNetWork import Network
import matplotlib.pyplot as plt

from transformer import Encoder,Decoder
import torch.nn as nn

seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 32  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 #15# epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(encoder, classifier, data_loader):
    """Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    encoder.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)

            # Forward pass through the encoder
            encoder_output = encoder(X)  # Output shape: [batch_size, seq_len, n_embd]

            # Mean pooling across the sequence dimension
            pooled_output = encoder_output.mean(dim=1)  # Shape: [batch_size, n_embd]

            # Pass the pooled output through the classifier
            outputs = classifier(pooled_output)

            # Compute predictions
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
            

    accuracy = (100 * total_correct / total_samples)
    
    # Set models back to training mode
    classifier.train()
    encoder.train()
    
    return accuracy



def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity
    
def main():

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)
   
    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)


  
    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)
    


     # for the classification  task, you will train for a fixed number of epochs like this:
    CLS_model = Encoder(tokenizer.vocab_size, n_embd, n_layer, n_head)
    CLS_model = CLS_model.to(device)
    classifier = Network(input_size = n_embd, hidden_size= 200).to(device)  
    
    decoder = Decoder(hidden_size=n_embd, ffn_dim=100, n_head=n_head)
    decoder = decoder.to(device)
    decoder_utilities = Utilities(tokenizer, CLS_model, decoder)
    

    CLS_loss_fn = torch.nn.CrossEntropyLoss()
    CLS_optimizer = torch.optim.Adam(list(CLS_model.parameters()) + list(classifier.parameters()), lr=0.001)
    encoder_utilities = Utilities(tokenizer, CLS_model)
    sample_sentence = "That is in Israel's interest, Palestine's interest, America's interest, and the world's interest." 
    sample_sentence_block_size = 50  
    
    # Training loop for the classifier
    for epoch in range(epochs_CLS):
      CLS_model.train()
      classifier.train()
      total_CLS_loss = 0
      
      for xb, yb in train_CLS_loader:
          xb, yb = xb.to(device), yb.to(device)
          
          CLS_optimizer.zero_grad()
          
          encoder_output = CLS_model(xb)  # Output shape: [batch_size, seq_len, n_embd]
          
          pooled_output = encoder_output.mean(dim=1)  # Shape: [batch_size, n_embd]
          
          output = classifier(pooled_output)  
          
          loss = CLS_loss_fn(output, yb)  
          loss.backward()
          CLS_optimizer.step()
          
          total_CLS_loss += loss.item()
      
      encoder_utilities.sanity_check(sample_sentence, sample_sentence_block_size)
      accuracy = compute_classifier_accuracy(CLS_model,classifier, test_CLS_loader) 
      print(f"Epoch [{epoch+1}/{epochs_CLS}], Loss: {total_CLS_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")


    # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
    LM_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)
    LM_loss_fn = nn.CrossEntropyLoss()
    
    total_CLS_loss = 0
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)

        LM_optimizer.zero_grad()

        encoder_output = CLS_model(xb)  
        seq_len = xb.size(1)
        self_attention_mask = torch.tril(torch.ones((seq_len, seq_len), device=device))

        self_attention_mask = self_attention_mask.unsqueeze(0)  
        self_attention_mask = self_attention_mask.expand(batch_size, seq_len, seq_len)  

        decoder_output = decoder(xb, encoder_output, self_attention_mask)


        decoder_output_flat = decoder_output.view(-1, decoder_output.size(-1))  
        project_output = nn.Linear(n_embd, tokenizer.vocab_size)

        if decoder_output_flat.size(-1) != tokenizer.vocab_size:
            decoder_output_flat = project_output(decoder_output_flat)  

        yb_flat = yb.view(-1)  
        yb_flat = yb_flat.clamp(min=0, max=tokenizer.vocab_size - 1)
        
        lm_loss = LM_loss_fn(decoder_output_flat, yb_flat)
        perplexity = torch.exp(lm_loss).item() 
        lm_loss.backward()
        LM_optimizer.step()
        
        # decoder_utilities.sanity_check_decoder(sample_sentence, sample_sentence_block_size)
        print(f"Iteration [{i+1}/{max_iters}], LM Loss: {lm_loss.item():.4f}, Perplexity: {perplexity:.4f}")
    
        
    
        # LM training code here

    



if __name__ == "__main__":
    main()
