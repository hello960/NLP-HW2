
import matplotlib.pyplot as plt
import torch

class Utilities:
    def __init__(self, tokenizer, model,decode_model=None):
        self.tokenizer = tokenizer
        self.model = model
        self.decode_model = decode_model
    def sanity_check(self, sentence, block_size):
        # Encode the sentence using the tokenizer
        wordids = self.tokenizer.encode(sentence)

        # Prepare the padded input for the model
        padded_sentence = wordids[:block_size] + [0] * (block_size - len(wordids))
        input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0)

        # Display input tensor shape
        print("Input tensor shape:", input_tensor.shape)

        # Process the input tensor through the encoder model
        _,  attn_maps = self.model(input_tensor, return_attention=True) # Ignore the output of the model, and only get the attention maps; make sure your encoder returns the attention maps

        # Display the number of attention maps
        print("Number of attention maps:", len(attn_maps))

        # Visualize and save the attention maps
        for j, attn_map in enumerate(attn_maps):
            att_map = attn_map.squeeze(0).detach().cpu().numpy()  # Remove batch dimension and convert to NumPy array

            # Check if the attention probabilities sum to 1 over rows
            total_prob_over_rows = torch.sum(attn_map[0], dim=1)
            if torch.any(total_prob_over_rows < 0.99) or torch.any(total_prob_over_rows > 1.01):
                print("Failed normalization test: probabilities do not sum to 1.0 over rows")
                print("Total probability over rows:", total_prob_over_rows.numpy())

            # Create a heatmap of the attention map
            fig, ax = plt.subplots()
            cax = ax.imshow(att_map, cmap='hot', interpolation='nearest')
            ax.xaxis.tick_top()  
            fig.colorbar(cax, ax=ax)  
            plt.title(f"Attention Map {j + 1}")
            
            # Save the plot
            plt.savefig(f"attention_map_{j + 1}.png")
            
            # Show the plot
            plt.show()
    def sanity_check_decoder(self, sentence, block_size):
        # Step 1: Tokenize input
        wordids = self.tokenizer.encode(sentence)
        padded_sentence = wordids[:block_size] + [0] * (block_size - len(wordids))
        input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0)  # [1, block_size]

        # Display input tensor shape
        print("Input tensor shape:", input_tensor.shape)

        # Step 2: Run input through encoder
        encoder_output = self.model(input_tensor)

        # Step 3: Prepare decoder input (shifted right, with embedding if needed)
        decoder_input = torch.zeros_like(input_tensor)  # Shifted inputs as needed
        decoder_input[:, 0] = 0  # Assuming 0 is the ID for a start token

        # Embed the decoder input if necessary
        if hasattr(self.decode_model, 'embedding'):
            decoder_input = self.decode_model.embedding(decoder_input)  # Shape: [1, block_size, embed_dim]

        # Step 4: Pass through the decoder with encoder output
        decoder_output, attn_maps = self.decode_model(decoder_input, encoder_output, self_attention_mask=True)

        # Display output tensor shape
        print("Decoder output shape:", decoder_output.shape)

        # Step 5: Check attention maps for each layer
        print("Number of decoder attention maps:", len(attn_maps))

        for j, attn_map in enumerate(attn_maps):
            att_map = attn_map.squeeze(0).detach().cpu().numpy()  # Convert to NumPy

            # Check if probabilities sum to 1
            total_prob_over_rows = torch.sum(attn_map[0], dim=1)
            if torch.any(total_prob_over_rows < 0.99) or torch.any(total_prob_over_rows > 1.01):
                print(f"Normalization issue in attention map {j+1}")

            # Plot and save attention maps
            plt.figure()
            plt.imshow(att_map, cmap='hot')
            plt.colorbar()
            plt.title(f"Decoder Attention Map {j + 1}")
            plt.savefig(f"decoder_attention_map_{j + 1}.png")
            plt.show()


            


