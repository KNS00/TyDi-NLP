import torch
from torch import nn 
from torch.utils.data import DataLoader
import numpy as np
import tqdm
class LSTM(nn.Module):
    """
    LSTM Language Model
    """
    def __init__(
            self,
            pretrained_embeddings: torch.tensor,
            lstm_dim: int,
            dropout_prob: float = 0.0,
            lstm_layers: int = 1,
            batch_size : int = 32,
            device: torch.device = torch.device('cpu')
    ):
        """
        Initializer for LSTM Language Model
        :param pretrained_embeddings: A tensor containing the pretrained BPE embeddings
        :param lstm_dim: The dimensionality of the BiLSTM network
        :param dropout_prob: Dropout probability
        :param lstm_layers: The number of stacked LSTM layers
        """

        # Initialize the super class
        super(LSTM, self).__init__()

        # We'll define the network in a ModuleDict, which makes organizing the model a bit nicer
        # The components are an embedding layer, an LSTM layer, a dropout layer, and a feed-forward output layer
        self.vocab_size = pretrained_embeddings.shape[0]
        self.model = nn.ModuleDict({
            'embeddings': nn.Embedding.from_pretrained(pretrained_embeddings, padding_idx=pretrained_embeddings.shape[0] - 1),
            'lstm': nn.LSTM(
                pretrained_embeddings.shape[1],
                lstm_dim,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=dropout_prob),
            'ff': nn.Linear(lstm_dim, pretrained_embeddings.shape[0]),
            'drop': nn.Dropout(dropout_prob)
        })

        # Initialize the weights of the model
        self._init_weights()

    def _init_weights(self):
        all_params = list(self.model['lstm'].named_parameters()) + \
                     list(self.model['ff'].named_parameters())
        for n, p in all_params:
            if 'weight' in n:
                nn.init.xavier_normal_(p)
            elif 'bias' in n:
                nn.init.zeros_(p)

    def forward(self, input_ids, input_lens, hidden_states):
        """
        Defines how tensors flow through the model
        :param input_ids: (b x sl) The IDs into the vocabulary of the input samples
        :param input_lens: (b x 1) The length of each instance's text
        :param hidden_states: (b x sl) x 2 Hidden states for the LSTM model
        :return: (lstm output, updated hidden stated)
        """

        # Get embeddings (b x sl x edim)
        embeds = self.model['drop'](self.model['embeddings'](input_ids))

        lstm_in = nn.utils.rnn.pack_padded_sequence(
            embeds,
            input_lens.to('cpu'),
            batch_first=True,
            enforce_sorted=False
        )

        # Pass the packed sequence through the BiLSTM
        lstm_out, hidden = self.model['lstm'](lstm_in)
        # Unpack the packed sequence --> (b x sl x 2*lstm_dim)
        lstm_out, hidden_states = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = self.model['drop'](lstm_out)
        # generate the prediction of each word in the vocabulary being the next
        lstm_out = self.model['ff'](lstm_out)
        lstm_out = lstm_out.reshape(-1, self.vocab_size)

        return lstm_out, hidden_states
    
    def evaluate(self, valid_dl: DataLoader):
        """
        Evaluates the model on the given dataset
        :param model: The model under evaluation
        :param valid_dl: A `DataLoader` reading validation data
        :param lstm_layers: The number of LSTM layers in the model
        :param batch_size: The batch size
        :param lstm_dim: The LSTM dimension
        :return: The accuracy of the model on the dataset
        """
        self.eval()
        loss_all = []
        states = (torch.zeros(self.lstm_layers, self.batch_size, self.lstm_dim).to(self.device),
                torch.zeros(self.lstm_layers, self.batch_size, self.lstm_dim).to(self.device))
        loss_fn = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in tqdm(valid_dl, desc='Evaluation'):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids = batch[0]
                input_lens = batch[1]
                targets = batch[2]
                states = self.detach(states)
                logits, states = self(input_ids, input_lens, states)
                loss = loss_fn(logits, targets.reshape(-1))

                loss_all.append(loss.detach().cpu().numpy())

        perplexity = np.exp(sum(loss_all) / (len(loss_all)))
        return perplexity

    # Truncated backpropagation
    def detach(self, states):
        return [state.detach() for state in states]

    def train(self, 
        model: nn.Module,
        train_dl: DataLoader,
        valid_dl: DataLoader,
        optimizer: torch.optim.Optimizer,
        n_epochs: int 
        ):
        """
        The main training loop which will optimize a given model on a given dataset
        :param model: The model being optimized
        :param train_dl: The training dataset
        :param valid_dl: A validation dataset
        :param optimizer: The optimizer used to update the model parameters
        :param n_epochs: Number of epochs to train for
        :param device: The device to train on
        :return: (model, losses) The best model and the losses per iteration
        """

        # Keep track of the loss and best accuracy
        losses = []
        best_perplexity = 2000
        # Set initial hidden and cell states
        loss_fn = nn.CrossEntropyLoss()
        # Iterate through epochs
        for ep in range(n_epochs):
            states = (torch.zeros(self.lstm_layers, self.batch_size, self.lstm_dim).to(self.device),
                    torch.zeros(self.lstm_layers, self.batch_size, self.lstm_dim).to(self.device))

            loss_epoch = []

            #Iterate through each batch in the dataloader
            for batch in tqdm(train_dl):
                # VERY IMPORTANT: Make sure the model is in training mode, which turns on
                # things like dropout and layer normalization
                model.train()

            # VERY IMPORTANT: zero out all of the gradients on each iteration -- PyTorch
            # keeps track of these dynamically in its computation graph so you need to explicitly
            # zero them out
            optimizer.zero_grad()

            # Place each tensor on the GPU
            batch = tuple(t.to(self.device) for t in batch)
            input_ids = batch[0]
            input_lens = batch[1]
            targets = batch[2]
            # Pass the inputs through the model, get the current loss and logits
            states = self.detach(states)
            logits, states = model(input_ids, input_lens, states)
            loss = loss_fn(logits, targets.reshape(-1))

            losses.append(loss.item())
            loss_epoch.append(loss.item())

            # Calculate all of the gradients and weight updates for the model
            loss.backward()

            # Optional: clip gradients, helps with exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Finally, update the weights of the model
            optimizer.step()
            #gc.collect()

            # Perform inline evaluation at the end of the epoch
            perplexity = self.evaluate(model, valid_dl)
            print(f'Validation perplexity: {perplexity}, train loss: {sum(loss_epoch) / len(loss_epoch)}')

            # Keep track of the best model based on the accuracy
            best_model = model.state_dict()
            if perplexity < best_perplexity:
                best_model = model.state_dict()
            best_perplexity = perplexity

            model.load_state_dict(best_model)
            return model, losses

    def opt_train(
        self,
        train_dl: DataLoader,
        valid_dl: DataLoader,
        optimizer: torch.optim.Optimizer,
        n_epochs: int,
        device: torch.device,
        lstm_layers: int,
        batch_size: int,
        lstm_dim: int
    ):
        """
        The main training loop which will optimize a given model on a given dataset
        :param model: The model being optimized
        :param train_dl: The training dataset
        :param valid_dl: A validation dataset
        :param optimizer: The optimizer used to update the model parameters
        :param n_epochs: Number of epochs to train for
        :param device: The device to train on
        :param lstm_layers: The number of LSTM layers
        :param batch_size: The batch size
        :param lstm_dim: The LSTM dimension
        :return: (model, losses) The best model and the losses per iteration
        """

        # Keep track of the loss and best accuracy
        losses = []
        best_perplexity = 480.0 # default value

        # Set initial hidden and cell states
        loss_fn = nn.CrossEntropyLoss()

        # Iterate through epochs
        for ep in range(n_epochs):
            states = (
                torch.zeros(lstm_layers, batch_size, lstm_dim).to(device),
                torch.zeros(lstm_layers, batch_size, lstm_dim).to(device)
            )

            loss_epoch = []

            # Iterate through each batch in the dataloader
            for batch in tqdm(train_dl):
                # VERY IMPORTANT: Make sure the model is in training mode, which turns on
                # things like dropout and layer normalization
                self.train()

                # VERY IMPORTANT: zero out all of the gradients on each iteration -- PyTorch
                # keeps track of these dynamically in its computation graph so you need to explicitly
                # zero them out
                optimizer.zero_grad()

                # Place each tensor on the GPU
                batch = tuple(t.to(device) for t in batch)
                input_ids = batch[0]
                input_lens = batch[1]
                targets = batch[2]

                # Pass the inputs through the model, get the current loss and logits
                states = self.detach(states)
                logits, states = self(input_ids, input_lens, states)
                loss = loss_fn(logits, targets.reshape(-1))

                losses.append(loss.item())
                loss_epoch.append(loss.item())

                # Calculate all of the gradients and weight updates for the model
                loss.backward()

                # Optional: clip gradients, helps with exploding gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

                # Finally, update the weights of the model
                optimizer.step()

            # Perform inline evaluation at the end of the epoch
            perplexity = self.evaluate(self, valid_dl, lstm_layers, batch_size, lstm_dim)

            print(f'Validation perplexity: {perplexity}, train loss: {sum(loss_epoch) / len(loss_epoch)}')

            # Keep track of the best model based on the accuracy
            best_model = self.state_dict()
            if perplexity < best_perplexity:
                best_model = self.state_dict()
                best_perplexity = perplexity

        self.load_state_dict(best_model)
        return self, losses
