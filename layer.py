import torch.nn as nn


class AutoEncoder(nn.Module):

    def __init__(self,
                 encoder_dim,
                 activation='relu',
                 layernorm=True,
                 dropout_prob=0.5):

        super(AutoEncoder, self).__init__()

        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._layernorm = layernorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim - 1:
                if self._layernorm:
                    encoder_layers.append(nn.LayerNorm(encoder_dim[i + 1], elementwise_affine=False))
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
                encoder_layers.append(nn.Dropout(dropout_prob))

        self._encoder = nn.Sequential(*encoder_layers)

        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if self._layernorm:
                decoder_layers.append(nn.LayerNorm(decoder_dim[i + 1], elementwise_affine=False))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
            decoder_layers.append(nn.Dropout(dropout_prob))

        self._decoder = nn.Sequential(*decoder_layers)

    def encoder(self, x):

        latent = self._encoder(x)
        return latent

    def decoder(self, latent):

        x_hat = self._decoder(latent)
        return x_hat

    def forward(self, x):

        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat, latent


class Classifier(nn.Module):

    def __init__(self, fc_layers, dropout_prob, cpi_hidden_dim):

        super(Classifier, self).__init__()

        self.fc_layers = nn.ModuleList()

        for i in range(fc_layers):
            if i == fc_layers - 1:
                self.fc_layers.append(nn.Linear(cpi_hidden_dim[i], 1))
            else:
                self.fc_layers.append(
                    nn.Linear(cpi_hidden_dim[i], cpi_hidden_dim[i + 1]))
                self.fc_layers.append(nn.ReLU())
                self.fc_layers.append(nn.Dropout(dropout_prob))

    def forward(self, output):

        for fc_layer in self.fc_layers:
            output = fc_layer(output)
        return output


class Prediction(nn.Module):

    def __init__(self,
                 prediction_dim,
                 activation='relu',
                 layernorm=True):

        super(Prediction, self).__init__()

        self._depth = len(prediction_dim) - 1
        self._activation = activation
        self._prediction_dim = prediction_dim

        encoder_layers = []
        for i in range(self._depth):
            encoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i + 1]))
            if layernorm:
                encoder_layers.append(nn.LayerNorm(self._prediction_dim[i + 1], elementwise_affine=False))
            if self._activation == 'sigmoid':
                encoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                encoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                encoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)

        self._encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(self._depth, 0, -1):
            decoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i - 1]))
            if i > 1:
                if layernorm:
                    decoder_layers.append(nn.LayerNorm(self._prediction_dim[i - 1], elementwise_affine=False))

                if self._activation == 'sigmoid':
                    decoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    decoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    decoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)

        self._decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):

        latent = self._encoder(x)
        output = self._decoder(latent)
        return output, latent
