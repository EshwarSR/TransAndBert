import torch
from torch import nn
import torch.nn.functional as F

from torchtext import data, datasets, vocab

import numpy as np
import Attention_models

import random
import tqdm
import sys
import math
import utils
import time
import os
import matplotlib.pyplot as plt


def get_train_valid_test_data(batch_size, device):
    inputs = data.Field(lower=True, include_lengths=True, batch_first=True)
    answers = data.Field(sequential=False)

    train, test = datasets.IMDB.splits(inputs, answers)
    train, valid = train.split(split_ratio=0.8)

    inputs.build_vocab(train, valid, test)
    answers.build_vocab(train)

    train_data, validation_data, test_data = data.BucketIterator.splits(
        (train, valid, test), batch_size=batch_size, device=device)
    return train_data, validation_data, test_data, inputs, answers


def validate(data):
    tot, cor = 0.0, 0.0

    for batch in data:

        input = batch.text[0]
        label = batch.label - 1

        if input.size(1) > mx:
            input = input[:, :mx]
        out = model(input).argmax(dim=1)

        tot += float(input.size(0))
        cor += float((label == out).sum().item())

    acc = cor / tot

    return acc


###########
# Configs #
###########
exp = sys.argv[1]
GPU = sys.argv[2]

print("Running the experiment", exp, "on GPU", GPU)

num_epochs = 20
save_every = 5
model_dir = "models/"
plot_folder = "plots/"

max_pool = False
embedding_size = 128
max_length = 128
lr_warmup = 10000
gradient_clipping = 1.0
learning_rate = 0.0001
batch_size = 32


if torch.cuda.is_available():
    torch.cuda.set_device(int(GPU))
    device = torch.device('cuda:'+str(GPU))
else:
    device = torch.device('cpu')

device = torch.device('cpu')

print("Running on the device", device)

LOG2E = math.log2(math.e)


###################
# Loading Dataset #
###################
print("Loading Dataset")
data_st = time.time()
train_data, validation_data, test_data, inputs, answers = get_train_valid_test_data(
    batch_size, device)
print("Finished loading Dataset", time.time() - data_st)

vocab_length = len(inputs.vocab)  # for pad and unk
num_classes = len(answers.vocab)


#################
# Main function #
#################

print(f'- batche size {batch_size}')
print(f'- nr. of training batches {len(train_data)}')
print(f'- nr. of {"validation"} batches {len(validation_data)}')

if max_length < 0:
    mx = max([input.text[0].size(1) for input in train_data])
    mx = mx * 2
    print(f'- maximum sequence length: {mx}')
else:
    mx = max_length


all_models = {
    "exp1": {
        "transformer_8_head_6_layers": {"num_heads": 8, "num_layers": 6},
        "transformer_8_head_12_layers": {"num_heads": 8, "num_layers": 12}
    },
    "exp2": {
        "transformer_16_head_6_layers": {"num_heads": 16, "num_layers": 6},
        "transformer_16_head_12_layers": {"num_heads": 16, "num_layers": 12}
    }
}

models = all_models[exp]

for model_name, configs in models.items():
    print("Creating model", model_name)
    num_heads = configs["num_heads"]
    num_layers = configs["num_layers"]
    # create the model
    model = Attention_models.TransformerClassifier(emb=embedding_size, heads=num_heads, depth=num_layers,
                                                   seq_length=mx, num_tokens=vocab_length, num_classes=num_classes, max_pool=max_pool, device=device)
    if torch.cuda.is_available():
        model.to(device)

    opt = torch.optim.Adam(lr=learning_rate, params=model.parameters())
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(i / (lr_warmup / batch_size), 1.0))

    # training loop
    epoch_ticks = []
    train_accuracies = []
    validation_accuracies = []
    for epoch in range(num_epochs):
        s = time.time()

        print(f'\n epoch {epoch}')
        model.train(True)

        for batch in train_data:

            opt.zero_grad()

            input = batch.text[0]
            label = batch.label - 1

            if input.size(1) > mx:
                input = input[:, :mx]
            out = model(input)
            loss = F.nll_loss(out, label)

            loss.backward()

            # clip gradients
            # - If the total gradient vector has a length > 1, we clip it back down to 1.
            if gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clipping)

            opt.step()
            sch.step()

        with torch.no_grad():
            model.train(False)
            train_acc = validate(train_data)
            validation_acc = validate(validation_data)

            epoch_ticks.append(epoch)
            train_accuracies.append(train_acc)
            validation_accuracies.append(validation_acc)
            print(f'-- Train accuracy {train_acc:.3}')
            print(f'-- Validation accuracy {validation_acc:.3}')

            test_acc = validate(test_data)
            print(f'-- Test accuracy {test_acc:.3}')
        print("Epoch time:", time.time() - s)

        if epoch % save_every == (save_every-1):
            final_dir = model_dir + model_name + "/"
            if not os.path.exists(final_dir):
                os.makedirs(final_dir)
            torch.save(model.state_dict(), final_dir +
                       "epoch_" + str(epoch) + ".pt")

        print("=="*25)

    # Plot metrics
    plt.plot(epoch_ticks, train_accuracies, label="training accuracies")
    plt.plot(epoch_ticks, validation_accuracies, label="validation accuracies")
    plt.legend(loc='best')
    plt.title(model_name + "Accuracies")
    plt.savefig(plot_folder + model_name + '_Accuracies.png')
    plt.clf()

    print("Final test accuracy")
    test_acc = validate(test_data)
    print(f'-- Test accuracy {test_acc:.3}')

    print("**"*25)
    print("**"*25)
    print("\n\n")
