import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertTokenizer
import torch
import json
import utils
from keras.preprocessing.sequence import pad_sequences
import time
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import numpy as np
import datetime
import utils
from sklearn.metrics import accuracy_score, confusion_matrix
import os
from torchtext import data, datasets, vocab


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def process_data(data):
    inputs = []
    labels = []
    token_type_ids = []
    max_len = 0

    for example in data.examples:
        text = " ".join(example.text)
        encoded_sent = tokenizer.encode(
            text, add_special_tokens=True, max_length=MAX_LEN)
        if len(encoded_sent) > max_len:
            max_len = len(encoded_sent)

        inputs.append(encoded_sent)
        labels.append(CLASSES[example.label])

        token_type_ids.append([0]*len(encoded_sent))

    return inputs, labels, token_type_ids, max_len


def get_attention_masks(X):
    attention_masks = []
    for sample in X:
        att_mask = [int(token_id > 0) for token_id in sample]
        attention_masks.append(att_mask)

    return attention_masks


def evaluate(model, data):
    all_y_truth = []
    all_y_pred = []
    loss_meter = utils.AverageMeter("loss")
    for batch in data:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_token_type_ids, b_labels = batch

        loss, logits = model(b_input_ids,
                             token_type_ids=b_token_type_ids,
                             attention_mask=b_input_mask,
                             labels=b_labels)
        _, pred_classes = torch.max(logits, 1)
        all_y_pred.extend(pred_classes.tolist())
        all_y_truth.extend(b_labels.tolist())
        loss_meter.update(loss, b_labels.shape[0])
    acc = accuracy_score(all_y_truth, all_y_pred)
    return acc, loss_meter.avg, all_y_pred, all_y_truth


###########
# Configs #
###########


GPU = 0
MAX_LEN = 512
BATCH_SIZE = 4
EPOCHS = 5
MODEL_NAME = "bert-base-uncased"
CLASSES = {"neg": 0, "pos": 1}


torch.cuda.set_device(GPU)
device = torch.device('cuda:'+str(GPU))
tokenizer = BertTokenizer.from_pretrained(
    MODEL_NAME, do_lower_case=True)



###########
# Dataset #
###########
st = time.time()
inputs = data.Field(lower=True, include_lengths=True, batch_first=True)
answers = data.Field(sequential=False)

train, test = datasets.IMDB.splits(inputs, answers)
train, valid = train.split(split_ratio=0.8)

X_train, y_train, token_type_ids_train, max_len_train = process_data(train)

X_valid, y_valid, token_type_ids_valid, max_len_valid = process_data(valid)

print("Time for loading and tokenizing data:", time.time() - st)
print("Lengths:", max_len_train, max_len_valid)
print(X_train[0])

#################
# Preprocessing #
#################

st = time.time()

# Padding
print('\nPadding/truncating all sentences to length', MAX_LEN)

X_train = pad_sequences(X_train, maxlen=MAX_LEN, dtype="long",
                        value=0, truncating="post", padding="post")
X_valid = pad_sequences(X_valid, maxlen=MAX_LEN, dtype="long",
                        value=0, truncating="post", padding="post")

token_type_ids_train = pad_sequences(token_type_ids_train, maxlen=MAX_LEN, dtype="long",
                                     value=1, truncating="post", padding="post")
token_type_ids_valid = pad_sequences(token_type_ids_valid, maxlen=MAX_LEN, dtype="long",
                                     value=1, truncating="post", padding="post")

# Attention masks
att_masks_train = get_attention_masks(X_train)
att_masks_valid = get_attention_masks(X_valid)


print("Time for pre processing:", time.time() - st)
print(X_train[0])
print(att_masks_train[0])
print(token_type_ids_train[0])

# Data Loaders
X_train = torch.tensor(X_train)
X_valid = torch.tensor(X_valid)
y_train = torch.tensor(y_train)
y_valid = torch.tensor(y_valid)
att_masks_train = torch.tensor(att_masks_train)
att_masks_valid = torch.tensor(att_masks_valid)
token_type_ids_train = torch.tensor(token_type_ids_train)
token_type_ids_valid = torch.tensor(token_type_ids_valid)

train_data = TensorDataset(X_train, att_masks_train,
                           token_type_ids_train, y_train)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(
    train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

validation_data = TensorDataset(
    X_valid, att_masks_valid, token_type_ids_valid, y_valid)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(
    validation_data, sampler=validation_sampler, batch_size=BATCH_SIZE)

#######################
# Model and optimizer #
#######################

model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    output_attentions=False,
    output_hidden_states=False
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# Total number of training steps is number of batches * number of EPOCHS.
total_steps = len(train_dataloader) * EPOCHS

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)


############
# Training #
############

train_losses = []
validation_losses = []
train_accuracies = []
validation_accuracies = []
epoch_ticks = []

start_time = time.time()

for epoch_i in range(0, EPOCHS):

    epoch_st = time.time()

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, EPOCHS))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_loss = 0

    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        if step % 100 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                step, len(train_dataloader), elapsed))

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_token_type_ids = batch[2].to(device)
        b_labels = batch[3].to(device)

        model.zero_grad()

        outputs = model(b_input_ids,
                        token_type_ids=b_token_type_ids,
                        attention_mask=b_input_mask,
                        labels=b_labels)

        loss = outputs[0]

        total_loss += loss.item()

        loss.backward()

        # Prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

    ##############
    # Validation #
    ##############

    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    with torch.no_grad():
        print("Starting Evaluation for epoch:", epoch_i+1)
        st = time.time()
        # Train data set evaluation
        train_acc, train_loss, train_y_pred, train_y_truth = evaluate(
            model, train_dataloader)
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        et = time.time()
        print("Time taken for train evaluation", et - st)

        # Validation data set evaluation
        st = time.time()
        validation_acc, validation_loss, validate_y_pred, validate_y_truth = evaluate(
            model, validation_dataloader)
        validation_accuracies.append(validation_acc)
        validation_losses.append(validation_loss)
        et = time.time()
        print("Time taken for validation evaluation", et - st)

        # Writing Confusion matrices
        utils.plot_confusion_matrices(
            MODEL_NAME, str(epoch_i+1), train_y_truth, train_y_pred, validate_y_truth, validate_y_pred)

        epoch_ticks.append(epoch_i+1)

    # Printing Epoch Metrics
    print("\n\n"+"="*25)
    print("Time elapsed:", time.time() - start_time)
    print("Epoch:", str(epoch_i + 1) + "/" + str(EPOCHS))
    print("Epoch time:", time.time() - epoch_st)
    print("Train Acc:", train_acc)
    print("Validation Acc:", validation_acc)
    print("Train Loss:", train_loss)
    print("Validation Loss:", validation_loss)
    print("="*25 + "\n\n")

    print("  Validation took: {:}".format(format_time(time.time() - t0)))

    ####################
    # Saving the model #
    ####################

    model_dir = "models/" + MODEL_NAME + "/" + str(epoch_i)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("Saving model to", model_dir)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

print("")
print("Training complete!")


###############################
# Plotting the Metrics Graphs #
###############################


def plot_metrics():
    plot_folder = "plots/" + MODEL_NAME + "/"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    # Losses
    plt.plot(epoch_ticks, train_losses, label="training loss")
    plt.plot(epoch_ticks, validation_losses, label="validation loss")
    plt.legend(loc='best')
    plt.title("Losses")
    plt.savefig(plot_folder + 'Losses.png')
    plt.clf()

    # Accuracies
    plt.plot(epoch_ticks, train_accuracies, label="training accuracies")
    plt.plot(epoch_ticks, validation_accuracies, label="validation accuracies")
    plt.legend(loc='best')
    plt.title("Accuracies")
    plt.savefig(plot_folder + 'Accuracies.png')
    plt.clf()


plot_metrics()

print("Done")


# Loading the model
"""
# Load a trained model and vocabulary that you have fine-tuned
model = model_class.from_pretrained(output_dir)
tokenizer = tokenizer_class.from_pretrained(output_dir)

# Copy the model to the GPU.
model.to(device)
"""


# Test accuracy
X_test_b, y_test_b, token_type_ids_test, max_len_test = process_data(test)

X_test_b = pad_sequences(X_test_b, maxlen=MAX_LEN, dtype="long",
                         value=0, truncating="post", padding="post")
token_type_ids_test = pad_sequences(token_type_ids_test, maxlen=MAX_LEN, dtype="long",
                                    value=1, truncating="post", padding="post")

att_masks_test = get_attention_masks(X_test_b)

X_test_b = torch.tensor(X_test_b)
y_test_b = torch.tensor(y_test_b)
att_masks_test = torch.tensor(att_masks_test)
token_type_ids_test = torch.tensor(token_type_ids_test)


test_data_b = TensorDataset(
    X_test_b, att_masks_test, token_type_ids_test, y_test_b)
test_sampler = SequentialSampler(test_data_b)
test_dataloader = DataLoader(
    test_data_b, sampler=test_sampler, batch_size=BATCH_SIZE)


with torch.no_grad():
    test_acc, test_loss, test_y_pred, test_y_truth = evaluate(
        model, test_dataloader)
print("TEST Accuracy:", round(test_acc, 3))
