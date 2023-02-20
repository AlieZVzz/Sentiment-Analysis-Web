import os
import argparse
from functools import partial
import paddle
import paddle.nn.functional as F
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Pad, Stack, Tuple
from paddlenlp.transformers import SkepTokenizer, SkepModel, LinearDecayWithWarmup
from utils.utils import set_seed
from utils import data_ext, data_cls

train_path = "./data/train_ext.txt"
dev_path = "./data/dev_ext.txt"
test_path = "./data/test_ext.txt"
label_path = "./data/label_ext.dict"

# load and process data
label2id, id2label = data_ext.load_dict(label_path)
train_ds = load_dataset(data_ext.read, data_path=train_path, lazy=False)
dev_ds = load_dataset(data_ext.read, data_path=dev_path, lazy=False)
test_ds = load_dataset(data_ext.read, data_path=test_path, lazy=False)

# print examples
for example in train_ds[9:11]:
    print(example)

model_name = "skep_ernie_1.0_large_ch"
batch_size = 8
max_seq_len = 512

tokenizer = SkepTokenizer.from_pretrained(model_name)
trans_func = partial(data_ext.convert_example_to_feature, tokenizer=tokenizer, label2id=label2id,
                     max_seq_len=max_seq_len)
train_ds = train_ds.map(trans_func, lazy=False)
dev_ds = dev_ds.map(trans_func, lazy=False)
test_ds = test_ds.map(trans_func, lazy=False)

# print examples
for example in train_ds[9:11]:
    print("input_ids: ", example[0])
    print("token_type_ids: ", example[1])
    print("seq_len: ", example[2])
    print("label: ", example[3])
    print()

batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
    Stack(dtype="int64"),
    Pad(axis=0, pad_val=-1)): fn(samples)

train_batch_sampler = paddle.io.BatchSampler(train_ds, batch_size=batch_size, shuffle=True)
dev_batch_sampler = paddle.io.BatchSampler(dev_ds, batch_size=batch_size, shuffle=False)
test_batch_sampler = paddle.io.BatchSampler(test_ds, batch_size=batch_size, shuffle=False)

train_loader = paddle.io.DataLoader(train_ds, batch_sampler=train_batch_sampler, collate_fn=batchify_fn)
dev_loader = paddle.io.DataLoader(dev_ds, batch_sampler=dev_batch_sampler, collate_fn=batchify_fn)
test_loader = paddle.io.DataLoader(test_ds, batch_sampler=test_batch_sampler, collate_fn=batchify_fn)


class SkepForTokenClassification(paddle.nn.Layer):
    def __init__(self, skep, num_classes=2, dropout=None):
        super(SkepForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.skep = skep
        self.dropout = paddle.nn.Dropout(dropout if dropout is not None else self.skep.config["hidden_dropout_prob"])
        self.classifier = paddle.nn.Linear(self.skep.config["hidden_size"], num_classes)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        sequence_output, _ = self.skep(input_ids, token_type_ids=token_type_ids, position_ids=position_ids,
                                       attention_mask=attention_mask)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


# model hyperparameter  setting
num_epoch = 3
learning_rate = 3e-5
weight_decay = 0.01
warmup_proportion = 0.1
max_grad_norm = 1.0
log_step = 20
eval_step = 100
seed = 1000
checkpoint = "./checkpoint/"

set_seed(seed)
use_gpu = True if paddle.get_device().startswith("gpu") else False
if use_gpu:
    paddle.set_device("gpu:0")
if not os.path.exists(checkpoint):
    os.mkdir(checkpoint)

skep = SkepModel.from_pretrained(model_name)
model = SkepForTokenClassification(skep, num_classes=len(label2id))

num_training_steps = len(train_loader) * num_epoch
lr_scheduler = LinearDecayWithWarmup(learning_rate=learning_rate, total_steps=num_training_steps,
                                     warmup=warmup_proportion)
decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
grad_clip = paddle.nn.ClipGradByGlobalNorm(max_grad_norm)
optimizer = paddle.optimizer.AdamW(learning_rate=lr_scheduler, parameters=model.parameters(), weight_decay=weight_decay,
                                   apply_decay_param_fun=lambda x: x in decay_params, grad_clip=grad_clip)

metric = ChunkEvaluator(label2id.keys())


def evaluate(model, data_loader, metric):
    model.eval()
    metric.reset()
    for idx, batch_data in enumerate(data_loader):
        input_ids, token_type_ids, seq_lens, labels = batch_data
        logits = model(input_ids, token_type_ids=token_type_ids)

        # count metric
        predictions = logits.argmax(axis=2)
        num_infer_chunks, num_label_chunks, num_correct_chunks = metric.compute(seq_lens, predictions, labels)
        metric.update(num_infer_chunks.numpy(), num_label_chunks.numpy(), num_correct_chunks.numpy())

    precision, recall, f1 = metric.accumulate()
    return precision, recall, f1


def train():
    # start to train model
    global_step, best_f1 = 1, 0.
    model.train()
    for epoch in range(1, num_epoch + 1):
        for batch_data in train_loader():
            input_ids, token_type_ids, _, labels = batch_data
            # logits: batch_size, seql_len, num_tags
            logits = model(input_ids, token_type_ids=token_type_ids)
            loss = F.cross_entropy(logits.reshape([-1, len(label2id)]), labels.reshape([-1]), ignore_index=-1)

            loss.backward()
            lr_scheduler.step()
            optimizer.step()
            optimizer.clear_grad()

            if global_step > 0 and global_step % log_step == 0:
                print(
                    f"epoch: {epoch} - global_step: {global_step}/{num_training_steps} - loss:{loss.numpy().item():.6f}")
            if (global_step > 0 and global_step % eval_step == 0) or global_step == num_training_steps:
                precision, recall, f1 = evaluate(model, dev_loader, metric)
                model.train()
                if f1 > best_f1:
                    print(f"best F1 performence has been updated: {best_f1:.5f} --> {f1:.5f}")
                    best_f1 = f1
                    paddle.save(model.state_dict(), f"{checkpoint}/best_ext.pdparams")
                print(f'evalution result: precision: {precision:.5f}, recall: {recall:.5f},  F1: {f1:.5f}')

            global_step += 1

    paddle.save(model.state_dict(), f"{checkpoint}/final_ext.pdparams")


train()

# load model
model_path = "./checkpoint/best_ext.pdparams"

loaded_state_dict = paddle.load(model_path)
skep = SkepModel.from_pretrained(model_name)
model = SkepForTokenClassification(skep, num_classes=len(label2id))
model.load_dict(loaded_state_dict)

# evalute on test data
precision, recall, f1 = evaluate(model, test_loader, metric)
print(f'evalution result: precision: {precision:.5f}, recall: {recall:.5f},  F1: {f1:.5f}')
