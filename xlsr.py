from huggingface_hub import login
from datasets import load_dataset,load_metric,Audio,Dataset 
import re
import json
import torch.version
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
import torch
from dataclasses import dataclass,field
from typing import Any,Dict,List,Optional,Union
import numpy as np
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer
print(torch.backends.cudnn.enabled)
print(torch.cuda.is_available())
print(torch.version.cuda)
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch



def removeChars(entry):
    entry['text']=re.sub('[\,\?\.\!\-\;\:\"\“\%\‘\”\�\…\']','',entry['text']).lower()
    return entry

def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = werMetric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

login('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')

vocab={}
vocabCount=0
allText=''

dataset=load_dataset('audiofolder',data_dir='parlaSmall/data')
ind=[]
count=0
for point in dataset['train']:
    text=point['text']
    if not any(chr.isdigit() for chr in text) and '[[' not in text:
        ind.append(count)
    count=count+1
dataset=dataset['train'].select(ind)
dataset=dataset.map(removeChars)

for point in dataset:
    text=point['text']
    allText=f'{allText}{text}'

charList=list(allText)
for char in charList:
    if char not in vocab:
        vocab[char]=vocabCount
        vocabCount=vocabCount+1    
vocab['|']=vocab[" "]
del vocab[' ']
vocab['[UNK]']=len(vocab)
vocab['[PAD]']=len(vocab)
with open('vocab.json','w') as vf:
    json.dump(vocab,vf)

tokenizer=Wav2Vec2CTCTokenizer.from_pretrained('./',unk_token='[UNK]',pad_token='[PAD]',word_delimiter_token="|")
repoName='parlaSmall_tokenizer'
#tokenizer.push_to_hub(repoName)
extractor=Wav2Vec2FeatureExtractor(feature_size=1,sampling_rate=16000,padding_value=0.0,do_normalize=True,return_attention_mask=True)
processor=Wav2Vec2Processor(feature_extractor=extractor,tokenizer=tokenizer)
dataset=dataset.map(prepare_dataset)
splits=dataset.train_test_split(0.1,0.9)
collator=DataCollatorCTCWithPadding(processor=processor,padding=True)
werMetric=load_metric('wer')
model=Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xls-r-300m",
                                     attention_dropout=0.0,hidden_dropout=0.0,
                                     feat_proj_dropout=0.0,
                                     mask_time_prob=0.05,
                                     layerdrop=0.0,
                                     ctc_loss_reduction='mean',
                                     pad_token_id=processor.tokenizer.pad_token_id,
                                     vocab_size=len(processor.tokenizer))
model.freeze_feature_encoder()
training_args = TrainingArguments(
  group_by_length=True,
  per_device_train_batch_size=1,
  gradient_accumulation_steps=2,
  evaluation_strategy="steps",
  num_train_epochs=30,
  gradient_checkpointing=True,
  save_steps=400,
  eval_steps=400,
  logging_steps=400,
  learning_rate=0.001,
  warmup_steps=500,
  eval_accumulation_steps=1,
  save_total_limit=2,
  push_to_hub=False,
  output_dir='./',
  fp16=True,
)

trainer = Trainer(
    model=model,
    data_collator=collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=splits['train'],
    eval_dataset=splits['test'],
    tokenizer=processor.feature_extractor
)
trainer.train()







