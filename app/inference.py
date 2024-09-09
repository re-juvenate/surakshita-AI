from transformers import AutoModelForTokenClassification
import torch

model = AutoModelForTokenClassification.from_pretrained("<path-to-checkpoint-folder>")
def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]

def infer_layoutlm(example, processor, model):
    image = example["image"]
    words = example["tokens"]
    boxes = example["bboxes"]
    word_labels = example["ner_tags"]


    encoding = processor(image, words, boxes=boxes, word_labels=word_labels, return_tensors="pt")
    for k,v in encoding.items():
    print(k,v.shape)

    with torch.no_grad():
        outputs = model(**encoding)
    logits = outputs.logits
    predictions = logits.argmax(-1).squeeze().tolist()
    labels = encoding.labels.squeeze().tolist()
    
    token_boxes = encoding.bbox.squeeze().tolist()
    width, height = image.size

    true_predictions = [model.config.id2label[pred] for pred, label in zip(predictions, labels) if label != - 100]
    true_labels = [model.config.id2label[label] for prediction, label in zip(predictions, labels) if label != -100]
    true_boxes = [unnormalize_box(box, width, height) for box, label in zip(token_boxes, labels) if label != -100]
