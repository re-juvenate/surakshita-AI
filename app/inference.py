from transformers import AutoModelForTokenClassification
import pytesseract
import torch

model = AutoModelForTokenClassification.from_pretrained("<path-to-checkpoint-folder>")
def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]

def get_polygon_from_bbox(bbox):
    x_min, y_min, x_max, y_max = bbox
    return Polygon([
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max)
    ])

def calculate_iou(polygon1, polygon2):
    intersection = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area
    return intersection / union if union != 0 else 0

def get_word_bounding_boxes(image_path):
    # Load image
    image = Image.open(image_path)
    
    # Perform OCR on the image
    ocr_data = pytesseract.image_to_data(image, output_type=Output.DICT)
    
    # Extract words, bounding boxes, and NER tags
    words = []
    bboxes = []
    word_labels = []
    
    for i, word in enumerate(ocr_data['text']):
        if word.strip() == '':
            continue
        
        words.append(word)
        
        # Extract bounding box coordinates
        x, y, width, height = (ocr_data['left'][i], ocr_data['top'][i],
                               ocr_data['width'][i], ocr_data['height'][i])
        bbox = (x, y, x + width, y + height)
        bboxes.append(bbox)
        
        # Extract NER tags (if available), default to 'O' if not available
        word_labels.append('O')
    
    # Convert bounding boxes to polygons
    polygons = [get_polygon_from_bbox(bbox) for bbox in bboxes]
    
    # Example: Calculate IoU between the first two words' bounding boxes if at least two exist
    iou = None
    if len(polygons) > 1:
        iou = calculate_iou(polygons[0], polygons[1])
    
    # Construct the schema object
    example = {
        "image": image,
        "tokens": words,
        "bboxes": bboxes,
        "bboxes_polygons": [list(p.exterior.coords) for p in polygons],  # convert polygons to list of coords for serialization
        "ner_tags": word_labels,
        "iou_example": iou
    }
    
    return example

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
