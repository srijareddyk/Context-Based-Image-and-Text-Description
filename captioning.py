import cv2
import pytesseract
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer, BartTokenizer, BartForConditionalGeneration
import torch
from translate import Translator 

# Preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    return blurred_image

# Extract text from the image using Tesseract
def extract_text(image):
    return pytesseract.image_to_string(image)

# Generate a caption for an image
def image_caption(image_paths):
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        images.append(i_image)
    
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    
    output_ids = model.generate(pixel_values)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    
    return preds

# Generate a summary of the text
def llm(text, desc):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    
    text_to_summarize = desc + "\n" + text 
    input_ids = tokenizer.encode("The image depicts " + text_to_summarize, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(input_ids, max_length=150, num_beams=4, length_penalty=2.0, min_length=30, early_stopping=True)
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Translate text to a target language
def translate_text(text, target_language):
    translator = Translator(to_lang=target_language)
    return translator.translate(text)

# Main function to integrate everything
def main(image_path):
    pre_image = preprocess_image(image_path)
    extracted_text = extract_text(pre_image)
    image_caption_text = image_caption([image_path])
    summary_text = llm(extracted_text, image_caption_text[0])
    translation_hindi = translate_text(summary_text, "hi")
    translation_telugu = translate_text(summary_text, "te")
    
    return summary_text, translation_hindi, translation_telugu
