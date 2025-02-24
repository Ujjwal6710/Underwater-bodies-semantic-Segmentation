import argparse
import logging
import os
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array, load_img
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU usage


def preprocess_image(image, scale_factor):
    target_size = (256, 256)  # Ensure the image is resized correctly
    img = image.resize(target_size)
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


def predict_img(model, full_img, scale_factor=1, out_threshold=0.5):
    img = preprocess_image(full_img, scale_factor)
    output = model.predict(img)[0]
    
    if output.shape[-1] > 1:
        mask = np.argmax(output, axis=-1)
    else:
        mask = (output[..., 0] > out_threshold).astype(np.uint8)
    
    return mask


def create_mixed_overlap(original, mask, output_path):
    plt.figure(figsize=(12, 5))
    
    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis('off')
    
    # Predicted Mask
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis('off')
    
    # Mixed Overlap
    plt.subplot(1, 3, 3)
    plt.imshow(original)
    plt.imshow(mask, alpha=0.5, cmap='jet')
    plt.title("Mixed Overlap")
    plt.axis('off')
    
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images using DeepLabV3')
    parser.add_argument('--model', '-m', default='/home/ujjwal6710/Underwater Semantic Segmentation/UnderWater-Semantic-Segmentation/saved_models/deeplab.pkl', metavar='FILE',
                        help='Specify the file in which the model is stored (.pkl format)')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--classes', '-c', type=int, default=21, help='Number of classes for DeepLabV3')
    
    return parser.parse_args()


def get_output_filenames(args):
    return args.output or [f'{os.path.splitext(fn)[0]}_OUT.png' for fn in args.input]


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = get_output_filenames(args)

    logging.info(f'Loading model {args.model}')
    try:
        with open(args.model, 'rb') as f:
            model = pickle.load(f)
        if not hasattr(model, 'predict'):
            raise ValueError("Loaded model does not have a 'predict' method. Ensure you are loading the correct DeepLabV3 model.")
    except Exception as e:
        logging.error(f'Error loading model: {e}')
        exit(1)
    
    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        try:
            img = Image.open(filename).convert("RGB")
            mask = predict_img(model=model,
                               full_img=img,
                               scale_factor=args.scale,
                               out_threshold=args.mask_threshold)
            
            if not args.no_save:
                out_filename = out_files[i]
                result = mask_to_image(mask)
                result.save(out_filename)
                logging.info(f'Mask saved to {out_filename}')
                
                # Generate mixed overlap visualization
                mixed_overlap_path = f'Outputs/{os.path.splitext(out_filename)[0]}_mixed.png'
                create_mixed_overlap(img, mask, mixed_overlap_path)
                logging.info(f'Mixed overlap saved to {mixed_overlap_path}')

            if args.viz:
                img.show(title='Original Image')
                result.show(title='Predicted Mask')
        except Exception as e:
            logging.error(f'Error processing {filename}: {e}')
