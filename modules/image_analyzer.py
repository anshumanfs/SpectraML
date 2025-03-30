import cv2
import numpy as np
import torch
import tensorflow as tf
from PIL import Image
import io
import os
from tensorflow.keras.applications import (
    VGG16, ResNet50, MobileNetV2, EfficientNetB0, InceptionV3
)
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.preprocessing import image as keras_image
import torchvision.transforms as transforms
import torchvision.models as models


class ImageAnalyzer:
    """
    Performs image analysis operations including preprocessing, 
    feature extraction, and deep learning model predictions.
    """
    
    def __init__(self):
        self.supported_operations = [
            'basic_info', 'histogram', 'edge_detection', 'contour_detection',
            'object_detection', 'segmentation', 'face_detection',
            'feature_extraction', 'classification', 'custom_model'
        ]
        
        # Load pre-trained models lazily when needed
        self.tf_models = {}
        self.torch_models = {}
        
        # Ensure model storage directory exists
        os.makedirs('storage/image_models', exist_ok=True)
    
    def analyze(self, image_file, analysis_type, params=None):
        """
        Analyze an image using the specified method
        
        Parameters:
        -----------
        image_file : FileStorage or str
            Image file to analyze (can be a file-like object or a file path)
        analysis_type : str
            Type of analysis to perform
        params : dict, optional
            Additional parameters for the analysis
            
        Returns:
        --------
        dict
            Results of the analysis
        """
        # Handle both file paths and file-like objects
        if isinstance(image_file, str):
            # Load image from file path
            if not os.path.exists(image_file):
                raise FileNotFoundError(f"Image file not found: {image_file}")
            img = cv2.imread(image_file)
            pil_img = Image.open(image_file)
        else:
            # Load image from file-like object
            img_bytes = image_file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            # Reset file pointer and load as PIL Image
            image_file.seek(0)
            pil_img = Image.open(io.BytesIO(img_bytes))
        
        # Check if image was loaded successfully
        if img is None or pil_img is None:
            raise ValueError("Failed to load the image. It may be corrupted or in an unsupported format.")
        
        # Use provided parameters or empty dict
        params = params or {}
        
        # Perform analysis based on type
        if analysis_type == 'basic_info':
            return self._get_basic_info(img, pil_img)
        elif analysis_type == 'histogram':
            return self._analyze_histogram(img, params)
        elif analysis_type == 'edge_detection':
            return self._detect_edges(img, params)
        elif analysis_type == 'contour_detection':
            return self._detect_contours(img, params)
        elif analysis_type == 'object_detection':
            return self._detect_objects(img, params)
        elif analysis_type == 'segmentation':
            return self._segment_image(img, params)
        elif analysis_type == 'face_detection':
            return self._detect_faces(img, params)
        elif analysis_type == 'feature_extraction':
            return self._extract_features(img, pil_img, params)
        elif analysis_type == 'classification':
            return self._classify_image(img, pil_img, params)
        elif analysis_type == 'custom_model':
            return self._apply_custom_model(img, pil_img, params)
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
    
    def _get_basic_info(self, img, pil_img):
        """Get basic information about the image"""
        height, width, channels = img.shape if len(img.shape) == 3 else (*img.shape, 1)
        
        # Calculate average color (if image is not grayscale)
        if channels == 3:
            avg_color_per_channel = np.mean(img, axis=(0, 1))
            avg_color = {
                'blue': float(avg_color_per_channel[0]),
                'green': float(avg_color_per_channel[1]),
                'red': float(avg_color_per_channel[2])
            }
        else:
            avg_color = float(np.mean(img))
        
        # Calculate image entropy (measure of randomness/information)
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_norm = hist / hist.sum()
        non_zero = hist_norm > 0
        entropy = -np.sum(hist_norm[non_zero] * np.log2(hist_norm[non_zero]))
        
        return {
            'width': width,
            'height': height,
            'channels': channels,
            'aspect_ratio': width / height,
            'format': pil_img.format,
            'mode': pil_img.mode,
            'dpi': pil_img.info.get('dpi'),
            'avg_color': avg_color,
            'entropy': float(entropy),
            'file_size_kb': os.fstat(pil_img.fp.fileno()).st_size / 1024 if hasattr(pil_img, 'fp') and hasattr(pil_img.fp, 'fileno') else None
        }
    
    def _analyze_histogram(self, img, params):
        """Analyze the image histogram"""
        # Convert to desired color space
        color_space = params.get('color_space', 'BGR')
        
        if color_space == 'BGR':
            color_img = img
        elif color_space == 'RGB':
            color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif color_space == 'HSV':
            color_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif color_space == 'LAB':
            color_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        elif color_space == 'GRAY':
            color_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unsupported color space: {color_space}")
        
        # Calculate histograms
        if len(color_img.shape) == 3 and color_img.shape[2] == 3:
            hist_data = []
            
            for i, color in enumerate(['first', 'second', 'third']):
                hist = cv2.calcHist([color_img], [i], None, [256], [0, 256])
                hist_data.append({
                    'channel': color,
                    'histogram': hist.flatten().tolist()
                })
        else:
            # Grayscale image
            hist = cv2.calcHist([color_img], [0], None, [256], [0, 256])
            hist_data = [{
                'channel': 'intensity',
                'histogram': hist.flatten().tolist()
            }]
        
        # Calculate histogram statistics
        basic_stats = []
        for i, hist in enumerate(hist_data):
            values = np.array(hist['histogram'])
            non_zero = values > 0
            if np.any(non_zero):
                # Find intensity range
                indices = np.arange(256)
                min_intensity = indices[non_zero][0]
                max_intensity = indices[non_zero][-1]
                
                # Calculate statistics
                total_pixels = values.sum()
                mean = np.sum(indices * values) / total_pixels
                variance = np.sum(((indices - mean) ** 2) * values) / total_pixels
                std_dev = np.sqrt(variance)
                
                basic_stats.append({
                    'channel': hist['channel'],
                    'min': int(min_intensity),
                    'max': int(max_intensity),
                    'mean': float(mean),
                    'std_dev': float(std_dev),
                    'entropy': float(-np.sum((values/total_pixels)[non_zero] * np.log2((values/total_pixels)[non_zero])))
                })
        
        return {
            'color_space': color_space,
            'histogram_data': hist_data,
            'statistics': basic_stats
        }
    
    def _detect_edges(self, img, params):
        """Detect edges in the image"""
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Get edge detection parameters
        method = params.get('method', 'canny')
        
        if method == 'canny':
            threshold1 = params.get('threshold1', 100)
            threshold2 = params.get('threshold2', 200)
            aperture_size = params.get('aperture_size', 3)
            
            edges = cv2.Canny(gray, threshold1, threshold2, apertureSize=aperture_size)
            
        elif method == 'sobel':
            ksize = params.get('ksize', 3)
            scale = params.get('scale', 1)
            delta = params.get('delta', 0)
            
            # Compute Sobel in x and y directions
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize, scale=scale, delta=delta)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize, scale=scale, delta=delta)
            
            # Compute gradient magnitude
            mag = np.sqrt(sobelx ** 2 + sobely ** 2)
            # Normalize to 0-255
            edges = np.uint8(255 * mag / np.max(mag))
        
        elif method == 'laplacian':
            ksize = params.get('ksize', 3)
            lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
            edges = np.uint8(np.absolute(lap))
        
        else:
            raise ValueError(f"Unsupported edge detection method: {method}")
        
        # Count edges (rough approximation)
        edge_count = np.count_nonzero(edges)
        edge_percentage = (edge_count / (edges.shape[0] * edges.shape[1])) * 100
        
        return {
            'method': method,
            'edge_count': int(edge_count),
            'edge_percentage': float(edge_percentage),
            'params': params,
            'edges_image': edges.tolist()  # Convert the edge image to a list for JSON serialization
        }
    
    def _detect_contours(self, img, params):
        """Detect contours in the image"""
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Apply thresholding
        threshold_type = params.get('threshold_type', 'binary')
        threshold_value = params.get('threshold_value', 127)
        
        if threshold_type == 'binary':
            _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        elif threshold_type == 'otsu':
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif threshold_type == 'adaptive':
            block_size = params.get('block_size', 11)
            c = params.get('c', 2)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, block_size, c)
        else:
            raise ValueError(f"Unsupported threshold type: {threshold_type}")
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process contours
        contour_data = []
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            
            # Skip very small contours
            if area < params.get('min_area', 10):
                continue
                
            # Calculate bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Convert contour to a simpler representation for serialization
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            simplified_contour = approx.reshape(-1, 2).tolist()
            
            contour_data.append({
                'id': i,
                'area': float(area),
                'perimeter': float(perimeter),
                'bounding_box': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                'contour_points': simplified_contour
            })
        
        return {
            'threshold_type': threshold_type,
            'contour_count': len(contour_data),
            'contours': contour_data
        }
    
    def _detect_objects(self, img, params):
        """Detect objects in the image using pre-trained models"""
        model_name = params.get('model', 'yolo')
        confidence_threshold = params.get('confidence', 0.5)
        
        if model_name == 'yolo':
            # Use YOLO model from OpenCV
            yolo_dir = 'storage/image_models/yolo'
            os.makedirs(yolo_dir, exist_ok=True)
            
            # Check if YOLO weights and configuration exist, download if not
            weights_path = os.path.join(yolo_dir, 'yolov3.weights')
            config_path = os.path.join(yolo_dir, 'yolov3.cfg')
            classes_path = os.path.join(yolo_dir, 'coco.names')
            
            # Load the model
            try:
                net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
            except:
                return {'error': 'YOLO model files not found. Please download yolov3.weights, yolov3.cfg, and coco.names to storage/image_models/yolo/'}
            
            # Read class names
            try:
                with open(classes_path, 'r') as f:
                    classes = [line.strip() for line in f.readlines()]
            except:
                return {'error': 'COCO class names file not found'}
            
            # Get output layer names
            layer_names = net.getLayerNames()
            try:
                # OpenCV 4.5.4 and earlier
                output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            except:
                # OpenCV 4.5.5 and later
                output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
            
            # Process image
            height, width, _ = img.shape
            blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            
            # Forward pass
            outputs = net.forward(output_layers)
            
            # Process outputs
            boxes = []
            confidences = []
            class_ids = []
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > confidence_threshold:
                        # YOLO returns coordinates relative to the center of the box
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # Calculate top-left corner
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply non-maximum suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
            
            # Prepare results
            detections = []
            for i in indices:
                try:
                    # OpenCV 4.5.4 and earlier
                    i = i[0]
                except:
                    # OpenCV 4.5.5 and later
                    pass
                
                box = boxes[i]
                detection = {
                    'class': classes[class_ids[i]],
                    'confidence': confidences[i],
                    'box': {
                        'x': box[0],
                        'y': box[1],
                        'width': box[2],
                        'height': box[3]
                    }
                }
                detections.append(detection)
            
            return {
                'model': 'YOLO',
                'detection_count': len(detections),
                'detections': detections
            }
        
        else:
            return {'error': f"Unsupported object detection model: {model_name}"}
    
    def _segment_image(self, img, params):
        """Segment the image into regions"""
        method = params.get('method', 'kmeans')
        
        if method == 'kmeans':
            # Reshape image for K-means
            pixel_vals = img.reshape((-1, 3))
            pixel_vals = np.float32(pixel_vals)
            
            # K-means parameters
            k = params.get('k', 5)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
            
            # Apply K-means
            _, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Convert back to uint8
            centers = np.uint8(centers)
            segmented_img = centers[labels.flatten()]
            segmented_img = segmented_img.reshape(img.shape)
            
            # Count pixels in each segment
            segment_counts = np.bincount(labels.flatten(), minlength=k)
            total_pixels = img.shape[0] * img.shape[1]
            
            segments = []
            for i in range(k):
                center = centers[i].tolist()
                segments.append({
                    'segment_id': i,
                    'center_color': center,
                    'pixel_count': int(segment_counts[i]),
                    'percentage': float(segment_counts[i] / total_pixels * 100)
                })
            
            return {
                'method': 'kmeans',
                'k': k,
                'segments': segments
            }
        
        elif method == 'watershed':
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Otsu's thresholding
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Noise removal with morphological operations
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Sure background area
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            
            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            
            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            
            # Marker labelling
            _, markers = cv2.connectedComponents(sure_fg)
            
            # Add 1 to all labels so that background is not 0, but 1
            markers = markers + 1
            
            # Mark the unknown region with 0
            markers[unknown == 255] = 0
            
            # Apply watershed
            markers = cv2.watershed(img, markers)
            
            # Count objects (excluding background and boundaries)
            unique_markers = np.unique(markers)
            num_objects = len(unique_markers) - 2  # Exclude background (1) and boundaries (-1)
            
            return {
                'method': 'watershed',
                'object_count': num_objects,
                'unique_segments': len(unique_markers)
            }
        
        else:
            return {'error': f"Unsupported segmentation method: {method}"}
    
    def _detect_faces(self, img, params):
        """Detect faces in the image"""
        # Load face cascade classifier
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        min_neighbors = params.get('min_neighbors', 5)
        scale_factor = params.get('scale_factor', 1.1)
        min_size = params.get('min_size', (30, 30))
        
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        
        # Process detected faces
        face_data = []
        for i, (x, y, w, h) in enumerate(faces):
            face_roi = gray[y:y+h, x:x+w]
            
            # Detect eyes within the face
            eyes = eye_cascade.detectMultiScale(face_roi)
            
            # Process eyes
            eye_data = []
            for (ex, ey, ew, eh) in eyes:
                eye_data.append({
                    'x': int(ex),
                    'y': int(ey),
                    'width': int(ew),
                    'height': int(eh)
                })
            
            face_data.append({
                'face_id': i,
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h),
                'eyes': eye_data,
                'eye_count': len(eye_data)
            })
        
        return {
            'face_count': len(faces),
            'faces': face_data
        }
    
    def _extract_features(self, img, pil_img, params):
        """Extract features from the image using pre-trained models"""
        model_name = params.get('model', 'vgg16')
        layer = params.get('layer', 'fc1')
        
        # Lazy-load TensorFlow model
        if model_name not in self.tf_models:
            if model_name == 'vgg16':
                self.tf_models[model_name] = VGG16(weights='imagenet', include_top=True)
                preprocess_fn = vgg_preprocess
            elif model_name == 'resnet50':
                self.tf_models[model_name] = ResNet50(weights='imagenet', include_top=True)
                preprocess_fn = resnet_preprocess
            elif model_name == 'mobilenet':
                self.tf_models[model_name] = MobileNetV2(weights='imagenet', include_top=True)
                preprocess_fn = mobilenet_preprocess
            elif model_name == 'efficientnet':
                self.tf_models[model_name] = EfficientNetB0(weights='imagenet', include_top=True)
                preprocess_fn = efficientnet_preprocess
            elif model_name == 'inception':
                self.tf_models[model_name] = InceptionV3(weights='imagenet', include_top=True)
                preprocess_fn = inception_preprocess
            else:
                raise ValueError(f"Unsupported model: {model_name}")
        
        # Create feature extraction model
        model = self.tf_models[model_name]
        
        # Convert OpenCV image to PIL image if needed and resize
        if isinstance(pil_img, np.ndarray):
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Resize and preprocess
        input_shape = model.input_shape[1:3]  # (height, width)
        img_array = keras_image.img_to_array(pil_img.resize(input_shape))
        img_array = np.expand_dims(img_array, axis=0)
        
        # Handle different models' preprocessing
        if model_name == 'vgg16':
            processed_img = vgg_preprocess(img_array)
        elif model_name == 'resnet50':
            processed_img = resnet_preprocess(img_array)
        elif model_name == 'mobilenet':
            processed_img = mobilenet_preprocess(img_array)
        elif model_name == 'efficientnet':
            processed_img = efficientnet_preprocess(img_array)
        elif model_name == 'inception':
            processed_img = inception_preprocess(img_array)
        
        # Extract features
        features = model.predict(processed_img)
        
        # Get specific statistics about the features
        feature_stats = {
            'min': float(np.min(features)),
            'max': float(np.max(features)),
            'mean': float(np.mean(features)),
            'std': float(np.std(features)),
            'feature_dimension': features.shape[1]
        }
        
        # Return summarized features (too large to return all)
        return {
            'model': model_name,
            'feature_stats': feature_stats
        }
    
    def _classify_image(self, img, pil_img, params):
        """Classify the image using pre-trained models"""
        model_name = params.get('model', 'vgg16')
        top_k = params.get('top_k', 5)
        
        # Lazy-load TensorFlow model for classification
        if model_name not in self.tf_models:
            if model_name == 'vgg16':
                self.tf_models[model_name] = VGG16(weights='imagenet')
                preprocess_fn = vgg_preprocess
            elif model_name == 'resnet50':
                self.tf_models[model_name] = ResNet50(weights='imagenet')
                preprocess_fn = resnet_preprocess
            elif model_name == 'mobilenet':
                self.tf_models[model_name] = MobileNetV2(weights='imagenet')
                preprocess_fn = mobilenet_preprocess
            elif model_name == 'efficientnet':
                self.tf_models[model_name] = EfficientNetB0(weights='imagenet')
                preprocess_fn = efficientnet_preprocess
            elif model_name == 'inception':
                self.tf_models[model_name] = InceptionV3(weights='imagenet')
                preprocess_fn = inception_preprocess
            else:
                raise ValueError(f"Unsupported model: {model_name}")
        
        model = self.tf_models[model_name]
        
        # Convert OpenCV image to PIL image if needed and resize
        if isinstance(pil_img, np.ndarray):
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Resize and preprocess
        input_shape = model.input_shape[1:3]  # (height, width)
        img_array = keras_image.img_to_array(pil_img.resize(input_shape))
        img_array = np.expand_dims(img_array, axis=0)
        
        # Handle different models' preprocessing
        if model_name == 'vgg16':
            processed_img = vgg_preprocess(img_array)
        elif model_name == 'resnet50':
            processed_img = resnet_preprocess(img_array)
        elif model_name == 'mobilenet':
            processed_img = mobilenet_preprocess(img_array)
        elif model_name == 'efficientnet':
            processed_img = efficientnet_preprocess(img_array)
        elif model_name == 'inception':
            processed_img = inception_preprocess(img_array)
        
        # Make predictions
        predictions = model.predict(processed_img)
        
        # Decode predictions
        from tensorflow.keras.applications.imagenet_utils import decode_predictions
        decoded = decode_predictions(predictions, top=top_k)[0]
        
        # Format results
        results = []
        for i, (imagenet_id, label, score) in enumerate(decoded):
            results.append({
                'rank': i + 1,
                'label': label,
                'score': float(score),
                'imagenet_id': imagenet_id
            })
        
        return {
            'model': model_name,
            'top_predictions': results
        }
    
    def _apply_custom_model(self, img, pil_img, params):
        """Apply a custom-trained model to the image"""
        model_path = params.get('model_path')
        model_format = params.get('model_format', 'tensorflow')
        
        if not model_path or not os.path.exists(model_path):
            return {'error': 'Model path is invalid or not provided'}
        
        try:
            if model_format == 'tensorflow':
                # Load TensorFlow model
                model = tf.keras.models.load_model(model_path)
                
                # Process image
                input_shape = model.input_shape[1:3]
                img_array = keras_image.img_to_array(pil_img.resize(input_shape))
                img_array = np.expand_dims(img_array, axis=0) / 255.0
                
                # Make prediction
                predictions = model.predict(img_array)
                
                return {
                    'model_format': 'tensorflow',
                    'raw_predictions': predictions.tolist(),
                    'output_shape': predictions.shape[1]
                }
                
            elif model_format == 'pytorch':
                # Load PyTorch model
                model = torch.load(model_path)
                model.eval()
                
                # Process image
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                ])
                
                input_tensor = transform(pil_img)
                input_batch = input_tensor.unsqueeze(0)
                
                # Make prediction
                with torch.no_grad():
                    output = model(input_batch)
                
                # Convert output to numpy for easier handling
                predictions = output.numpy()
                
                return {
                    'model_format': 'pytorch',
                    'raw_predictions': predictions.tolist(),
                    'output_shape': predictions.shape[1]
                }
                
            else:
                return {'error': f"Unsupported model format: {model_format}"}
                
        except Exception as e:
            return {'error': f"Error applying custom model: {str(e)}"}
