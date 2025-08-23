#!/usr/bin/env python3
"""
Enhanced Advanced DeepFake Detection & Secure Browsing Platform v3.0
A comprehensive AI-powered solution for media authenticity verification and secure content analysis

Features:
- Advanced neural network detection with CNN-LSTM hybrid architecture
- Real-time video monitoring with multi-source support
- Multi-modal analysis fusion (video, audio, metadata)
- Comprehensive analytics dashboard with interactive visualizations
- Enhanced face manipulation detection with 8 analysis metrics
- Performance optimization with hardware detection
- Professional GUI with tabbed interface
- Database integration for analysis history
- Privacy protection with data anonymization
- Comprehensive logging and error handling

Author: AI Security Solutions Team
Version: 3.0 Enhanced
License: MIT
"""

import os
import sys
import subprocess
import pkg_resources
from packaging import version
import warnings
warnings.filterwarnings('ignore')

def check_and_install_dependencies():
    """Check and install required packages with user confirmation"""
    required_packages = [
        'opencv-python',
        'torch',
        'torchvision', 
        'torchaudio',
        'numpy',
        'pillow',
        'matplotlib',
        'seaborn',
        'scipy',
        'scikit-learn',
        'cryptography',
        'requests',
        'psutil',
        'selenium',
        'webdriver-manager'
    ]
    
    try:
        installed_packages = [pkg.project_name.lower() for pkg in pkg_resources.working_set]
        missing_packages = []
        
        for package in required_packages:
            if package.lower() not in installed_packages:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"Missing packages detected: {missing_packages}")
            response = input("Install missing packages automatically? (y/n): ").lower().strip()
            
            if response == 'y':
                print("Installing required packages...")
                for package in missing_packages:
                    try:
                        print(f"Installing {package}...")
                        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
                        print(f"✓ Successfully installed {package}")
                    except subprocess.CalledProcessError as e:
                        print(f"✗ Failed to install {package}: {e}")
                        return False
                print("All packages installed successfully!")
            else:
                print("Please install missing packages manually:")
                for package in missing_packages:
                    print(f"  pip install {package}")
                return False
                
        return True
        
    except Exception as e:
        print(f"Dependency check failed: {e}")
        print("Continuing with available packages...")
        return True

# Install dependencies if needed
print("Enhanced DeepFake Detection Platform v3.0")
print("=========================================")
print("Checking dependencies...")

if not check_and_install_dependencies():
    print("Some dependencies are missing. The application may not function properly.")
    input("Press Enter to continue anyway, or Ctrl+C to exit...")

# Import all required modules
print("Loading modules...")

try:
    import cv2
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    import torchvision.models as models  # Added for pre-trained models
    from torch.utils.data import DataLoader, Dataset
except ImportError as e:
    print(f"Critical import error: {e}")
    print("Please install PyTorch and OpenCV: pip install torch torchvision opencv-python")
    sys.exit(1)

import sqlite3
import hashlib
import json
import threading
import time
import requests
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import base64
from cryptography.fernet import Fernet
import tempfile
import shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import simpledialog  # Fixed: Imported simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    import seaborn as sns
    sns.set_style("whitegrid")
    SEABORN_AVAILABLE = True
except ImportError:
    print("Seaborn not available, using matplotlib defaults")
    SEABORN_AVAILABLE = False

from PIL import Image, ImageTk
import io
import uuid
from collections import deque
from scipy import stats
import pickle
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import secrets
import wave
import struct

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    print("psutil not available - system monitoring will be limited")
    PSUTIL_AVAILABLE = False

# Try to import selenium with fallback
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    print("Selenium or webdriver-manager not available. Secure browsing features will be limited.")
    SELENIUM_AVAILABLE = False

# Configure comprehensive logging
def setup_logging():
    """Setup comprehensive logging system"""
    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # File handler
    file_handler = logging.FileHandler('logs/deepfake_platform.log')
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler  
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    console_handler.setLevel(logging.INFO)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)

logger = setup_logging()
logger.info("Enhanced DeepFake Detection Platform v3.0 starting...")

class AdvancedDeepFakeDetector(nn.Module):
    """
    Advanced CNN-LSTM hybrid model for deepfake detection
    
    Architecture:
    - Pre-trained EfficientNet-B0 feature extractor (improved for better detection)
    - Multi-head self-attention mechanism
    - Bidirectional LSTM for temporal analysis
    - Multi-layer classification head with dropout
    """
    
    def __init__(self, input_channels=3, hidden_size=512, num_classes=2):
        super(AdvancedDeepFakeDetector, self).__init__()
        
        logger.info(f"Initializing AdvancedDeepFakeDetector with {num_classes} classes")
        
        # Use pre-trained EfficientNet-B0 for better feature extraction
        self.feature_extractor = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # Replace classifier to output 512 features
        self.feature_extractor.classifier = nn.Linear(self.feature_extractor.classifier[1].in_features, 512)
        
        # Multi-head self-attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=512, 
            num_heads=8, 
            dropout=0.1,
            batch_first=True
        )
        
        # Bidirectional LSTM for temporal analysis
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Classification head with dropout and batch normalization
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights using best practices
        self._initialize_weights()
        
        logger.info("AdvancedDeepFakeDetector initialized with pre-trained EfficientNet backbone")
        
    def _initialize_weights(self):
        """Initialize model weights using Xavier/He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, channels, height, width]
            
        Returns:
            output: Classification logits [batch_size, num_classes]
            attention_weights: Attention weights for interpretability
        """
        batch_size, seq_len, c, h, w = x.size()
        
        # Process each frame through pre-trained feature extractor
        features = []
        for i in range(seq_len):
            frame_features = self.feature_extractor(x[:, i])
            features.append(frame_features)
        
        # Stack features for sequence processing
        features = torch.stack(features, dim=1)  # [batch_size, seq_len, 512]
        
        # Apply multi-head self-attention
        attended_features, attention_weights = self.attention(features, features, features)
        
        # Add residual connection
        attended_features = attended_features + features
        
        # LSTM processing for temporal analysis
        lstm_out, (hidden, cell) = self.lstm(attended_features)
        
        # Use the last time step output for classification
        final_features = lstm_out[:, -1, :]
        
        # Classification
        output = self.classifier(final_features)
        
        return output, attention_weights

class EnhancedFaceAnalyzer:
    """
    Enhanced face analysis system with 8 comprehensive detection methods:
    1. Compression artifact detection
    2. Blending artifact detection  
    3. Lighting consistency analysis
    4. Texture naturalness analysis
    5. Geometric consistency analysis
    6. Color naturalness analysis
    7. Edge consistency analysis
    8. Frequency domain analysis
    """
    
    def __init__(self):
        logger.info("Initializing EnhancedFaceAnalyzer")
        
        # Initialize OpenCV cascade classifiers
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            self.profile_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_profileface.xml'
            )
            logger.info("OpenCV cascade classifiers loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load cascade classifiers: {e}")
            self.face_cascade = None
            self.eye_cascade = None
            self.profile_cascade = None
        
        # Initialize feature detectors
        try:
            self.sift = cv2.SIFT_create(nfeatures=500)
            self.orb = cv2.ORB_create(nfeatures=500)
            logger.info("Feature detectors initialized")
        except Exception as e:
            logger.warning(f"Feature detectors not available: {e}")
            self.sift = None
            self.orb = None
            
        # Analysis thresholds and parameters
        self.analysis_params = {
            'compression_threshold': 0.6,
            'blending_threshold': 0.5,
            'lighting_threshold': 0.4,
            'texture_threshold': 0.5,
            'geometry_threshold': 0.3,
            'color_threshold': 0.4,
            'edge_threshold': 0.5,
            'frequency_threshold': 0.4
        }
        
        logger.info("EnhancedFaceAnalyzer initialized successfully")
    
    def comprehensive_face_analysis(self, frame, return_visualizations=False):
        """
        Perform comprehensive face analysis with multiple detection methods
        
        Args:
            frame: Input image frame
            return_visualizations: Whether to return visualization overlays
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        try:
            analysis_start = time.time()
            
            results = {
                'faces_detected': 0,
                'face_regions': [],
                'analysis_results': [],
                'overall_confidence': 0.0,
                'processing_time': 0.0,
                'timestamp': datetime.now().isoformat(),
                'analysis_methods': [
                    'compression_artifacts',
                    'blending_artifacts', 
                    'lighting_consistency',
                    'texture_naturalness',
                    'geometric_consistency',
                    'color_naturalness',
                    'edge_consistency',
                    'frequency_analysis'
                ],
                'visualizations': {} if return_visualizations else None
            }
            
            if self.face_cascade is None:
                logger.warning("Face cascade not available, using center region analysis")
                # Analyze center region as fallback
                h, w = frame.shape[:2]
                center_region = (w//4, h//4, w//2, h//2)
                results['face_regions'] = [{'id': 0, 'bbox': center_region, 'type': 'center_region'}]
                results['faces_detected'] = 1
            else:
                # Convert to grayscale for detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Multi-scale face detection with multiple cascades
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5,
                    minSize=(50, 50),
                    maxSize=(500, 500),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                # Try profile detection if no frontal faces found
                if len(faces) == 0 and self.profile_cascade is not None:
                    faces = self.profile_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(50, 50)
                    )
                
                results['faces_detected'] = len(faces)
                
                # Store face region information
                for i, (x, y, w, h) in enumerate(faces):
                    results['face_regions'].append({
                        'id': i,
                        'bbox': (x, y, w, h),
                        'size': w * h,
                        'aspect_ratio': w / h,
                        'center': (x + w//2, y + h//2),
                        'type': 'detected_face'
                    })
            
            # Analyze each detected face region
            total_suspicion = 0.0
            visualized_frame = frame.copy() if return_visualizations else None
            
            for i, face_info in enumerate(results['face_regions']):
                x, y, w, h = face_info['bbox']
                
                # Extract face region
                face_region = frame[y:y+h, x:x+w]
                face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                
                # Perform comprehensive analysis
                analysis = self._analyze_single_face(face_region, face_gray)
                analysis.update({
                    'face_id': i,
                    'bbox': (x, y, w, h),
                    'face_info': face_info
                })
                
                results['analysis_results'].append(analysis)
                total_suspicion += analysis.get('overall_suspicion', 0.0)
                
                # Add visualization if requested
                if return_visualizations and visualized_frame is not None:
                    self._add_face_visualization(visualized_frame, (x, y, w, h), analysis)
            
            # Calculate overall confidence
            if len(results['face_regions']) > 0:
                results['overall_confidence'] = total_suspicion / len(results['face_regions'])
            
            results['processing_time'] = time.time() - analysis_start
            
            # Store visualizations
            if return_visualizations:
                results['visualizations']['annotated_frame'] = visualized_frame
                results['visualizations']['detection_overlay'] = self._create_detection_overlay(frame, results)
            
            logger.debug(f"Face analysis completed in {results['processing_time']:.3f}s, "
                        f"detected {results['faces_detected']} faces with "
                        f"{results['overall_confidence']:.1%} overall suspicion")
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive face analysis failed: {e}")
            return {
                'error': str(e),
                'faces_detected': 0,
                'overall_confidence': 0.0,
                'processing_time': time.time() - analysis_start if 'analysis_start' in locals() else 0.0,
                'timestamp': datetime.now().isoformat()
            }
    
    def _analyze_single_face(self, face_region, face_gray):
        """
        Analyze a single face region using all 8 detection methods
        
        Args:
            face_region: Color face region (BGR)
            face_gray: Grayscale face region
            
        Returns:
            Dictionary with analysis results for all methods
        """
        try:
            analysis = {
                'compression_artifacts': 0.0,
                'blending_artifacts': 0.0, 
                'lighting_consistency': 0.0,
                'texture_naturalness': 0.0,
                'geometric_consistency': 0.0,
                'color_naturalness': 0.0,
                'edge_consistency': 0.0,
                'frequency_analysis': 0.0,
                'overall_suspicion': 0.0,
                'analysis_details': {},
                'confidence_scores': {}
            }
            
            # Method 1: Compression artifact detection using DCT analysis
            analysis['compression_artifacts'] = self._detect_compression_artifacts(face_region)
            
            # Method 2: Blending artifact detection using edge analysis
            analysis['blending_artifacts'] = self._detect_blending_artifacts(face_region)
            
            # Method 3: Lighting consistency analysis
            analysis['lighting_consistency'] = self._analyze_lighting_consistency(face_region)
            
            # Method 4: Texture naturalness using LBP analysis
            analysis['texture_naturalness'] = self._analyze_texture_naturalness(face_gray)
            
            # Method 5: Geometric consistency analysis
            analysis['geometric_consistency'] = self._analyze_geometric_consistency(face_gray)
            
            # Method 6: Color naturalness analysis
            analysis['color_naturalness'] = self._analyze_color_naturalness(face_region)
            
            # Method 7: Edge consistency analysis
            analysis['edge_consistency'] = self._analyze_edge_consistency(face_gray)
            
            # Method 8: Frequency domain analysis
            analysis['frequency_analysis'] = self._frequency_domain_analysis(face_gray)
            
            # Calculate overall suspicion score with weighted average
            weights = {
                'compression_artifacts': 0.15,
                'blending_artifacts': 0.15,
                'lighting_consistency': 0.12,
                'texture_naturalness': 0.15,
                'geometric_consistency': 0.10,
                'color_naturalness': 0.13,
                'edge_consistency': 0.10,
                'frequency_analysis': 0.10
            }
            
            weighted_score = 0.0
            for method, weight in weights.items():
                score = analysis.get(method, 0.0)
                weighted_score += score * weight
                analysis['confidence_scores'][method] = score
            
            analysis['overall_suspicion'] = min(max(weighted_score, 0.0), 1.0)
            
            # Store analysis details
            analysis['analysis_details'] = {
                'region_size': face_region.shape,
                'analysis_weights': weights,
                'suspicious_methods': [
                    method for method, score in analysis['confidence_scores'].items() 
                    if score > self.analysis_params.get(f"{method.split('_')[0]}_threshold", 0.5)
                ]
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Single face analysis failed: {e}")
            return {
                'error': str(e),
                'overall_suspicion': 0.0,
                'analysis_details': {'error': str(e)}
            }
    
    def _detect_compression_artifacts(self, face_region):
        """
        Detect compression artifacts using DCT (Discrete Cosine Transform) analysis
        
        Compression artifacts appear as:
        - Blocking effects in 8x8 DCT blocks
        - Reduced high-frequency content
        - Quantization noise patterns
        """
        try:
            # Convert to grayscale and standardize size
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Resize to multiple of 8 for DCT analysis (JPEG uses 8x8 blocks)
            target_size = 128  # 128x128 = 16x16 blocks of 8x8
            gray_face = cv2.resize(gray_face, (target_size, target_size))
            
            # Apply DCT to the entire image
            dct_full = cv2.dct(np.float32(gray_face))
            
            # Analyze DCT coefficient distribution
            # Split into frequency bands
            block_size = 8
            compression_scores = []
            
            # Analyze each 8x8 block
            for i in range(0, target_size, block_size):
                for j in range(0, target_size, block_size):
                    block = dct_full[i:i+block_size, j:j+block_size]
                    
                    # Calculate energy distribution
                    # DC component (top-left)
                    dc_energy = abs(block[0, 0])
                    
                    # Low frequency (top-left 3x3)
                    low_freq_energy = np.sum(np.abs(block[:3, :3])) - dc_energy
                    
                    # High frequency (remaining coefficients)
                    high_freq_energy = np.sum(np.abs(block)) - dc_energy - low_freq_energy
                    
                    total_energy = dc_energy + low_freq_energy + high_freq_energy
                    
                    if total_energy > 0:
                        # Compression typically reduces high-frequency content
                        high_freq_ratio = high_freq_energy / total_energy
                        
                        # Normal images have moderate high-frequency content (0.1-0.3)
                        # Heavily compressed images have very low high-frequency content (<0.1)
                        if high_freq_ratio < 0.05:  # Very low high frequencies
                            compression_scores.append(0.8)
                        elif high_freq_ratio < 0.1:  # Low high frequencies  
                            compression_scores.append(0.6)
                        elif high_freq_ratio > 0.5:  # Unusually high (possible artifact)
                            compression_scores.append(0.4)
                        else:
                            compression_scores.append(0.1)
                    else:
                        compression_scores.append(0.0)
            
            # Average compression artifact score
            avg_compression_score = np.mean(compression_scores) if compression_scores else 0.0
            
            # Additional analysis: Look for blocking artifacts
            blocking_score = self._detect_blocking_artifacts(gray_face, block_size)
            
            # Combine scores
            final_score = (avg_compression_score * 0.7 + blocking_score * 0.3)
            
            return min(max(final_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Compression artifact detection failed: {e}")
            return 0.0
    
    def _detect_blocking_artifacts(self, image, block_size=8):
        """Detect JPEG-like blocking artifacts"""
        try:
            h, w = image.shape
            blocking_scores = []
            
            # Check horizontal block boundaries
            for i in range(block_size, h, block_size):
                if i < h - 1:
                    # Calculate difference across block boundary
                    row_above = image[i-1, :]
                    row_below = image[i, :]
                    boundary_diff = np.mean(np.abs(row_above.astype(float) - row_below.astype(float)))
                    
                    # Calculate difference within blocks
                    within_diff_above = np.mean(np.abs(np.diff(row_above.astype(float))))
                    within_diff_below = np.mean(np.abs(np.diff(row_below.astype(float))))
                    within_diff = (within_diff_above + within_diff_below) / 2
                    
                    # Blocking artifacts show high boundary differences relative to within-block differences
                    if within_diff > 0:
                        blocking_ratio = boundary_diff / within_diff
                        blocking_scores.append(min(blocking_ratio / 3.0, 1.0))  # Normalize
            
            # Check vertical block boundaries  
            for j in range(block_size, w, block_size):
                if j < w - 1:
                    col_left = image[:, j-1]
                    col_right = image[:, j]
                    boundary_diff = np.mean(np.abs(col_left.astype(float) - col_right.astype(float)))
                    
                    within_diff_left = np.mean(np.abs(np.diff(col_left.astype(float))))
                    within_diff_right = np.mean(np.abs(np.diff(col_right.astype(float))))
                    within_diff = (within_diff_left + within_diff_right) / 2
                    
                    if within_diff > 0:
                        blocking_ratio = boundary_diff / within_diff
                        blocking_scores.append(min(blocking_ratio / 3.0, 1.0))
            
            return np.mean(blocking_scores) if blocking_scores else 0.0
            
        except Exception as e:
            logger.error(f"Blocking artifact detection failed: {e}")
            return 0.0
    
    def _detect_blending_artifacts(self, face_region):
        """
        Detect blending artifacts using multi-scale edge analysis
        
        Blending artifacts appear as:
        - Inconsistent edge patterns
        - Unnatural smoothing
        - Sharp transitions at blend boundaries
        """
        try:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray_face, (3, 3), 0)
            
            # Multi-scale edge detection
            edge_maps = []
            
            # Canny edge detection with different thresholds
            for low_thresh in [30, 50, 70]:
                high_thresh = low_thresh * 2
                edges = cv2.Canny(blurred, low_thresh, high_thresh)
                edge_maps.append(edges)
            
            # Sobel edge detection
            sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            edge_maps.append((sobel_magnitude > np.percentile(sobel_magnitude, 70)).astype(np.uint8) * 255)
            
            # Laplacian edge detection
            laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
            edge_maps.append((np.abs(laplacian) > np.percentile(np.abs(laplacian), 70)).astype(np.uint8) * 255)
            
            # Analyze edge consistency across different methods
            inconsistency_scores = []
            
            for i in range(len(edge_maps)):
                for j in range(i+1, len(edge_maps)):
                    # Normalize edge maps to [0, 1]
                    edge_map1 = edge_maps[i].astype(np.float32) / 255.0
                    edge_map2 = edge_maps[j].astype(np.float32) / 255.0
                    
                    # Calculate correlation between edge maps
                    correlation = np.corrcoef(edge_map1.flatten(), edge_map2.flatten())[0, 1]
                    
                    if not np.isnan(correlation):
                        # Low correlation indicates inconsistent edge detection
                        inconsistency = 1.0 - abs(correlation)
                        inconsistency_scores.append(inconsistency)
            
            # Calculate mean inconsistency
            avg_inconsistency = np.mean(inconsistency_scores) if inconsistency_scores else 0.0
            
            # Additional analysis: Look for unnatural smoothing patterns
            smoothing_score = self._detect_unnatural_smoothing(gray_face)
            
            # Combine scores
            blending_score = (avg_inconsistency * 0.6 + smoothing_score * 0.4)
            
            return min(max(blending_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Blending artifact detection failed: {e}")
            return 0.0
    
    def _detect_unnatural_smoothing(self, gray_image):
        """Detect unnatural smoothing patterns"""
        try:
            # Calculate local variance using a sliding window
            kernel_size = 5
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
            
            # Mean and squared mean
            mean = cv2.filter2D(gray_image.astype(np.float32), -1, kernel)
            sqr_mean = cv2.filter2D((gray_image.astype(np.float32))**2, -1, kernel)
            
            # Local variance
            variance = sqr_mean - mean**2
            
            # Natural images have varying local variance
            # Over-smoothed regions have very low variance
            low_variance_ratio = np.sum(variance < 10) / variance.size
            
            # High ratio of low-variance regions indicates unnatural smoothing
            return min(low_variance_ratio * 2, 1.0)
            
        except Exception as e:
            logger.error(f"Unnatural smoothing detection failed: {e}")
            return 0.0
    
    def _analyze_lighting_consistency(self, face_region):
        """
        Analyze lighting consistency across face regions
        
        Inconsistent lighting appears as:
        - Unnatural shadow patterns
        - Inconsistent light direction
        - Abrupt illumination changes
        """
        try:
            # Convert to LAB color space for better lighting analysis
            lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0].astype(np.float32)  # Lightness channel
            
            h, w = l_channel.shape
            
            # Divide face into 3x3 grid for regional analysis
            grid_size = 3
            regions = []
            region_means = []
            region_stds = []
            
            for i in range(grid_size):
                for j in range(grid_size):
                    y_start = i * h // grid_size
                    y_end = (i + 1) * h // grid_size
                    x_start = j * w // grid_size
                    x_end = (j + 1) * w // grid_size
                    
                    region = l_channel[y_start:y_end, x_start:x_end]
                    regions.append(region)
                    region_means.append(np.mean(region))
                    region_stds.append(np.std(region))
            
            # Analyze lighting gradients
            lighting_variance = np.var(region_means)
            
            # Natural faces have smooth lighting transitions
            # Check for abrupt changes between adjacent regions
            gradient_inconsistencies = []
            
            # Check horizontal gradients
            for i in range(grid_size):
                for j in range(grid_size - 1):
                    idx1 = i * grid_size + j
                    idx2 = i * grid_size + j + 1
                    gradient = abs(region_means[idx1] - region_means[idx2])
                    gradient_inconsistencies.append(gradient)
            
            # Check vertical gradients
            for i in range(grid_size - 1):
                for j in range(grid_size):
                    idx1 = i * grid_size + j
                    idx2 = (i + 1) * grid_size + j
                    gradient = abs(region_means[idx1] - region_means[idx2])
                    gradient_inconsistencies.append(gradient)
            
            # Calculate gradient consistency
            if gradient_inconsistencies:
                avg_gradient = np.mean(gradient_inconsistencies)
                gradient_variance = np.var(gradient_inconsistencies)
                
                # High gradient variance indicates inconsistent lighting
                gradient_score = min(gradient_variance / 1000.0, 1.0)
            else:
                gradient_score = 0.0
            
            # Analyze overall lighting distribution
            lighting_score = min(lighting_variance / 2000.0, 1.0)
            
            # Check for unnatural shadow patterns
            shadow_score = self._analyze_shadow_patterns(l_channel)
            
            # Combine scores
            final_score = (lighting_score * 0.4 + gradient_score * 0.4 + shadow_score * 0.2)
            
            return min(max(final_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Lighting consistency analysis failed: {e}")
            return 0.0
    
    def _analyze_shadow_patterns(self, lightness_channel):
        """Analyze shadow patterns for naturalness"""
        try:
            # Find dark regions (potential shadows)
            shadow_threshold = np.percentile(lightness_channel, 20)
            shadow_mask = lightness_channel < shadow_threshold
            
            if np.sum(shadow_mask) == 0:
                return 0.0
            
            # Analyze shadow connectivity and shape
            # Natural shadows tend to be connected and have smooth boundaries
            
            # Find connected components in shadow regions
            shadow_mask_uint8 = shadow_mask.astype(np.uint8) * 255
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(shadow_mask_uint8)
            
            # Analyze shadow characteristics
            shadow_areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background
            
            if len(shadow_areas) == 0:
                return 0.0
            
            # Too many small disconnected shadows is suspicious
            small_shadows = np.sum(shadow_areas < np.max(shadow_areas) * 0.1)
            fragmentation_score = min(small_shadows / len(shadow_areas), 1.0)
            
            return fragmentation_score
            
        except Exception as e:
            logger.error(f"Shadow pattern analysis failed: {e}")
            return 0.0
    
    def _analyze_texture_naturalness(self, face_gray):
        """
        Analyze texture naturalness using Local Binary Patterns (LBP)
        
        Unnatural textures show:
        - Too uniform or too random patterns
        - Inconsistent texture entropy
        - Missing natural skin micro-patterns
        """
        try:
            # Calculate LBP with multiple parameters for robustness
            lbp_results = []
            
            # Different radius and point combinations
            lbp_configs = [
                (1, 8),   # Traditional LBP
                (2, 16),  # Extended LBP
                (3, 24),  # Large-scale LBP
            ]
            
            for radius, n_points in lbp_configs:
                try:
                    lbp = self._calculate_lbp(face_gray, radius=radius, n_points=n_points)
                    if lbp is not None:
                        # Calculate texture statistics
                        hist, _ = np.histogram(lbp.flatten(), bins=256, range=(0, 256))
                        hist = hist.astype(np.float32)
                        hist = hist / (hist.sum() + 1e-7)  # Normalize
                        
                        # Calculate entropy (texture complexity)
                        entropy = -np.sum(hist * np.log2(hist + 1e-7))
                        
                        # Calculate uniformity (texture regularity)
                        uniformity = np.sum(hist**2)
                        
                        lbp_results.append({
                            'radius': radius,
                            'n_points': n_points,
                            'entropy': entropy,
                            'uniformity': uniformity,
                            'histogram': hist
                        })
                        
                except Exception as e:
                    logger.debug(f"LBP calculation failed for r={radius}, p={n_points}: {e}")
                    continue
            
            if not lbp_results:
                logger.warning("All LBP calculations failed")
                return 0.0
            
            # Analyze texture characteristics
            entropies = [r['entropy'] for r in lbp_results]
            uniformities = [r['uniformity'] for r in lbp_results]
            
            avg_entropy = np.mean(entropies)
            avg_uniformity = np.mean(uniformities)
            entropy_variance = np.var(entropies)
            
            # Natural skin texture typically has:
            # - Moderate entropy (6-8 bits)
            # - Low to moderate uniformity
            # - Consistent entropy across scales
            
            # Entropy analysis
            optimal_entropy = 7.0
            entropy_deviation = abs(avg_entropy - optimal_entropy) / optimal_entropy
            
            # Uniformity analysis (too high uniformity is suspicious)
            uniformity_score = min(avg_uniformity * 5, 1.0)
            
            # Entropy variance analysis (too high variance indicates inconsistency)
            variance_score = min(entropy_variance / 2.0, 1.0)
            
            # Additional texture analysis using statistical measures
            texture_stats_score = self._analyze_texture_statistics(face_gray)
            
            # Combine scores
            unnaturalness_score = (
                entropy_deviation * 0.3 + 
                uniformity_score * 0.3 + 
                variance_score * 0.2 + 
                texture_stats_score * 0.2
            )
            
            return min(max(unnaturalness_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Texture naturalness analysis failed: {e}")
            return 0.0
    
    def _calculate_lbp(self, img, radius=1, n_points=8):
        """Calculate Local Binary Pattern with improved implementation"""
        try:
            h, w = img.shape
            
            # Ensure sufficient image size
            if h < 2 * radius + 1 or w < 2 * radius + 1:
                return None
            
            # Initialize LBP array
            lbp = np.zeros((h - 2 * radius, w - 2 * radius), dtype=np.uint8)
            
            # Pre-calculate sampling points for efficiency
            angles = 2 * np.pi * np.arange(n_points) / n_points
            sample_points = []
            for angle in angles:
                dy = -radius * np.sin(angle)
                dx = radius * np.cos(angle)
                sample_points.append((dy, dx))
            
            # Calculate LBP
            for i in range(radius, h - radius):
                for j in range(radius, w - radius):
                    center_pixel = img[i, j]
                    lbp_value = 0
                    
                    for k, (dy, dx) in enumerate(sample_points):
                        # Bilinear interpolation for sub-pixel sampling
                        y = i + dy
                        x = j + dx
                        
                        # Get integer parts
                        y1, x1 = int(y), int(x)
                        y2, x2 = min(y1 + 1, h - 1), min(x1 + 1, w - 1)
                        
                        # Get fractional parts
                        fy, fx = y - y1, x - x1
                        
                        # Bilinear interpolation
                        interpolated_value = (
                            img[y1, x1] * (1 - fx) * (1 - fy) +
                            img[y1, x2] * fx * (1 - fy) +
                            img[y2, x1] * (1 - fx) * fy +
                            img[y2, x2] * fx * fy
                        )
                        
                        # Compare with center pixel
                        if interpolated_value >= center_pixel:
                            lbp_value |= (1 << k)
                    
                    lbp[i - radius, j - radius] = lbp_value
            
            return lbp
            
        except Exception as e:
            logger.error(f"LBP calculation failed: {e}")
            return None
    
    def _analyze_texture_statistics(self, gray_image):
        """Analyze texture using statistical measures"""
        try:
            # Calculate Gray Level Co-occurrence Matrix (GLCM) features
            # This is a simplified version - in practice, use skimage.feature.greycomatrix
            
            # Calculate local variance
            kernel = np.ones((5, 5), np.float32) / 25
            local_mean = cv2.filter2D(gray_image.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((gray_image.astype(np.float32) - local_mean)**2, -1, kernel)
            
            # Calculate contrast using standard deviation
            contrast = np.std(local_variance)
            
            # Calculate homogeneity (inverse difference moment)
            diff_image = np.abs(cv2.filter2D(gray_image.astype(np.float32), -1, 
                                           np.array([[-1, 0, 1]]))) # Horizontal gradient
            homogeneity = 1.0 / (1.0 + np.mean(diff_image))
            
            # Natural skin has moderate contrast and homogeneity
            # Too low or too high values are suspicious
            
            # Normalize contrast (typical range 0-50)
            contrast_score = abs(contrast - 25) / 25
            
            # Homogeneity score (optimal around 0.7-0.9)
            homogeneity_score = abs(homogeneity - 0.8) / 0.8
            
            combined_score = (contrast_score + homogeneity_score) / 2
            return min(combined_score, 1.0)
            
        except Exception as e:
            logger.error(f"Texture statistics analysis failed: {e}")
            return 0.0
    
    def _analyze_geometric_consistency(self, face_gray):
        """
        Analyze geometric consistency of facial features
        
        Inconsistencies appear as:
        - Asymmetric eye positions
        - Unnatural facial proportions
        - Inconsistent feature scaling
        """
        try:
            inconsistency_score = 0.0
            
            # Eye detection for symmetry analysis
            if self.eye_cascade is not None:
                eyes = self.eye_cascade.detectMultiScale(
                    face_gray, 
                    scaleFactor=1.1, 
                    minNeighbors=3,
                    minSize=(10, 10)
                )
                
                if len(eyes) >= 2:
                    # Analyze eye positions for symmetry
                    eye_centers = []
                    eye_sizes = []
                    
                    for (ex, ey, ew, eh) in eyes:
                        center = (ex + ew//2, ey + eh//2)
                        eye_centers.append(center)
                        eye_sizes.append((ew, eh))
                    
                    # Calculate distances between eye centers
                    if len(eye_centers) >= 2:
                        distances = []
                        for i in range(len(eye_centers)):
                            for j in range(i+1, len(eye_centers)):
                                dist = np.sqrt(
                                    (eye_centers[i][0] - eye_centers[j][0])**2 + 
                                    (eye_centers[i][1] - eye_centers[j][1])**2
                                )
                                distances.append(dist)
                        
                        # Analyze distance variation
                        if distances:
                            distance_variance = np.var(distances)
                            inconsistency_score += min(distance_variance / 1000.0, 0.3)
                    
                    # Analyze eye size consistency
                    if len(eye_sizes) >= 2:
                        width_ratios = []
                        height_ratios = []
                        
                        for i in range(len(eye_sizes)):
                            for j in range(i+1, len(eye_sizes)):
                                w1, h1 = eye_sizes[i]
                                w2, h2 = eye_sizes[j]
                                
                                if w2 > 0 and h2 > 0:
                                    width_ratios.append(w1 / w2)
                                    height_ratios.append(h1 / h2)
                        
                        # Check for consistent ratios (should be close to 1.0)
                        if width_ratios:
                            width_inconsistency = np.mean([abs(r - 1.0) for r in width_ratios])
                            inconsistency_score += min(width_inconsistency, 0.2)
                        
                        if height_ratios:
                            height_inconsistency = np.mean([abs(r - 1.0) for r in height_ratios])
                            inconsistency_score += min(height_inconsistency, 0.2)
            
            # Overall facial symmetry analysis using pixel intensities
            h, w = face_gray.shape
            left_half = face_gray[:, :w//2]
            right_half = np.fliplr(face_gray[:, w//2:])
            
            # Ensure same dimensions
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
            
            # Calculate structural similarity
            diff = np.abs(left_half.astype(np.float32) - right_half.astype(np.float32))
            asymmetry_score = np.mean(diff) / 255.0
            
            inconsistency_score += asymmetry_score * 0.3
            
            # Additional geometric analysis using contour detection
            contour_score = self._analyze_facial_contours(face_gray)
            inconsistency_score += contour_score * 0.2
            
            return min(max(inconsistency_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Geometric consistency analysis failed: {e}")
            return 0.0
    
    def _analyze_facial_contours(self, face_gray):
        """Analyze facial contours for geometric consistency"""
        try:
            # Edge detection for contour analysis
            edges = cv2.Canny(face_gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 0.0
            
            # Analyze contour properties
            contour_areas = [cv2.contourArea(contour) for contour in contours]
            contour_perimeters = [cv2.arcLength(contour, True) for contour in contours]
            
            # Filter out very small contours
            significant_contours = [(area, perim) for area, perim in zip(contour_areas, contour_perimeters) if area > 50]
            
            if not significant_contours:
                return 0.0
            
            # Calculate circularity for each significant contour
            circularities = []
            for area, perimeter in significant_contours:
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    circularities.append(circularity)
            
            # Unnatural faces may have too regular or too irregular contours
            if circularities:
                circularity_variance = np.var(circularities)
                # High variance indicates inconsistent contour shapes
                return min(circularity_variance * 2, 1.0)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Facial contour analysis failed: {e}")
            return 0.0
    
    def _analyze_color_naturalness(self, face_region):
        """
        Analyze color distribution naturalness in multiple color spaces
        
        Unnatural colors appear as:
        - Incorrect skin tone ranges
        - Too uniform color distribution  
        - Unusual color relationships
        """
        try:
            # Convert to multiple color spaces for comprehensive analysis
            lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            
            # LAB color space analysis (best for skin tones)
            l_channel = lab[:, :, 0].astype(np.float32)  # Lightness
            a_channel = lab[:, :, 1].astype(np.float32)  # Green-Red axis
            b_channel = lab[:, :, 2].astype(np.float32)  # Blue-Yellow axis
            
            # Calculate color statistics
            a_mean, a_std = np.mean(a_channel), np.std(a_channel)
            b_mean, b_std = np.mean(b_channel), np.std(b_channel)
            l_mean, l_std = np.mean(l_channel), np.std(l_channel)
            
            # Natural skin tone ranges in LAB space (empirically determined)
            natural_ranges = {
                'a': (120, 145),  # Typical range for various skin tones
                'b': (115, 140),  # Typical range for skin yellow-blue
                'l': (30, 85)     # Lightness range for most faces
            }
            
            # Calculate deviation from natural ranges
            deviations = []
            
            # A-channel deviation
            if not (natural_ranges['a'][0] <= a_mean <= natural_ranges['a'][1]):
                a_dev = min(abs(a_mean - 132.5) / 25.0, 1.0)  # 132.5 is center of range
                deviations.append(a_dev)
            
            # B-channel deviation  
            if not (natural_ranges['b'][0] <= b_mean <= natural_ranges['b'][1]):
                b_dev = min(abs(b_mean - 127.5) / 25.0, 1.0)
                deviations.append(b_dev)
            
            # L-channel deviation
            if not (natural_ranges['l'][0] <= l_mean <= natural_ranges['l'][1]):
                l_dev = min(abs(l_mean - 57.5) / 35.0, 1.0)
                deviations.append(l_dev)
            
            # Color variance analysis
            # Natural faces have moderate color variance
            # Synthetic faces often have too uniform colors
            color_variance = (a_std + b_std + l_std) / 3.0
            
            # Too low variance indicates unnatural uniformity
            variance_score = 1.0 / (1.0 + color_variance / 10.0)
            
            # HSV analysis for additional validation
            hsv_score = self._analyze_hsv_naturalness(hsv)
            
            # Color relationship analysis
            relationship_score = self._analyze_color_relationships(lab)
            
            # Combine all scores
            color_unnaturalness = (
                np.mean(deviations) * 0.4 if deviations else 0.0 +
                variance_score * 0.3 +
                hsv_score * 0.2 +
                relationship_score * 0.1
            )
            
            return min(max(color_unnaturalness, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Color naturalness analysis failed: {e}")
            return 0.0
    
    def _analyze_hsv_naturalness(self, hsv_image):
        """Analyze HSV color space for naturalness"""
        try:
            h_channel = hsv_image[:, :, 0].astype(np.float32)
            s_channel = hsv_image[:, :, 1].astype(np.float32)
            v_channel = hsv_image[:, :, 2].astype(np.float32)
            
            # Skin tones typically have hue in certain ranges
            # Hue is in range 0-179 in OpenCV
            skin_hue_ranges = [(0, 25), (160, 179)]  # Red-orange range for skin
            
            # Check hue distribution
            hue_hist, _ = np.histogram(h_channel, bins=180, range=(0, 180))
            
            # Calculate percentage of pixels in skin hue ranges
            skin_hue_pixels = 0
            for h_min, h_max in skin_hue_ranges:
                skin_hue_pixels += np.sum(hue_hist[h_min:h_max+1])
            
            total_pixels = np.sum(hue_hist)
            skin_hue_ratio = skin_hue_pixels / total_pixels if total_pixels > 0 else 0
            
            # Lower ratio indicates less natural skin tones
            hue_unnaturalness = max(0, 1.0 - skin_hue_ratio * 2)
            
            # Saturation analysis - too high saturation is unnatural for skin
            s_mean = np.mean(s_channel)
            saturation_score = min(s_mean / 128.0, 1.0)  # Normalize to 0-1
            
            return (hue_unnaturalness * 0.6 + saturation_score * 0.4)
            
        except Exception as e:
            logger.error(f"HSV naturalness analysis failed: {e}")
            return 0.0
    
    def _analyze_color_relationships(self, lab_image):
        """Analyze relationships between color channels"""
        try:
            l_channel = lab_image[:, :, 0].astype(np.float32)
            a_channel = lab_image[:, :, 1].astype(np.float32)
            b_channel = lab_image[:, :, 2].astype(np.float32)
            
            # Calculate correlations between channels
            # Natural images have certain expected correlations
            correlation_la = np.corrcoef(l_channel.flatten(), a_channel.flatten())[0, 1]
            correlation_lb = np.corrcoef(l_channel.flatten(), b_channel.flatten())[0, 1]
            correlation_ab = np.corrcoef(a_channel.flatten(), b_channel.flatten())[0, 1]
            
            # Remove NaN correlations
            correlations = [c for c in [correlation_la, correlation_lb, correlation_ab] if not np.isnan(c)]
            
            if not correlations:
                return 0.0
            
            # Unusual correlation patterns can indicate synthetic generation
            # Natural faces typically have moderate correlations
            unusual_correlations = [abs(c) for c in correlations if abs(c) > 0.8]
            
            return min(len(unusual_correlations) / len(correlations), 1.0)
            
        except Exception as e:
            logger.error(f"Color relationship analysis failed: {e}")
            return 0.0
    
    def _analyze_edge_consistency(self, face_gray):
        """
        Analyze edge consistency and sharpness patterns
        
        Inconsistent edges appear as:
        - Varying sharpness across the face
        - Unnatural edge strength patterns
        - Inconsistent edge directions
        """
        try:
            # Multi-scale edge detection
            edge_detectors = []
            
            # Canny with different parameters
            for threshold in [(30, 90), (50, 150), (70, 210)]:
                edges = cv2.Canny(face_gray, threshold[0], threshold[1])
                edge_detectors.append(edges)
            
            # Sobel operators
            sobel_x = cv2.Sobel(face_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(face_gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Normalize and threshold
            sobel_norm = ((sobel_magnitude - np.min(sobel_magnitude)) / 
                         (np.max(sobel_magnitude) - np.min(sobel_magnitude)) * 255).astype(np.uint8)
            edge_detectors.append(sobel_norm > np.percentile(sobel_norm, 80))
            
            # Laplacian
            laplacian = cv2.Laplacian(face_gray, cv2.CV_64F)
            lap_norm = ((np.abs(laplacian) - np.min(np.abs(laplacian))) / 
                       (np.max(np.abs(laplacian)) - np.min(np.abs(laplacian))) * 255).astype(np.uint8)
            edge_detectors.append(lap_norm > np.percentile(lap_norm, 80))
            
            # Analyze edge strength distribution
            edge_strengths = []
            for detector in edge_detectors:
                if detector.dtype != np.uint8:
                    detector = (detector * 255).astype(np.uint8)
                edge_density = np.sum(detector > 0) / detector.size
                edge_strengths.append(edge_density)
            
            # Calculate edge consistency metrics
            edge_mean = np.mean(edge_strengths)
            edge_std = np.std(edge_strengths)
            
            # Analyze spatial edge distribution
            spatial_score = self._analyze_spatial_edge_distribution(face_gray, sobel_magnitude)
            
            # Edge direction consistency
            direction_score = self._analyze_edge_directions(sobel_x, sobel_y)
            
            # Combine metrics
            if edge_mean > 0:
                edge_variation = edge_std / edge_mean
                # Unnatural if edges are too uniform or too chaotic
                variation_score = abs(edge_variation - 0.5) / 0.5  # Optimal variation around 0.5
            else:
                variation_score = 1.0  # No edges is highly suspicious
            
            # Final edge consistency score
            consistency_score = (variation_score * 0.4 + spatial_score * 0.3 + direction_score * 0.3)
            
            return min(max(consistency_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Edge consistency analysis failed: {e}")
            return 0.0
    
    def _analyze_spatial_edge_distribution(self, gray_image, edge_magnitude):
        """Analyze spatial distribution of edges"""
        try:
            h, w = gray_image.shape
            
            # Divide image into regions
            regions = []
            for i in range(2):  # 2x2 grid
                for j in range(2):
                    y_start = i * h // 2
                    y_end = (i + 1) * h // 2
                    x_start = j * w // 2
                    x_end = (j + 1) * w // 2
                    
                    region_edges = edge_magnitude[y_start:y_end, x_start:x_end]
                    region_mean = np.mean(region_edges)
                    regions.append(region_mean)
            
            # Calculate variance across regions
            region_variance = np.var(regions)
            
            # Normalize (typical variance range 0-1000)
            spatial_inconsistency = min(region_variance / 1000.0, 1.0)
            
            return spatial_inconsistency
            
        except Exception as e:
            logger.error(f"Spatial edge distribution analysis failed: {e}")
            return 0.0
    
    def _analyze_edge_directions(self, sobel_x, sobel_y):
        """Analyze edge direction consistency"""
        try:
            # Calculate edge directions
            edge_directions = np.arctan2(sobel_y, sobel_x)
            
            # Flatten and remove zero magnitude areas (where sobel_x and sobel_y are both 0)
            magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            valid_mask = magnitude.flatten() > 0
            valid_directions = edge_directions.flatten()[valid_mask]
            
            if len(valid_directions) < 10:
                return 0.0  # Not enough edges
            
            # Calculate circular variance for directions (angles in radians)
            # Circular mean
            sin_mean = np.mean(np.sin(valid_directions))
            cos_mean = np.mean(np.cos(valid_directions))
            circular_mean = np.arctan2(sin_mean, cos_mean)
            
            # Circular variance (0 = consistent, 1 = random)
            r = np.sqrt(sin_mean**2 + cos_mean**2)
            circular_variance = 1 - r
            
            # High variance indicates inconsistent directions, suspicious for deepfakes
            return circular_variance
            
        except Exception as e:
            logger.error(f"Edge direction analysis failed: {e}")
            return 0.0
    
    def _frequency_domain_analysis(self, face_gray):
        """Perform frequency domain analysis for synthetic artifacts"""
        try:
            # Apply FFT
            f = np.fft.fft2(face_gray)
            fshift = np.fft.fftshift(f)
            
            # Calculate magnitude spectrum
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-7)
            
            # Analyze high-frequency content
            h, w = face_gray.shape
            center_y, center_x = h // 2, w // 2
            
            # Define high-frequency region (outer 30% radius)
            radius = min(center_x, center_y)
            high_freq_radius = int(radius * 0.7)
            
            # Create mask for high-frequency area
            y, x = np.ogrid[0:h, 0:w]
            dist_from_center = np.sqrt((y - center_y)**2 + (x - center_x)**2)
            high_freq_mask = dist_from_center > high_freq_radius
            
            # Calculate high-frequency energy
            high_freq_energy = np.mean(magnitude_spectrum[high_freq_mask])
            
            # Total energy
            total_energy = np.mean(magnitude_spectrum)
            
            # Ratio of high-frequency energy
            high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
            
            # Synthetic images often have reduced or unnatural high-frequency content
            # Natural images have balanced frequency distribution
            # Too low or too high is suspicious
            optimal_ratio = 0.25  # Empirical value for natural faces
            freq_deviation = abs(high_freq_ratio - optimal_ratio) / optimal_ratio
            
            # Additional analysis: Look for periodic patterns (common in generated images)
            periodic_score = self._detect_periodic_patterns(magnitude_spectrum)
            
            # Combine scores
            final_score = (freq_deviation * 0.7 + periodic_score * 0.3)
            
            return min(max(final_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Frequency domain analysis failed: {e}")
            return 0.0
    
    def _detect_periodic_patterns(self, magnitude_spectrum):
        """Detect periodic patterns in frequency spectrum"""
        try:
            # Threshold magnitude to find peaks
            threshold = np.percentile(magnitude_spectrum, 95)
            peaks = magnitude_spectrum > threshold
            
            # Find connected components in peaks
            num_labels, _ = cv2.connectedComponents(peaks.astype(np.uint8))
            
            # Too many disconnected peaks indicate periodic noise
            if num_labels > 20:  # Empirical threshold
                return min((num_labels - 20) / 30.0, 1.0)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Periodic pattern detection failed: {e}")
            return 0.0
    
    def _add_face_visualization(self, frame, bbox, analysis):
        """Add visualization overlays to frame"""
        try:
            x, y, w, h = bbox
            suspicion = analysis.get('overall_suspicion', 0.0)
            
            # Draw bounding box colored by suspicion level
            color = (0, int(255 * (1 - suspicion)), int(255 * suspicion))  # Green to Red
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Add suspicion score text
            text = f"Suspicion: {suspicion:.1%}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Highlight suspicious methods
            suspicious = analysis['analysis_details'].get('suspicious_methods', [])
            if suspicious:
                methods_text = ", ".join([m[:3].upper() for m in suspicious])
                cv2.putText(frame, methods_text, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
        except Exception as e:
            logger.error(f"Face visualization failed: {e}")
    
    def _create_detection_overlay(self, frame, results):
        """Create detection overlay image"""
        try:
            overlay = frame.copy()
            alpha = 0.4  # Transparency
            
            for analysis in results['analysis_results']:
                x, y, w, h = analysis['bbox']
                suspicion = analysis.get('overall_suspicion', 0.0)
                color = (0, int(255 * (1 - suspicion)), int(255 * suspicion))
                
                # Draw semi-transparent rectangle
                cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)
            
            # Blend with original frame
            blended = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            # Add overall text
            overall_text = f"Overall Suspicion: {results['overall_confidence']:.1%}"
            cv2.putText(blended, overall_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            return blended
            
        except Exception as e:
            logger.error(f"Detection overlay creation failed: {e}")
            return frame

class DatabaseManager:
    """
    Enhanced database management for analysis results and history
    
    Features:
    - SQLite database integration
    - Analysis result storage with JSON metadata
    - Privacy protection with data hashing
    - Automatic cleanup of old records
    - Query support for analytics
    - Known fake/real sample storage for improved detection
    """
    
    def __init__(self, db_path='deepfake_db.sqlite'):
        self.db_path = db_path
        logger.info(f"Initializing DatabaseManager with {db_path}")
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database schema with additional tables for known samples"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Analysis results table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS analysis_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_hash TEXT UNIQUE NOT NULL,
                        media_type TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        is_deepfake INTEGER NOT NULL,
                        processing_time REAL,
                        model_version TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        metadata JSON
                    )
                ''')
                
                # Alert history table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS alert_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        message TEXT NOT NULL,
                        confidence REAL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        metadata JSON
                    )
                ''')
                
                # System metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        cpu_usage REAL,
                        memory_usage REAL,
                        gpu_usage REAL,
                        temperature REAL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Known samples table for improved detection (new)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS known_samples (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_hash TEXT UNIQUE NOT NULL,
                        label TEXT NOT NULL,  -- 'real' or 'fake'
                        source TEXT,
                        added_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        metadata JSON
                    )
                ''')
                
                conn.commit()
            
            logger.info("Database initialized successfully")
        
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
    
    def store_analysis_result(self, result):
        """Store analysis result with enhanced metadata"""
        try:
            file_hash = hashlib.md5(result.get('file_path', str(uuid.uuid4())).encode()).hexdigest()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO analysis_results 
                    (file_hash, media_type, confidence, is_deepfake, processing_time, model_version, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    file_hash,
                    result.get('media_type'),
                    result.get('confidence'),
                    1 if result.get('is_deepfake') else 0,
                    result.get('processing_time'),
                    result.get('model_version'),
                    json.dumps(result.get('metadata', {}))
                ))
                
                conn.commit()
            
            logger.debug(f"Stored analysis result for hash {file_hash}")
            
        except Exception as e:
            logger.error(f"Failed to store analysis result: {e}")
    
    def store_known_sample(self, file_path, label, source=None, metadata=None):
        """Store known fake/real sample for future reference"""
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO known_samples 
                    (file_hash, label, source, metadata)
                    VALUES (?, ?, ?, ?)
                ''', (
                    file_hash,
                    label,
                    source,
                    json.dumps(metadata or {})
                ))
                
                conn.commit()
            
            logger.info(f"Stored known {label} sample: {file_hash}")
            
        except Exception as e:
            logger.error(f"Failed to store known sample: {e}")
    
    def check_known_sample(self, file_path):
        """Check if file is a known fake/real sample"""
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('SELECT label FROM known_samples WHERE file_hash = ?', (file_hash,))
                result = cursor.fetchone()
                
                return result[0] if result else None
            
        except Exception as e:
            logger.error(f"Known sample check failed: {e}")
            return None
    
    def store_alert(self, alert):
        """Store alert in history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO alert_history 
                    (alert_type, severity, message, confidence, metadata)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    alert.get('type'),
                    alert.get('severity'),
                    alert.get('message'),
                    alert.get('confidence'),
                    json.dumps(alert.get('metadata', {}))
                ))
                
                conn.commit()
            
            logger.debug(f"Stored alert: {alert.get('type')}")
            
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")
    
    def store_system_metrics(self, metrics):
        """Store system performance metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO system_metrics 
                    (cpu_usage, memory_usage, gpu_usage, temperature)
                    VALUES (?, ?, ?, ?)
                ''', (
                    metrics.get('cpu_usage'),
                    metrics.get('memory_usage'),
                    metrics.get('gpu_usage'),
                    metrics.get('temperature')
                ))
                
                conn.commit()
            
            logger.debug("Stored system metrics")
            
        except Exception as e:
            logger.error(f"Failed to store system metrics: {e}")
    
    def get_analysis_history(self, limit=100):
        """Get recent analysis history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM analysis_results 
                    ORDER BY timestamp DESC LIMIT ?
                ''', (limit,))
                
                return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"Failed to get analysis history: {e}")
            return []
    
    def get_alert_history(self, limit=50):
        """Get recent alert history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM alert_history 
                    ORDER BY timestamp DESC LIMIT ?
                ''', (limit,))
                
                return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"Failed to get alert history: {e}")
            return []
    
    def get_system_metrics_history(self, hours=24):
        """Get system metrics history"""
        try:
            start_time = datetime.now() - timedelta(hours=hours)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM system_metrics 
                    WHERE timestamp >= ?
                    ORDER BY timestamp ASC
                ''', (start_time.isoformat(),))
                
                return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"Failed to get system metrics history: {e}")
            return []
    
    def cleanup_old_data(self, days=30):
        """Cleanup old records"""
        try:
            old_time = datetime.now() - timedelta(days=days)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Cleanup analysis results
                cursor.execute('''
                    DELETE FROM analysis_results 
                    WHERE timestamp < ?
                ''', (old_time.isoformat(),))
                
                # Cleanup alerts
                cursor.execute('''
                    DELETE FROM alert_history 
                    WHERE timestamp < ?
                ''', (old_time.isoformat(),))
                
                # Cleanup metrics
                cursor.execute('''
                    DELETE FROM system_metrics 
                    WHERE timestamp < ?
                ''', (old_time.isoformat(),))
                
                conn.commit()
                
                deleted = cursor.rowcount
                logger.info(f"Cleaned up {deleted} old records")
                
                return deleted
            
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
            return 0

class RealTimeAnalyticsEngine:
    """
    Enhanced real-time analytics and reporting system
    
    Features:
    - Real-time metric updates from actual data
    - Comprehensive statisticscalculation
    - Alert history management
    - System performance monitoring
    - Report generation with visualizations
    - Integration with known samples for accuracy tracking
    """
    
    def __init__(self, database=None):
        logger.info("Initializing RealTimeAnalyticsEngine")
        
        self.database = database
        self.analytics_state = {
            'detection_statistics': {
                'total_analyses': 0,
                'detection_count': 0,
                'average_confidence': 0.0,
                'media_types': {},
                'detection_trend': deque(maxlen=1000)
            },
            'alert_history': [],
            'system_metrics': {
                'current': {},
                'history': deque(maxlen=1000)
            },
            'accuracy_metrics': {  # New for known samples
                'true_positives': 0,
                'true_negatives': 0,
                'false_positives': 0,
                'false_negatives': 0
            }
        }
        
        # Monitoring interval in seconds
        self.monitoring_interval = 60
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._system_monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info("RealTimeAnalyticsEngine initialized successfully")
    
    def update_metrics(self, analysis_result):
        """Update analytics with new analysis result"""
        try:
            self.analytics_state['detection_statistics']['total_analyses'] += 1
            
            if analysis_result.get('is_deepfake', False):
                self.analytics_state['detection_statistics']['detection_count'] += 1
            
            # Update average confidence
            current_avg = self.analytics_state['detection_statistics']['average_confidence']
            n = self.analytics_state['detection_statistics']['total_analyses']
            new_conf = analysis_result.get('confidence', 0.0)
            self.analytics_state['detection_statistics']['average_confidence'] = (
                current_avg * (n - 1) + new_conf
            ) / n
            
            # Update media types
            media_type = analysis_result.get('media_type', 'unknown')
            self.analytics_state['detection_statistics']['media_types'][media_type] = (
                self.analytics_state['detection_statistics']['media_types'].get(media_type, 0) + 1
            )
            
            # Add to trend
            self.analytics_state['detection_statistics']['detection_trend'].append({
                'timestamp': datetime.now(),
                'confidence': new_conf,
                'is_deepfake': analysis_result.get('is_deepfake', False)
            })
            
            # Check against known samples for accuracy
            file_path = analysis_result.get('file_path')
            if file_path:
                known_label = self.database.check_known_sample(file_path)
                if known_label:
                    predicted = 'fake' if analysis_result.get('is_deepfake') else 'real'
                    if predicted == known_label:
                        if known_label == 'fake':
                            self.analytics_state['accuracy_metrics']['true_positives'] += 1
                        else:
                            self.analytics_state['accuracy_metrics']['true_negatives'] += 1
                    else:
                        if predicted == 'fake':
                            self.analytics_state['accuracy_metrics']['false_positives'] += 1
                        else:
                            self.analytics_state['accuracy_metrics']['false_negatives'] += 1
            
            logger.debug("Analytics metrics updated")
            
        except Exception as e:
            logger.error(f"Metrics update failed: {e}")
    
    def _system_monitoring_loop(self):
        """Continuous system performance monitoring"""
        while True:
            try:
                metrics = self._collect_system_metrics()
                self.analytics_state['system_metrics']['current'] = metrics
                self.analytics_state['system_metrics']['history'].append({
                    'timestamp': datetime.now(),
                    'metrics': metrics
                })
                
                if self.database:
                    self.database.store_system_metrics(metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"System monitoring failed: {e}")
                time.sleep(10)  # Retry delay
    
    def _collect_system_metrics(self):
        """Collect current system metrics"""
        metrics = {
            'cpu_usage': psutil.cpu_percent() if PSUTIL_AVAILABLE else None,
            'memory_usage': psutil.virtual_memory().percent if PSUTIL_AVAILABLE else None,
            'gpu_usage': None,
            'temperature': None
        }
        
        # GPU metrics if available
        if torch.cuda.is_available():
            try:
                gpu = torch.cuda
                metrics['gpu_usage'] = gpu.utilization.gpu[0] if hasattr(gpu.utilization, 'gpu') else None
            except:
                pass
        
        # Temperature (requires additional libraries like py-cpuinfo, but skip for now)
        
        return metrics
    
    def generate_analytics_report(self):
        """Generate comprehensive analytics report from real data"""
        try:
            report = {
                'detection_statistics': self.analytics_state['detection_statistics'].copy(),
                'system_metrics': self.analytics_state['system_metrics']['current'],
                'accuracy_metrics': self.analytics_state['accuracy_metrics'].copy(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Load from database for persistent data
            if self.database:
                report['detection_statistics']['total_analyses'] = len(self.database.get_analysis_history())
                report['alert_history'] = self.database.get_alert_history()
                report['system_metrics_history'] = self.database.get_system_metrics_history()
                
                # Recalculate averages from DB
                history = self.database.get_analysis_history()
                if history:
                    confidences = [h['confidence'] for h in history]
                    report['detection_statistics']['average_confidence'] = np.mean(confidences)
                    report['detection_statistics']['detection_count'] = sum(1 for h in history if h['is_deepfake'])
                
                # Accuracy from known samples
                report['accuracy'] = self._calculate_accuracy(report['accuracy_metrics'])
            
            # Add trends
            report['detection_trend'] = list(self.analytics_state['detection_statistics']['detection_trend'])
            
            logger.info("Generated analytics report")
            return report
            
        except Exception as e:
            logger.error(f"Analytics report generation failed: {e}")
            return {}
    
    def _calculate_accuracy(self, metrics):
        """Calculate accuracy metrics"""
        total = sum(metrics.values())
        if total == 0:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
        
        tp = metrics['true_positives']
        tn = metrics['true_negatives']
        fp = metrics['false_positives']
        fn = metrics['false_negatives']
        
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

class SecureBrowserManager:
    """
    Enhanced secure browsing manager with fallback support
    
    Features:
    - Isolated browser sessions
    - Content analysis for deepfake risks
    - Privacy protection
    - Fallback to requests when Selenium unavailable
    - Session logging
    """
    
    def __init__(self, database=None):
        logger.info("Initializing SecureBrowserManager")
        
        self.database = database
        self.active_sessions = {}
        self.session_timeout = timedelta(minutes=15)
        
        logger.info("SecureBrowserManager initialized")
    
    def create_secure_session(self, url):
        """Create secure browsing session with fallback"""
        try:
            session_id = str(uuid.uuid4())
            start_time = datetime.now()
            
            session_info = {
                'session_id': session_id,
                'url': url,
                'start_time': start_time,
                'status': 'active',
                'content': None,
                'analysis': None,
                'risk_level': 'low',
                'driver': None
            }
            
            if SELENIUM_AVAILABLE:
                # Use Selenium for full browser automation
                options = Options()
                options.add_argument("--headless")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--disable-gpu")
                
                driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
                driver.get(url)
                
                # Wait for content
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                session_info['content'] = driver.page_source
                session_info['driver'] = driver
                
            else:
                # Fallback to requests (fixed: no raise, use HTTP fetch)
                logger.warning("Selenium not available, using requests fallback")
                response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
                response.raise_for_status()
                session_info['content'] = response.text
            
            # Analyze content
            page_analysis = self._analyze_page_content(session_info['content'])
            session_info['analysis'] = page_analysis
            session_info['risk_level'] = self._determine_risk_level(page_analysis)
            
            self.active_sessions[session_id] = session_info
            
            logger.info(f"Secure session created: {session_id} for {url}")
            
            return {
                'success': True,
                'session_id': session_id,
                'risk_level': session_info['risk_level'],
                'page_analysis': page_analysis
            }
            
        except Exception as e:
            logger.error(f"Secure session creation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _analyze_page_content(self, content):
        """Analyze page content for deepfake risks"""
        try:
            analysis = {
                'has_video': '<video' in content.lower(),
                'has_images': '<img' in content.lower(),
                'suspicious_keywords': 0,
                'links_count': content.lower().count('<a '),
                'scripts_count': content.lower().count('<script')
            }
            
            keywords = ['deepfake', 'fake', 'manipulated', 'ai generated', 'synthetic']
            for kw in keywords:
                analysis['suspicious_keywords'] += content.lower().count(kw)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Page content analysis failed: {e}")
            return {}
    
    def _determine_risk_level(self, analysis):
        """Determine risk level based on analysis"""
        if analysis.get('suspicious_keywords', 0) > 5 or analysis.get('scripts_count', 0) > 20:
            return 'high'
        elif analysis.get('has_video') or analysis.get('has_images'):
            return 'medium'
        return 'low'
    
    def close_session(self, session_id):
        """Close secure session"""
        try:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                
                if session.get('driver'):
                    session['driver'].quit()
                
                del self.active_sessions[session_id]
                
                duration = (datetime.now() - session['start_time']).total_seconds()
                
                logger.info(f"Session {session_id} closed after {duration:.1f}s")
                
                return {'success': True, 'status': 'closed', 'duration': duration}
            
            return {'success': False, 'status': 'not_found'}
            
        except Exception as e:
            logger.error(f"Session close failed: {e}")
            return {'success': False, 'error': str(e)}

class StreamMonitoringEngine:
    """
    Enhanced stream monitoring engine for real-time deepfake detection
    
    Features:
    - Multi-stream support
    - Real-time analysis and alerts for detected deepfakes
    - Performance optimization for continuous processing
    - Recording capabilities for suspicious content
    """
    
    def __init__(self, detector_model=None, face_analyzer=None, analytics_engine=None):
        logger.info("Initializing StreamMonitoringEngine")
        
        self.detector_model = detector_model
        self.face_analyzer = face_analyzer
        self.analytics_engine = analytics_engine
        
        # Stream sources
        self.active_streams = {}
        self.stream_configs = {
            'default': {
                'analysis_interval': 1.0,  # Analyze every 1 second
                'detection_threshold': 0.7,  # Alert threshold
                'max_fps': 15,  # Limit FPS for performance
                'buffer_size': 30,  # Frame buffer size
                'record_on_detection': True,
                'auto_restart': True
            }
        }
        
        # Monitoring state
        self.monitoring_active = False
        self.detection_history = deque(maxlen=1000)
        self.alert_callbacks = []
        
        # Recording settings
        self.recording_config = {
            'format': 'mp4',
            'quality': 'medium',
            'duration_seconds': 30,  # Record 30 seconds on detection
            'pre_buffer_seconds': 5   # Include 5 seconds before detection
        }
        
        logger.info("StreamMonitoringEngine initialized successfully")
    
    def add_stream_source(self, source_id, source_config):
        """
        Add a new stream source for monitoring
        
        Args:
            source_id: Unique identifier for the stream
            source_config: Configuration dictionary for the stream
        """
        try:
            logger.info(f"Adding stream source: {source_id}")
            
            # Validate source configuration
            required_fields = ['source_type', 'source_path']
            for field in required_fields:
                if field not in source_config:
                    raise ValueError(f"Missing required field: {field}")
            
            # Merge with default config
            config = self.stream_configs['default'].copy()
            config.update(source_config)
            
            # Initialize stream
            stream_info = {
                'source_id': source_id,
                'config': config,
                'status': 'initialized',
                'start_time': None,
                'frame_count': 0,
                'detection_count': 0,
                'last_analysis': None,
                'frame_buffer': deque(maxlen=config['buffer_size']),
                'recording_buffer': deque(maxlen=config['max_fps'] * self.recording_config['pre_buffer_seconds']),
                'active_recording': None,
                'capture': None,
                'thread': None
            }
            
            # Test stream connectivity
            if not self._test_stream_connectivity(config):
                raise Exception("Stream connectivity test failed")
            
            self.active_streams[source_id] = stream_info
            
            logger.info(f"Stream source {source_id} added successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add stream source {source_id}: {e}")
            return False
    
    def _test_stream_connectivity(self, config):
        """Test if stream source is accessible"""
        try:
            source_type = config['source_type']
            source_path = config['source_path']
            
            if source_type == 'webcam':
                # Test webcam access
                cap = cv2.VideoCapture(int(source_path))
                if not cap.isOpened():
                    return False
                
                ret, frame = cap.read()
                cap.release()
                return ret and frame is not None
                
            elif source_type == 'ip_camera':
                # Test IP camera access
                cap = cv2.VideoCapture(source_path)
                if not cap.isOpened():
                    return False
                
                ret, frame = cap.read()
                cap.release()
                return ret and frame is not None
                
            elif source_type == 'file':
                # Test file access
                return os.path.exists(source_path)
                
            elif source_type == 'rtsp':
                # Test RTSP stream
                cap = cv2.VideoCapture(source_path)
                success = cap.isOpened()
                cap.release()
                return success
                
            else:
                logger.warning(f"Unknown source type: {source_type}")
                return False
                
        except Exception as e:
            logger.error(f"Stream connectivity test failed: {e}")
            return False
    
    def start_monitoring(self, source_id=None):
        """
        Start monitoring streams
        
        Args:
            source_id: Specific stream to start (None for all streams)
        """
        try:
            if source_id:
                # Start specific stream
                if source_id not in self.active_streams:
                    raise ValueError(f"Stream {source_id} not found")
                
                return self._start_stream_monitoring(source_id)
            else:
                # Start all streams
                success_count = 0
                for stream_id in self.active_streams:
                    if self._start_stream_monitoring(stream_id):
                        success_count += 1
                
                self.monitoring_active = success_count > 0
                
                logger.info(f"Started monitoring {success_count}/{len(self.active_streams)} streams")
                return success_count > 0
                
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            return False
    
    def _start_stream_monitoring(self, source_id):
        """Start monitoring for a specific stream"""
        try:
            stream_info = self.active_streams[source_id]
            
            if stream_info['status'] == 'running':
                logger.warning(f"Stream {source_id} is already running")
                return True
            
            # Initialize video capture
            config = stream_info['config']
            source_type = config['source_type']
            source_path = config['source_path']
            
            if source_type == 'webcam':
                cap = cv2.VideoCapture(int(source_path))
            else:
                cap = cv2.VideoCapture(source_path)
            
            if not cap.isOpened():
                raise Exception(f"Failed to open stream: {source_path}")
            
            # Configure capture settings
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.get('width', 640))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.get('height', 480))
            cap.set(cv2.CAP_PROP_FPS, config.get('max_fps', 15))
            
            stream_info['capture'] = cap
            stream_info['start_time'] = datetime.now()
            stream_info['status'] = 'running'
            
            # Start monitoring thread
            monitoring_thread = threading.Thread(
                target=self._stream_monitoring_loop,
                args=(source_id,),
                daemon=True
            )
            monitoring_thread.start()
            stream_info['thread'] = monitoring_thread
            
            logger.info(f"Started monitoring stream: {source_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start stream monitoring for {source_id}: {e}")
            if source_id in self.active_streams:
                self.active_streams[source_id]['status'] = 'error'
            return False
    
    def _stream_monitoring_loop(self, source_id):
        """Main monitoring loop for a stream"""
        try:
            stream_info = self.active_streams[source_id]
            config = stream_info['config']
            
            last_analysis_time = time.time()
            
            logger.info(f"Starting monitoring loop for stream: {source_id}")
            
            while stream_info['status'] == 'running':
                try:
                    # Read frame
                    ret, frame = stream_info['capture'].read()
                    
                    if not ret or frame is None:
                        if config.get('auto_restart', True):
                            logger.warning(f"Stream {source_id} disconnected, attempting restart...")
                            self._restart_stream(source_id)
                            continue
                        else:
                            break
                    
                    stream_info['frame_count'] += 1
                    current_time = time.time()
                    
                    # Add to frame buffer
                    frame_entry = {
                        'frame': frame.copy(),
                        'timestamp': datetime.now(),
                        'frame_number': stream_info['frame_count']
                    }
                    stream_info['frame_buffer'].append(frame_entry)
                    stream_info['recording_buffer'].append(frame_entry)
                    
                    # Check if it's time for analysis
                    if current_time - last_analysis_time >= config['analysis_interval']:
                        analysis_result = self._analyze_frame(source_id, frame)
                        
                        if analysis_result:
                            stream_info['last_analysis'] = analysis_result
                            
                            # Check for detection
                            if analysis_result.get('is_deepfake', False) and analysis_result.get('confidence', 0) > config['detection_threshold']:
                                self._handle_detection(source_id, analysis_result, frame_entry)
                        
                        last_analysis_time = current_time
                    
                    # Small delay to prevent excessive CPU usage
                    time.sleep(1.0 / config['max_fps'])
                    
                except Exception as frame_error:
                    logger.error(f"Frame processing error for stream {source_id}: {frame_error}")
                    continue
            
            # Cleanup
            self._cleanup_stream(source_id)
            
        except Exception as e:
            logger.error(f"Stream monitoring loop failed for {source_id}: {e}")
            if source_id in self.active_streams:
                self.active_streams[source_id]['status'] = 'error'
    
    def _analyze_frame(self, source_id, frame):
        """Analyze a single frame for deepfake detection"""
        try:
            analysis_start = time.time()
            
            # Comprehensive analysis combining neural network and face analysis
            result = {
                'source_id': source_id,
                'timestamp': datetime.now(),
                'analysis_type': 'stream_monitoring',
                'confidence': 0.0,
                'is_deepfake': False,
                'processing_time': 0.0
            }
            
            # Face analysis
            if self.face_analyzer:
                face_analysis = self.face_analyzer.comprehensive_face_analysis(frame)
                result['face_analysis'] = face_analysis
                
                if face_analysis.get('faces_detected', 0) > 0:
                    face_conf = face_analysis.get('overall_confidence', 0.0)
                    result['confidence'] = max(result['confidence'], face_conf)
            
            # Neural network analysis (if available and face detected)
            if self.detector_model and result.get('face_analysis', {}).get('faces_detected', 0) > 0:
                try:
                    # Prepare frame for neural network
                    processed_frame = self._prepare_frame_for_nn(frame)
                    
                    if processed_frame is not None:
                        with torch.no_grad():
                            device = next(self.detector_model.parameters()).device
                            frame_tensor = processed_frame.to(device)
                            
                            # Add batch and sequence dimensions
                            frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, C, H, W]
                            
                            output, attention = self.detector_model(frame_tensor)
                            
                            # Get prediction
                            probabilities = F.softmax(output, dim=1)
                            confidence = float(probabilities[0, 1])  # Deepfake probability
                            prediction = int(torch.argmax(output, dim=1)[0])
                            
                            result['neural_network_result'] = {
                                'confidence': confidence,
                                'prediction': prediction,
                                'model_name': 'AdvancedDeepFakeDetector'
                            }
                            
                            # Update overall confidence
                            result['confidence'] = max(result['confidence'], confidence)
                
                except Exception as nn_error:
                    logger.warning(f"Neural network analysis failed for stream {source_id}: {nn_error}")
            
            # Determine final result
            result['is_deepfake'] = result['confidence'] > 0.5
            result['processing_time'] = time.time() - analysis_start
            
            # Update analytics
            if self.analytics_engine:
                self.analytics_engine.update_metrics(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Frame analysis failed for stream {source_id}: {e}")
            return None
    
    def _prepare_frame_for_nn(self, frame):
        """Prepare frame for neural network processing"""
        try:
            # Resize to expected input size
            target_size = (224, 224)
            resized_frame = cv2.resize(frame, target_size)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor and normalize
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            frame_tensor = transform(rgb_frame)
            return frame_tensor
            
        except Exception as e:
            logger.error(f"Frame preparation for NN failed: {e}")
            return None
    
    def _handle_detection(self, source_id, analysis_result, frame_entry):
        """Handle deepfake detection in stream"""
        try:
            stream_info = self.active_streams[source_id]
            stream_info['detection_count'] += 1
            
            detection_info = {
                'source_id': source_id,
                'detection_time': datetime.now(),
                'confidence': analysis_result.get('confidence', 0.0),
                'frame_number': frame_entry['frame_number'],
                'analysis_result': analysis_result
            }
            
            self.detection_history.append(detection_info)
            
            logger.warning(f"DEEPFAKE DETECTED in stream {source_id}! "
                          f"Confidence: {analysis_result.get('confidence', 0.0):.1%}")
            
            # Start recording if enabled
            if stream_info['config'].get('record_on_detection', True):
                self._start_detection_recording(source_id, detection_info)
            
            # Trigger alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(detection_info)
                except Exception as callback_error:
                    logger.error(f"Alert callback failed: {callback_error}")
            
            # Generate system alert
            alert = {
                'type': 'LIVE_STREAM_DETECTION',
                'severity': 'HIGH',
                'message': f"Live deepfake detected in stream {source_id}",
                'confidence': analysis_result.get('confidence', 0.0),
                'timestamp': datetime.now(),
                'metadata': {
                    'source_id': source_id,
                    'frame_number': frame_entry['frame_number'],
                    'faces_detected': analysis_result.get('face_analysis', {}).get('faces_detected', 0)
                }
            }
            
            if self.analytics_engine:
                self.analytics_engine.analytics_state['alert_history'].append(alert)
            
        except Exception as e:
            logger.error(f"Detection handling failed for stream {source_id}: {e}")
    
    def _start_detection_recording(self, source_id, detection_info):
        """Start recording when detection occurs"""
        try:
            stream_info = self.active_streams[source_id]
            
            # Check if already recording
            if stream_info.get('active_recording'):
                logger.debug(f"Stream {source_id} already recording")
                return
            
            # Create recording configuration
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"recordings/detection_{source_id}_{timestamp}.{self.recording_config['format']}"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Initialize video writer
            config = stream_info['config']
            frame_size = (config.get('width', 640), config.get('height', 480))
            fps = config.get('max_fps', 15)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
            
            recording_info = {
                'output_path': output_path,
                'video_writer': video_writer,
                'start_time': datetime.now(),
                'detection_info': detection_info,
                'frames_written': 0,
                'target_duration': self.recording_config['duration_seconds']
            }
            
            # Write pre-buffer frames
            for buffered_frame in list(stream_info['recording_buffer']):
                video_writer.write(buffered_frame['frame'])
                recording_info['frames_written'] += 1
            
            stream_info['active_recording'] = recording_info
            
            # Start recording thread
            recording_thread = threading.Thread(
                target=self._recording_loop,
                args=(source_id,),
                daemon=True
            )
            recording_thread.start()
            
            logger.info(f"Started recording for stream {source_id}: {output_path}")
            
        except Exception as e:
            logger.error(f"Detection recording start failed for stream {source_id}: {e}")
    
    def _recording_loop(self, source_id):
        """Recording loop for detected content"""
        try:
            stream_info = self.active_streams[source_id]
            recording_info = stream_info['active_recording']
            
            start_time = time.time()
            target_duration = recording_info['target_duration']
            
            while (time.time() - start_time) < target_duration and stream_info['status'] == 'running':
                try:
                    # Get latest frame from buffer
                    if stream_info['frame_buffer']:
                        latest_frame = stream_info['frame_buffer'][-1]
                        recording_info['video_writer'].write(latest_frame['frame'])
                        recording_info['frames_written'] += 1
                    
                    time.sleep(1.0 / stream_info['config']['max_fps'])
                    
                except Exception as frame_error:
                    logger.error(f"Recording frame error for stream {source_id}: {frame_error}")
                    continue
            
            # Finalize recording
            recording_info['video_writer'].release()
            recording_info['end_time'] = datetime.now()
            
            duration = (recording_info['end_time'] - recording_info['start_time']).total_seconds()
            
            logger.info(f"Recording completed for stream {source_id}: "
                       f"{recording_info['output_path']} ({duration:.1f}s, {recording_info['frames_written']} frames)")
            
            # Clear active recording
            stream_info['active_recording'] = None
            
        except Exception as e:
            logger.error(f"Recording loop failed for stream {source_id}: {e}")
            if source_id in self.active_streams and self.active_streams[source_id].get('active_recording'):
                try:
                    self.active_streams[source_id]['active_recording']['video_writer'].release()
                except:
                    pass
                self.active_streams[source_id]['active_recording'] = None
    
    def _restart_stream(self, source_id):
        """Restart a disconnected stream"""
        try:
            stream_info = self.active_streams[source_id]
            
            # Close current capture
            if stream_info.get('capture'):
                stream_info['capture'].release()
            
            # Wait before restart
            time.sleep(2)
            
            # Reinitialize capture
            config = stream_info['config']
            source_type = config['source_type']
            source_path = config['source_path']
            
            if source_type == 'webcam':
                cap = cv2.VideoCapture(int(source_path))
            else:
                cap = cv2.VideoCapture(source_path)
            
            if cap.isOpened():
                # Configure capture settings
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.get('width', 640))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.get('height', 480))
                cap.set(cv2.CAP_PROP_FPS, config.get('max_fps', 15))
                
                stream_info['capture'] = cap
                logger.info(f"Stream {source_id} restarted successfully")
                return True
            else:
                logger.error(f"Failed to restart stream {source_id}")
                stream_info['status'] = 'error'
                return False
                
        except Exception as e:
            logger.error(f"Stream restart failed for {source_id}: {e}")
            return False
    
    def stop_monitoring(self, source_id=None):
        """Stop monitoring streams"""
        try:
            if source_id:
                # Stop specific stream
                if source_id in self.active_streams:
                    self.active_streams[source_id]['status'] = 'stopped'
                    self._cleanup_stream(source_id)
                    return True
                return False
            else:
                # Stop all streams
                for stream_id in list(self.active_streams.keys()):
                    self.active_streams[stream_id]['status'] = 'stopped'
                    self._cleanup_stream(stream_id)
                
                self.monitoring_active = False
                logger.info("All stream monitoring stopped")
                return True
                
        except Exception as e:
            logger.error(f"Failed to stop monitoring: {e}")
            return False
    
    def _cleanup_stream(self, source_id):
        """Cleanup stream resources"""
        try:
            if source_id not in self.active_streams:
                return
            
            stream_info = self.active_streams[source_id]
            
            # Release video capture
            if stream_info.get('capture'):
                stream_info['capture'].release()
            
            # Finalize any active recording
            if stream_info.get('active_recording'):
                try:
                    stream_info['active_recording']['video_writer'].release()
                except:
                    pass
            
            # Wait for thread to finish
            if stream_info.get('thread') and stream_info['thread'].is_alive():
                stream_info['thread'].join(timeout=2)
            
            logger.debug(f"Stream {source_id} cleanup completed")
            
        except Exception as e:
            logger.error(f"Stream cleanup failed for {source_id}: {e}")
    
    def add_alert_callback(self, callback):
        """Add callback function for detection alerts"""
        self.alert_callbacks.append(callback)
    
    def get_monitoring_status(self):
        """Get comprehensive monitoring status"""
        try:
            status = {
                'monitoring_active': self.monitoring_active,
                'active_streams': len(self.active_streams),
                'total_detections': len(self.detection_history),
                'streams': {}
            }
            
            for source_id, stream_info in self.active_streams.items():
                stream_status = {
                    'status': stream_info.get('status', 'unknown'),
                    'frame_count': stream_info.get('frame_count', 0),
                    'detection_count': stream_info.get('detection_count', 0),
                    'start_time': stream_info.get('start_time', '').isoformat() if stream_info.get('start_time') else '',
                    'last_analysis': stream_info.get('last_analysis', {}).get('timestamp', '').isoformat() if stream_info.get('last_analysis', {}).get('timestamp') else '',
                    'recording_active': stream_info.get('active_recording') is not None
                }
                status['streams'][source_id] = stream_status
            
            return status
            
        except Exception as e:
            logger.error(f"Monitoring status retrieval failed: {e}")
            return {'error': str(e)}

class AuthenticationManager:
    """
    Multi-factor authentication and access control system
    
    Features:
    - Username/password authentication
    - TOTP (Time-based One-Time Password) support
    - Session management with secure tokens
    - Role-based access control
    - Login attempt monitoring and lockout
    - Secure password hashing with salt
    """
    
    def __init__(self, database=None):
        logger.info("Initializing AuthenticationManager")
        
        self.database = database
        self.active_sessions = {}
        self.login_attempts = {}
        
        # Security configuration
        self.security_config = {
            'max_login_attempts': 5,
            'lockout_duration_minutes': 15,
            'session_timeout_hours': 24,
            'password_min_length': 8,
            'require_special_chars': True,
            'token_length': 32
        }
        
        # Initialize user database
        self._initialize_auth_database()
        
        logger.info("AuthenticationManager initialized successfully")
    
    def _initialize_auth_database(self):
        """Initialize authentication database tables"""
        try:
            if not self.database:
                logger.warning("No database provided for authentication")
                return
            
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.cursor()
                
                # Users table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        salt TEXT NOT NULL,
                        totp_secret TEXT,
                        role TEXT DEFAULT 'user',
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        last_login DATETIME,
                        is_active INTEGER DEFAULT 1,
                        failed_attempts INTEGER DEFAULT 0,
                        locked_until DATETIME
                    )
                ''')
                
                # Sessions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        session_token TEXT UNIQUE NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        expires_at DATETIME NOT NULL,
                        last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
                        ip_address TEXT,
                        user_agent TEXT,
                        is_active INTEGER DEFAULT 1,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                ''')
                
                # Login attempts table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS login_attempts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT,
                        ip_address TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        success INTEGER,
                        failure_reason TEXT
                    )
                ''')
                
                conn.commit()
                
                # Create default admin user if none exists
                cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
                admin_count = cursor.fetchone()[0]
                
                if admin_count == 0:
                    self._create_default_admin()
                
            logger.debug("Authentication database initialized")
            
        except Exception as e:
            logger.error(f"Authentication database initialization failed: {e}")
    
    def _create_default_admin(self):
        """Create default admin user"""
        try:
            default_username = "admin"
            default_password = "DeepFakeAdmin2024!"  # Should be changed immediately
            
            success = self.create_user(
                username=default_username,
                password=default_password,
                role='admin'
            )
            
            if success:
                logger.warning("Default admin user created with username 'admin'. "
                             "Please change the password immediately!")
            
        except Exception as e:
            logger.error(f"Default admin creation failed: {e}")
    
    def create_user(self, username, password, role='user', totp_secret=None):
        """
        Create a new user account
        
        Args:
            username: Unique username
            password: Plain text password (will be hashed)
            role: User role (user, admin, analyst)
            totp_secret: TOTP secret for 2FA (optional)
            
        Returns:
            Boolean indicating success
        """
        try:
            if not self.database:
                return False
            
            # Validate password strength
            if not self._validate_password_strength(password):
                logger.error("Password does not meet security requirements")
                return False
            
            # Generate salt and hash password
            salt = secrets.token_hex(16)
            password_hash = self._hash_password(password, salt)
            
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if username already exists
                cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
                if cursor.fetchone():
                    logger.error(f"Username '{username}' already exists")
                    return False
                
                # Create user
                cursor.execute('''
                    INSERT INTO users (username, password_hash, salt, totp_secret, role)
                    VALUES (?, ?, ?, ?, ?)
                ''', (username, password_hash, salt, totp_secret, role))
                
                conn.commit()
                user_id = cursor.lastrowid
                
                logger.info(f"User '{username}' created successfully with ID {user_id}")
                return True
                
        except Exception as e:
            logger.error(f"User creation failed: {e}")
            return False
    
    def authenticate_user(self, username, password, totp_code=None, ip_address=None):
        """
        Authenticate user with username/password and optional TOTP
        
        Args:
            username: Username
            password: Password
            totp_code: TOTP code for 2FA (optional)
            ip_address: Client IP address for logging
            
        Returns:
            Dictionary with authentication result and session info
        """
        try:
            if not self.database:
                return {'success': False, 'error': 'Authentication system not available'}
            
            # Check if user is locked out
            if self._is_user_locked(username):
                self._log_login_attempt(username, ip_address, False, 'Account locked')
                return {'success': False, 'error': 'Account temporarily locked due to failed login attempts'}
            
            with sqlite3.connect(self.database.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get user information
                cursor.execute('''
                    SELECT id, username, password_hash, salt, totp_secret, role, is_active
                    FROM users WHERE username = ?
                ''', (username,))
                
                user = cursor.fetchone()
                
                if not user:
                    self._increment_failed_attempts(username)
                    self._log_login_attempt(username, ip_address, False, 'User not found')
                    return {'success': False, 'error': 'Invalid username or password'}
                
                # Check if user is active
                if not user['is_active']:
                    self._log_login_attempt(username, ip_address, False, 'Account disabled')
                    return {'success': False, 'error': 'Account is disabled'}
                
                # Verify password
                password_hash = self._hash_password(password, user['salt'])
                if password_hash != user['password_hash']:
                    self._increment_failed_attempts(username)
                    self._log_login_attempt(username, ip_address, False, 'Invalid password')
                    return {'success': False, 'error': 'Invalid username or password'}
                
                # Verify TOTP if enabled
                if user['totp_secret'] and not totp_code:
                    return {'success': False, 'error': 'TOTP code required', 'requires_totp': True}
                
                if user['totp_secret'] and totp_code:
                    if not self._verify_totp(user['totp_secret'], totp_code):
                        self._increment_failed_attempts(username)
                        self._log_login_attempt(username, ip_address, False, 'Invalid TOTP')
                        return {'success': False, 'error': 'Invalid TOTP code'}
                
                # Authentication successful
                self._reset_failed_attempts(username)
                
                # Create session
                session_info = self._create_session(user, ip_address)
                
                # Update last login
                cursor.execute('''
                    UPDATE users SET last_login = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (user['id'],))
                
                conn.commit()
                
                self._log_login_attempt(username, ip_address, True)
                
                logger.info(f"User '{username}' authenticated successfully")
                
                return {
                    'success': True,
                    'user_id': user['id'],
                    'username': user['username'],
                    'role': user['role'],
                    'session_token': session_info['session_token'],
                    'session_expires': session_info['expires_at']
                }
                
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return {'success': False, 'error': 'Authentication system error'}
    
    def _validate_password_strength(self, password):
        """Validate password meets security requirements"""
        try:
            if len(password) < self.security_config['password_min_length']:
                return False
            
            if self.security_config['require_special_chars']:
                special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
                if not any(char in special_chars for char in password):
                    return False
                
                # Require at least one number and one letter
                if not any(char.isdigit() for char in password):
                    return False
                
                if not any(char.isalpha() for char in password):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Password validation failed: {e}")
            return False
    
    def _hash_password(self, password, salt):
        """Hash password with salt using SHA-256"""
        try:
            return hashlib.sha256((password + salt).encode()).hexdigest()
        except Exception as e:
            logger.error(f"Password hashing failed: {e}")
            return None
    
    def _is_user_locked(self, username):
        """Check if user account is locked"""
        try:
            if not self.database:
                return False
            
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT failed_attempts, locked_until FROM users 
                    WHERE username = ?
                ''', (username,))
                
                result = cursor.fetchone()
                
                if not result:
                    return False
                
                failed_attempts, locked_until = result
                
                # Check if account is locked
                if locked_until:
                    lock_time = datetime.fromisoformat(locked_until)
                    if datetime.now() < lock_time:
                        return True
                    else:
                        # Lock expired, reset
                        cursor.execute('''
                            UPDATE users SET failed_attempts = 0, locked_until = NULL
                            WHERE username = ?
                        ''', (username,))
                        conn.commit()
                
                return False
                
        except Exception as e:
            logger.error(f"User lock check failed: {e}")
            return False
    
    def _increment_failed_attempts(self, username):
        """Increment failed login attempts and lock account if necessary"""
        try:
            if not self.database:
                return
            
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE users SET failed_attempts = failed_attempts + 1
                    WHERE username = ?
                ''', (username,))
                
                # Check if account should be locked
                cursor.execute('''
                    SELECT failed_attempts FROM users WHERE username = ?
                ''', (username,))
                
                result = cursor.fetchone()
                
                if result and result[0] >= self.security_config['max_login_attempts']:
                    lock_until = datetime.now() + timedelta(minutes=self.security_config['lockout_duration_minutes'])
                    
                    cursor.execute('''
                        UPDATE users SET locked_until = ?
                        WHERE username = ?
                    ''', (lock_until.isoformat(), username))
                    
                    logger.warning(f"Account '{username}' locked due to failed login attempts")
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed attempts increment failed: {e}")
    
    def _reset_failed_attempts(self, username):
        """Reset failed login attempts counter"""
        try:
            if not self.database:
                return
            
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE users SET failed_attempts = 0, locked_until = NULL
                    WHERE username = ?
                ''', (username,))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed attempts reset failed: {e}")
    
    def _create_session(self, user, ip_address=None):
        """Create user session"""
        try:
            session_token = secrets.token_urlsafe(self.security_config['token_length'])
            expires_at = datetime.now() + timedelta(hours=self.security_config['session_timeout_hours'])
            
            session_info = {
                'user_id': user['id'],
                'username': user['username'],
                'role': user['role'],
                'session_token': session_token,
                'created_at': datetime.now(),
                'expires_at': expires_at,
                'ip_address': ip_address
            }
            
            # Store in memory
            self.active_sessions[session_token] = session_info
            
            # Store in database
            if self.database:
                with sqlite3.connect(self.database.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        INSERT INTO user_sessions 
                        (user_id, session_token, expires_at, ip_address)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        user['id'],
                        session_token,
                        expires_at.isoformat(),
                        ip_address
                    ))
                    
                    conn.commit()
            
            return session_info
            
        except Exception as e:
            logger.error(f"Session creation failed: {e}")
            return None
    
    def validate_session(self, session_token):
        """Validate and refresh session"""
        try:
            # Check in-memory sessions first
            if session_token in self.active_sessions:
                session = self.active_sessions[session_token]
                
                # Check if session is expired
                if datetime.now() > session['expires_at']:
                    del self.active_sessions[session_token]
                    return None
                
                # Update last activity
                session['last_activity'] = datetime.now()
                return session
            
            # Check database sessions
            if self.database:
                with sqlite3.connect(self.database.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        SELECT us.*, u.username, u.role
                        FROM user_sessions us
                        JOIN users u ON us.user_id = u.id
                        WHERE us.session_token = ? AND us.is_active = 1
                    ''', (session_token,))
                    result = cursor.fetchone()
                    
                    if not result:
                        return None
                    
                    session = dict(result)
                    expires_at = datetime.fromisoformat(session['expires_at'])
                    
                    if datetime.now() > expires_at:
                        cursor.execute('UPDATE user_sessions SET is_active = 0 WHERE id = ?', (session['id'],))
                        conn.commit()
                        return None
                    
                    # Update last activity
                    cursor.execute('UPDATE user_sessions SET last_activity = CURRENT_TIMESTAMP WHERE id = ?', (session['id'],))
                    conn.commit()
                    
                    # Cache in memory
                    self.active_sessions[session_token] = {
                        'user_id': session['user_id'],
                        'username': session['username'],
                        'role': session['role'],
                        'expires_at': expires_at,
                        'last_activity': datetime.now(),
                        'ip_address': session.get('ip_address')
                    }
                    
                    return self.active_sessions[session_token]
            
        except Exception as e:
            logger.error(f"Session validation failed: {e}")
            return None
    
    def logout(self, session_token):
        """Logout user and invalidate session"""
        try:
            if session_token in self.active_sessions:
                del self.active_sessions[session_token]
            
            if self.database:
                with sqlite3.connect(self.database.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('UPDATE user_sessions SET is_active = 0 WHERE session_token = ?', (session_token,))
                    conn.commit()
            
            logger.info(f"Session {session_token} logged out successfully")
            return True
            
        except Exception as e:
            logger.error(f"Logout failed: {e}")
            return False
    
    def change_password(self, username, old_password, new_password):
        """Change user password"""
        try:
            if not self.database:
                return False
            
            if not self._validate_password_strength(new_password):
                return False
            
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('SELECT password_hash, salt FROM users WHERE username = ?', (username,))
                user = cursor.fetchone()
                
                if not user:
                    return False
                
                old_hash = self._hash_password(old_password, user[1])
                if old_hash != user[0]:
                    return False
                
                new_salt = secrets.token_hex(16)
                new_hash = self._hash_password(new_password, new_salt)
                
                cursor.execute('UPDATE users SET password_hash = ?, salt = ? WHERE username = ?', (new_hash, new_salt, username))
                conn.commit()
            
            logger.info(f"Password changed for user {username}")
            return True
            
        except Exception as e:
            logger.error(f"Password change failed: {e}")
            return False
    
    def _verify_totp(self, secret, code):
        """Verify TOTP code"""
        # Note: Requires 'pyotp' library, assume installed or add to dependencies
        try:
            import pyotp
            totp = pyotp.TOTP(secret)
            return totp.verify(code)
        except ImportError:
            logger.error("pyotp not available for TOTP verification")
            return False
        except Exception as e:
            logger.error(f"TOTP verification failed: {e}")
            return False
    
    def _log_login_attempt(self, username, ip_address, success, failure_reason=None):
        """Log login attempt"""
        try:
            if not self.database:
                return
            
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO login_attempts (username, ip_address, success, failure_reason)
                    VALUES (?, ?, ?, ?)
                ''', (username, ip_address, 1 if success else 0, failure_reason))
                conn.commit()
        except Exception as e:
            logger.error(f"Login attempt logging failed: {e}")

class DeepFakePlatformGUI:
    def __init__(self):
        self.database = DatabaseManager()
        self.analytics = RealTimeAnalyticsEngine(database=self.database)
        self.face_analyzer = EnhancedFaceAnalyzer()
        self.detector_model = AdvancedDeepFakeDetector()
        if torch.cuda.is_available():
            self.detector_model.to('cuda')
        self.detector_model.eval()
        self.browser_manager = SecureBrowserManager(database=self.database)
        self.stream_engine = StreamMonitoringEngine(self.detector_model, self.face_analyzer, self.analytics)
        self.auth_manager = AuthenticationManager(database=self.database)

        self.root = tk.Tk()
        self.root.title("Enhanced DeepFake Detection Platform v3.0")
        self.root.geometry("1200x800")

        self.logged_in = False
        self.current_user = None

        self._create_login_screen()

    def _create_login_screen(self):
        self.login_frame = ttk.Frame(self.root)
        self.login_frame.pack(fill='both', expand=True, padx=20, pady=20)

        ttk.Label(self.login_frame, text="Username:", font=('Arial', 12)).pack(pady=10)
        self.username_entry = ttk.Entry(self.login_frame, font=('Arial', 12))
        self.username_entry.pack(pady=10)

        ttk.Label(self.login_frame, text="Password:", font=('Arial', 12)).pack(pady=10)
        self.password_entry = ttk.Entry(self.login_frame, show="*", font=('Arial', 12))
        self.password_entry.pack(pady=10)

        self.totp_label = ttk.Label(self.login_frame, text="TOTP Code (if enabled):", font=('Arial', 12))
        self.totp_entry = ttk.Entry(self.login_frame, font=('Arial', 12))

        self.login_button = ttk.Button(self.login_frame, text="Login", command=self._handle_login)
        self.login_button.pack(pady=20)

    def _handle_login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        totp = self.totp_entry.get() if self.totp_entry.winfo_ismapped() else None

        auth_result = self.auth_manager.authenticate_user(username, password, totp)

        if auth_result['success']:
            self.current_user = auth_result
            self.logged_in = True
            self.login_frame.destroy()
            self._create_main_interface()
        else:
            messagebox.showerror("Login Failed", auth_result.get('error', 'Unknown error'))
            if auth_result.get('requires_totp'):
                self.totp_label.pack(pady=10)
                self.totp_entry.pack(pady=10)

    def _create_main_interface(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True)

        self.detection_tab = ttk.Frame(self.notebook)
        self.browser_tab = ttk.Frame(self.notebook)
        self.stream_tab = ttk.Frame(self.notebook)
        self.analytics_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.detection_tab, text='Media Detection')
        self.notebook.add(self.browser_tab, text='Secure Browsing')
        self.notebook.add(self.stream_tab, text='Stream Monitoring')
        self.notebook.add(self.analytics_tab, text='Analytics Dashboard')
        self.notebook.add(self.settings_tab, text='Settings & Privacy')

        self._setup_detection_tab()
        self._setup_browser_tab()
        self._setup_stream_tab()
        self._setup_analytics_tab()
        self._setup_settings_tab()

    def _setup_detection_tab(self):
        ttk.Label(self.detection_tab, text="Select Media File for DeepFake Detection", font=('Arial', 14)).pack(pady=20)

        self.file_path_var = tk.StringVar()
        ttk.Entry(self.detection_tab, textvariable=self.file_path_var, width=50).pack(pady=10)
        ttk.Button(self.detection_tab, text="Browse", command=self._browse_file).pack(pady=10)
        ttk.Button(self.detection_tab, text="Analyze Media", command=self._analyze_media).pack(pady=20)

        self.result_text = tk.Text(self.detection_tab, height=15, width=80)
        self.result_text.pack(pady=20)

    def _browse_file(self):
        file = filedialog.askopenfilename(filetypes=[("Media Files", "*.mp4 *.avi *.mov *.jpg *.png")])
        if file:
            self.file_path_var.set(file)

    def _analyze_media(self):
        file_path = self.file_path_var.get()
        if not file_path:
            messagebox.showerror("Error", "Please select a file")
            return

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Analyzing...\n")

        try:
            # Check if known sample
            known_label = self.database.check_known_sample(file_path)
            if known_label:
                result_str = f"Known sample detected: {known_label.upper()}\n"
                self.result_text.insert(tk.END, result_str)
                return

            is_video = file_path.lower().endswith(('.mp4', '.avi', '.mov'))
            frames = []
            if is_video:
                cap = cv2.VideoCapture(file_path)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                cap.release()
                selected_frames = frames[::max(1, len(frames)//10)]  # Select up to 10 frames
            else:
                frame = cv2.imread(file_path)
                selected_frames = [frame]

            confidences = []
            for frame in selected_frames:
                if frame is None:
                    continue
                face_analysis = self.face_analyzer.comprehensive_face_analysis(frame)
                face_conf = face_analysis.get('overall_confidence', 0.0)

                frame_tensor = self._prepare_frame_for_nn(frame)
                nn_conf = 0.0
                if frame_tensor is not None:
                    with torch.no_grad():
                        frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0)
                        device = next(self.detector_model.parameters()).device
                        frame_tensor = frame_tensor.to(device)
                        output, _ = self.detector_model(frame_tensor)
                        probs = F.softmax(output, dim=1)
                        nn_conf = float(probs[0, 1])

                overall_conf = max(face_conf, nn_conf) if face_analysis['faces_detected'] > 0 else nn_conf
                confidences.append(overall_conf)

            if not confidences:
                raise Exception("No frames analyzed")

            avg_conf = np.mean(confidences)
            is_deepfake = avg_conf > 0.5

            result_str = f"Analysis Complete\nAverage Confidence (Deepfake Probability): {avg_conf:.2%}\nIs DeepFake: {'Yes' if is_deepfake else 'No'}\n"
            self.result_text.insert(tk.END, result_str)

            # Store result
            analysis_result = {
                'file_path': file_path,
                'media_type': 'video' if is_video else 'image',
                'confidence': avg_conf,
                'is_deepfake': is_deepfake,
                'processing_time': 0.0,  # Placeholder
                'model_version': '3.0',
                'face_analysis': face_analysis if not is_video else {},
                'neural_network_result': {'confidence': nn_conf if not is_video else np.mean(confidences)}
            }
            self.database.store_analysis_result(analysis_result)
            self.analytics.update_metrics(analysis_result)

        except Exception as e:
            messagebox.showerror("Analysis Error", str(e))
            self.result_text.insert(tk.END, f"Error: {str(e)}\n")

    def _prepare_frame_for_nn(self, frame):
        try:
            resized = cv2.resize(frame, (224, 224))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            return transform(rgb)
        except:
            return None

    def _setup_browser_tab(self):
        ttk.Label(self.browser_tab, text="Secure Browsing", font=('Arial', 14)).pack(pady=20)

        ttk.Label(self.browser_tab, text="Enter URL:").pack(pady=10)
        self.url_var = tk.StringVar()
        ttk.Entry(self.browser_tab, textvariable=self.url_var, width=50).pack(pady=10)

        ttk.Button(self.browser_tab, text="Open in Secure Browser", command=self._open_secure_browser).pack(pady=20)

        self.browser_result = tk.Text(self.browser_tab, height=15, width=80)
        self.browser_result.pack(pady=20)

    def _open_secure_browser(self):
        url = self.url_var.get()
        if not url:
            messagebox.showerror("Error", "Please enter a URL")
            return

        self.browser_result.delete(1.0, tk.END)
        self.browser_result.insert(tk.END, "Opening secure session...\n")

        try:
            session_result = self.browser_manager.create_secure_session(url)
            if session_result.get('success', False):
                self.browser_result.insert(tk.END, f"Session ID: {session_result['session_id']}\nRisk Level: {session_result['risk_level']}\n")
                self.browser_result.insert(tk.END, f"Page Analysis: {json.dumps(session_result['page_analysis'], indent=2)}\n")
            else:
                self.browser_result.insert(tk.END, f"Error: {session_result.get('error')}\n")

            # Close session after
            if 'session_id' in session_result:
                close_result = self.browser_manager.close_session(session_result['session_id'])
                self.browser_result.insert(tk.END, f"Session closed: {close_result['status']}\n")

        except Exception as e:
            self.browser_result.insert(tk.END, f"Error: {str(e)}\n")

    def _setup_stream_tab(self):
        ttk.Label(self.stream_tab, text="Stream Monitoring", font=('Arial', 14)).pack(pady=20)
        ttk.Label(self.stream_tab, text="Enter webcam index (e.g., 0) or RTSP URL for IP camera. This monitors live streams for deepfakes in real-time, alerting and recording suspicious content.", wraplength=600, font=('Arial', 10)).pack(pady=10)

        ttk.Label(self.stream_tab, text="Stream Source:").pack(pady=10)
        self.stream_source_var = tk.StringVar()
        ttk.Entry(self.stream_tab, textvariable=self.stream_source_var, width=50).pack(pady=10)

        ttk.Button(self.stream_tab, text="Start Monitoring", command=self._start_stream).pack(pady=10)
        ttk.Button(self.stream_tab, text="Stop Monitoring", command=self._stop_stream).pack(pady=10)

        self.stream_status = tk.Text(self.stream_tab, height=15, width=80)
        self.stream_status.pack(pady=20)

    def _start_stream(self):
        source = self.stream_source_var.get()
        if not source:
            messagebox.showerror("Error", "Please enter a stream source")
            return

        source_id = "main_stream"
        source_config = {
            'source_type': 'webcam' if source.isdigit() else 'rtsp',
            'source_path': source
        }

        self.stream_engine.add_stream_source(source_id, source_config)
        self.stream_engine.start_monitoring(source_id)

        self.stream_status.insert(tk.END, "Stream monitoring started\n")

    def _stop_stream(self):
        self.stream_engine.stop_monitoring()
        self.stream_status.insert(tk.END, "Stream monitoring stopped\n")

    def _setup_analytics_tab(self):
        ttk.Label(self.analytics_tab, text="Analytics Dashboard", font=('Arial', 14)).pack(pady=20)

        ttk.Button(self.analytics_tab, text="Generate Report & Charts", command=self._generate_analytics_report).pack(pady=20)
        self.analytics_canvas_frame = ttk.Frame(self.analytics_tab)
        self.analytics_canvas_frame.pack(fill='both', expand=True)
    def _generate_analytics_report(self):
        report = self.analytics.generate_analytics_report()
        # Clear previous charts
        for widget in self.analytics_canvas_frame.winfo_children():
            widget.destroy()
        if not report:
            messagebox.showerror("Error", "No analytics data available")
            return
        # Create figures for charts
        fig1 = Figure(figsize=(5, 4))
        ax1 = fig1.add_subplot(111)
        ax1.bar(['Detections', 'Total Analyses'], [report['detection_statistics']['detection_count'], report['detection_statistics']['total_analyses']])
        ax1.set_title('Detection Statistics')
        canvas1 = FigureCanvasTkAgg(fig1, master=self.analytics_canvas_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(side=tk.LEFT, padx=10)
        fig2 = Figure(figsize=(5, 4))
        ax2 = fig2.add_subplot(111)
        media_types = list(report['detection_statistics']['media_types'].keys())
        counts = list(report['detection_statistics']['media_types'].values())
        ax2.pie(counts, labels=media_types, autopct='%1.1f%%')
        ax2.set_title('Media Types')
        canvas2 = FigureCanvasTkAgg(fig2, master=self.analytics_canvas_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(side=tk.LEFT, padx=10)
        if 'detection_trend' in report and report['detection_trend']:
            fig3 = Figure(figsize=(5, 4))
            ax3 = fig3.add_subplot(111)
            times = [t['timestamp'] for t in report['detection_trend']]
            confs = [t['confidence'] for t in report['detection_trend']]
            ax3.plot(times, confs)
            ax3.set_title('Detection Trend')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Confidence')
            canvas3 = FigureCanvasTkAgg(fig3, master=self.analytics_canvas_frame)
            canvas3.draw()
            canvas3.get_tk_widget().pack(side=tk.LEFT, padx=10)
    def _setup_settings_tab(self):
        ttk.Label(self.settings_tab, text="Settings & Privacy", font=('Arial', 14)).pack(pady=20)
        ttk.Button(self.settings_tab, text="Change Password", command=self._change_password).pack(pady=10)
        ttk.Button(self.settings_tab, text="Cleanup Old Data", command=self._cleanup_data).pack(pady=10)
        ttk.Button(self.settings_tab, text="Logout", command=self._logout).pack(pady=10)
    def _change_password(self):
        if not self.current_user:
            return
        old_pass = simpledialog.askstring("Change Password", "Old Password:", show="*")
        new_pass = simpledialog.askstring("Change Password", "New Password:", show="*")
        if new_pass:
            success = self.auth_manager.change_password(self.current_user['username'], old_pass, new_pass)
            if success:
                messagebox.showinfo("Success", "Password changed")
            else:
                messagebox.showerror("Error", "Password change failed")
    def _cleanup_data(self):
        if messagebox.askyesno("Confirm", "Cleanup old data?"):
            self.database.cleanup_old_data()
            messagebox.showinfo("Success", "Old data cleaned")
    def _logout(self):
        if self.current_user and 'session_token' in self.current_user:
            self.auth_manager.logout(self.current_user['session_token'])
        self.root.destroy()
if __name__ == "__main__":
    print("Starting Enhanced DeepFake Detection Platform v3.0")
    platform = DeepFakePlatformGUI()
    platform.root.mainloop()
