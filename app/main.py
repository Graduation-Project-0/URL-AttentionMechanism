from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import re
from urllib.parse import urlparse, parse_qs
from collections import Counter
import math
from contextlib import asynccontextmanager
import logging
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
scaler = None
feature_names = None
device = None

SUSPICIOUS_TLDS = {
    'tk', 'ml', 'ga', 'cf', 'gq', 'work', 'click', 'link', 'top', 
    'bid', 'loan', 'win', 'download', 'racing', 'date', 'stream'
}

SENSITIVE_FINANCIAL_WORDS = [
    'bank', 'account', 'paypal', 'payment', 'credit', 'card', 
    'wallet', 'bitcoin', 'transaction', 'secure'
]

SENSITIVE_WORDS = [
    'login', 'signin', 'password', 'verify', 'update', 'confirm', 
    'secure', 'account', 'admin', 'client', 'server'
]

COMMON_DOMAINS = [
    'google', 'facebook', 'amazon', 'microsoft', 'apple', 'netflix',
    'paypal', 'ebay', 'twitter', 'instagram', 'linkedin', 'yahoo'
]

class MaliciousURLDetector(nn.Module):
    def __init__(self, input_dim, hidden_dims=[768, 512, 256, 128], dropout_rate=0.4, use_batch_norm=True):
        super(MaliciousURLDetector, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )
        
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layer_block = []
            layer_block.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layer_block.append(nn.BatchNorm1d(hidden_dim))
            
            layer_block.append(nn.ReLU())
            layer_block.append(nn.Dropout(dropout_rate))
            
            setattr(self, f'layer_{i}', nn.Sequential(*layer_block))
            
            # Residual
            if prev_dim == hidden_dim:
                setattr(self, f'residual_{i}', nn.Identity())
            else:
                setattr(self, f'residual_{i}', nn.Sequential(
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
                ))
            
            prev_dim = hidden_dim
        
        self.output_dropout = nn.Dropout(0.3)
        self.output_layer = nn.Linear(hidden_dims[-1], 2)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # attention
        attention_weights = self.feature_attention(x)
        x = x * attention_weights
        
        for i in range(len(self.hidden_dims)):
            layer = getattr(self, f'layer_{i}')
            residual = getattr(self, f'residual_{i}')
            
            identity = residual(x)
            out = layer(x)
            x = out + identity  # Residual
        
        x = self.output_dropout(x)
        x = self.output_layer(x)
        return x


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, feature_names, device
    
    try:
        TIMESTAMP = "20251023_003423"
        MODEL_PATH = r"H:\GradProject\url\model_artifacts\best_model.pth"
        FEATURES_PATH = rf"H:\GradProject\url\model_artifacts\features_{TIMESTAMP}.pkl"
        SCALER_PATH = rf"H:\GradProject\url\model_artifacts\scaler_{TIMESTAMP}.pkl"
        CONFIG_PATH = rf"H:\GradProject\url\model_artifacts\config_{TIMESTAMP}.pkl"
        
        checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        
        with open(FEATURES_PATH, 'rb') as f:
            feature_names = pickle.load(f)
        
        if 'tld' in feature_names:
            logger.warning("'tld' found in feature_names but should be excluded")
            feature_names = [f for f in feature_names if f != 'tld']
        
        import os
        import joblib
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            logger.info("✓ Scaler loaded successfully")
        else:
            scaler = None
            logger.warning("Scaler file not found. Running without scaling.")
        
        use_scaling = True  # Default
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'rb') as f:
                config_dict = pickle.load(f)
                use_scaling = config_dict.get('USE_SCALING', True)
                logger.info(f"✓ Config loaded. USE_SCALING: {use_scaling}")
        
        if 'input_dim' in checkpoint:
            input_dim = checkpoint['input_dim']
            hidden_dims = checkpoint.get('hidden_dims', [768, 512, 256, 128])
            dropout_rate = checkpoint.get('dropout_rate', 0.4)
            use_batch_norm = checkpoint.get('use_batch_norm', True)
        else:
            input_dim = len(feature_names)
            hidden_dims = [768, 512, 256, 128]
            dropout_rate = 0.4
            use_batch_norm = True
            logger.warning("Checkpoint doesn't contain architecture info. Using enhanced defaults.")
        
        model = MaliciousURLDetector(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm
        )
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
    except Exception as e:
        logger.error(f"ERROR LOADING MODEL: {str(e)}", exc_info=True)
        raise
    
    yield
    logger.info("Shutting down API...")

app = FastAPI(
    title="Malicious URL Detection API",
    description="Enhanced deep learning-based API for detecting malicious URLs with improved architecture, feature scaling, and comprehensive metrics",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class URLRequest(BaseModel):
    url: str
    debug: Optional[bool] = False
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('URL cannot be empty')
        v = v.strip()
        if not v.startswith(('http://', 'https://')):
            v = 'http://' + v
        return v

class PredictionResponse(BaseModel):
    url: str
    prediction: str
    is_malicious: bool
    confidence: float
    malicious_probability: float
    benign_probability: float
    timestamp: str
    details: Dict[str, Any]
    debug_info: Optional[Dict[str, Any]] = None

class CSVPredictionResult(BaseModel):
    total_urls: int
    malicious_count: int
    benign_count: int
    predictions: List[Dict[str, Any]]
    timestamp: str
    summary: Dict[str, Any]

def calculate_entropy(text):
    """Calculate Shannon entropy"""
    if not text:
        return 0
    counter = Counter(text)
    length = len(text)
    entropy = -sum((count/length) * math.log2(count/length) for count in counter.values())
    return entropy

def calculate_hamming_features(url):
    """Calculate Hamming distance based features"""
    if len(url) < 2:
        return 0, 0, 0, 0, 0
    
    pairs = [(url[i], url[i+1]) for i in range(len(url)-1)]
    
    hamming_1 = sum(1 for a, b in pairs if a != b) / len(pairs)
    hamming_00 = sum(1 for a, b in pairs if a == '0' and b == '0') / len(pairs)
    hamming_10 = sum(1 for a, b in pairs if a == '1' and b == '0') / len(pairs)
    hamming_01 = sum(1 for a, b in pairs if a == '0' and b == '1') / len(pairs)
    hamming_11 = sum(1 for a, b in pairs if a == '1' and b == '1') / len(pairs)
    
    return hamming_1, hamming_00, hamming_10, hamming_01, hamming_11

def calculate_n_gram_entropy(text, n):
    """Calculate n-gram entropy"""
    if len(text) < n:
        return 0
    
    ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
    counter = Counter(ngrams)
    length = len(ngrams)
    
    if length == 0:
        return 0
    
    entropy = -sum((count/length) * math.log2(count/length) for count in counter.values())
    return entropy

def levenshtein_distance(s1, s2):
    """Calculate Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def min_distance_to_common_domains(domain):
    """Calculate minimum Levenshtein distance to common legitimate domains"""
    if not domain:
        return 100
    
    distances = [levenshtein_distance(domain.lower(), common.lower()) 
                 for common in COMMON_DOMAINS]
    return min(distances) if distances else 100

def count_sensitive_words(text, word_list):
    """Count occurrences of sensitive words in text"""
    text_lower = text.lower()
    return sum(1 for word in word_list if word in text_lower)

def extract_features_from_url(url: str) -> pd.DataFrame:
    features = {}
    
    parsed = urlparse(url)
    hostname = parsed.netloc
    path = parsed.path
    query = parsed.query
    
    domain_parts = hostname.split('.')
    
    if len(domain_parts) >= 2:
        tld = domain_parts[-1]
        pdomain = domain_parts[-2]
        subdomain = '.'.join(domain_parts[:-2]) if len(domain_parts) > 2 else ''
    else:
        tld = domain_parts[0] if domain_parts else ''
        pdomain = ''
        subdomain = ''
    
    features['url_has_login'] = int('login' in url.lower())
    features['url_has_client'] = int('client' in url.lower())
    features['url_has_server'] = int('server' in url.lower())
    features['url_has_admin'] = int('admin' in url.lower())
    
    ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    features['url_has_ip'] = int(bool(re.search(ip_pattern, hostname)))
    
    shorteners = ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly', 'is.gd']
    features['url_isshorted'] = int(any(short in hostname.lower() for short in shorteners))
    
    features['url_len'] = len(url)
    features['url_entropy'] = calculate_entropy(url)
    
    h1, h00, h10, h01, h11 = calculate_hamming_features(url)
    features['url_hamming_1'] = h1
    features['url_hamming_00'] = h00
    features['url_hamming_10'] = h10
    features['url_hamming_01'] = h01
    features['url_hamming_11'] = h11
    
    features['url_2bentropy'] = calculate_n_gram_entropy(url, 2)
    features['url_3bentropy'] = calculate_n_gram_entropy(url, 3)
    
    features['url_count_dot'] = url.count('.')
    features['url_count_https'] = url.lower().count('https')
    features['url_count_http'] = url.lower().count('http')
    features['url_count_perc'] = url.count('%')
    features['url_count_hyphen'] = url.count('-')
    features['url_count_www'] = url.lower().count('www')
    features['url_count_atrate'] = url.count('@')
    features['url_count_hash'] = url.count('#')
    features['url_count_semicolon'] = url.count(';')
    features['url_count_underscore'] = url.count('_')
    features['url_count_ques'] = url.count('?')
    features['url_count_equal'] = url.count('=')
    features['url_count_amp'] = url.count('&')
    
    features['url_count_letter'] = sum(1 for c in url if c.isalpha())
    features['url_count_digit'] = sum(1 for c in url if c.isdigit())
    
    features['url_count_sensitive_financial_words'] = count_sensitive_words(url, SENSITIVE_FINANCIAL_WORDS)
    features['url_count_sensitive_words'] = count_sensitive_words(url, SENSITIVE_WORDS)
    
    features['url_nunique_chars_ratio'] = len(set(url)) / len(url) if len(url) > 0 else 0
    
    features['path_len'] = len(path)
    features['path_count_no_of_dir'] = path.count('/') - 1 if path else 0
    features['path_count_no_of_embed'] = path.count('//')
    features['path_count_zero'] = path.count('0')
    features['path_count_pertwent'] = path.count('%20')
    features['path_has_any_sensitive_words'] = int(count_sensitive_words(path, SENSITIVE_WORDS) > 0)
    features['path_count_lower'] = sum(1 for c in path if c.islower())
    features['path_count_upper'] = sum(1 for c in path if c.isupper())
    features['path_count_nonascii'] = sum(1 for c in path if ord(c) > 127)
    
    path_dirs = [d for d in path.split('/') if d]
    features['path_has_singlechardir'] = int(any(len(d) == 1 for d in path_dirs))
    features['path_has_upperdir'] = int(any(d[0].isupper() for d in path_dirs if d))
    
    features['query_len'] = len(query)
    features['query_count_components'] = len(parse_qs(query)) if query else 0
    
    features['pdomain_len'] = len(pdomain)
    features['pdomain_count_hyphen'] = pdomain.count('-')
    features['pdomain_count_atrate'] = pdomain.count('@')
    features['pdomain_count_non_alphanum'] = sum(1 for c in pdomain if not c.isalnum())
    features['pdomain_count_digit'] = sum(1 for c in pdomain if c.isdigit())
    
    features['tld_len'] = len(tld)
    

    features['tld'] = tld
    features['tld_is_sus'] = int(tld.lower() in SUSPICIOUS_TLDS)
    features['pdomain_min_distance'] = min_distance_to_common_domains(pdomain)
    
    features['subdomain_len'] = len(subdomain)
    features['subdomain_count_dot'] = subdomain.count('.')
    
    df = pd.DataFrame([features])

    if 'tld' in df.columns:
        df = df.drop(columns=['tld'])
    
    return df

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Enhanced Malicious URL Detection API v3.0",
        "version": "3.0.0",
        "model_loaded": model is not None,
        "device": str(device) if device else "not loaded",
        "num_features": len(feature_names) if feature_names else 0,
        "scaling_enabled": scaler is not None,
        "architecture": {
            "hidden_dims": [768, 512, 256, 128],
            "dropout_rate": 0.4,
            "attention_mechanism": "enhanced",
            "residual_connections": True
        },
        "enhancements": [
            "Feature scaling with StandardScaler",
            "Enhanced attention mechanism",
            "Improved residual connections",
            "Gradient clipping",
            "Label smoothing",
            "Class weight balancing"
        ],
        "endpoints": {
            "predict": "/predict (POST) - add 'debug: true' for detailed info",
            "predict_csv": "/predict-csv (POST) - Upload CSV file with features",
            "health": "/health (GET)",
            "docs": "/docs"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_url(request: URLRequest):
    if model is None or feature_names is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        features_df = extract_features_from_url(request.url)
        
        parsed = urlparse(request.url)
        hostname = parsed.netloc
        domain_parts = hostname.split('.')
        original_tld = domain_parts[-1] if len(domain_parts) >= 1 else ''
        
        missing_features = []
        for feature in feature_names:
            if feature not in features_df.columns:
                features_df[feature] = 0
                missing_features.append(feature)
        
        if missing_features and request.debug:
            logger.warning(f"Missing features filled with 0: {missing_features}")
        
        features_df = features_df[feature_names]
        
        raw_features = features_df.iloc[0].to_dict()
        
        # scaling
        if scaler is not None:
            X_scaled = scaler.transform(features_df.values)
            X_for_model = X_scaled
        else:
            X_for_model = features_df.values
        
        X_tensor = torch.tensor(X_for_model, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            outputs = model(X_tensor)
            probabilities = F.softmax(outputs, dim=1)
            _, predictions = torch.max(probabilities, 1)
        
        prediction = predictions.cpu().numpy()[0]
        probs = probabilities.cpu().numpy()[0]
        raw_outputs = outputs.cpu().numpy()[0]
        
        is_malicious = bool(prediction == 1)
        confidence = float(probs[prediction])
        malicious_prob = float(probs[1])
        benign_prob = float(probs[0])
        
        # Debug
        debug_info = None
        if request.debug:
            # top 10 most influential features
            feature_importance = [(name, raw_features[name], X_for_model[0][i]) 
                                  for i, name in enumerate(feature_names)]

            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            debug_info = {
                "tld_info": {
                    "original": original_tld,
                    "tld_is_suspicious": bool(raw_features.get('tld_is_sus', 0))
                },
                "raw_model_outputs": {
                    "benign_logit": float(raw_outputs[0]),
                    "malicious_logit": float(raw_outputs[1]),
                    "logit_difference": float(raw_outputs[1] - raw_outputs[0])
                },
                "feature_stats": {
                    "num_features": len(feature_names),
                    "num_nonzero_raw": int(sum(1 for v in raw_features.values() if v != 0)),
                    "raw_mean": float(np.mean(features_df.values[0])),
                    "raw_std": float(np.std(features_df.values[0])),
                    "raw_min": float(np.min(features_df.values[0])),
                    "raw_max": float(np.max(features_df.values[0])),
                    "scaled_mean": float(np.mean(X_for_model[0])) if scaler is not None else None,
                    "scaled_std": float(np.std(X_for_model[0])) if scaler is not None else None,
                    "scaling_applied": scaler is not None
                },
                "top_10_features": [
                    {
                        "name": name,
                        "raw_value": float(raw),
                        "scaled_value": float(scaled),
                        "note": "StandardScaler applied" if scaler is not None else "No scaling"
                    }
                    for name, raw, scaled in feature_importance[:10]
                ],
                "missing_features": missing_features if missing_features else []
            }
            
            logger.info(f"URL: {request.url}")
            logger.info(f"Prediction: {'Malicious' if is_malicious else 'Benign'} ({confidence:.4f})")
            logger.info(f"Raw outputs - Benign: {raw_outputs[0]:.4f}, Malicious: {raw_outputs[1]:.4f}")
            logger.info(f"Scaling applied: {scaler is not None}")
        
        response = PredictionResponse(
            url=request.url,
            prediction="Malicious" if is_malicious else "Benign",
            is_malicious=is_malicious,
            confidence=round(confidence, 4),
            malicious_probability=round(malicious_prob, 4),
            benign_probability=round(benign_prob, 4),
            timestamp=datetime.now().isoformat(),
            details={
                "url_length": len(request.url),
                "domain": parsed.netloc,
                "tld": original_tld,
                "has_suspicious_keywords": bool(raw_features.get('url_count_sensitive_words', 0) > 0),
                "has_ip_address": bool(raw_features.get('url_has_ip', 0)),
                "is_shortened": bool(raw_features.get('url_isshorted', 0)),
                "suspicious_tld": bool(raw_features.get('tld_is_sus', 0)),
                "url_entropy": round(float(raw_features.get('url_entropy', 0)), 4),
                "scaling_applied": scaler is not None,
                "device_used": str(device)
            },
            debug_info=debug_info
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict-csv", response_model=CSVPredictionResult)
async def predict_csv(file: UploadFile = File(...)):
    if model is None or feature_names is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        logger.info(f"CSV uploaded: {file.filename}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        original_urls = df['url'].tolist() if 'url' in df.columns else [f"row_{i}" for i in range(len(df))]
        
        columns_to_drop = ['label', 'url', 'source', 'tld']
        df_features = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
        
        missing_features = []
        for feature in feature_names:
            if feature not in df_features.columns:
                df_features[feature] = 0
                missing_features.append(feature)
        
        if missing_features:
            logger.warning(f"Missing features filled with 0: {missing_features}")
        
        df_features = df_features[feature_names]
        
        logger.info(f"✓ Features prepared: {df_features.shape}")
        
        if scaler is not None:
            X_scaled = scaler.transform(df_features.values)
            X_for_model = X_scaled
            logger.info("✓ Feature scaling applied")
        else:
            X_for_model = df_features.values
            logger.info("No scaling applied (scaler not loaded)")
        
        X_tensor = torch.tensor(X_for_model, dtype=torch.float32).to(device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)
            probabilities = F.softmax(outputs, dim=1)
            _, predictions = torch.max(probabilities, 1)
        
        predictions_np = predictions.cpu().numpy()
        probabilities_np = probabilities.cpu().numpy()
        
        results = []
        malicious_count = 0
        benign_count = 0
        
        for i in range(len(df)):
            pred = int(predictions_np[i])
            is_malicious = bool(pred == 1)
            
            if is_malicious:
                malicious_count += 1
            else:
                benign_count += 1
            
            result = {
                "row_index": i,
                "url": original_urls[i],
                "prediction": "Malicious" if is_malicious else "Benign",
                "is_malicious": is_malicious,
                "confidence": round(float(probabilities_np[i][pred]), 4),
                "malicious_probability": round(float(probabilities_np[i][1]), 4),
                "benign_probability": round(float(probabilities_np[i][0]), 4)
            }
            results.append(result)
        
        malicious_probs = probabilities_np[:, 1]
        summary = {
            "avg_malicious_probability": round(float(np.mean(malicious_probs)), 4),
            "std_malicious_probability": round(float(np.std(malicious_probs)), 4),
            "min_malicious_probability": round(float(np.min(malicious_probs)), 4),
            "max_malicious_probability": round(float(np.max(malicious_probs)), 4),
            "malicious_percentage": round((malicious_count / len(df)) * 100, 2),
            "benign_percentage": round((benign_count / len(df)) * 100, 2),
            "missing_features": missing_features if missing_features else [],
            "scaling_applied": scaler is not None,
            "model_version": "3.0.0",
            "architecture": "enhanced"
        }
        
        logger.info(f"Predictions completed: {malicious_count} malicious, {benign_count} benign")
        
        response = CSVPredictionResult(
            total_urls=len(df),
            malicious_count=malicious_count,
            benign_count=benign_count,
            predictions=results,
            timestamp=datetime.now().isoformat(),
            summary=summary
        )
        
        return response
        
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid CSV format: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )