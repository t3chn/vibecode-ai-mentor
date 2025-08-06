"""
Live Code Analysis Demo Scenarios
=================================

This file contains carefully crafted code samples and scenarios for demonstrating 
VibeCode AI Mentor's live code analysis capabilities during the hackathon presentation.
"""

# Demo Scenario 1: Authentication Middleware (Real-world complexity)
AUTHENTICATION_MIDDLEWARE_DEMO = '''
"""
Authentication middleware with multiple improvement opportunities for impressive demo.
This code intentionally contains various patterns that the AI can detect and improve.
"""

import jwt
import logging
from functools import wraps
from flask import request, jsonify, current_app
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import redis
import hashlib
import time

# Global variables (anti-pattern for demo)
user_cache = {}
failed_attempts = {}

def authenticate_user(username, password):
    """Authenticate user with multiple issues for demo purposes."""
    # Issue 1: No input validation
    # Issue 2: Hardcoded credentials check (security)
    # Issue 3: No rate limiting
    # Issue 4: Poor error handling
    
    if username == "" or password == "":
        return False
    
    # Issue 5: Hardcoded admin credentials (major security issue)
    if username == "admin" and password == "admin123":
        return True
    
    # Issue 6: SQL injection vulnerable (simulated)
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    
    # Issue 7: No password hashing
    # Issue 8: Synchronous database call in async context
    try:
        # Simulate database call
        time.sleep(0.1)  # Blocking call (performance issue)
        
        # Issue 9: Using global variable for caching
        if username in user_cache:
            return user_cache[username]['valid']
        
        # Issue 10: Mock authentication logic with issues
        valid = len(username) > 3 and len(password) > 6
        user_cache[username] = {'valid': valid, 'timestamp': time.time()}
        
        return valid
    except Exception as e:
        # Issue 11: Bare except clause
        print(f"Auth error: {e}")  # Issue 12: Using print instead of logging
        return False

def generate_jwt_token(user_id, username):
    """Generate JWT token with security issues."""
    # Issue 13: Hardcoded secret key
    secret_key = "supersecret123"
    
    # Issue 14: No token expiration
    payload = {
        'user_id': user_id,
        'username': username,
        'generated_at': datetime.now().timestamp()
    }
    
    # Issue 15: Using deprecated algorithm
    token = jwt.encode(payload, secret_key, algorithm='HS256')
    
    # Issue 16: Logging sensitive information
    logging.info(f"Generated token for {username}: {token}")
    
    return token

def validate_token(token):
    """Validate JWT token with multiple issues."""
    # Issue 17: No try-catch for token validation
    # Issue 18: Same hardcoded secret
    secret_key = "supersecret123"
    
    decoded = jwt.decode(token, secret_key, algorithms=['HS256'])
    
    # Issue 19: No token expiration check
    # Issue 20: No token blacklist check
    
    return decoded

def rate_limit_check(username):
    """Rate limiting with inefficient implementation."""
    # Issue 21: Using global dictionary instead of proper cache
    current_time = time.time()
    
    if username not in failed_attempts:
        failed_attempts[username] = []
    
    # Issue 22: Inefficient list operations
    # Remove old attempts (older than 1 hour)
    failed_attempts[username] = [
        attempt for attempt in failed_attempts[username] 
        if current_time - attempt < 3600
    ]
    
    # Issue 23: Magic number (should be configurable)
    if len(failed_attempts[username]) > 5:
        return False
    
    return True

def require_auth(f):
    """Authentication decorator with issues."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        
        # Issue 24: Inconsistent header checking
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]  # Issue 25: No bounds check
            except IndexError:
                pass
        
        if not token:
            return jsonify({'error': 'Token missing'}), 401
        
        try:
            user_data = validate_token(token)
            # Issue 26: No user existence verification
            current_user = user_data
        except:  # Issue 27: Bare except again
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(*args, **kwargs)
    
    return decorated_function

# Demo API endpoints using the flawed authentication
@require_auth
def get_user_profile():
    """Get user profile endpoint."""
    # Issue 28: No input validation
    # Issue 29: No proper error responses
    user_id = request.args.get('user_id')
    
    # Issue 30: Potential None type error
    profile = {
        'id': user_id,
        'name': f'User {user_id}',
        'permissions': ['read', 'write'] if user_id == '1' else ['read']
    }
    
    return jsonify(profile)

@require_auth 
def update_user_settings():
    """Update user settings with validation issues."""
    # Issue 31: No CSRF protection
    # Issue 32: No input sanitization
    # Issue 33: Direct request data usage
    
    data = request.get_json()
    
    # Issue 34: No data validation
    # Issue 35: Potential key errors
    user_id = data['user_id']
    settings = data['settings']
    
    # Issue 36: No database transaction handling
    # Issue 37: No rollback mechanism
    
    # Simulate settings update
    success = True  # Always succeeds (unrealistic)
    
    if success:
        return jsonify({'status': 'updated'})
    else:
        return jsonify({'error': 'Update failed'}), 500

# Additional helper functions with issues
def hash_password(password):
    """Weak password hashing for demo."""
    # Issue 38: Using MD5 instead of bcrypt/argon2
    # Issue 39: No salt
    return hashlib.md5(password.encode()).hexdigest()

def log_security_event(event_type, user_id, details):
    """Security logging with issues."""
    # Issue 40: Logging to file without rotation
    # Issue 41: No structured logging
    # Issue 42: Potential log injection
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {event_type}: User {user_id} - {details}"
    
    # Issue 43: File I/O without error handling
    with open('security.log', 'a') as f:
        f.write(log_entry + '\n')

# Configuration with hardcoded values
class Config:
    """Configuration with security issues."""
    # Issue 44: Hardcoded secrets in code
    SECRET_KEY = 'dev-secret-key-123'
    DATABASE_URL = 'mysql://admin:password123@localhost/app'
    REDIS_URL = 'redis://localhost:6379'
    
    # Issue 45: Debug mode in production-like code
    DEBUG = True
    
    # Issue 46: Weak session configuration
    SESSION_COOKIE_SECURE = False
    SESSION_COOKIE_HTTPONLY = False
'''

# Demo Scenario 2: Data Processing Pipeline (Performance Issues)
DATA_PROCESSING_PIPELINE_DEMO = '''
"""
Data processing pipeline with performance bottlenecks and optimization opportunities.
Perfect for demonstrating AI-powered performance recommendations.
"""

import pandas as pd
import numpy as np
import json
import time
from typing import List, Dict, Any
import requests
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import os

class DataProcessor:
    """Data processor with multiple performance issues for demo."""
    
    def __init__(self):
        # Issue 1: Loading large data in constructor
        self.lookup_data = self._load_lookup_data()
        # Issue 2: No connection pooling
        self.db_connections = []
        # Issue 3: No caching mechanism
        self.processed_cache = {}
    
    def _load_lookup_data(self):
        """Load lookup data inefficiently."""
        # Issue 4: Reading entire file into memory
        # Issue 5: No error handling for file operations
        lookup = {}
        
        # Simulate loading large lookup table
        for i in range(100000):  # Issue 6: Magic number
            lookup[f"key_{i}"] = {
                'value': i * 2,
                'category': f"cat_{i % 10}",
                'metadata': f"meta_data_for_key_{i}" * 10  # Issue 7: Memory waste
            }
        
        return lookup
    
    def process_data_batch(self, data_batch: List[Dict]) -> List[Dict]:
        """Process data batch with multiple inefficiencies."""
        results = []
        
        # Issue 8: Sequential processing instead of parallel
        for item in data_batch:
            try:
                # Issue 9: Expensive lookup in loop
                processed_item = self._process_single_item(item)
                results.append(processed_item)
                
                # Issue 10: I/O in tight loop
                self._log_processing_step(item['id'])
                
            except Exception as e:
                # Issue 11: Continuing with partial results
                print(f"Error processing item: {e}")
                continue
        
        return results
    
    def _process_single_item(self, item: Dict) -> Dict:
        """Process single item with optimization opportunities."""
        # Issue 12: Multiple database queries per item
        user_info = self._get_user_info(item['user_id'])
        permissions = self._get_user_permissions(item['user_id'])
        preferences = self._get_user_preferences(item['user_id'])
        
        # Issue 13: Inefficient data transformation
        transformed_data = {}
        for key, value in item.items():
            if isinstance(value, str):
                # Issue 14: Repeated string operations
                transformed_data[key] = value.strip().lower().replace(' ', '_')
                # Issue 15: Regex would be more efficient
                transformed_data[key] = transformed_data[key].replace('-', '_')
                transformed_data[key] = transformed_data[key].replace('.', '_')
        
        # Issue 16: Expensive lookup for each item
        lookup_key = f"key_{item.get('lookup_id', 0)}"
        if lookup_key in self.lookup_data:
            enrichment = self.lookup_data[lookup_key]
        else:
            # Issue 17: Expensive fallback operation
            enrichment = self._fetch_enrichment_data(lookup_key)
        
        # Issue 18: Inefficient dictionary merging
        result = {}
        result.update(item)
        result.update(transformed_data)
        result.update(user_info)
        result.update(permissions)
        result.update(preferences)
        result.update(enrichment)
        
        return result
    
    def _get_user_info(self, user_id: int) -> Dict:
        """Get user info with N+1 query problem."""
        # Issue 19: Creating new connection for each query
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        # Issue 20: No prepared statements
        query = f"SELECT * FROM users WHERE id = {user_id}"
        cursor.execute(query)
        
        result = cursor.fetchone()
        conn.close()  # Issue 21: Connection not reused
        
        if result:
            # Issue 22: Manual row to dict conversion
            return {
                'user_name': result[1],
                'user_email': result[2],
                'user_status': result[3]
            }
        
        return {}
    
    def _get_user_permissions(self, user_id: int) -> Dict:
        """Another database query - N+1 problem continues."""
        conn = sqlite3.connect('users.db')  # Issue 23: Duplicate connection code
        cursor = conn.cursor()
        
        cursor.execute(f"SELECT permission FROM user_permissions WHERE user_id = {user_id}")
        permissions = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        return {'permissions': permissions}
    
    def _get_user_preferences(self, user_id: int) -> Dict:
        """Third database query for same user."""
        conn = sqlite3.connect('users.db')  # Issue 24: More connection overhead
        cursor = conn.cursor()
        
        cursor.execute(f"SELECT pref_key, pref_value FROM user_prefs WHERE user_id = {user_id}")
        prefs = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        return {'preferences': prefs}
    
    def _fetch_enrichment_data(self, lookup_key: str) -> Dict:
        """Fetch enrichment data with network inefficiencies."""
        # Issue 25: Synchronous HTTP requests
        # Issue 26: No request timeout
        # Issue 27: No retry mechanism
        
        try:
            response = requests.get(
                f'https://api.example.com/enrich/{lookup_key}',
                # Issue 28: No timeout specified
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                # Issue 29: No proper error handling
                return {}
                
        except requests.RequestException:
            # Issue 30: Swallowing exceptions
            return {}
    
    def _log_processing_step(self, item_id: str):
        """Logging with I/O inefficiency."""
        # Issue 31: File I/O for each log entry
        # Issue 32: No log rotation
        # Issue 33: No structured logging
        
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] Processed item: {item_id}\n"
        
        # Issue 34: Opening/closing file repeatedly
        with open('processing.log', 'a') as f:
            f.write(log_entry)
    
    def export_results(self, results: List[Dict], format: str = 'json'):
        """Export results with format inefficiencies."""
        if format == 'json':
            # Issue 35: Loading all data in memory for export
            json_data = json.dumps(results, indent=2)
            
            # Issue 36: No streaming for large datasets
            with open('results.json', 'w') as f:
                f.write(json_data)
                
        elif format == 'csv':
            # Issue 37: Using pandas for simple CSV writing
            df = pd.DataFrame(results)
            df.to_csv('results.csv', index=False)
        
        elif format == 'excel':
            # Issue 38: Inefficient Excel writing
            df = pd.DataFrame(results)
            # Issue 39: No Excel optimization settings
            df.to_excel('results.xlsx', index=False, engine='openpyxl')

# Demo usage with inefficient patterns
def process_large_dataset():
    """Main processing function with orchestration issues."""
    processor = DataProcessor()
    
    # Issue 40: Loading all data at once
    # Simulate loading large dataset
    all_data = []
    for i in range(10000):  # Issue 41: Another magic number
        all_data.append({
            'id': f'item_{i}',
            'user_id': i % 1000,  # Issue 42: Limited user variety
            'data_field': f'data_value_{i}',
            'lookup_id': i % 50000,
            'timestamp': time.time()
        })
    
    # Issue 43: No batch processing
    results = processor.process_data_batch(all_data)
    
    # Issue 44: Multiple export formats (wasteful)
    processor.export_results(results, 'json')
    processor.export_results(results, 'csv')
    processor.export_results(results, 'excel')
    
    print(f"Processed {len(results)} items")  # Issue 45: Print instead of proper logging

if __name__ == "__main__":
    start_time = time.time()
    process_large_dataset()
    end_time = time.time()
    
    # Issue 46: Manual timing instead of profiler
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
'''

# Demo Scenario 3: Machine Learning Model Training (Complex Issues)
ML_MODEL_TRAINING_DEMO = '''
"""
Machine Learning model training code with various improvement opportunities.
Showcases AI recommendations for ML best practices and performance optimization.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class MLModelTrainer:
    """ML Model trainer with multiple issues for demonstration."""
    
    def __init__(self, model_type='neural_network'):
        # Issue 1: No input validation
        self.model_type = model_type
        # Issue 2: No proper seed setting for reproducibility
        np.random.seed(42)  # Issue 3: Hardcoded seed
        tf.random.set_seed(42)
        
        # Issue 4: Model created in constructor
        self.model = self._create_model()
        # Issue 5: No configuration management
        self.config = {
            'epochs': 100,  # Issue 6: Hardcoded hyperparameters
            'batch_size': 32,
            'learning_rate': 0.001,
            'validation_split': 0.2
        }
        
        # Issue 7: Preprocessors as instance variables
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def _create_model(self):
        """Create model with architecture issues."""
        if self.model_type == 'neural_network':
            model = tf.keras.Sequential([
                # Issue 8: Hardcoded layer sizes
                tf.keras.layers.Dense(128, activation='relu', input_shape=(10,)),
                # Issue 9: No dropout for regularization
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                # Issue 10: No output layer size configuration
                tf.keras.layers.Dense(3, activation='softmax')
            ])
            
            # Issue 11: Hardcoded optimizer configuration
            model.compile(
                optimizer='adam',  # Issue 12: No learning rate specification
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
        else:
            # Issue 13: Limited model type support
            raise ValueError("Only neural_network supported")
    
    def load_and_preprocess_data(self, data_path: str):
        """Load and preprocess data with multiple issues."""
        # Issue 14: No file existence check
        # Issue 15: Assuming CSV format without validation
        data = pd.read_csv(data_path)
        
        # Issue 16: Hardcoded column assumptions
        # Issue 17: No missing value handling strategy
        data = data.dropna()  # Issue 18: Simply dropping all missing values
        
        # Issue 19: Hardcoded feature/target split
        features = data.iloc[:, :-1]
        target = data.iloc[:, -1]
        
        # Issue 20: No data validation
        # Issue 21: Preprocessing without proper pipeline
        features_scaled = self.scaler.fit_transform(features)
        target_encoded = self.label_encoder.fit_transform(target)
        
        # Issue 22: No stratification consideration
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, target_encoded, 
            test_size=0.2,  # Issue 23: Hardcoded split ratio
            random_state=42  # Issue 24: Same hardcoded seed
        )
        
        # Issue 25: Storing as instance variables (memory inefficient)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # Issue 26: No data quality checks
        # Issue 27: No feature engineering
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self):
        """Train model with various issues."""
        # Issue 28: No early stopping
        # Issue 29: No model checkpointing
        # Issue 30: No validation monitoring
        
        print("Starting model training...")  # Issue 31: Print instead of logging
        
        start_time = datetime.now()
        
        # Issue 32: No callbacks for monitoring
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_split=self.config['validation_split'],
            verbose=1  # Issue 33: Hardcoded verbosity
        )
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        # Issue 34: Manual time tracking
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Issue 35: No training metrics persistence
        self.training_history = history.history
        
        return history
    
    def evaluate_model(self):
        """Evaluate model with limited metrics."""
        # Issue 36: Only basic evaluation
        # Issue 37: No cross-validation
        
        train_predictions = self.model.predict(self.X_train)
        test_predictions = self.model.predict(self.X_test)
        
        # Issue 38: Manual prediction processing
        train_pred_classes = np.argmax(train_predictions, axis=1)
        test_pred_classes = np.argmax(test_predictions, axis=1)
        
        # Issue 39: Limited metrics calculation
        train_accuracy = accuracy_score(self.y_train, train_pred_classes)
        test_accuracy = accuracy_score(self.y_test, test_pred_classes)
        
        # Issue 40: No statistical significance testing
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Issue 41: No overfitting detection
        if train_accuracy - test_accuracy > 0.1:
            print("Warning: Possible overfitting detected")
        
        # Issue 42: No confusion matrix
        # Issue 43: No precision/recall analysis
        
        evaluation_results = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'overfitting_gap': train_accuracy - test_accuracy
        }
        
        return evaluation_results
    
    def save_model(self, model_path: str = 'model.h5'):
        """Save model with basic approach."""
        # Issue 44: No model versioning
        # Issue 45: No metadata saving
        # Issue 46: Hardcoded format
        
        self.model.save(model_path)
        
        # Issue 47: Separate files for preprocessors (inconsistent)
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.label_encoder, 'label_encoder.pkl')
        
        # Issue 48: No model configuration saving
        # Issue 49: No training metadata
        
        print(f"Model saved to {model_path}")
    
    def create_training_plots(self):
        """Create training plots with basic implementation."""
        # Issue 50: No plot customization
        # Issue 51: Hardcoded figure sizes
        
        plt.figure(figsize=(12, 4))  # Issue 52: Magic numbers
        
        # Issue 53: No error handling for missing history
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['loss'], label='Training Loss')
        plt.plot(self.training_history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')  # Issue 54: Generic titles
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.training_history['accuracy'], label='Training Accuracy')
        plt.plot(self.training_history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Issue 55: Hardcoded file name
        plt.savefig('training_plots.png')
        plt.show()
    
    def hyperparameter_tuning(self):
        """Basic hyperparameter tuning with issues."""
        # Issue 56: Manual grid search implementation
        # Issue 57: No proper cross-validation
        # Issue 58: Limited parameter space
        
        learning_rates = [0.001, 0.01, 0.1]  # Issue 59: Small search space
        batch_sizes = [16, 32, 64]
        
        best_accuracy = 0
        best_params = {}
        
        # Issue 60: Nested loops instead of proper grid search
        for lr in learning_rates:
            for batch_size in batch_sizes:
                print(f"Testing lr={lr}, batch_size={batch_size}")
                
                # Issue 61: Creating new model each time (inefficient)
                test_model = self._create_model()
                test_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # Issue 62: No validation strategy
                history = test_model.fit(
                    self.X_train, self.y_train,
                    epochs=10,  # Issue 63: Too few epochs for tuning
                    batch_size=batch_size,
                    validation_split=0.2,
                    verbose=0
                )
                
                # Issue 64: Using final epoch accuracy (not best)
                accuracy = max(history.history['val_accuracy'])
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {'lr': lr, 'batch_size': batch_size}
        
        print(f"Best parameters: {best_params}")
        print(f"Best accuracy: {best_accuracy:.4f}")
        
        return best_params

# Demo usage showing workflow issues
def main_training_pipeline():
    """Main training pipeline with orchestration issues."""
    # Issue 65: No argument parsing
    # Issue 66: Hardcoded file paths
    data_path = 'training_data.csv'
    
    # Issue 67: No error handling for file operations
    trainer = MLModelTrainer()
    
    # Issue 68: Sequential execution (no parallelization opportunities)
    print("Loading and preprocessing data...")
    trainer.load_and_preprocess_data(data_path)
    
    print("Training model...")
    trainer.train_model()
    
    print("Evaluating model...")
    results = trainer.evaluate_model()
    
    print("Creating visualizations...")
    trainer.create_training_plots()
    
    print("Saving model...")
    trainer.save_model()
    
    # Issue 69: No experiment tracking
    # Issue 70: No model deployment preparation
    
    print("Training pipeline completed!")
    
    # Issue 71: No return value for pipeline result
    return results

if __name__ == "__main__":
    # Issue 72: No command line interface
    # Issue 73: No configuration file support
    results = main_training_pipeline()
'''

# AI-Improved versions for before/after comparison
IMPROVED_AUTH_MIDDLEWARE = '''
"""
IMPROVED VERSION - Authentication middleware following security best practices.
This demonstrates the quality of AI-generated recommendations.
"""

import jwt
import logging
import bcrypt
import redis
from functools import wraps
from flask import request, jsonify, current_app
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import re
from contextlib import asynccontextmanager
import secrets

# Configure structured logging
logger = logging.getLogger(__name__)

class AuthError(Enum):
    """Authentication error types for better error handling."""
    INVALID_CREDENTIALS = "invalid_credentials"
    TOKEN_EXPIRED = "token_expired"
    TOKEN_INVALID = "token_invalid"
    RATE_LIMITED = "rate_limited"
    INSUFFICIENT_PERMISSIONS = "insufficient_permissions"

@dataclass
class AuthConfig:
    """Authentication configuration with proper defaults."""
    SECRET_KEY: str
    TOKEN_EXPIRY_MINUTES: int = 60
    MAX_LOGIN_ATTEMPTS: int = 5
    RATE_LIMIT_WINDOW_MINUTES: int = 60
    PASSWORD_MIN_LENGTH: int = 8
    REDIS_URL: str = "redis://localhost:6379"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.SECRET_KEY or len(self.SECRET_KEY) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")

class RateLimiter:
    """Redis-based rate limiter for authentication attempts."""
    
    def __init__(self, redis_client: redis.Redis, config: AuthConfig):
        self.redis = redis_client
        self.config = config
    
    async def is_rate_limited(self, identifier: str) -> bool:
        """Check if identifier is rate limited."""
        key = f"auth_attempts:{identifier}"
        attempts = await self.redis.get(key)
        
        if attempts is None:
            return False
        
        return int(attempts) >= self.config.MAX_LOGIN_ATTEMPTS
    
    async def record_attempt(self, identifier: str, success: bool):
        """Record authentication attempt."""
        key = f"auth_attempts:{identifier}"
        
        if success:
            # Clear attempts on successful login
            await self.redis.delete(key)
        else:
            # Increment failed attempts with TTL
            pipe = self.redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, self.config.RATE_LIMIT_WINDOW_MINUTES * 60)
            await pipe.execute()

class SecureAuthenticator:
    """Secure authentication service with proper error handling."""
    
    def __init__(self, config: AuthConfig, redis_client: redis.Redis):
        self.config = config
        self.redis = redis_client
        self.rate_limiter = RateLimiter(redis_client, config)
        
        # Email validation regex
        self.email_pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )
    
    def _validate_password_strength(self, password: str) -> bool:
        """Validate password meets security requirements."""
        if len(password) < self.config.PASSWORD_MIN_LENGTH:
            return False
        
        # Check for complexity requirements
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password)
        
        return all([has_upper, has_lower, has_digit, has_special])
    
    def _hash_password(self, password: str) -> str:
        """Securely hash password using bcrypt."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def _verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    async def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user with proper security measures."""
        # Input validation
        if not username or not password:
            logger.warning(f"Authentication attempt with empty credentials from {username}")
            return None
        
        # Rate limiting check
        if await self.rate_limiter.is_rate_limited(username):
            logger.warning(f"Rate limited authentication attempt for {username}")
            raise AuthError.RATE_LIMITED
        
        try:
            # Use parameterized query to prevent SQL injection
            user_data = await self._get_user_by_username(username)
            
            if not user_data:
                await self.rate_limiter.record_attempt(username, False)
                return None
            
            # Verify password using secure comparison
            if self._verify_password(password, user_data['password_hash']):
                await self.rate_limiter.record_attempt(username, True)
                logger.info(f"Successful authentication for {username}")
                
                return {
                    'user_id': user_data['id'],
                    'username': user_data['username'],
                    'email': user_data['email'],
                    'permissions': user_data['permissions']
                }
            else:
                await self.rate_limiter.record_attempt(username, False)
                return None
                
        except Exception as e:
            logger.error(f"Authentication error for {username}: {str(e)}")
            await self.rate_limiter.record_attempt(username, False)
            return None
    
    def generate_jwt_token(self, user_data: Dict[str, Any]) -> str:
        """Generate secure JWT token with proper expiration."""
        now = datetime.utcnow()
        payload = {
            'user_id': user_data['user_id'],
            'username': user_data['username'],
            'permissions': user_data['permissions'],
            'iat': now,
            'exp': now + timedelta(minutes=self.config.TOKEN_EXPIRY_MINUTES),
            'jti': secrets.token_urlsafe(32)  # Unique token ID for revocation
        }
        
        token = jwt.encode(payload, self.config.SECRET_KEY, algorithm='HS256')
        
        # Store token in Redis for revocation capability
        token_key = f"token:{payload['jti']}"
        self.redis.setex(
            token_key, 
            self.config.TOKEN_EXPIRY_MINUTES * 60,
            user_data['user_id']
        )
        
        logger.info(f"JWT token generated for user {user_data['username']}")
        return token
    
    async def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token with comprehensive checks."""
        try:
            # Decode token
            payload = jwt.decode(
                token, 
                self.config.SECRET_KEY, 
                algorithms=['HS256']
            )
            
            # Check if token is revoked
            token_key = f"token:{payload['jti']}"
            if not await self.redis.exists(token_key):
                logger.warning(f"Revoked token used by {payload['username']}")
                return None
            
            # Verify user still exists and is active
            user_data = await self._get_user_by_id(payload['user_id'])
            if not user_data or not user_data['active']:
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Expired token used")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {str(e)}")
            return None
    
    async def revoke_token(self, token: str):
        """Revoke a specific token."""
        try:
            payload = jwt.decode(
                token, 
                self.config.SECRET_KEY, 
                algorithms=['HS256'],
                options={"verify_exp": False}  # Don't verify expiration for revocation
            )
            
            token_key = f"token:{payload['jti']}"
            await self.redis.delete(token_key)
            
            logger.info(f"Token revoked for user {payload['username']}")
            
        except jwt.InvalidTokenError:
            pass  # Invalid tokens don't need revocation

def require_auth(permissions: Optional[List[str]] = None):
    """Enhanced authentication decorator with permission checking."""
    def decorator(f):
        @wraps(f)
        async def decorated_function(*args, **kwargs):
            auth_header = request.headers.get('Authorization')
            
            if not auth_header or not auth_header.startswith('Bearer '):
                return jsonify({'error': 'Missing or invalid authorization header'}), 401
            
            try:
                token = auth_header.split(' ')[1]
                authenticator = current_app.authenticator
                
                user_data = await authenticator.validate_token(token)
                if not user_data:
                    return jsonify({'error': 'Invalid or expired token'}), 401
                
                # Check permissions if specified
                if permissions:
                    user_permissions = user_data.get('permissions', [])
                    if not any(perm in user_permissions for perm in permissions):
                        return jsonify({'error': 'Insufficient permissions'}), 403
                
                # Add user data to request context
                request.current_user = user_data
                
                return await f(*args, **kwargs)
                
            except Exception as e:
                logger.error(f"Authentication error: {str(e)}")
                return jsonify({'error': 'Authentication failed'}), 401
        
        return decorated_function
    return decorator
'''

print("✅ Demo Scenario 1: Live Code Analysis - Created comprehensive code samples with 70+ improvement opportunities")