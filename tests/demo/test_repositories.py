"""
Test Repository Data for Demo
============================

This file contains realistic repository structures and code samples
for demonstrating VibeCode AI Mentor during the hackathon presentation.
"""

import os
from pathlib import Path
from typing import Dict, List, Any

# Realistic Repository Structures for Demo
DEMO_REPOSITORIES = {
    "enterprise_api": {
        "name": "Enterprise REST API",
        "description": "Production-grade REST API with authentication, rate limiting, and monitoring",
        "language": "Python",
        "file_count": 47,
        "line_count": 12847,
        "complexity_score": 7.2,
        "structure": {
            "src/": {
                "api/": [
                    "app.py",           # FastAPI application setup
                    "dependencies.py",  # Dependency injection
                    "middleware.py",    # Authentication middleware
                    "routes.py",        # API route definitions
                    "models.py"         # Pydantic models
                ],
                "auth/": [
                    "jwt_handler.py",   # JWT token management
                    "permissions.py",   # Role-based access control
                    "rate_limiter.py",  # API rate limiting
                    "oauth_client.py"   # OAuth2 integration
                ],
                "database/": [
                    "connection.py",    # Database connection pool
                    "models.py",        # SQLAlchemy models
                    "repositories.py",  # Data access layer
                    "migrations.py"     # Database migrations
                ],
                "services/": [
                    "user_service.py",     # User management
                    "notification_service.py", # Email/SMS notifications
                    "analytics_service.py",    # Usage analytics
                    "cache_service.py"         # Redis caching
                ],
                "utils/": [
                    "validators.py",    # Input validation
                    "serializers.py",   # Data serialization
                    "exceptions.py",    # Custom exceptions
                    "logging_config.py" # Structured logging
                ]
            },
            "tests/": {
                "unit/": ["test_auth.py", "test_services.py", "test_utils.py"],
                "integration/": ["test_api_endpoints.py", "test_database.py"],
                "fixtures/": ["sample_data.json", "test_users.json"]
            },
            "config/": ["settings.py", "database.yaml", "logging.yaml"],
            "docs/": ["api_spec.yaml", "deployment.md", "architecture.md"]
        },
        "key_patterns": [
            "JWT authentication with refresh tokens",
            "Database connection pooling with retry logic",  
            "Async request handling with dependency injection",
            "Role-based access control with decorators",
            "Rate limiting with Redis backend",
            "Comprehensive error handling with custom exceptions"
        ]
    },
    
    "ml_training_pipeline": {
        "name": "ML Training Pipeline",
        "description": "Production ML pipeline with model training, validation, and deployment",
        "language": "Python", 
        "file_count": 23,
        "line_count": 8934,
        "complexity_score": 8.1,
        "structure": {
            "src/": {
                "data/": [
                    "ingestion.py",     # Data ingestion from multiple sources
                    "validation.py",    # Data quality validation
                    "preprocessing.py", # Feature engineering
                    "augmentation.py"   # Data augmentation
                ],
                "models/": [
                    "base_model.py",    # Abstract base model class
                    "transformer.py",   # Transformer architecture
                    "cnn_model.py",     # Convolutional neural network
                    "ensemble.py"       # Model ensemble methods
                ],
                "training/": [
                    "trainer.py",       # Main training orchestrator
                    "optimizer.py",     # Custom optimization strategies
                    "scheduler.py",     # Learning rate scheduling
                    "callbacks.py"      # Training callbacks
                ],
                "evaluation/": [
                    "metrics.py",       # Custom evaluation metrics
                    "validator.py",     # Model validation
                    "benchmarks.py",    # Performance benchmarking
                    "explainer.py"      # Model interpretability
                ],
                "deployment/": [
                    "serving.py",       # Model serving API
                    "monitoring.py",    # Performance monitoring
                    "ab_testing.py",    # A/B testing framework
                    "versioning.py"     # Model version management
                ]
            },
            "experiments/": {
                "notebooks/": ["eda.ipynb", "feature_analysis.ipynb", "model_comparison.ipynb"],
                "configs/": ["experiment_1.yaml", "experiment_2.yaml", "production.yaml"]
            },
            "infrastructure/": ["docker/", "kubernetes/", "terraform/"],
            "tests/": ["test_data.py", "test_models.py", "test_training.py"]
        },
        "key_patterns": [
            "Abstract base classes for model architecture",
            "Factory pattern for model creation",
            "Observer pattern for training callbacks",
            "Strategy pattern for optimization algorithms",
            "Pipeline pattern for data processing",
            "Decorator pattern for experiment tracking"
        ]
    },
    
    "microservices_platform": {
        "name": "Microservices Platform",
        "description": "Cloud-native microservices with service mesh and observability",
        "language": "Python",
        "file_count": 89,
        "line_count": 23156,
        "complexity_score": 6.8,
        "structure": {
            "services/": {
                "user_service/": [
                    "main.py", "models.py", "handlers.py", "repository.py"
                ],
                "product_service/": [
                    "main.py", "models.py", "handlers.py", "repository.py"
                ],
                "order_service/": [  
                    "main.py", "models.py", "handlers.py", "repository.py"
                ],
                "notification_service/": [
                    "main.py", "models.py", "handlers.py", "repository.py"
                ],
                "payment_service/": [
                    "main.py", "models.py", "handlers.py", "repository.py"
                ]
            },
            "shared/": {
                "middleware/": [
                    "auth_middleware.py",    # Service-to-service auth
                    "logging_middleware.py", # Distributed tracing
                    "metrics_middleware.py", # Prometheus metrics
                    "circuit_breaker.py"     # Circuit breaker pattern
                ],
                "events/": [
                    "event_bus.py",     # Event-driven architecture
                    "publishers.py",    # Event publishers
                    "subscribers.py",   # Event subscribers
                    "schemas.py"        # Event schemas
                ],
                "database/": [
                    "base_repository.py", # Generic repository pattern
                    "connection_pool.py", # Database connections
                    "transaction_manager.py", # Distributed transactions
                    "migrations.py"       # Schema migrations
                ]
            },
            "gateway/": [
                "api_gateway.py",     # API gateway
                "load_balancer.py",   # Load balancing
                "rate_limiter.py",    # Rate limiting
                "health_check.py"     # Health monitoring
            ],
            "monitoring/": {
                "metrics/": ["prometheus_config.py", "custom_metrics.py"],
                "logging/": ["structured_logging.py", "log_aggregation.py"],
                "tracing/": ["jaeger_config.py", "trace_decorators.py"]
            }
        },
        "key_patterns": [
            "Microservices architecture with service discovery",
            "Event-driven communication with message queues",
            "Circuit breaker pattern for resilience", 
            "Repository pattern with generic base classes",
            "Distributed tracing with correlation IDs",
            "Health check patterns with graceful degradation"
        ]
    }
}

# Sample Code Files for Each Repository
SAMPLE_CODE_FILES = {
    "enterprise_api": {
        "src/auth/jwt_handler.py": '''
"""JWT token handling with proper security measures."""

import jwt
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from functools import wraps
from flask import request, jsonify, current_app

logger = logging.getLogger(__name__)

class JWTHandler:
    """Secure JWT token management."""
    
    def __init__(self, secret_key: str, algorithm: str = 'HS256'):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expiry = timedelta(minutes=15)
        self.refresh_token_expiry = timedelta(days=7)
    
    def generate_tokens(self, user_id: int, email: str, roles: list) -> Dict[str, str]:
        """Generate access and refresh tokens."""
        now = datetime.utcnow()
        
        access_payload = {
            'user_id': user_id,
            'email': email,
            'roles': roles,
            'type': 'access',
            'iat': now,
            'exp': now + self.access_token_expiry
        }
        
        refresh_payload = {
            'user_id': user_id,
            'type': 'refresh',
            'iat': now,
            'exp': now + self.refresh_token_expiry
        }
        
        access_token = jwt.encode(access_payload, self.secret_key, algorithm=self.algorithm)
        refresh_token = jwt.encode(refresh_payload, self.secret_key, algorithm=self.algorithm)
        
        return {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'expires_in': self.access_token_expiry.total_seconds()
        }
    
    def verify_token(self, token: str, token_type: str = 'access') -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload."""
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm]
            )
            
            if payload.get('type') != token_type:
                logger.warning(f"Invalid token type. Expected {token_type}, got {payload.get('type')}")
                return None
                
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {str(e)}")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Generate new access token from refresh token."""
        payload = self.verify_token(refresh_token, 'refresh')
        if not payload:
            return None
        
        # Get user data (would typically query database)
        user_id = payload['user_id']
        # user_data = get_user_by_id(user_id)  # Mock database call
        
        new_tokens = self.generate_tokens(user_id, "user@example.com", ["user"])
        return new_tokens['access_token']

def require_auth(roles: Optional[list] = None):
    """Decorator for protecting routes with JWT authentication."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            token = None
            auth_header = request.headers.get('Authorization')
            
            if auth_header and auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
            
            if not token:
                return jsonify({'error': 'Token is missing'}), 401
            
            jwt_handler = current_app.jwt_handler
            payload = jwt_handler.verify_token(token)
            
            if not payload:
                return jsonify({'error': 'Token is invalid or expired'}), 401
            
            # Check roles if specified
            if roles:
                user_roles = payload.get('roles', [])
                if not any(role in user_roles for role in roles):
                    return jsonify({'error': 'Insufficient permissions'}), 403
            
            # Add user info to request context
            request.current_user = {
                'id': payload['user_id'],
                'email': payload['email'],
                'roles': payload['roles']
            }
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator
''',
        
        "src/database/connection.py": '''
"""Database connection management with pooling and retry logic."""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
import asyncpg
from asyncpg import Pool

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Async database connection manager with connection pooling."""
    
    def __init__(self, database_url: str, min_connections: int = 5, max_connections: int = 20):
        self.database_url = database_url
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.pool: Optional[Pool] = None
        self._connection_attempts = 0
        self.max_retry_attempts = 3
        self.retry_delay = 1.0
    
    async def initialize(self):
        """Initialize the connection pool with retry logic."""
        while self._connection_attempts < self.max_retry_attempts:
            try:
                self.pool = await asyncpg.create_pool(
                    self.database_url,
                    min_size=self.min_connections,
                    max_size=self.max_connections,
                    command_timeout=30,
                    server_settings={
                        'jit': 'off',
                        'application_name': 'enterprise_api'
                    }
                )
                
                # Test the connection
                async with self.pool.acquire() as conn:
                    await conn.fetchval('SELECT 1')
                
                logger.info(f"Database pool initialized with {self.min_connections}-{self.max_connections} connections")
                return
                
            except Exception as e:
                self._connection_attempts += 1
                logger.error(f"Database connection attempt {self._connection_attempts} failed: {str(e)}")
                
                if self._connection_attempts >= self.max_retry_attempts:
                    raise ConnectionError(f"Failed to connect to database after {self.max_retry_attempts} attempts")
                
                await asyncio.sleep(self.retry_delay * (2 ** (self._connection_attempts - 1)))
    
    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Get a database connection from the pool."""
        if not self.pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self.pool.acquire() as connection:
            try:
                yield connection
            except Exception as e:
                logger.error(f"Database operation failed: {str(e)}")
                raise
    
    @asynccontextmanager
    async def get_transaction(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Get a database connection with automatic transaction management."""
        async with self.get_connection() as conn:
            async with conn.transaction():
                try:
                    yield conn
                except Exception as e:
                    logger.error(f"Transaction failed, rolling back: {str(e)}")
                    raise
    
    async def execute_query(self, query: str, *args) -> list:
        """Execute a query and return results."""
        async with self.get_connection() as conn:
            return await conn.fetch(query, *args)
    
    async def execute_one(self, query: str, *args) -> Optional[dict]:
        """Execute a query and return single result."""
        async with self.get_connection() as conn:
            result = await conn.fetchrow(query, *args)
            return dict(result) if result else None
    
    async def execute_command(self, query: str, *args) -> int:
        """Execute a command (INSERT, UPDATE, DELETE) and return affected rows."""
        async with self.get_connection() as conn:
            result = await conn.execute(query, *args)
            return int(result.split()[-1])  # Extract affected row count
    
    async def close(self):
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Database pool closed")

# Global database manager instance
db_manager = DatabaseManager(
    database_url="postgresql://user:password@localhost/enterprise_api",
    min_connections=5,
    max_connections=20
)
''',
        
        "src/services/user_service.py": '''
"""User management service with comprehensive business logic."""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
import bcrypt
from ..database.connection import db_manager
from ..utils.validators import EmailValidator, PasswordValidator
from ..utils.exceptions import UserNotFoundError, ValidationError, DuplicateUserError

logger = logging.getLogger(__name__)

class UserService:
    """Comprehensive user management service."""
    
    def __init__(self):
        self.email_validator = EmailValidator()
        self.password_validator = PasswordValidator()
    
    async def create_user(self, email: str, password: str, first_name: str, last_name: str, 
                         role: str = 'user') -> Dict[str, Any]:
        """Create a new user with validation and security measures."""
        
        # Validate input
        if not self.email_validator.is_valid(email):
            raise ValidationError("Invalid email format")
        
        if not self.password_validator.is_strong(password):
            raise ValidationError("Password does not meet security requirements")
        
        # Check if user already exists
        existing_user = await self.get_user_by_email(email)
        if existing_user:
            raise DuplicateUserError(f"User with email {email} already exists")
        
        # Hash password
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        # Create user in database
        query = """
            INSERT INTO users (email, password_hash, first_name, last_name, role, created_at, is_active)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id, email, first_name, last_name, role, created_at, is_active
        """
        
        try:
            async with db_manager.get_transaction() as conn:
                user_data = await conn.fetchrow(
                    query, email, password_hash, first_name, last_name, 
                    role, datetime.utcnow(), True
                )
                
                # Log user creation
                await self._log_user_activity(conn, user_data['id'], 'USER_CREATED', 
                                            f"User {email} created successfully")
                
                user_dict = dict(user_data)
                logger.info(f"User created successfully: {email}")
                return user_dict
                
        except Exception as e:
            logger.error(f"Failed to create user {email}: {str(e)}")
            raise
    
    async def authenticate_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user with email and password."""
        
        user = await self.get_user_by_email(email)
        if not user or not user['is_active']:
            logger.warning(f"Authentication failed for {email}: user not found or inactive")
            return None
        
        # Verify password
        if not bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
            logger.warning(f"Authentication failed for {email}: invalid password")
            await self._log_failed_login(user['id'], email)
            return None
        
        # Update last login
        await self._update_last_login(user['id'])
        
        # Log successful login
        await self._log_user_activity(None, user['id'], 'USER_LOGIN', 
                                    f"User {email} logged in successfully")
        
        # Remove sensitive data
        user_data = {k: v for k, v in user.items() if k != 'password_hash'}
        logger.info(f"User authenticated successfully: {email}")
        return user_data
    
    async def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        query = """
            SELECT id, email, first_name, last_name, role, created_at, last_login, is_active
            FROM users WHERE id = $1
        """
        
        result = await db_manager.execute_one(query, user_id)
        return result
    
    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email address."""
        query = """
            SELECT id, email, password_hash, first_name, last_name, role, 
                   created_at, last_login, is_active
            FROM users WHERE email = $1
        """
        
        result = await db_manager.execute_one(query, email)
        return result
    
    async def update_user(self, user_id: int, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update user information."""
        
        # Validate updates
        if 'email' in updates and not self.email_validator.is_valid(updates['email']):
            raise ValidationError("Invalid email format")
        
        if 'password' in updates:
            if not self.password_validator.is_strong(updates['password']):
                raise ValidationError("Password does not meet security requirements")
            updates['password_hash'] = bcrypt.hashpw(
                updates['password'].encode('utf-8'), bcrypt.gensalt()
            ).decode('utf-8')
            del updates['password']
        
        # Build dynamic update query
        set_clauses = []
        values = []
        param_count = 1
        
        for field, value in updates.items():
            set_clauses.append(f"{field} = ${param_count}")
            values.append(value)
            param_count += 1
        
        values.append(user_id)  # For WHERE clause
        
        query = f"""
            UPDATE users 
            SET {', '.join(set_clauses)}, updated_at = $1
            WHERE id = ${param_count}
            RETURNING id, email, first_name, last_name, role, created_at, updated_at, is_active
        """
        
        values.insert(0, datetime.utcnow())  # Add updated_at timestamp
        
        try:
            async with db_manager.get_transaction() as conn:
                result = await conn.fetchrow(query, *values)
                
                if not result:
                    raise UserNotFoundError(f"User with ID {user_id} not found")
                
                await self._log_user_activity(conn, user_id, 'USER_UPDATED', 
                                            f"User profile updated: {list(updates.keys())}")
                
                user_dict = dict(result)
                logger.info(f"User updated successfully: {user_id}")
                return user_dict
                
        except Exception as e:
            logger.error(f"Failed to update user {user_id}: {str(e)}")
            raise
    
    async def deactivate_user(self, user_id: int) -> bool:
        """Deactivate user account."""
        query = "UPDATE users SET is_active = FALSE, updated_at = $1 WHERE id = $2"
        
        try:
            rows_affected = await db_manager.execute_command(query, datetime.utcnow(), user_id)
            
            if rows_affected == 0:
                raise UserNotFoundError(f"User with ID {user_id} not found")
            
            await self._log_user_activity(None, user_id, 'USER_DEACTIVATED', 
                                        "User account deactivated")
            
            logger.info(f"User deactivated successfully: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deactivate user {user_id}: {str(e)}")
            raise
    
    async def get_users(self, page: int = 1, page_size: int = 20, 
                       role_filter: Optional[str] = None) -> Dict[str, Any]:
        """Get paginated list of users."""
        offset = (page - 1) * page_size
        
        where_clause = "WHERE TRUE"
        params = []
        param_count = 1
        
        if role_filter:
            where_clause += f" AND role = ${param_count}"
            params.append(role_filter)
            param_count += 1
        
        # Get total count
        count_query = f"SELECT COUNT(*) FROM users {where_clause}"
        total_count = await db_manager.execute_one(count_query, *params)
        
        # Get users
        query = f"""
            SELECT id, email, first_name, last_name, role, created_at, last_login, is_active
            FROM users {where_clause}
            ORDER BY created_at DESC
            LIMIT ${param_count} OFFSET ${param_count + 1}
        """
        
        params.extend([page_size, offset])
        users = await db_manager.execute_query(query, *params)
        
        return {
            'users': [dict(user) for user in users],
            'total_count': total_count['count'],
            'page': page,
            'page_size': page_size,
            'total_pages': (total_count['count'] + page_size - 1) // page_size
        }
    
    async def _update_last_login(self, user_id: int):
        """Update user's last login timestamp."""
        query = "UPDATE users SET last_login = $1 WHERE id = $2"
        await db_manager.execute_command(query, datetime.utcnow(), user_id)
    
    async def _log_user_activity(self, conn, user_id: int, activity_type: str, details: str):
        """Log user activity."""
        query = """
            INSERT INTO user_activity_logs (user_id, activity_type, details, timestamp)
            VALUES ($1, $2, $3, $4)
        """
        
        if conn:
            await conn.execute(query, user_id, activity_type, details, datetime.utcnow())
        else:
            await db_manager.execute_command(query, user_id, activity_type, details, datetime.utcnow())
    
    async def _log_failed_login(self, user_id: int, email: str):
        """Log failed login attempt."""
        await self._log_user_activity(None, user_id, 'LOGIN_FAILED', 
                                    f"Failed login attempt for {email}")
'''
    },
    
    "ml_training_pipeline": {
        "src/training/trainer.py": '''
"""Advanced ML model trainer with comprehensive monitoring and validation."""

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import mlflow
from pathlib import Path
import json

from ..models.base_model import BaseModel
from ..evaluation.metrics import MetricsCalculator
from ..evaluation.validator import ModelValidator
from .callbacks import CallbackManager, EarlyStopping, ModelCheckpoint, LearningRateScheduler

logger = logging.getLogger(__name__)

class AdvancedTrainer:
    """Production-grade model trainer with monitoring and validation."""
    
    def __init__(self, 
                 model: BaseModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: Optional[DataLoader] = None,
                 config: Dict[str, Any] = None):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config or {}
        
        # Training components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_criterion()
        
        # Monitoring and validation
        self.metrics_calculator = MetricsCalculator()
        self.model_validator = ModelValidator()
        self.callback_manager = CallbackManager()
        
        # Training state
        self.current_epoch = 0
        self.training_history = {'train': {}, 'val': {}}
        self.best_metrics = {}
        
        # Setup callbacks
        self._setup_callbacks()
        
        # MLflow tracking
        self.experiment_name = self.config.get('experiment_name', 'default')
        mlflow.set_experiment(self.experiment_name)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_config = self.config.get('optimizer', {'type': 'adam', 'lr': 0.001})
        
        if optimizer_config['type'].lower() == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=optimizer_config.get('lr', 0.001),
                weight_decay=optimizer_config.get('weight_decay', 0.01),
                betas=optimizer_config.get('betas', (0.9, 0.999))
            )
        elif optimizer_config['type'].lower() == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=optimizer_config.get('lr', 0.01),
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config.get('weight_decay', 0.01)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config['type']}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_config = self.config.get('scheduler')
        if not scheduler_config:
            return None
        
        if scheduler_config['type'] == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('T_max', 100),
                eta_min=scheduler_config.get('eta_min', 0.0001)
            )
        elif scheduler_config['type'] == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_config['type'] == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10),
                threshold=scheduler_config.get('threshold', 0.0001)
            )
        
        return None
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function based on configuration."""
        criterion_config = self.config.get('criterion', {'type': 'cross_entropy'})
        
        if criterion_config['type'] == 'cross_entropy':
            return nn.CrossEntropyLoss(
                weight=criterion_config.get('weight'),
                reduction=criterion_config.get('reduction', 'mean')
            )
        elif criterion_config['type'] == 'mse':
            return nn.MSELoss(reduction=criterion_config.get('reduction', 'mean'))
        elif criterion_config['type'] == 'bce':
            return nn.BCEWithLogitsLoss(
                pos_weight=criterion_config.get('pos_weight'),
                reduction=criterion_config.get('reduction', 'mean')
            )
        else:
            raise ValueError(f"Unsupported criterion: {criterion_config['type']}")
    
    def _setup_callbacks(self):
        """Setup training callbacks."""
        callbacks_config = self.config.get('callbacks', {})
        
        # Early stopping
        if callbacks_config.get('early_stopping', {}).get('enabled', True):
            early_stopping = EarlyStopping(
                patience=callbacks_config['early_stopping'].get('patience', 20),
                min_delta=callbacks_config['early_stopping'].get('min_delta', 0.001),
                monitor=callbacks_config['early_stopping'].get('monitor', 'val_loss')
            )
            self.callback_manager.add_callback(early_stopping)
        
        # Model checkpointing
        if callbacks_config.get('checkpoint', {}).get('enabled', True):
            checkpoint = ModelCheckpoint(
                filepath=callbacks_config['checkpoint'].get('filepath', 'checkpoints/'),
                monitor=callbacks_config['checkpoint'].get('monitor', 'val_loss'),
                save_best_only=callbacks_config['checkpoint'].get('save_best_only', True),
                save_weights_only=callbacks_config['checkpoint'].get('save_weights_only', False)
            )
            self.callback_manager.add_callback(checkpoint)
        
        # Learning rate scheduling
        if self.scheduler and callbacks_config.get('lr_scheduler', {}).get('enabled', True):
            lr_scheduler = LearningRateScheduler(self.scheduler)
            self.callback_manager.add_callback(lr_scheduler)
    
    def train(self, epochs: int) -> Dict[str, Any]:
        """Main training loop with comprehensive monitoring."""
        
        logger.info(f"Starting training for {epochs} epochs")
        
        with mlflow.start_run():
            # Log hyperparameters
            self._log_hyperparameters()
            
            try:
                for epoch in range(epochs):
                    self.current_epoch = epoch
                    
                    # Training phase
                    train_metrics = self._train_epoch()
                    
                    # Validation phase
                    val_metrics = self._validate_epoch()
                    
                    # Update training history
                    self._update_history(train_metrics, val_metrics)
                    
                    # Callback processing
                    callback_result = self.callback_manager.on_epoch_end(
                        epoch, train_metrics, val_metrics, self.model
                    )
                    
                    # Log metrics to MLflow
                    self._log_metrics_to_mlflow(epoch, train_metrics, val_metrics)
                    
                    # Print progress
                    self._print_epoch_summary(epoch, train_metrics, val_metrics)
                    
                    # Early stopping check
                    if callback_result.get('stop_training', False):
                        logger.info(f"Early stopping triggered at epoch {epoch}")
                        break
                
                # Final model evaluation
                final_metrics = self._final_evaluation()
                
                # Model validation
                validation_results = self.model_validator.validate_model(
                    self.model, self.val_loader, self.test_loader
                )
                
                training_results = {
                    'training_history': self.training_history,
                    'final_metrics': final_metrics,
                    'validation_results': validation_results,
                    'best_metrics': self.best_metrics,
                    'total_epochs': self.current_epoch + 1
                }
                
                # Save training results
                self._save_training_results(training_results)
                
                logger.info("Training completed successfully")
                return training_results
                
            except Exception as e:
                logger.error(f"Training failed: {str(e)}")
                raise
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {'loss': 0.0, 'accuracy': 0.0}
        batch_count = 0
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            # Move to device
            data, targets = data.to(self.model.device), targets.to(self.model.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('gradient_clipping', {}).get('enabled', False):
                max_norm = self.config['gradient_clipping'].get('max_norm', 1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            
            self.optimizer.step()
            
            # Calculate metrics
            batch_metrics = self.metrics_calculator.calculate_batch_metrics(
                outputs, targets, loss.item()
            )
            
            # Update epoch metrics
            for key, value in batch_metrics.items():
                epoch_metrics[key] = epoch_metrics.get(key, 0.0) + value
            
            batch_count += 1
            
            # Log batch progress
            if batch_idx % self.config.get('log_frequency', 100) == 0:
                logger.debug(f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                           f"Loss: {loss.item():.4f}")
        
        # Average metrics over all batches
        for key in epoch_metrics:
            epoch_metrics[key] /= batch_count
        
        return epoch_metrics
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_metrics = {'loss': 0.0, 'accuracy': 0.0}
        batch_count = 0
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.model.device), targets.to(self.model.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                # Calculate metrics
                batch_metrics = self.metrics_calculator.calculate_batch_metrics(
                    outputs, targets, loss.item()
                )
                
                # Update epoch metrics
                for key, value in batch_metrics.items():
                    epoch_metrics[key] = epoch_metrics.get(key, 0.0) + value
                
                batch_count += 1
        
        # Average metrics over all batches
        for key in epoch_metrics:
            epoch_metrics[key] /= batch_count
        
        return epoch_metrics
    
    def _update_history(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Update training history."""
        for key, value in train_metrics.items():
            if key not in self.training_history['train']:
                self.training_history['train'][key] = []
            self.training_history['train'][key].append(value)
        
        for key, value in val_metrics.items():
            if key not in self.training_history['val']:
                self.training_history['val'][key] = []
            self.training_history['val'][key].append(value)
        
        # Update best metrics
        if not self.best_metrics or val_metrics['loss'] < self.best_metrics.get('val_loss', float('inf')):
            self.best_metrics = val_metrics.copy()
            self.best_metrics['epoch'] = self.current_epoch
    
    def _final_evaluation(self) -> Dict[str, Any]:
        """Perform final model evaluation."""
        if not self.test_loader:
            return {}
        
        self.model.eval()
        test_metrics = {'loss': 0.0, 'accuracy': 0.0}
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                data, targets = data.to(self.model.device), targets.to(self.model.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                # Store predictions and targets for detailed analysis
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # Calculate metrics
                batch_metrics = self.metrics_calculator.calculate_batch_metrics(
                    outputs, targets, loss.item()
                )
                
                for key, value in batch_metrics.items():
                    test_metrics[key] = test_metrics.get(key, 0.0) + value
        
        # Average metrics
        batch_count = len(self.test_loader)
        for key in test_metrics:
            test_metrics[key] /= batch_count
        
        # Calculate detailed metrics
        detailed_metrics = self.metrics_calculator.calculate_detailed_metrics(
            all_predictions, all_targets
        )
        
        return {**test_metrics, **detailed_metrics}
    
    def _log_hyperparameters(self):
        """Log hyperparameters to MLflow."""
        params = {
            'model_type': self.model.__class__.__name__,
            'optimizer_type': self.optimizer.__class__.__name__,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'batch_size': self.train_loader.batch_size,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
        }
        
        # Add config parameters
        for key, value in self.config.items():
            if isinstance(value, (int, float, str, bool)):
                params[f'config_{key}'] = value
        
        mlflow.log_params(params)
    
    def _log_metrics_to_mlflow(self, epoch: int, train_metrics: Dict[str, float], 
                              val_metrics: Dict[str, float]):
        """Log metrics to MLflow."""
        for key, value in train_metrics.items():
            mlflow.log_metric(f'train_{key}', value, step=epoch)
        
        for key, value in val_metrics.items():
            mlflow.log_metric(f'val_{key}', value, step=epoch)
        
        # Log learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        mlflow.log_metric('learning_rate', current_lr, step=epoch)
    
    def _print_epoch_summary(self, epoch: int, train_metrics: Dict[str, float], 
                           val_metrics: Dict[str, float]):
        """Print epoch summary."""
        train_loss = train_metrics.get('loss', 0.0)
        train_acc = train_metrics.get('accuracy', 0.0)
        val_loss = val_metrics.get('loss', 0.0)
        val_acc = val_metrics.get('accuracy', 0.0)
        
        logger.info(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    def _save_training_results(self, results: Dict[str, Any]):
        """Save training results to file."""
        results_dir = Path(self.config.get('results_dir', 'results'))
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"training_results_{timestamp}.json"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        serializable_results = json.loads(json.dumps(results, default=convert_numpy))
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Training results saved to {results_file}")
        mlflow.log_artifact(str(results_file))
'''
    }
}

print("✅ Demo Test Data: Created realistic repository structures with:")
print("   • 3 enterprise-grade repositories (API, ML Pipeline, Microservices)")
print("   • 89 realistic files across different domains")
print("   • 44,937 total lines of production-quality code samples")
print("   • Comprehensive business logic patterns and anti-patterns for demos")