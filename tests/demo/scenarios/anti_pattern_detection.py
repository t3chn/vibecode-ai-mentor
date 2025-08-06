"""
Anti-Pattern Detection Demo Scenarios
====================================

This file contains carefully crafted anti-patterns and their improved versions
for demonstrating VibeCode AI Mentor's pattern recognition and recommendation capabilities.
"""

# Anti-Pattern 1: God Object/Class
GOD_CLASS_ANTIPATTERN = '''
"""
God Class Anti-Pattern: A single class that does everything.
Perfect for demonstrating architectural improvement recommendations.
"""

import json
import sqlite3
import smtplib
import logging
import hashlib
import datetime
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any, Optional
import os
import csv
import xml.etree.ElementTree as ET

class UserManagementSystem:
    """God class that handles everything - ANTI-PATTERN for demo."""
    
    def __init__(self):
        # Database connection
        self.db_connection = sqlite3.connect('users.db')
        self.cursor = self.db_connection.cursor()
        
        # Email configuration
        self.smtp_server = 'smtp.gmail.com'
        self.smtp_port = 587
        self.email_username = 'system@example.com'
        self.email_password = 'password123'
        
        # Logging setup
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # API configuration
        self.api_base_url = 'https://api.external-service.com'
        self.api_key = 'secret-api-key-123'
        
        # File paths
        self.user_data_file = 'user_data.json'
        self.reports_directory = 'reports/'
        
        # Initialize database tables
        self.init_database()
    
    def init_database(self):
        """Initialize database tables."""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE,
                email TEXT,
                password_hash TEXT,
                created_at TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN,
                role TEXT,
                profile_data TEXT
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                session_token TEXT,
                created_at TIMESTAMP,
                expires_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                action TEXT,
                timestamp TIMESTAMP,
                details TEXT
            )
        ''')
        
        self.db_connection.commit()
    
    def create_user(self, username: str, email: str, password: str, role: str = 'user') -> bool:
        """Create a new user."""
        try:
            # Validate email format
            if '@' not in email or '.' not in email.split('@')[1]:
                self.logger.error(f"Invalid email format: {email}")
                return False
            
            # Validate password strength
            if len(password) < 8:
                self.logger.error("Password too short")
                return False
            
            # Hash password
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            # Check if user exists
            self.cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
            if self.cursor.fetchone():
                self.logger.error(f"User already exists: {username}")
                return False
            
            # Insert user
            created_at = datetime.datetime.now()
            self.cursor.execute('''
                INSERT INTO users (username, email, password_hash, created_at, is_active, role)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (username, email, password_hash, created_at, True, role))
            
            user_id = self.cursor.lastrowid
            self.db_connection.commit()
            
            # Send welcome email
            self.send_welcome_email(email, username)
            
            # Log audit event
            self.log_audit_event(user_id, 'USER_CREATED', f'User {username} created')
            
            # Sync with external service
            self.sync_user_with_external_service(user_id, username, email)
            
            # Generate initial report
            self.generate_user_report(user_id)
            
            self.logger.info(f"User created successfully: {username}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating user: {str(e)}")
            self.db_connection.rollback()
            return False
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user and create session."""
        try:
            # Hash provided password
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            # Find user
            self.cursor.execute('''
                SELECT id, username, email, password_hash, is_active, role, last_login
                FROM users WHERE username = ? AND password_hash = ?
            ''', (username, password_hash))
            
            user_data = self.cursor.fetchone()
            if not user_data:
                self.logger.warning(f"Failed login attempt for: {username}")
                return None
            
            user_id, username, email, stored_hash, is_active, role, last_login = user_data
            
            if not is_active:
                self.logger.warning(f"Inactive user login attempt: {username}")
                return None
            
            # Update last login
            now = datetime.datetime.now()
            self.cursor.execute("UPDATE users SET last_login = ? WHERE id = ?", (now, user_id))
            
            # Create session
            session_token = hashlib.md5(f"{user_id}{now}".encode()).hexdigest()
            expires_at = now + datetime.timedelta(hours=24)
            
            self.cursor.execute('''
                INSERT INTO user_sessions (user_id, session_token, created_at, expires_at)
                VALUES (?, ?, ?, ?)
            ''', (user_id, session_token, now, expires_at))
            
            self.db_connection.commit()
            
            # Log audit event
            self.log_audit_event(user_id, 'USER_LOGIN', f'User {username} logged in')
            
            # Send login notification email
            self.send_login_notification_email(email, username, now)
            
            # Update external service
            self.update_user_last_activity(user_id, now)
            
            # Generate activity report
            self.generate_activity_report(user_id)
            
            return {
                'user_id': user_id,
                'username': username,
                'email': email,
                'role': role,
                'session_token': session_token,
                'expires_at': expires_at
            }
            
        except Exception as e:
            self.logger.error(f"Error authenticating user: {str(e)}")
            return None
    
    def send_welcome_email(self, email: str, username: str):
        """Send welcome email to new user."""
        try:
            smtp = smtplib.SMTP(self.smtp_server, self.smtp_port)
            smtp.starttls()
            smtp.login(self.email_username, self.email_password)
            
            msg = MIMEMultipart()
            msg['From'] = self.email_username
            msg['To'] = email
            msg['Subject'] = 'Welcome to Our System!'
            
            body = f"""
            Dear {username},
            
            Welcome to our system! Your account has been created successfully.
            
            Best regards,
            The System Team
            """
            
            msg.attach(MIMEText(body, 'plain'))
            smtp.send_message(msg)
            smtp.quit()
            
            self.logger.info(f"Welcome email sent to: {email}")
            
        except Exception as e:
            self.logger.error(f"Error sending welcome email: {str(e)}")
    
    def send_login_notification_email(self, email: str, username: str, login_time: datetime.datetime):
        """Send login notification email."""
        try:
            smtp = smtplib.SMTP(self.smtp_server, self.smtp_port)
            smtp.starttls()
            smtp.login(self.email_username, self.email_password)
            
            msg = MIMEMultipart()
            msg['From'] = self.email_username
            msg['To'] = email
            msg['Subject'] = 'Login Notification'
            
            body = f"""
            Dear {username},
            
            You have successfully logged in at {login_time}.
            
            If this wasn't you, please contact support immediately.
            
            Best regards,
            The System Team
            """
            
            msg.attach(MIMEText(body, 'plain'))
            smtp.send_message(msg)
            smtp.quit()
            
        except Exception as e:
            self.logger.error(f"Error sending login notification: {str(e)}")
    
    def log_audit_event(self, user_id: int, action: str, details: str):
        """Log audit event."""
        try:
            timestamp = datetime.datetime.now()
            self.cursor.execute('''
                INSERT INTO audit_logs (user_id, action, timestamp, details)
                VALUES (?, ?, ?, ?)
            ''', (user_id, action, timestamp, details))
            self.db_connection.commit()
            
        except Exception as e:
            self.logger.error(f"Error logging audit event: {str(e)}")
    
    def sync_user_with_external_service(self, user_id: int, username: str, email: str):
        """Sync user data with external service."""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'user_id': user_id,
                'username': username,
                'email': email,
                'sync_timestamp': datetime.datetime.now().isoformat()
            }
            
            response = requests.post(
                f'{self.api_base_url}/users/sync',
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                self.logger.info(f"User synced with external service: {username}")
            else:
                self.logger.error(f"Failed to sync user: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error syncing with external service: {str(e)}")
    
    def update_user_last_activity(self, user_id: int, activity_time: datetime.datetime):
        """Update user's last activity in external service."""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'user_id': user_id,
                'last_activity': activity_time.isoformat()
            }
            
            response = requests.put(
                f'{self.api_base_url}/users/{user_id}/activity',
                headers=headers,
                json=data
            )
            
            if response.status_code != 200:
                self.logger.error(f"Failed to update user activity: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error updating user activity: {str(e)}")
    
    def generate_user_report(self, user_id: int):
        """Generate user report."""
        try:
            # Get user data
            self.cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            user_data = self.cursor.fetchone()
            
            if not user_data:
                return
            
            # Get audit logs
            self.cursor.execute('''
                SELECT action, timestamp, details FROM audit_logs 
                WHERE user_id = ? ORDER BY timestamp DESC LIMIT 10
            ''', (user_id,))
            audit_logs = self.cursor.fetchall()
            
            # Create report
            report = {
                'user_id': user_id,
                'username': user_data[1],
                'email': user_data[2],
                'created_at': user_data[4],
                'last_login': user_data[5],
                'is_active': user_data[6],
                'role': user_data[7],
                'recent_activities': [
                    {'action': log[0], 'timestamp': log[1], 'details': log[2]}
                    for log in audit_logs
                ]
            }
            
            # Save report to file
            os.makedirs(self.reports_directory, exist_ok=True)
            report_file = f"{self.reports_directory}user_{user_id}_report.json"
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"User report generated: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating user report: {str(e)}")
    
    def generate_activity_report(self, user_id: int):
        """Generate user activity report."""
        try:
            self.cursor.execute('''
                SELECT DATE(timestamp) as date, COUNT(*) as activity_count
                FROM audit_logs WHERE user_id = ?
                GROUP BY DATE(timestamp)
                ORDER BY date DESC LIMIT 30
            ''', (user_id,))
            
            activity_data = self.cursor.fetchall()
            
            # Generate CSV report
            csv_file = f"{self.reports_directory}user_{user_id}_activity.csv"
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Date', 'Activity Count'])
                writer.writerows(activity_data)
            
            # Generate XML report
            xml_file = f"{self.reports_directory}user_{user_id}_activity.xml"
            root = ET.Element('ActivityReport')
            root.set('user_id', str(user_id))
            
            for date, count in activity_data:
                activity = ET.SubElement(root, 'Activity')
                activity.set('date', date)
                activity.set('count', str(count))
            
            tree = ET.ElementTree(root)
            tree.write(xml_file, encoding='utf-8', xml_declaration=True)
            
        except Exception as e:
            self.logger.error(f"Error generating activity report: {str(e)}")
    
    def backup_user_data(self):
        """Backup all user data to file."""
        try:
            self.cursor.execute("SELECT * FROM users")
            users = self.cursor.fetchall()
            
            backup_data = {
                'backup_timestamp': datetime.datetime.now().isoformat(),
                'users': [
                    {
                        'id': user[0],
                        'username': user[1],
                        'email': user[2],
                        'created_at': user[4],
                        'last_login': user[5],
                        'is_active': user[6],
                        'role': user[7]
                    }
                    for user in users
                ]
            }
            
            with open(self.user_data_file, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)
            
            self.logger.info("User data backup completed")
            
        except Exception as e:
            self.logger.error(f"Error backing up user data: {str(e)}")
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        try:
            now = datetime.datetime.now()
            self.cursor.execute("DELETE FROM user_sessions WHERE expires_at < ?", (now,))
            deleted_count = self.cursor.rowcount
            self.db_connection.commit()
            
            self.logger.info(f"Cleaned up {deleted_count} expired sessions")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up sessions: {str(e)}")
    
    def close_connection(self):
        """Close database connection."""
        self.db_connection.close()
'''

# Improved Version using SOLID Principles
IMPROVED_USER_SYSTEM = '''
"""
IMPROVED VERSION - Applying SOLID Principles and Separation of Concerns
This demonstrates how the God Class can be refactored into focused, maintainable components.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Protocol
import logging

# Domain Models
@dataclass
class User:
    """User domain model."""
    id: Optional[int] = None
    username: str = ""
    email: str = ""
    password_hash: str = ""
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    is_active: bool = True
    role: str = "user"

@dataclass
class UserSession:
    """User session model."""
    id: Optional[int] = None
    user_id: int = 0
    session_token: str = ""
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

@dataclass
class AuditLog:
    """Audit log model."""
    id: Optional[int] = None
    user_id: int = 0
    action: str = ""
    timestamp: Optional[datetime] = None
    details: str = ""

# Protocols for dependency injection
class UserRepository(Protocol):
    """User repository interface."""
    async def create(self, user: User) -> User: ...
    async def get_by_username(self, username: str) -> Optional[User]: ...
    async def get_by_id(self, user_id: int) -> Optional[User]: ...
    async def update(self, user: User) -> User: ...
    async def delete(self, user_id: int) -> bool: ...

class SessionRepository(Protocol):
    """Session repository interface."""
    async def create(self, session: UserSession) -> UserSession: ...
    async def get_by_token(self, token: str) -> Optional[UserSession]: ...
    async def cleanup_expired(self) -> int: ...

class AuditLogger(Protocol):
    """Audit logging interface."""
    async def log_event(self, user_id: int, action: str, details: str): ...
    async def get_user_logs(self, user_id: int, limit: int = 10) -> List[AuditLog]: ...

class EmailService(Protocol):
    """Email service interface."""
    async def send_welcome_email(self, email: str, username: str): ...
    async def send_login_notification(self, email: str, username: str, login_time: datetime): ...

class ExternalSyncService(Protocol):
    """External synchronization service interface."""
    async def sync_user(self, user: User): ...
    async def update_user_activity(self, user_id: int, activity_time: datetime): ...

# Core Services
class UserService:
    """Core user management service."""
    
    def __init__(
        self,
        user_repo: UserRepository,
        session_repo: SessionRepository,
        audit_logger: AuditLogger,
        email_service: EmailService,
        external_sync: ExternalSyncService,
        password_hasher: 'PasswordHasher',
        validator: 'UserValidator'
    ):
        self.user_repo = user_repo
        self.session_repo = session_repo
        self.audit_logger = audit_logger
        self.email_service = email_service
        self.external_sync = external_sync
        self.password_hasher = password_hasher
        self.validator = validator
        self.logger = logging.getLogger(__name__)
    
    async def create_user(self, username: str, email: str, password: str, role: str = 'user') -> bool:
        """Create a new user with proper validation and side effects."""
        try:
            # Validate input
            if not self.validator.validate_email(email):
                self.logger.error(f"Invalid email format: {email}")
                return False
            
            if not self.validator.validate_password_strength(password):
                self.logger.error("Password does not meet strength requirements")
                return False
            
            # Check if user exists
            if await self.user_repo.get_by_username(username):
                self.logger.error(f"User already exists: {username}")
                return False
            
            # Create user
            password_hash = self.password_hasher.hash_password(password)
            user = User(
                username=username,
                email=email,
                password_hash=password_hash,
                created_at=datetime.now(),
                role=role
            )
            
            created_user = await self.user_repo.create(user)
            
            # Handle side effects asynchronously
            await self._handle_user_creation_side_effects(created_user)
            
            self.logger.info(f"User created successfully: {username}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating user: {str(e)}")
            return False
    
    async def _handle_user_creation_side_effects(self, user: User):
        """Handle side effects of user creation."""
        try:
            # These can be done concurrently
            await asyncio.gather(
                self.email_service.send_welcome_email(user.email, user.username),
                self.audit_logger.log_event(user.id, 'USER_CREATED', f'User {user.username} created'),
                self.external_sync.sync_user(user)
            )
        except Exception as e:
            self.logger.error(f"Error handling user creation side effects: {str(e)}")

class AuthenticationService:
    """Dedicated authentication service."""
    
    def __init__(
        self,
        user_repo: UserRepository,
        session_repo: SessionRepository,
        audit_logger: AuditLogger,
        password_hasher: 'PasswordHasher',
        session_generator: 'SessionTokenGenerator'
    ):
        self.user_repo = user_repo
        self.session_repo = session_repo
        self.audit_logger = audit_logger
        self.password_hasher = password_hasher
        self.session_generator = session_generator
        self.logger = logging.getLogger(__name__)
    
    async def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user and create session."""
        try:
            user = await self.user_repo.get_by_username(username)
            if not user or not user.is_active:
                await self.audit_logger.log_event(0, 'FAILED_LOGIN', f'Failed login for {username}')
                return None
            
            if not self.password_hasher.verify_password(password, user.password_hash):
                await self.audit_logger.log_event(user.id, 'FAILED_LOGIN', f'Invalid password for {username}')
                return None
            
            # Create session
            session = await self._create_user_session(user)
            
            # Update last login
            user.last_login = datetime.now()
            await self.user_repo.update(user)
            
            # Log successful login
            await self.audit_logger.log_event(user.id, 'USER_LOGIN', f'User {username} logged in')
            
            return {
                'user_id': user.id,
                'username': user.username,
                'email': user.email,
                'role': user.role,
                'session_token': session.session_token,
                'expires_at': session.expires_at
            }
            
        except Exception as e:
            self.logger.error(f"Error authenticating user: {str(e)}")
            return None
    
    async def _create_user_session(self, user: User) -> UserSession:
        """Create a new user session."""
        session = UserSession(
            user_id=user.id,
            session_token=self.session_generator.generate_token(),
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24)
        )
        
        return await self.session_repo.create(session)

# Supporting Services
class PasswordHasher:
    """Secure password hashing service."""
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        import bcrypt
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        import bcrypt
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

class UserValidator:
    """User input validation service."""
    
    def validate_email(self, email: str) -> bool:
        """Validate email format."""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def validate_password_strength(self, password: str) -> bool:
        """Validate password meets security requirements."""
        if len(password) < 8:
            return False
        
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password)
        
        return all([has_upper, has_lower, has_digit, has_special])

class SessionTokenGenerator:
    """Session token generation service."""
    
    def generate_token(self) -> str:
        """Generate secure session token."""
        import secrets
        return secrets.token_urlsafe(32)

# Factory for creating the complete system
class UserManagementSystemFactory:
    """Factory for creating a properly configured user management system."""
    
    @staticmethod
    def create_system(
        user_repo: UserRepository,
        session_repo: SessionRepository,
        audit_logger: AuditLogger,
        email_service: EmailService,
        external_sync: ExternalSyncService
    ) -> tuple[UserService, AuthenticationService]:
        """Create a complete user management system."""
        
        password_hasher = PasswordHasher()
        validator = UserValidator()
        session_generator = SessionTokenGenerator()
        
        user_service = UserService(
            user_repo=user_repo,
            session_repo=session_repo,
            audit_logger=audit_logger,
            email_service=email_service,
            external_sync=external_sync,
            password_hasher=password_hasher,
            validator=validator
        )
        
        auth_service = AuthenticationService(
            user_repo=user_repo,
            session_repo=session_repo,
            audit_logger=audit_logger,
            password_hasher=password_hasher,
            session_generator=session_generator
        )
        
        return user_service, auth_service
'''

# Anti-Pattern 2: Copy-Paste Programming
COPY_PASTE_ANTIPATTERN = '''
"""
Copy-Paste Programming Anti-Pattern: Duplicated code with slight variations.
Perfect for demonstrating DRY principle and refactoring recommendations.
"""

import requests
import json
import logging
from datetime import datetime
from typing import Dict, List, Any

class DataProcessingService:
    """Service with extensive code duplication - ANTI-PATTERN for demo."""
    
    def __init__(self):
        self.api_base_url = 'https://api.dataservice.com'
        self.api_key = 'secret-key-123'
        self.logger = logging.getLogger(__name__)
    
    def process_user_data(self, user_ids: List[int]) -> List[Dict]:
        """Process user data with duplicated code."""
        results = []
        
        for user_id in user_ids:
            try:
                # API call setup - DUPLICATED CODE BLOCK 1
                headers = {
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
                
                # Make API request
                response = requests.get(
                    f'{self.api_base_url}/users/{user_id}',
                    headers=headers,
                    timeout=30
                )
                
                # Response validation - DUPLICATED CODE BLOCK 2
                if response.status_code != 200:
                    self.logger.error(f"API error for user {user_id}: {response.status_code}")
                    continue
                
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    self.logger.error(f"Invalid JSON response for user {user_id}")
                    continue
                
                # Data processing - DUPLICATED CODE BLOCK 3
                processed_data = {
                    'id': data.get('id'),
                    'name': data.get('name', '').strip(),
                    'email': data.get('email', '').lower(),
                    'created_at': data.get('created_at'),
                    'updated_at': datetime.now().isoformat(),
                    'status': 'processed'
                }
                
                # Validation - DUPLICATED CODE BLOCK 4
                if not processed_data['id']:
                    self.logger.error(f"Missing ID for user {user_id}")
                    continue
                
                if not processed_data['name']:
                    self.logger.warning(f"Missing name for user {user_id}")
                
                if not processed_data['email']:
                    self.logger.warning(f"Missing email for user {user_id}")
                
                results.append(processed_data)
                
            except requests.RequestException as e:
                self.logger.error(f"Request error for user {user_id}: {str(e)}")
                continue
            except Exception as e:
                self.logger.error(f"Unexpected error for user {user_id}: {str(e)}")
                continue
        
        return results
    
    def process_product_data(self, product_ids: List[int]) -> List[Dict]:
        """Process product data with duplicated code."""
        results = []
        
        for product_id in product_ids:
            try:
                # API call setup - DUPLICATED CODE BLOCK 1 (EXACT COPY)
                headers = {
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
                
                # Make API request
                response = requests.get(
                    f'{self.api_base_url}/products/{product_id}',
                    headers=headers,
                    timeout=30
                )
                
                # Response validation - DUPLICATED CODE BLOCK 2 (EXACT COPY)
                if response.status_code != 200:
                    self.logger.error(f"API error for product {product_id}: {response.status_code}")
                    continue
                
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    self.logger.error(f"Invalid JSON response for product {product_id}")
                    continue
                
                # Data processing - SIMILAR TO BLOCK 3 with minor differences
                processed_data = {
                    'id': data.get('id'),
                    'name': data.get('name', '').strip(),
                    'description': data.get('description', '').strip(),
                    'price': data.get('price', 0),
                    'created_at': data.get('created_at'),
                    'updated_at': datetime.now().isoformat(),
                    'status': 'processed'
                }
                
                # Validation - SIMILAR TO BLOCK 4 with minor differences
                if not processed_data['id']:
                    self.logger.error(f"Missing ID for product {product_id}")
                    continue
                
                if not processed_data['name']:
                    self.logger.warning(f"Missing name for product {product_id}")
                
                if processed_data['price'] < 0:
                    self.logger.warning(f"Invalid price for product {product_id}")
                
                results.append(processed_data)
                
            except requests.RequestException as e:
                self.logger.error(f"Request error for product {product_id}: {str(e)}")
                continue
            except Exception as e:
                self.logger.error(f"Unexpected error for product {product_id}: {str(e)}")
                continue
        
        return results
    
    def process_order_data(self, order_ids: List[int]) -> List[Dict]:
        """Process order data with duplicated code."""
        results = []
        
        for order_id in order_ids:
            try:
                # API call setup - DUPLICATED CODE BLOCK 1 (EXACT COPY AGAIN)
                headers = {
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
                
                # Make API request
                response = requests.get(
                    f'{self.api_base_url}/orders/{order_id}',
                    headers=headers,
                    timeout=30
                )
                
                # Response validation - DUPLICATED CODE BLOCK 2 (EXACT COPY AGAIN)
                if response.status_code != 200:
                    self.logger.error(f"API error for order {order_id}: {response.status_code}")
                    continue
                
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    self.logger.error(f"Invalid JSON response for order {order_id}")
                    continue
                
                # Data processing - SIMILAR TO PREVIOUS BLOCKS
                processed_data = {
                    'id': data.get('id'),
                    'customer_id': data.get('customer_id'),
                    'total_amount': data.get('total_amount', 0),
                    'status': data.get('status', '').upper(),
                    'items': data.get('items', []),
                    'created_at': data.get('created_at'),
                    'updated_at': datetime.now().isoformat(),
                    'processing_status': 'processed'
                }
                
                # Validation - SIMILAR PATTERN REPEATED
                if not processed_data['id']:
                    self.logger.error(f"Missing ID for order {order_id}")
                    continue
                
                if not processed_data['customer_id']:
                    self.logger.warning(f"Missing customer ID for order {order_id}")
                
                if processed_data['total_amount'] <= 0:
                    self.logger.warning(f"Invalid total amount for order {order_id}")
                
                if not processed_data['items']:
                    self.logger.warning(f"No items found for order {order_id}")
                
                results.append(processed_data)
                
            except requests.RequestException as e:
                self.logger.error(f"Request error for order {order_id}: {str(e)}")
                continue
            except Exception as e:
                self.logger.error(f"Unexpected error for order {order_id}: {str(e)}")
                continue
        
        return results
    
    def process_category_data(self, category_ids: List[int]) -> List[Dict]:
        """Process category data with more duplicated code."""
        results = []
        
        for category_id in category_ids:
            try:
                # API call setup - DUPLICATED AGAIN
                headers = {
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
                
                # Make API request
                response = requests.get(
                    f'{self.api_base_url}/categories/{category_id}',
                    headers=headers,
                    timeout=30
                )
                
                # Response validation - DUPLICATED AGAIN
                if response.status_code != 200:
                    self.logger.error(f"API error for category {category_id}: {response.status_code}")
                    continue
                
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    self.logger.error(f"Invalid JSON response for category {category_id}")
                    continue
                
                # Data processing - VARIATION OF SAME PATTERN
                processed_data = {
                    'id': data.get('id'),
                    'name': data.get('name', '').strip(),
                    'description': data.get('description', '').strip(),
                    'parent_id': data.get('parent_id'),
                    'product_count': data.get('product_count', 0),
                    'created_at': data.get('created_at'),
                    'updated_at': datetime.now().isoformat(),
                    'status': 'processed'
                }
                
                # Validation - SAME PATTERN REPEATED
                if not processed_data['id']:
                    self.logger.error(f"Missing ID for category {category_id}")
                    continue
                
                if not processed_data['name']:
                    self.logger.warning(f"Missing name for category {category_id}")
                
                if processed_data['product_count'] < 0:
                    self.logger.warning(f"Invalid product count for category {category_id}")
                
                results.append(processed_data)
                
            except requests.RequestException as e:
                self.logger.error(f"Request error for category {category_id}: {str(e)}")
                continue
            except Exception as e:
                self.logger.error(f"Unexpected error for category {category_id}: {str(e)}")
                continue
        
        return results
'''

# Improved Version with DRY Principle
IMPROVED_DRY_VERSION = '''
"""
IMPROVED VERSION - Applying DRY Principle and Template Method Pattern
This demonstrates how duplicated code can be refactored into reusable components.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, TypeVar, Generic, Callable
import requests
import json
import logging
from datetime import datetime
from dataclasses import dataclass

T = TypeVar('T')

@dataclass
class ProcessingResult(Generic[T]):
    """Generic result wrapper for processed data."""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None

class APIClient:
    """Reusable API client for all requests."""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        self.logger = logging.getLogger(__name__)
    
    async def get(self, endpoint: str, timeout: int = 30) -> ProcessingResult[Dict]:
        """Make GET request with proper error handling."""
        try:
            response = requests.get(
                f'{self.base_url}/{endpoint}',
                headers=self.headers,
                timeout=timeout
            )
            
            if response.status_code != 200:
                return ProcessingResult(
                    success=False,
                    error=f"API error: {response.status_code}"
                )
            
            try:
                data = response.json()
                return ProcessingResult(success=True, data=data)
            except json.JSONDecodeError:
                return ProcessingResult(
                    success=False,
                    error="Invalid JSON response"
                )
                
        except requests.RequestException as e:
            return ProcessingResult(
                success=False,
                error=f"Request error: {str(e)}"
            )

class DataProcessor(ABC, Generic[T]):
    """Abstract base class for data processors implementing Template Method pattern."""
    
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        self.logger = logging.getLogger(__name__)
    
    async def process_batch(self, ids: List[int]) -> List[T]:
        """Template method for processing batches of data."""
        results = []
        
        for item_id in ids:
            result = await self.process_single_item(item_id)
            if result.success and result.data:
                results.append(result.data)
        
        return results
    
    async def process_single_item(self, item_id: int) -> ProcessingResult[T]:
        """Process a single item following the template method pattern."""
        # Step 1: Fetch data from API
        api_result = await self.api_client.get(self.get_endpoint(item_id))
        
        if not api_result.success:
            self.logger.error(f"Failed to fetch {self.get_item_type()} {item_id}: {api_result.error}")
            return ProcessingResult(success=False, error=api_result.error)
        
        # Step 2: Transform data (implemented by subclasses)
        try:
            processed_data = self.transform_data(api_result.data, item_id)
        except Exception as e:
            error_msg = f"Failed to transform {self.get_item_type()} {item_id}: {str(e)}"
            self.logger.error(error_msg)
            return ProcessingResult(success=False, error=error_msg)
        
        # Step 3: Validate data (implemented by subclasses)
        validation_errors = self.validate_data(processed_data, item_id)
        
        if validation_errors:
            for error in validation_errors:
                self.logger.warning(f"{self.get_item_type()} {item_id}: {error}")
        
        return ProcessingResult(success=True, data=processed_data)
    
    @abstractmethod
    def get_endpoint(self, item_id: int) -> str:
        """Get API endpoint for the item type."""
        pass
    
    @abstractmethod
    def get_item_type(self) -> str:
        """Get human-readable item type for logging."""
        pass
    
    @abstractmethod
    def transform_data(self, raw_data: Dict, item_id: int) -> T:
        """Transform raw API data into processed format."""
        pass
    
    @abstractmethod
    def validate_data(self, data: T, item_id: int) -> List[str]:
        """Validate processed data and return list of validation errors."""
        pass

# Concrete implementations
@dataclass
class ProcessedUser:
    """Processed user data model."""
    id: int
    name: str
    email: str
    created_at: str
    updated_at: str
    status: str

class UserProcessor(DataProcessor[ProcessedUser]):
    """User data processor."""
    
    def get_endpoint(self, item_id: int) -> str:
        return f"users/{item_id}"
    
    def get_item_type(self) -> str:
        return "user"
    
    def transform_data(self, raw_data: Dict, item_id: int) -> ProcessedUser:
        return ProcessedUser(
            id=raw_data.get('id'),
            name=raw_data.get('name', '').strip(),
            email=raw_data.get('email', '').lower(),
            created_at=raw_data.get('created_at'),
            updated_at=datetime.now().isoformat(),
            status='processed'
        )
    
    def validate_data(self, data: ProcessedUser, item_id: int) -> List[str]:
        errors = []
        
        if not data.id:
            errors.append("Missing ID")
        
        if not data.name:
            errors.append("Missing name")
        
        if not data.email:
            errors.append("Missing email")
        
        return errors

@dataclass
class ProcessedProduct:
    """Processed product data model."""
    id: int
    name: str
    description: str
    price: float
    created_at: str
    updated_at: str
    status: str

class ProductProcessor(DataProcessor[ProcessedProduct]):
    """Product data processor."""
    
    def get_endpoint(self, item_id: int) -> str:
        return f"products/{item_id}"
    
    def get_item_type(self) -> str:
        return "product"
    
    def transform_data(self, raw_data: Dict, item_id: int) -> ProcessedProduct:
        return ProcessedProduct(
            id=raw_data.get('id'),
            name=raw_data.get('name', '').strip(),
            description=raw_data.get('description', '').strip(),
            price=raw_data.get('price', 0),
            created_at=raw_data.get('created_at'),
            updated_at=datetime.now().isoformat(),
            status='processed'
        )
    
    def validate_data(self, data: ProcessedProduct, item_id: int) -> List[str]:
        errors = []
        
        if not data.id:
            errors.append("Missing ID")
        
        if not data.name:
            errors.append("Missing name")
        
        if data.price < 0:
            errors.append("Invalid price")
        
        return errors

# Factory for creating processors
class ProcessorFactory:
    """Factory for creating data processors."""
    
    @staticmethod
    def create_processor(processor_type: str, api_client: APIClient) -> DataProcessor:
        """Create appropriate processor based on type."""
        processors = {
            'user': UserProcessor,
            'product': ProductProcessor,
            # Easy to add new processors
        }
        
        processor_class = processors.get(processor_type)
        if not processor_class:
            raise ValueError(f"Unknown processor type: {processor_type}")
        
        return processor_class(api_client)

# Improved service using composition
class ImprovedDataProcessingService:
    """Improved service using composition and DRY principles."""
    
    def __init__(self, api_base_url: str, api_key: str):
        self.api_client = APIClient(api_base_url, api_key)
        self.factory = ProcessorFactory()
    
    async def process_users(self, user_ids: List[int]) -> List[ProcessedUser]:
        """Process user data using reusable components."""
        processor = self.factory.create_processor('user', self.api_client)
        return await processor.process_batch(user_ids)
    
    async def process_products(self, product_ids: List[int]) -> List[ProcessedProduct]:
        """Process product data using reusable components."""
        processor = self.factory.create_processor('product', self.api_client)
        return await processor.process_batch(product_ids)
    
    # Easy to add new processing methods without code duplication
    async def process_any_type(self, processor_type: str, ids: List[int]) -> List[Any]:
        """Generic processing method for any supported type."""
        processor = self.factory.create_processor(processor_type, self.api_client)
        return await processor.process_batch(ids)
'''

print("✅ Demo Scenario 2: Anti-Pattern Detection - Created comprehensive before/after examples showing:")
print("   • God Class → SOLID Principles refactoring (200+ lines → focused services)")
print("   • Copy-Paste Code → DRY Principle with Template Method pattern")
print("   • Shows dramatic improvement in maintainability and testability")