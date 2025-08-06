# DevOps Engineer Tasks - VibeCode AI Mentor

## 🎯 Role Overview
You are the DevOps engineer responsible for infrastructure setup, TiDB Cloud configuration, Docker containerization, deployment, and ensuring the system runs smoothly for the VibeCode AI Mentor hackathon demo.

## 📋 Primary Responsibilities

### 1. TiDB Cloud Setup
- Create and configure TiDB Cloud Serverless cluster
- Design optimal database schema with vector indexes
- Setup connection pooling and security
- Monitor query performance
- Implement backup strategy

### 2. Docker Configuration
- Create multi-stage Dockerfile for Python app
- Setup docker-compose for local development
- Configure environment variables
- Optimize image size (< 500MB)
- Add health checks

### 3. Infrastructure & Deployment
- Setup CI/CD pipeline (GitHub Actions)
- Configure cloud deployment (Railway/Render)
- Implement monitoring and logging
- Setup SSL/TLS certificates
- Create deployment scripts

### 4. Performance Optimization
- Database query optimization
- Connection pool tuning
- Resource monitoring
- Cost optimization
- Load testing setup

## 🛠 Technical Requirements

### TiDB Configuration
```sql
-- Required tables with vector support
CREATE TABLE code_snippets (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    content TEXT NOT NULL,
    embedding VECTOR(1536) NOT NULL COMMENT 'Code embedding vector',
    language VARCHAR(50) NOT NULL,
    file_path VARCHAR(500),
    complexity_score INT,
    repo_name VARCHAR(200),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    VECTOR INDEX idx_embedding (embedding) COMMENT 'HNSW index for similarity search'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE best_practices (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    pattern_name VARCHAR(200) NOT NULL,
    embedding VECTOR(1536) NOT NULL,
    example_code TEXT,
    severity ENUM('info', 'warning', 'error'),
    VECTOR INDEX idx_pattern_embedding (embedding)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Performance indexes
CREATE INDEX idx_language ON code_snippets(language);
CREATE INDEX idx_complexity ON code_snippets(complexity_score);
CREATE INDEX idx_created ON code_snippets(created_at);
```

### Infrastructure Stack
```yaml
# Core Services
- TiDB Cloud Serverless (Free Tier)
- Docker + Docker Compose
- GitHub Actions (CI/CD)
- Railway.app or Render.com (Deployment)

# Monitoring
- TiDB Cloud Monitoring
- Docker health checks
- Application metrics

# Security
- Environment variables
- SSL/TLS encryption
- API rate limiting
```

## 📁 Files You Own

```
vibecode/
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── deploy.yml
├── scripts/
│   ├── setup_db.py      # Database initialization
│   ├── migrate.py       # Schema migrations
│   ├── backup.sh        # Backup script
│   └── deploy.sh        # Deployment script
├── infrastructure/
│   ├── tidb_config.sql  # Database schema
│   ├── monitoring.py    # Metrics collection
│   └── health_check.py  # Health endpoints
├── .env.example
└── deployment.md
```

## 🔧 Configuration Templates

### 1. Docker Compose
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - TIDB_HOST=${TIDB_HOST}
      - TIDB_PORT=4000
      - TIDB_USER=${TIDB_USER}
      - TIDB_PASSWORD=${TIDB_PASSWORD}
      - TIDB_DATABASE=vibecode
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 2. Environment Variables
```bash
# TiDB Configuration
TIDB_HOST=gateway01.us-west-2.prod.aws.tidbcloud.com
TIDB_PORT=4000
TIDB_USER=
TIDB_PASSWORD=
TIDB_DATABASE=vibecode
TIDB_SSL_CA=/etc/ssl/certs/ca-certificates.crt

# API Keys
GEMINI_API_KEY=
OPENAI_API_KEY=

# Application
ENVIRONMENT=production
LOG_LEVEL=INFO
WORKERS=4
```

## 🔄 Integration Points

### With Backend Developer
- Provide database connection strings
- Setup connection pooling
- Monitor query performance
- Optimize slow queries

### With ML Engineer
- Monitor API usage and costs
- Setup GPU resources if needed
- Optimize vector operations
- Cache embeddings

### With QA Tester
- Provide test environments
- Setup staging deployment
- Monitor application logs
- Track error rates

## 📊 Success Metrics

### Infrastructure Targets
- Uptime > 99.9% during demo
- Database queries < 100ms
- Docker image < 500MB
- Deployment time < 5 minutes

### Cost Optimization
- TiDB free tier limits respected
- API costs tracked
- Resource usage monitored
- No unexpected charges

## 🚀 Quick Start Commands

```bash
# TiDB Setup
mysql --connect-timeout 15 -u root -h $TIDB_HOST -P 4000 -D test --ssl-ca=/etc/ssl/certs/ca-certificates.crt -p

# Docker Operations
docker-compose up -d
docker-compose logs -f api
docker-compose down

# Deployment
./scripts/deploy.sh production

# Database Migration
python scripts/setup_db.py
python scripts/migrate.py

# Monitoring
docker stats
python infrastructure/monitoring.py
```

## ⏰ Timeline

### Phase 1 (Hours 0-6)
- [ ] Setup TiDB Cloud account
- [ ] Create database schema
- [ ] Configure vector indexes
- [ ] Test connections

### Phase 2 (Hours 6-14)
- [ ] Create Dockerfile
- [ ] Setup docker-compose
- [ ] Configure environment
- [ ] Add health checks

### Phase 3 (Hours 42-48)
- [ ] Deploy to cloud
- [ ] Setup monitoring
- [ ] Performance testing
- [ ] Demo preparation

## 🎯 Definition of Done

- [ ] TiDB Cloud cluster running with vector support
- [ ] Docker containers building successfully
- [ ] CI/CD pipeline working
- [ ] Application deployed and accessible
- [ ] Monitoring dashboard available
- [ ] Zero downtime during demo

## 🚨 Deployment Checklist

### Pre-deployment
- [ ] All environment variables set
- [ ] Database migrations run
- [ ] SSL certificates configured
- [ ] Health checks passing

### Post-deployment
- [ ] Application accessible via HTTPS
- [ ] API endpoints responding
- [ ] Vector search working
- [ ] Logs aggregated
- [ ] Backup configured