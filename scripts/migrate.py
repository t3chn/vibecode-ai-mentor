#!/usr/bin/env python
"""Simple migration script for database schema changes."""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text

from src.core.config import settings
from src.core.logging import logger
from src.db.schema import db_manager


# Migration tracking table
MIGRATION_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(255) PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);
"""

# Define migrations
MIGRATIONS = [
    {
        "version": "001_initial_schema",
        "description": "Initial database schema",
        "up": [
            # This migration is handled by setup_db.py
            "-- Initial schema created by setup_db.py"
        ],
        "down": [
            "DROP TABLE IF EXISTS analysis_results;",
            "DROP TABLE IF EXISTS code_snippets;",
            "DROP TABLE IF EXISTS repositories;",
            "DROP TABLE IF EXISTS embedding_cache;",
            "DROP TABLE IF EXISTS search_cache;",
        ],
    },
    # Add future migrations here
    # {
    #     "version": "002_add_user_table",
    #     "description": "Add user authentication table",
    #     "up": [
    #         """
    #         CREATE TABLE users (
    #             id CHAR(36) PRIMARY KEY,
    #             email VARCHAR(255) UNIQUE NOT NULL,
    #             ...
    #         );
    #         """
    #     ],
    #     "down": ["DROP TABLE IF EXISTS users;"],
    # },
]


class MigrationRunner:
    """Handles database migrations."""

    def __init__(self):
        self.engine = None

    async def init(self):
        """Initialize migration runner."""
        self.engine = await db_manager.create_async_engine()
        
        # Create migration tracking table
        async with self.engine.begin() as conn:
            await conn.execute(text(MIGRATION_TABLE_DDL))

    async def get_applied_migrations(self) -> set[str]:
        """Get list of applied migrations."""
        async with self.engine.begin() as conn:
            result = await conn.execute(
                text("SELECT version FROM schema_migrations ORDER BY version;")
            )
            return {row[0] for row in result}

    async def apply_migration(self, migration: dict) -> None:
        """Apply a single migration."""
        version = migration["version"]
        
        async with self.engine.begin() as conn:
            # Run migration statements
            for statement in migration["up"]:
                if statement.strip() and not statement.startswith("--"):
                    await conn.execute(text(statement))
            
            # Record migration
            await conn.execute(
                text(
                    """
                    INSERT INTO schema_migrations (version, description)
                    VALUES (:version, :description);
                    """
                ),
                {"version": version, "description": migration["description"]},
            )
        
        logger.info(f"✅ Applied migration: {version}")

    async def rollback_migration(self, migration: dict) -> None:
        """Rollback a single migration."""
        version = migration["version"]
        
        async with self.engine.begin() as conn:
            # Run rollback statements
            for statement in migration["down"]:
                if statement.strip() and not statement.startswith("--"):
                    await conn.execute(text(statement))
            
            # Remove migration record
            await conn.execute(
                text("DELETE FROM schema_migrations WHERE version = :version;"),
                {"version": version},
            )
        
        logger.info(f"↩️  Rolled back migration: {version}")

    async def run_migrations(self) -> None:
        """Run all pending migrations."""
        applied = await self.get_applied_migrations()
        pending = [m for m in MIGRATIONS if m["version"] not in applied]
        
        if not pending:
            logger.info("✨ Database is up to date!")
            return
        
        logger.info(f"Found {len(pending)} pending migrations")
        
        for migration in pending:
            await self.apply_migration(migration)
        
        logger.info("✅ All migrations completed!")

    async def rollback(self, steps: int = 1) -> None:
        """Rollback the last N migrations."""
        applied = await self.get_applied_migrations()
        
        # Get migrations to rollback in reverse order
        to_rollback = []
        for migration in reversed(MIGRATIONS):
            if migration["version"] in applied:
                to_rollback.append(migration)
                if len(to_rollback) >= steps:
                    break
        
        if not to_rollback:
            logger.info("No migrations to rollback")
            return
        
        for migration in to_rollback:
            await self.rollback_migration(migration)

    async def status(self) -> None:
        """Show migration status."""
        applied = await self.get_applied_migrations()
        
        logger.info("Migration Status:")
        logger.info("-" * 60)
        
        for migration in MIGRATIONS:
            version = migration["version"]
            description = migration["description"]
            status = "✅ Applied" if version in applied else "⏳ Pending"
            logger.info(f"{status} | {version} | {description}")

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.engine:
            await self.engine.dispose()


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Database migration tool")
    parser.add_argument(
        "command",
        choices=["migrate", "rollback", "status"],
        help="Migration command to run",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Number of migrations to rollback (default: 1)",
    )
    
    args = parser.parse_args()
    
    runner = MigrationRunner()
    
    try:
        await runner.init()
        
        if args.command == "migrate":
            await runner.run_migrations()
        elif args.command == "rollback":
            await runner.rollback(args.steps)
        elif args.command == "status":
            await runner.status()
            
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Migration interrupted by user")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)