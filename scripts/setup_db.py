#!/usr/bin/env python
"""Initialize the VibeCode AI Mentor database with tables and indexes."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text

from src.core.config import settings
from src.core.logging import logger
from src.db.schema import db_manager


async def setup_database():
    """Set up the database with all required tables and indexes."""
    logger.info("Starting database setup...")
    logger.info(f"Connecting to TiDB at {settings.tidb_host}:{settings.tidb_port}")
    logger.info(f"Database: {settings.tidb_database}")

    try:
        # Initialize database
        await db_manager.initialize_database()
        
        logger.info("✅ Database setup completed successfully!")
        logger.info("Tables created:")
        logger.info("  - repositories")
        logger.info("  - code_snippets (with vector index)")
        logger.info("  - analysis_results")
        logger.info("  - embedding_cache (with vector index)")
        logger.info("  - search_cache")
        
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        raise


async def verify_setup():
    """Verify database setup by checking tables exist."""
    engine = await db_manager.create_async_engine()
    
    try:
        async with engine.begin() as conn:
            # Check tables exist
            result = await conn.execute(
                text(
                    """
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = :db_name
                    AND table_name IN ('repositories', 'code_snippets', 'analysis_results')
                    ORDER BY table_name;
                    """
                ),
                {"db_name": settings.tidb_database}
            )
            
            tables = [row[0] for row in result]
            logger.info(f"Found tables: {tables}")
            
            # Check vector indexes
            result = await conn.execute(
                text(
                    """
                    SELECT index_name, table_name
                    FROM information_schema.statistics
                    WHERE table_schema = :db_name
                    AND index_name LIKE '%embedding%'
                    GROUP BY index_name, table_name;
                    """
                ),
                {"db_name": settings.tidb_database}
            )
            
            indexes = [(row[0], row[1]) for row in result]
            logger.info(f"Found vector indexes: {indexes}")
            
    finally:
        await engine.dispose()


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup VibeCode AI Mentor database")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify database setup without creating tables",
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop existing tables before creating (WARNING: destroys data)",
    )
    
    args = parser.parse_args()
    
    if args.verify:
        await verify_setup()
    else:
        if args.drop_existing:
            response = input(
                "⚠️  WARNING: This will DROP all existing tables and data. Continue? (yes/no): "
            )
            if response.lower() != "yes":
                logger.info("Aborted.")
                return
            
            # Drop tables
            engine = await db_manager.create_async_engine()
            try:
                async with engine.begin() as conn:
                    await conn.execute(text("SET FOREIGN_KEY_CHECKS = 0;"))
                    for table in [
                        "analysis_results",
                        "code_snippets", 
                        "repositories",
                        "embedding_cache",
                        "search_cache",
                    ]:
                        await conn.execute(text(f"DROP TABLE IF EXISTS {table};"))
                    await conn.execute(text("SET FOREIGN_KEY_CHECKS = 1;"))
                    logger.info("Existing tables dropped.")
            finally:
                await engine.dispose()
        
        await setup_database()
        await verify_setup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)