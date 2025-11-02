import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# allow importing project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    try:
        fileConfig(config.config_file_name)
    except Exception:
        # If logging config is missing expected sections, ignore and continue
        pass

# Import the metadata from the project's models
try:
    from index.store import Base

    target_metadata = Base.metadata
except Exception:
    target_metadata = None


def get_url():
    # Prefer environment variable, fall back to config.yml
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    try:
        import yaml

        cfg = yaml.safe_load(open("config.yml"))
        path = cfg.get("index", {}).get("sqlite_path", "data/meta.db")
        return f"sqlite:///{path}"
    except Exception:
        return "sqlite:///data/meta.db"


def run_migrations_offline():
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_url()
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
