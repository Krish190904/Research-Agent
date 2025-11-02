"""create initial documents and embeddings tables

Revision ID: 0001_initial
Revises:
Create Date: 2025-11-02 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("faiss_id", sa.Integer, nullable=False, unique=True, index=True),
        sa.Column("doc_id", sa.String, index=True),
        sa.Column("chunk_index", sa.Integer),
        sa.Column("text", sa.Text),
        sa.Column("meta", sa.Text),
    )
    op.create_table(
        "embeddings",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("faiss_id", sa.Integer, nullable=False, unique=True, index=True),
        sa.Column("vector", sa.LargeBinary),
    )


def downgrade() -> None:
    op.drop_table("embeddings")
    op.drop_table("documents")
