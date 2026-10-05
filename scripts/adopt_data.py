"""Move the data of an old placeholder user to a real account.

Before Version 3 there was no sign-in: everything belonged to a name sent in
the X-User-Id header (the default was "demo-user"). After you create an
account, this script gives it the documents, investigations, conversations
and memory of such a name.

    docker compose exec api python -m scripts.adopt_data --list
    docker compose exec api python -m scripts.adopt_data --from-name demo-user --to-email you@example.com

It is safe to run twice. Documents and notes the account already has are left
where they are instead of being duplicated.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import uuid
from pathlib import Path

from sqlalchemy import func, select, text

from app.config import get_settings
from app.database.models import Document, Investigation, User
from app.database.session import dispose_engine, get_session_factory, init_db


async def list_placeholder_users() -> list[tuple[str, int, int]]:
    """(name, documents, investigations) for every user without an email address."""
    async with get_session_factory()() as session:
        users = (await session.execute(select(User).where(User.email.is_(None)).order_by(User.created_at))).scalars().all()
        rows = []
        for user in users:
            documents = await session.scalar(select(func.count(Document.id)).where(Document.user_id == user.id))
            investigations = await session.scalar(select(func.count(Investigation.id)).where(Investigation.user_id == user.id))
            rows.append((user.name, int(documents or 0), int(investigations or 0)))
        return rows


async def adopt(from_name: str, to_email: str) -> dict[str, int]:
    """Move everything from the placeholder user to the account. Returns how many rows moved."""
    settings = get_settings()
    async with get_session_factory()() as session:
        source = (await session.execute(select(User).where(User.name == from_name, User.email.is_(None)))).scalars().first()
        target = (await session.execute(select(User).where(User.email == to_email.strip().lower()))).scalars().first()
        if source is None:
            raise SystemExit(f"There is no placeholder user named '{from_name}'. Use --list to see the names.")
        if target is None:
            raise SystemExit(f"There is no account with the email address '{to_email}'. Create it first, then run this again.")
        ids = {"source": source.id, "target": target.id}

        moved_documents = (await session.execute(
            text(
                "UPDATE documents SET user_id = :target WHERE user_id = :source AND file_hash NOT IN "
                "(SELECT file_hash FROM documents WHERE user_id = :target) RETURNING id"
            ), ids,
        )).scalars().all()
        counts = {"documents": len(moved_documents)}
        for table in ("conversations", "investigations"):
            result = await session.execute(text(f"UPDATE {table} SET user_id = :target WHERE user_id = :source"), ids)
            counts[table] = result.rowcount
        result = await session.execute(
            text(
                "UPDATE user_memory SET user_id = :target WHERE user_id = :source AND memory_key NOT IN "
                "(SELECT memory_key FROM user_memory WHERE user_id = :target)"
            ), ids,
        )
        counts["preferences"] = result.rowcount
        result = await session.execute(
            text(
                "UPDATE semantic_memory SET user_id = :target WHERE user_id = :source AND text_hash NOT IN "
                "(SELECT text_hash FROM semantic_memory WHERE user_id = :target)"
            ), ids,
        )
        counts["notes"] = result.rowcount
        await session.commit()

    # The stored files live in a folder per user.
    source_dir, target_dir = Path(settings.UPLOAD_DIR) / str(ids["source"]), Path(settings.UPLOAD_DIR) / str(ids["target"])
    for document_id in moved_documents:
        for path in source_dir.glob(f"{uuid.UUID(str(document_id))}.*"):
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target_dir / path.name))
    return counts


async def run(args: argparse.Namespace) -> int:
    await init_db()
    try:
        if args.list or not (args.from_name and args.to_email):
            rows = await list_placeholder_users()
            if not rows:
                print("There are no placeholder users. Nothing to adopt.")
            for name, documents, investigations in rows:
                print(f"{name:<30} {documents} document(s), {investigations} investigation(s)")
            if not args.list:
                print("\nTo move one of them: --from-name <name> --to-email <your account's email>")
            return 0
        counts = await adopt(args.from_name, args.to_email)
        print(f"Moved from '{args.from_name}' to {args.to_email}: " + ", ".join(f"{count} {what}" for what, count in counts.items()))
        return 0
    finally:
        await dispose_engine()


def main() -> int:
    parser = argparse.ArgumentParser(description="Give an account the data of an old placeholder user.")
    parser.add_argument("--list", action="store_true", help="Show the placeholder users and what they own.")
    parser.add_argument("--from-name", help="Name of the placeholder user, for example demo-user.")
    parser.add_argument("--to-email", help="Email address of the account that should receive the data.")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
