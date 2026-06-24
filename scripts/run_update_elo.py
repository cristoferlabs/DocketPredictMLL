#!/usr/bin/env python3
"""Run update_elo job directly (no worker required)."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apps.worker.tasks.update_elo import update_elo_after_finished_matches


async def main() -> None:
    result = await update_elo_after_finished_matches({})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
