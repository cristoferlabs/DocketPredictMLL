"""Quick Odds API quota check."""
import asyncio

from apps.worker.ingest.odds_api import OddsApiClient


async def main() -> None:
    OddsApiClient._live_disabled = None
    client = OddsApiClient()
    status = await client.check_status()
    print("key_configured:", bool(client.api_key))
    for k in ("ok", "reason", "remaining", "used", "events", "detail"):
        print(f"{k}: {status.get(k)}")


if __name__ == "__main__":
    asyncio.run(main())
