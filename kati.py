import aiohttp
import asyncio
async def main():
    async with aiohttp.ClientSession() as session:
        async with session.get("https://google.com/") as response:
            print(await response.text())
asyncio.run(main())