import asyncio
import aiohttp
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import os

# ==== CONFIG ====
GOOGLE_API_KEY = "AIzaSyCHy80eWH_N7Q9Xc0niq9OpxdNKaCoJmBQ"
CSV_URL = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== GEMINI (LLM TOOL) ====
async def gemini_prompt(prompt: str):
    async with aiohttp.ClientSession() as session:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GOOGLE_API_KEY}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        async with session.post(url, json=payload) as resp:
            result = await resp.json()
            try:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            except:
                return "[Error in Gemini response]"

# ==== DATA FETCHER AGENT ====
class DataFetcherAgent:
    async def fetch_csv(self, url: str) -> pd.DataFrame:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                text = await response.text()
                return pd.read_csv(StringIO(text))

# ==== ANALYST AGENT ====
class AnalystAgent:
    def describe(self, df):
        return df.describe()

    def plot_histogram(self, df):
        df.hist(figsize=(10, 8))
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/histogram.png")

    async def analyze(self, df):
        description = self.describe(df)
        self.plot_histogram(df)
        summary_text = f"Summary Statistics:\n{description}"
        gemini_summary = await gemini_prompt(f"Summarize this data:\n{description}")
        return summary_text, gemini_summary

# ==== ROUND ROBIN GROUP CHAT (Controller) ====
class RoundRobinGroupChat:
    def __init__(self):  # ✅ Corrected method name
        self.fetcher = DataFetcherAgent()
        self.analyst = AnalystAgent()

    async def run(self, url):
        print("[1/3] Fetching CSV data...")
        df = await self.fetcher.fetch_csv(url)

        print("[2/3] Running analysis...")
        stats, gemini_insight = await self.analyst.analyze(df)

        print("[3/3] Analysis complete.\n")
        print(stats)
        print("\n--- Gemini Summary ---\n")
        print(gemini_insight)
        print(f"\nHistogram saved to {OUTPUT_DIR}/histogram.png")

# ==== MAIN ====
async def main():
    orchestrator = RoundRobinGroupChat()
    await orchestrator.run(CSV_URL)

if __name__ == "__main__":  # ✅ Corrected main check
    asyncio.run(main())
