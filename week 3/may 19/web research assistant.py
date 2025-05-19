from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiohttp

class AsyncWebBrowser:
    def __init__(self, headless=True):
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        self.driver = webdriver.Chrome(options=chrome_options)
        self.executor = ThreadPoolExecutor(max_workers=5)

    async def fetch_page_text(self, url: str) -> str:
        loop = asyncio.get_event_loop()

        def load_url():
            self.driver.get(url)
            self.driver.implicitly_wait(3)
            body = self.driver.find_element(By.TAG_NAME, 'body')
            return body.text

        return await loop.run_in_executor(self.executor, load_url)

    async def close(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self.driver.quit)


class GeminiClient:
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model = model
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

    async def generate_async(self, prompt: str, max_tokens: int = 300) -> str:
        headers = {"Content-Type": "application/json"}

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.7
            }
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    raise Exception(f"Gemini API Error: {resp.status} - {await resp.text()}")
                data = await resp.json()
                return data['candidates'][0]['content']['parts'][0]['text']


class GeminiSummarizer:
    def __init__(self, gemini_client: GeminiClient):
        self.gemini = gemini_client

    async def summarize_text(self, text: str, max_tokens=300) -> str:
        # Truncate text if too long for prompt size limits (adjust as needed)
        max_input_length = 3000
        if len(text) > max_input_length:
            text = text[:max_input_length] + "\n\n[Truncated]"
        prompt = f"Summarize the following content concisely:\n\n{text}\n\nSummary:"
        summary = await self.gemini.generate_async(prompt, max_tokens=max_tokens)
        return summary.strip()


async def main(url: str, api_key: str):
    browser = AsyncWebBrowser(headless=True)
    gemini_client = GeminiClient(api_key=api_key)
    summarizer = GeminiSummarizer(gemini_client)

    try:
        print(f"Fetching page text from: {url}")
        page_text = await browser.fetch_page_text(url)
        print(f"Fetched text length: {len(page_text)} characters")

        summary = await summarizer.summarize_text(page_text)
        print("\nSummary:\n", summary)
    finally:
        await browser.close()


if __name__ == "__main__":
    # Replace with your actual Gemini API key
    API_KEY = ""
    URL_TO_SUMMARIZE = "https://en.wikipedia.org/wiki/Cristiano_Ronaldo"  # Replace with any URL

    asyncio.run(main(URL_TO_SUMMARIZE, API_KEY))