import asyncio
import os
import time
import ast
import google.generativeai as genai
from autogen import UserProxyAgent, AssistantAgent
from autogen.agentchat import GroupChat, GroupChatManager
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# === SETUP GEMINI LLM ===
GOOGLE_API_KEY = "AIzaSyDDaUTEWZGkEvfT46SVH_qOs_QPQJcHLsg"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# === WEB BROWSER TOOL (Selenium) ===
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class WebBrowserTool:
    def __init__(self):
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")
        self.driver = webdriver.Chrome(options=options)

    def search_and_scrape(self, query: str, num_results: int = 3):
        self.driver.get(f"https://www.google.com/search?q={query}")
        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div#search a[href*="http"]'))
            )
        except Exception as e:
            print(f"Error waiting for search results: {e}")
            return []
        
        links = self.driver.find_elements(By.CSS_SELECTOR, 'div#search a[href*="http"]')
        print(f"Number of links found: {len(links)}")
        results = []
        for link in links[:num_results]:
            url = link.get_attribute('href')
            try:
                print(f"Attempting to scrape: {url}")
                self.driver.get(url)
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, 'body'))
                )
                text_element = self.driver.find_element(By.TAG_NAME, 'body')
                text = text_element.text
                results.append({"url": url, "content": text[:3000]})
                print(f"Successfully scraped: {url}")
            except Exception as e:
                print(f"Error scraping {url}: {e}")
                continue
        return results

    def close(self):
        self.driver.quit()

# === AGENT: Researcher ===
class ResearcherAgent(AssistantAgent):
    def __init__(self, name, browser_tool):
        super().__init__(name=name)
        self.browser_tool = browser_tool

    async def a_generate_reply(self, sender):
        topic = self.chat_messages[sender][-1]["content"]
        print(f"[{self.name}] Searching the web for: {topic}")
        results = self.browser_tool.search_and_scrape(topic)
        return {
            "name": self.name,
            "content": str(results),
            "is_termination_msg": False,
        }

# === AGENT: Summarizer ===
class SummarizerAgent(AssistantAgent):
    def __init__(self, name):
        super().__init__(name=name)

    async def a_generate_reply(self, sender):
        print(f"[{self.name}] Summarizing search results...")
        try:
            search_data = ast.literal_eval(self.chat_messages[sender][-1]["content"])
        except Exception as e:
            return {
                "name": self.name,
                "content": f"Failed to parse input: {e}",
                "is_termination_msg": True,
            }

        summaries = []
        for item in search_data:
            prompt = (
                f"Summarize the following webpage content:\n\n"
                f"URL: {item['url']}\n\n"
                f"Content:\n{item['content'][:2000]}"
            )
            try:
                response = model.generate_content(prompt)
                summary = response.text.strip()
                summaries.append({"url": item['url'], "summary": summary})
            except Exception as e:
                summaries.append({"url": item['url'], "summary": f"Error: {e}"})

        return {
            "name": self.name,
            "content": str(summaries),
            "is_termination_msg": True,
        }

# === MAIN PROGRAM ===
async def main():
    topic = input("Enter a topic to research: ")

    browser_tool = WebBrowserTool()

    user = UserProxyAgent(name="User", human_input_mode="NEVER",code_execution_config={"use_docker": False})
    researcher = ResearcherAgent(name="Researcher", browser_tool=browser_tool)
    summarizer = SummarizerAgent(name="Summarizer")

    # ✅ Use GroupChat with round_robin mode
    groupchat = GroupChat(
        agents=[user, researcher, summarizer],
        messages=[],
        max_round=5,
        speaker_selection_method="round_robin"
    )
    manager = GroupChatManager(groupchat=groupchat, llm_config=False)

    await user.a_initiate_chat(manager, message=topic)
    browser_tool.close()

if __name__ == "__main__":
    asyncio.run(main())