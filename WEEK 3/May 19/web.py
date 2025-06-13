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
class WebBrowserTool:
    def __init__(self):
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        self.driver = webdriver.Chrome(options=options)

    def search_and_scrape(self, query: str, num_results: int = 3):
        self.driver.get(f"https://www.google.com/search?q={query}")
        time.sleep(2)
        links = self.driver.find_elements(By.CSS_SELECTOR, 'div.yuRUbf > a')
        results = []
        for link in links[:num_results]:
            url = link.get_attribute('href')
            try:
                self.driver.get(url)
                time.sleep(2)
                text = self.driver.find_element(By.TAG_NAME, 'body').text
                results.append({"url": url, "content": text[:3000]})
            except Exception:
                continue
        return results

    def close(self):
        self.driver.quit()

# === AGENT: Researcher ===
class ResearcherAgent(AssistantAgent):
    def __init__(self, name, browser_tool):
        super().__init__(name=name)
        self.browser_tool = browser_tool

    async def a_generate_reply(self, messages, sender, config):
        topic = messages[-1]["content"]
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

    async def a_generate_reply(self, messages, sender, config):
        print(f"[{self.name}] Summarizing search results...")
        try:
            search_data = ast.literal_eval(messages[-1]["content"])
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
