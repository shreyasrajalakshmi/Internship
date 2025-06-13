import asyncio
import json
import google.generativeai as genai
from autogen import UserProxyAgent, AssistantAgent
from autogen.agentchat import GroupChat, GroupChatManager
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# === SETUP GEMINI LLM ===
GOOGLE_API_KEY = "AIzaSyDDaUTEWZGkEvfT46SVH_qOs_QPQJcHLsg"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# === WEB BROWSER TOOL (Selenium) ===
class WebBrowserTool:
    def __init__(self):
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
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

        link_elements = self.driver.find_elements(By.CSS_SELECTOR, 'div#search a[href*="http"]')
        print(f"Number of links found: {len(link_elements)}")

        urls = [link.get_attribute('href') for link in link_elements[:num_results] if link.get_attribute('href')]

        results = []
        for url in urls:
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
        self.done = False

    async def a_generate_reply(self, sender):
        last_message = self.chat_messages[sender][-1]["content"]

        # Check if the message is a summary from Summarizer
        if "Hope this summary helps!" in last_message:
            self.done = True
            return {
                "name": self.name,
                "content": "Thank you for the summary. It looks great! Ready to end the chat.",
                "is_termination_msg": False,  # Let Summarizer terminate
            }

        # Perform search for the topic
        topic = last_message
        print(f"[{self.name}] Searching the web for: {topic}")
        results = self.browser_tool.search_and_scrape(topic)

        return {
            "name": self.name,
            "content": json.dumps(results),
            "is_termination_msg": False,
        }

# === AGENT: Summarizer ===
class SummarizerAgent(AssistantAgent):
    def __init__(self, name):
        super().__init__(name=name)

    async def a_generate_reply(self, sender):
        last_message = self.chat_messages[sender][-1]["content"]

        # Check if Researcher acknowledged the summary
        if "Thank you for the summary" in last_message:
            return {
                "name": self.name,
                "content": "Thanks for acknowledging the summary. Ending chat now.",
                "is_termination_msg": True,  # Terminate the chat
            }

        # Generate summary
        print(f"[{self.name}] Summarizing search results...")
        try:
            search_data = json.loads(last_message)
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
            "content": f"Summary:\n{json.dumps(summaries, indent=2)}\n\nHope this summary helps!",
            "is_termination_msg": False,
        }

# === AGENT: User Proxy ===
class CustomUserProxyAgent(UserProxyAgent):
    async def a_generate_reply(self, sender):
        last_message = self.chat_messages[sender][-1]["content"]
        # Only send the initial topic; skip further rounds
        if last_message == self._initial_message or not self.chat_messages[sender]:
            return {
                "name": self.name,
                "content": self._initial_message,
                "is_termination_msg": False,
            }
        # Return None to avoid interfering with Researcher-Summarizer flow
        return None

# === MAIN PROGRAM ===
async def main():
    topic = input("Enter a topic to research: ")

    browser_tool = WebBrowserTool()

    user = CustomUserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        code_execution_config={"use_docker": False}
    )
    user._initial_message = topic  # Store initial topic
    researcher = ResearcherAgent(name="Researcher", browser_tool=browser_tool)
    summarizer = SummarizerAgent(name="Summarizer")

    # Use round_robin with only Researcher and Summarizer after User's initial message
    groupchat = GroupChat(
        agents=[user, researcher, summarizer],
        messages=[],
        max_round=5,
        speaker_selection_method="round_robin"
    )
    manager = GroupChatManager(groupchat=groupchat, llm_config=False)

    # Start with the user's initial message (topic)
    await user.a_initiate_chat(manager, message=topic)

    browser_tool.close()

if __name__ == "__main__":
    asyncio.run(main())