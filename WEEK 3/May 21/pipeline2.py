import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import os
import tempfile
import google.generativeai as genai
from autogen import UserProxyAgent, AssistantAgent
from autogen.agentchat import GroupChat, GroupChatManager
from google.api_core.exceptions import ResourceExhausted
import time

# === Gemini Config ===
genai.configure(api_key="AIzaSyDDaUTEWZGkEvfT46SVH_qOs_QPQJcHLsg")  # 🔒 Use env vars in real apps
model = genai.GenerativeModel("gemini-1.5-flash")

def gemini_generate(prompt, retries=5, delay=60):
    for i in range(retries):
        try:
            return model.generate_content(prompt).text
        except ResourceExhausted:
            print(f"[Gemini] Quota hit. Retry in {delay} sec ({i+1}/{retries})")
            time.sleep(delay)
    return "ERROR: Gemini API quota exceeded."

# === TOOL: Visualizer ===
class Visualizer:
    def __init__(self):
        self.plot_dir = os.path.join(os.getcwd(), "plots")
        os.makedirs(self.plot_dir, exist_ok=True)

    def plot_column(self, df, column):
        if column not in df.columns:
            return f"Column '{column}' not found."
        plt.figure()
        df[column].plot(kind="hist", title=f"Histogram of {column}")
        img_path = os.path.join(self.plot_dir, f"{column}_hist.png")
        plt.savefig(img_path)
        plt.close()
        return img_path

# === AGENTS ===
class DataFetcherAgent(AssistantAgent):
    def __init__(self, name):
        super().__init__(name=name)
        self.df = None
        self.terminated = False

    async def a_generate_reply(self, sender):
        if self.terminated:
            return None  # Don't reply after termination

        last_msg = self.chat_messages[sender][-1]["content"]
        if last_msg.startswith("UPLOAD_CSV:"):
            csv_path = last_msg.split("UPLOAD_CSV:")[1].strip()
            try:
                self.df = pd.read_csv(csv_path)
                summary = self.df.describe(include="all").to_string()
                # After loading, send termination message to agree to stop
                self.terminated = True
                return {"name": self.name, "content": f"Data fetched successfully.\n\nSummary:\n{summary}", "is_termination_msg": True}
            except Exception as e:
                self.terminated = True
                return {"name": self.name, "content": f"Failed to read CSV: {e}", "is_termination_msg": True}
        # If no proper upload command, ignore further messages after termination
        return None

class AnalystAgent(AssistantAgent):
    def __init__(self, name, fetcher: DataFetcherAgent, visualizer: Visualizer):
        super().__init__(name=name)
        self.fetcher = fetcher
        self.visualizer = visualizer
        self.terminated = False

    async def a_generate_reply(self, sender):
        if self.terminated:
            return None  # No further replies after termination

        if self.fetcher.df is None:
            # Wait until DataFetcher finishes
            return None

        # Compose prompt with data summary
        prompt = f"Given the following data summary:\n{self.fetcher.df.describe().to_string()}\n\n"
        prompt += "Suggest which column(s) to visualize and why with matplotlib. List multiple charts (histogram, scatter, boxplot) if suitable."

        response = gemini_generate(prompt)

        # Parse suggested columns from response heuristically
        columns_to_plot = []
        for col in self.fetcher.df.columns:
            if col.lower() in response.lower():
                columns_to_plot.append(col)

        result = f"Gemini suggests visualizing: {', '.join(columns_to_plot)}\n\nExplanation:\n{response}\n"
        for col in columns_to_plot:
            img_path = self.visualizer.plot_column(self.fetcher.df, col)
            result += f"\n📊 Plot saved at: {img_path}"

        self.terminated = True
        return {"name": self.name, "content": result, "is_termination_msg": True}

# === MAIN ===
async def main():
    csv_path = input("📂 Enter the path to your CSV file: ").strip()
    if not os.path.exists(csv_path):
        print("❌ File not found. Exiting.")
        return

    user = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        code_execution_config={"use_docker": False}
    )
    fetcher = DataFetcherAgent(name="DataFetcher")
    analyst = AnalystAgent(name="Analyst", fetcher=fetcher, visualizer=Visualizer())

    groupchat = GroupChat(
        agents=[user, fetcher, analyst],
        messages=[],
        max_round=50,
        speaker_selection_method="round_robin"
    )
    manager = GroupChatManager(groupchat=groupchat)

    # User initiates the chat by uploading CSV path
    await user.a_initiate_chat(manager, message=f"UPLOAD_CSV:{csv_path}")

    terminated_agents = set()

    while True:
        await manager.async_step()

        # Check last messages for termination flags
        for agent in groupchat.agents:
            msgs = groupchat.get_messages(agent.name)
            if msgs and isinstance(msgs[-1], dict):
                if msgs[-1].get("is_termination_msg"):
                    terminated_agents.add(agent.name)

        # Check if DataFetcher and Analyst both terminated
        if {"DataFetcher", "Analyst"}.issubset(terminated_agents):
            print("All agents agreed to terminate. Ending chat.")
            break

    print("✅ Pipeline completed.")

if __name__ == "__main__":
    asyncio.run(main())
