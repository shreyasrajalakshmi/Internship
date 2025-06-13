import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import os
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
    def __init__(self, output_dir="plots"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def plot_column(self, df, column):
        if column not in df.columns:
            return f"Column '{column}' not found."
        plt.figure()
        df[column].plot(kind="hist", title=f"Histogram of {column}")
        img_path = os.path.join(self.output_dir, f"{column}_hist.png")
        plt.savefig(img_path)
        plt.close()
        return img_path

# === AGENTS ===
class DataFetcherAgent(AssistantAgent):
    def __init__(self, name):
        super().__init__(name=name)
        self.df = None

    async def a_generate_reply(self, sender):
        last_msg = self.chat_messages[sender][-1]["content"]
        if last_msg.startswith("UPLOAD_CSV:"):
            csv_path = last_msg.split("UPLOAD_CSV:")[1].strip()
            try:
                self.df = pd.read_csv(csv_path)
                summary = self.df.describe(include="all").to_string()
                return {"name": self.name, "content": f"Data fetched successfully.\n\nSummary:\n{summary}"}
            except Exception as e:
                return {"name": self.name, "content": f"Failed to read CSV: {e}"}
        return {"name": self.name, "content": "Please upload CSV using: UPLOAD_CSV:<path>"}

class AnalystAgent(AssistantAgent):
    def __init__(self, name, fetcher: DataFetcherAgent, visualizer: Visualizer):
        super().__init__(name=name)
        self.fetcher = fetcher
        self.visualizer = visualizer

    async def a_generate_reply(self, sender):
        if self.fetcher.df is None:
            return {"name": self.name, "content": "No data yet. Waiting for fetcher to load CSV."}

        prompt = f"Given the following data summary:\n{self.fetcher.df.describe().to_string()}\n\n"
        prompt += "Which column(s) should I visualize and why?"
        response = gemini_generate(prompt)

        columns_to_plot = []
        for col in self.fetcher.df.columns:
            if col.lower() in response.lower():
                columns_to_plot.append(col)

        result = f"Gemini suggests visualizing: {', '.join(columns_to_plot)}\n"
        for col in columns_to_plot:
            img_path = self.visualizer.plot_column(self.fetcher.df, col)
            result += f"\n📊 Plot saved at: {img_path}"
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
        code_execution_config={"use_docker": False}  # 👈 Disable Docker for code execution
    )

    fetcher = DataFetcherAgent(name="DataFetcher")
    visualizer = Visualizer()  # Will save plots inside ./plots/
    analyst = AnalystAgent(name="Analyst", fetcher=fetcher, visualizer=visualizer)

    groupchat = GroupChat(
        agents=[user, fetcher, analyst],
        messages=[],
        max_round=20,
        speaker_selection_method="round_robin"
    )
    manager = GroupChatManager(groupchat=groupchat)

    await user.a_initiate_chat(manager, message=f"UPLOAD_CSV:{csv_path}")
    print("✅ Pipeline completed.")

if __name__ == "__main__":
    asyncio.run(main())
