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
    def __init__(self, output_dir="plots"):
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def sanitize_filename(self, name):
        import re
        return re.sub(r'[\\/*?:"<>|]', "_", name)

    def plot_column(self, df, column, plot_type="hist"):
        plt.figure()
        try:
            if plot_type == "hist":
                df[column].dropna().plot(kind="hist", title=f"{column} - Histogram")
            elif plot_type == "box":
                df[column].dropna().plot(kind="box", title=f"{column} - Boxplot")
            elif plot_type == "bar":
                df[column].value_counts().plot(kind="bar", title=f"{column} - Bar Chart")
            elif plot_type == "scatter":
                numeric_cols = df.select_dtypes(include="number").columns
                x_col = next((c for c in numeric_cols if c != column), None)
                if x_col:
                    df.plot.scatter(x=x_col, y=column, title=f"{column} vs {x_col} - Scatter")
                else:
                    return f"No suitable numeric column to plot scatter for {column}"
            else:
                return f"Unsupported plot type: {plot_type}"
        except Exception as e:
            return f"Error plotting {column} as {plot_type}: {e}"

        safe_filename = self.sanitize_filename(f"{column}_{plot_type}.png")
        img_path = os.path.join(self.output_dir, safe_filename)
        plt.tight_layout()
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
        self.done = False

    async def a_generate_reply(self, sender):
        if self.done:
            return None  # Stop generating replies after final message

        if self.fetcher.df is None:
            return {"name": self.name, "content": "No data yet. Waiting for fetcher to load CSV."}

        df = self.fetcher.df
        summary = df.describe(include="all").to_string()
        prompt = (
            f"Given this dataset summary:\n\n{summary}\n\n"
            "Generate a list of plots (histogram, boxplot, scatter, bar chart) for each column "
            "that best helps to understand the data. For each plot, explain why it's chosen."
        )
        response = gemini_generate(prompt)
        explanation = response.strip()

        result = f"{explanation}\n\n"

        # Choose plots based on column type
        for column in df.columns:
            col_data = df[column]
            col_type = str(col_data.dtypes)

            plots = []
            if col_data.dtype in [int, float]:
                plots = ["hist", "box"]
                if len(df.select_dtypes(include="number").columns) > 1:
                    plots.append("scatter")
            elif col_data.dtype == "object" or col_data.nunique() < 20:
                plots = ["bar"]

            for ptype in plots:
                path = self.visualizer.plot_column(df, column, ptype)
                result += f"📊 Plot ({ptype}) saved at: {path}\n"

        self.done = True
        return {"name": self.name, "content": result + "\nAll visualizations completed. Goodbye from Analyst!", "is_termination_msg": True}

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
        max_round=3,
        speaker_selection_method="round_robin"
    )
    manager = GroupChatManager(groupchat=groupchat)

    await user.a_initiate_chat(manager, message=f"UPLOAD_CSV:{csv_path}")

    while True:
        await manager.step()
        if any(
            m.get("is_termination_msg", False) and "Goodbye from Analyst" in m.get("content", "")
            for m in groupchat.messages
        ):
            print("✅ Analyst has completed all visualizations. Exiting.")
            break

if __name__ == "__main__":
    asyncio.run(main())
