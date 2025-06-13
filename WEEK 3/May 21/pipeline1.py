import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
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

    def plot_histogram(self, df, column):
        plt.figure()
        df[column].plot(kind="hist", title=f"Histogram of {column}")
        img_path = os.path.join(self.output_dir, f"{column}_hist.png")
        plt.savefig(img_path)
        plt.close()
        return img_path

    def plot_boxplot(self, df, column):
        plt.figure()
        df.boxplot(column=column)
        plt.title(f"Boxplot of {column}")
        img_path = os.path.join(self.output_dir, f"{column}_boxplot.png")
        plt.savefig(img_path)
        plt.close()
        return img_path

    def plot_scatter(self, df, x_col, y_col):
        plt.figure()
        plt.scatter(df[x_col], df[y_col])
        plt.title(f"Scatter Plot: {x_col} vs {y_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        img_path = os.path.join(self.output_dir, f"{x_col}_vs_{y_col}_scatter.png")
        plt.savefig(img_path)
        plt.close()
        return img_path

# === AGENTS ===
class DataFetcherAgent(AssistantAgent):
    def __init__(self, name):
        super().__init__(name=name)
        self.df = None

    async def a_generate_reply(self, sender):
        last_msg = self.chat_messages[sender][-1]["content"].strip()
        if last_msg == "TERMINATE":
            return {"name": self.name, "content": "DataFetcher acknowledges termination.", "is_termination_msg": True}

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
        last_msg = self.chat_messages[sender][-1]["content"].strip()
        if last_msg == "TERMINATE":
            return {"name": self.name, "content": "Analyst acknowledges termination.", "is_termination_msg": True}

        if self.fetcher.df is None:
            return {"name": self.name, "content": "No data yet. Waiting for fetcher to load CSV."}

        prompt = f"""Given the following data summary:
{self.fetcher.df.describe().to_string()}

Please suggest for each column whether and how it should be visualized.
For each suggested column, specify:
- The type of plot (histogram, boxplot, scatterplot, line plot, etc.)
- The reason why this visualization helps understanding this column's data
If scatter plots involve two columns, specify both columns.
Format your answer as:

Column: <column_name>
Plot: <plot_type>
Reason: <your explanation>

Example:

Column: Age
Plot: Histogram
Reason: To understand distribution of ages

Column: Height and Weight
Plot: Scatterplot
Reason: To analyze correlation between height and weight
"""

        response = gemini_generate(prompt)

        pattern = r"Column:\s*(.+?)\nPlot:\s*(.+?)\nReason:\s*(.+?)(?:\n|$)"
        matches = re.findall(pattern, response, flags=re.IGNORECASE | re.DOTALL)

        if not matches:
            return {"name": self.name, "content": "Gemini did not provide plot suggestions in the expected format."}

        result = ""
        for col_part, plot_type, reason in matches:
            col_part = col_part.strip()
            plot_type = plot_type.strip().lower()
            reason = reason.strip()

            # Handle scatterplots with two columns separated by "and"
            if plot_type == "scatterplot":
                cols = [c.strip() for c in re.split(r'and|,', col_part)]
                if len(cols) < 2 or any(c not in self.fetcher.df.columns for c in cols):
                    result += f"\nInvalid columns for scatterplot: {col_part}\n"
                    continue
                img_path = self.visualizer.plot_scatter(self.fetcher.df, cols[0], cols[1])
                result += f"\nScatter Plot: {cols[0]} vs {cols[1]}\nReason: {reason}\nPlot saved at: {img_path}\n"
                continue

            # Single column plots
            col = col_part
            if col not in self.fetcher.df.columns:
                result += f"\nColumn '{col}' not found in data.\n"
                continue

            if plot_type == "histogram":
                img_path = self.visualizer.plot_histogram(self.fetcher.df, col)
                result += f"\nHistogram of {col}\nReason: {reason}\nPlot saved at: {img_path}\n"
            elif plot_type == "boxplot":
                img_path = self.visualizer.plot_boxplot(self.fetcher.df, col)
                result += f"\nBoxplot of {col}\nReason: {reason}\nPlot saved at: {img_path}\n"
            else:
                result += f"\nPlot type '{plot_type}' not supported yet for column '{col}'. Reason: {reason}\n"

        # Mark this as the last message to terminate chat
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
    analyst = AnalystAgent(name="Analyst", fetcher=fetcher, visualizer=Visualizer())

    groupchat = GroupChat(
        agents=[user, fetcher, analyst],
        messages=[],
        max_round=20,
        speaker_selection_method="round_robin"
    )
    manager = GroupChatManager(groupchat=groupchat)

    # Step 1: Upload CSV
    await user.a_initiate_chat(manager, message=f"UPLOAD_CSV:{csv_path}")

    # Wait a bit for processing (optional)
    await asyncio.sleep(3)

    # Step 2: Signal termination explicitly after pipeline
    await user.a_send_message(manager, message="TERMINATE")

    print("✅ Pipeline completed.")

if __name__ == "__main__":
    asyncio.run(main())
