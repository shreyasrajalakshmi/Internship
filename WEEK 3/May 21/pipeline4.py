import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import os
import google.generativeai as genai
from autogen import UserProxyAgent, AssistantAgent
from autogen.agentchat import GroupChat, GroupChatManager
import re

# === Gemini Setup ===
genai.configure(api_key="AIzaSyDDaUTEWZGkEvfT46SVH_qOs_QPQJcHLsg")  # Use env vars in real apps
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# === Global Shared Context ===
context = {}

# === Agents ===

class DataFetcher(AssistantAgent):
    async def a_generate_reply(self, sender):
        msg = self.chat_messages[sender][-1]["content"]
        if "csv" in msg.lower():
            path = msg.strip().replace("Please process the following CSV file:", "").strip()
            try:
                df = pd.read_csv(path)
                context["df"] = df
                return {
                    "name": self.name,
                    "content": f"✅ CSV loaded with {df.shape[0]} rows and {df.shape[1]} columns.",
                    "is_termination_msg": False
                }
            except Exception as e:
                return {
                    "name": self.name,
                    "content": f"❌ Failed to load CSV: {e}",
                    "is_termination_msg": True
                }
        return None

class DataCleaner(AssistantAgent):
    async def a_generate_reply(self, sender):
        if "df" in context:
            df = context["df"]
            df_cleaned = df.dropna()
            context["df"] = df_cleaned
            return {
                "name": self.name,
                "content": f"✅ Cleaned dataset: removed NA rows, now {df_cleaned.shape[0]} rows.",
                "is_termination_msg": False
            }
        return {
            "name": self.name,
            "content": "❌ No data to clean.",
            "is_termination_msg": True
        }

def sanitize_filename(name):
    """Remove unsafe characters from filename"""
    return re.sub(r'[\\/*?:"<>|]', "_", name)

class Analyst(AssistantAgent):
    async def a_generate_reply(self, sender):
        if "df" not in context:
            return {
                "name": self.name,
                "content": "❌ No data to analyze.",
                "is_termination_msg": True
            }

        df = context["df"]
        plot_dir = "plots"
        os.makedirs(plot_dir, exist_ok=True)

        plot_files = []

        for col in df.columns:
            plt.figure(figsize=(6, 4))
            safe_col_name = sanitize_filename(col)
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].hist(bins=30)
                plt.title(f"Histogram of {col}")
                plot_path = os.path.join(plot_dir, f"{safe_col_name}_hist.png")
                plt.savefig(plot_path)
                plt.close()
                plot_files.append(plot_path)
            elif pd.api.types.is_string_dtype(df[col]) and df[col].nunique() < 20:
                df[col].value_counts().plot(kind='bar')
                plt.title(f"Bar Chart of {col}")
                plot_path = os.path.join(plot_dir, f"{safe_col_name}_bar.png")
                plt.savefig(plot_path)
                plt.close()
                plot_files.append(plot_path)

        # Prepare explanation
        columns_analyzed = [
            col for col in df.columns if (
                pd.api.types.is_numeric_dtype(df[col]) or
                (pd.api.types.is_string_dtype(df[col]) and df[col].nunique() < 20)
            )
        ]

        prompt = (
            "Explain why histograms were used for numeric columns and bar charts for categorical columns "
            f"in the following dataset columns: {', '.join(columns_analyzed)}"
        )
        gemini_response = gemini_model.generate_content(prompt)
        explanation = gemini_response.text.strip()

        return {
            "name": self.name,
            "content": (
                f"📊 Generated {len(plot_files)} plots saved in `{plot_dir}/`.\n\n"
                f"🧠 Gemini explanation:\n{explanation}\n\n"
                "All visualizations completed. Goodbye from Analyst!"
            ),
            "is_termination_msg": True
        }



# === Main Chat Loop ===

async def main():
    csv_path = input("Enter path to CSV file: ").strip()

    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        code_execution_config={"use_docker": False},
        max_consecutive_auto_reply=3,
    )

    fetcher = DataFetcher(name="DataFetcher")
    cleaner = DataCleaner(name="DataCleaner")
    analyst = Analyst(name="Analyst")

    groupchat = GroupChat(
        agents=[user_proxy, fetcher, cleaner, analyst],
        messages=[],
        max_round=20,
        speaker_selection_method="round_robin"
    )

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=False,
        is_termination_msg=lambda msg: "is_termination_msg" in msg and msg["is_termination_msg"]
    )

    await user_proxy.a_initiate_chat(
        manager,
        message=f"Please process the following CSV file: {csv_path}"
    )

if __name__ == "__main__":
    asyncio.run(main())
