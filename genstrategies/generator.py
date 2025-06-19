import openai
class Generator:
    def __init__(self, local_embeddings=None, outsource_embeddings=None, cloud_embeddings=None, trend_prediction=None, model="gpt-4"):
        self.openai = openai
        self.local_embeddings = local_embeddings
        self.outsource_embeddings = outsource_embeddings
        self.cloud_embeddings = cloud_embeddings
        self.trend_prediction = trend_prediction
        self.status_local_embeddings = bool(local_embeddings)
        self.status_outsource_embeddings = bool(outsource_embeddings)
        self.status_cloud_embeddings = bool(cloud_embeddings)
        self.status_trend_prediction = bool(trend_prediction)
        self.model = "gpt-4"
        self.strategy_prompt = """
        You are a business strategy expert.
        Based on the following:
        1. Summary of internal documents: {local_summary}
        2. Online insights and trends: {online_summary}
        
        Generate a strategic recommendation, including:
        - Risk and Opportunities
        - Suggested Actions
        - Market Positioning Ideas
        """

    def _format_embeddings(self, data):
        if isinstance(data, dict):
            return "\n\n".join([f"{k}:\n{v}" for k, v in data.items()])
        elif isinstance(data, (list, tuple)):
            return "\n".join(str(item) for item in data)
        return data if isinstance(data, str) else "N/A"


    def generate_strategies(self):
        if not any([self.status_local_embeddings, self.status_outsource_embeddings, self.status_cloud_embeddings, self.status_trend_prediction]):
            raise ValueError("No valid data provided to generate strategy")

        
        prompt = self.strategy_prompt.format(
            # local_summary=self.local_embeddings or "N/A",
            # online_summary=self.outsource_embeddings or "N/A",
            # cloud_embeddings=self.cloud_embeddings or "N/A",
            # trend_prediction=self.trend_prediction or "N/A"
            local_summary=self._format_embeddings(self.local_embeddings),
            online_summary=self._format_embeddings(self.outsource_embeddings),
            cloud_summary=self._format_embeddings(self.cloud_embeddings),
            trend_prediction=self._format_embeddings(self.trend_prediction)
        )
        try:
            response = self.openai.chat.completions.create(
                model = self.model,
                messages = [{"role":"user", "content": prompt}],
                temperature=0.4
            )
            result = {
                "response": response.choices[0].message.content,
                "prompt": self.strategy_prompt
            }
            return result

        except Exception as e:
            print(f"Error geenrating strategies: {e}")
            return None

    def set_strategy_prompt(self, prompt: str):
        self.strategy_prompt = prompt
        return self.strategy_prompt