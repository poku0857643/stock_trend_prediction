class Generator:
    def __init__(self, local_embeddings=None, outsource_embeddings=None, cloud_embeddings=None, trend_prediction=None, model="gpt-4"):
        self.local_embeddings = local_embeddings
        self.outsource_embeddings = outsource_embeddings
        self.cloud_embeddings = cloud_embeddings
        self.trend_prediction = trend_prediction
        self.status_local_embeddings = bool(local_embeddings)
        self.status_outsource_embeddings = bool(outsource_embeddings)
        self.status_cloud_embeddings = bool(cloud_embeddings)
        self.status_trend_prediction = bool(trend_prediction)
        self.model = "gpt-4"

    def generate_strategies(self):
        if not any([self.status_local_embeddings, self.status_outsource_embeddings, self.status_cloud_embeddings, self.status_trend_prediction]):
            raise ValueError("No valid data provided to generate strategy")

        prompt = STRATEGY_PROMPT.format(
            local_summary=self.local_embeddings or "N/A",
            online_summary=self.outsource_embeddings or "N/A",
            cloud_embeddings=self.cloud_embeddings or "N/A",
            trend_prediction=self.trend_prediction or "N/A"
        )
        try:
            response = openai.chat.completions.create(
                model = self.model,
                messages = [{"role":"user", "content": prompt}],
                temperature=0.4
            )
            return response.choices[0].message.content
        except Exception as e:
            return F"Error during strategy generation: {e}"

    def set_strategy_prompt(self):
        STRATEGY_PROMPT = """
        You are a business strategy expert.
        Based on the following:
        1. Summary of internal documents: {local_summary}
        2. Online insights and trends: {online_summary}
        
        Generate a strategic recommendation, including:
        - Risk and Opportunities
        - Suggested Actions
        - Market Positioning Ideas
        """
        strategy_prompt = STRATEGY_PROMPT
        return strategy_prompt