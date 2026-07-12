"""Tool calling and agent routing benchmark task."""
import re

from benchmark_plugin import BenchmarkTaskPlugin


class ToolCallingPlugin(BenchmarkTaskPlugin):
    @property
    def id(self):
        return "tool-calling"

    @property
    def version(self):
        return "0.1.0"

    @property
    def name(self):
        return "Tool Calling Agent"

    @property
    def max_score(self):
        return 15.0

    @property
    def supports_streaming(self):
        return True

    def get_prompt(self):
        return (
            "You are an agent with access to the following tools:\n"
            "1. get_weather(location: str)\n"
            "2. get_stock_price(ticker: str)\n"
            "3. convert_currency(amount: float, from_curr: str, to_curr: str)\n\n"
            "User Request: 'I am planning a trip to Tokyo. What's the weather there right now? "
            "Also, check the stock price of Sony (SONY) and tell me how much 1000 USD is in JPY.'\n\n"
            "Output your tool calls exactly in this format block:\n"
            "<tool_call>{\"name\": \"tool_name\", \"args\": {\"arg1\": \"val1\"}}</tool_call>\n\n"
            "After outputting the necessary tool calls sequentially, synthesize a mock final response "
            "assuming hypothetical return values."
        )

    def get_temperature(self, global_config):
        if "tool_calling_temperature" in global_config:
            return global_config["tool_calling_temperature"]
        if "general_temperature" in global_config:
            return global_config["general_temperature"]
        return None

    def score(self, response_text):
        t = response_text
        s = 0.0

        # 1. Output format compliance (0-3)
        if re.search(r'<tool_call>.*?</tool_call>', t, re.DOTALL):
            s += 3.0

        # 2. Weather tool call (0-3)
        if re.search(r'"name"\s*:\s*"get_weather"', t):
            s += 1.5
        if re.search(r'"location"\s*:\s*"[^"]*[Tt]okyo[^"]*"', t):
            s += 1.5

        # 3. Stock ticker tool (0-3)
        if re.search(r'"name"\s*:\s*"get_stock_price"', t):
            s += 1.5
        if re.search(r'"ticker"\s*:\s*"SONY"', t):
            s += 1.5

        # 4. Currency conversion tool (0-3)
        if re.search(r'"name"\s*:\s*"convert_currency"', t):
            s += 1.0
        if re.search(r'"amount"\s*:\s*1000', t):
            s += 1.0
        if re.search(r'"from_curr"\s*:\s*"USD"', t) or re.search(r'"to_curr"\s*:\s*"JPY"', t):
            s += 1.0

        # 5. Synthesis / mock final response (0-3)
        has_weather = re.search(r'(?:weather|degrees|celsius|fahrenheit|sunny|rain|cloud)', t.lower())
        has_stock = re.search(r'(?:price|shares|stock|sony)', t.lower())
        has_currency = re.search(r'(?:yen|jpy|usd|currency|exchange)', t.lower())
        if has_weather and has_stock and has_currency:
            s += 3.0
        elif (has_weather and has_stock) or (has_stock and has_currency) or (has_weather and has_currency):
            s += 1.5

        return round(min(s, self.max_score), 1)
