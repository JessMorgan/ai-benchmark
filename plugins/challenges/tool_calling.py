"""Tool calling and agent routing benchmark task."""
import json
import re

from benchmark_plugin import BenchmarkTaskPlugin
from plugins.challenges._rubric import Rubric


class ToolCallingPlugin(BenchmarkTaskPlugin):
    @property
    def id(self):
        return "tool-calling"

    @property
    def version(self):
        return "0.2.0"

    @property
    def name(self):
        return "Tool Calling Agent"

    @property
    def max_score(self):
        return 25.0

    @property
    def supports_streaming(self):
        return True

    def get_prompt(self):
        return (
            "You are a travel-planning agent with access to the following tools:\n"
            "1. get_weather(location: str, unit: str = 'celsius')\n"
            "2. search_flights(origin: str, destination: str, date: str)\n"
            "3. book_hotel(city: str, check_in: str, check_out: str, guests: int)\n"
            "4. get_stock_price(ticker: str)\n"
            "5. convert_currency(amount: float, from_curr: str, to_curr: str)\n"
            "6. send_email(to: str, subject: str, body: str)\n\n"
            "User Request: 'I am planning a business trip from New York (JFK) to Tokyo "
            "departing on 2024-08-15. I need a hotel in Tokyo from 2024-08-16 to "
            "2024-08-20 for 2 guests. Please check the weather in Tokyo in celsius, "
            "search for the flight, book the hotel, check Sony's stock price (SONY), "
            "convert 1000 USD to JPY, and email the itinerary to alice@example.com with "
            "subject 'Tokyo Trip Itinerary'.'\n\n"
            "First, briefly plan which tools you will call and in what order "
            "inside a <plan>...</plan> block. "
            "Then output each tool call exactly in this format block:\n"
            "<tool_call>{\"name\": \"tool_name\", \"args\": {\"arg1\": \"val1\"}}</tool_call>\n\n"
            "After outputting the necessary tool calls sequentially, synthesize a mock final response "
            "assuming hypothetical return values. Include the converted amount in JPY."
        )

    def get_temperature(self, global_config):
        if "tool_calling_temperature" in global_config:
            return global_config["tool_calling_temperature"]
        return None

    @staticmethod
    def _extract_tool_calls(response_text):
        """Extract all <tool_call>...</tool_call> blocks and parse JSON inside."""
        calls = []
        for match in re.finditer(r'<tool_call>(.*?)</tool_call>', response_text, re.DOTALL):
            raw = match.group(1).strip()
            try:
                calls.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        return calls

    def evaluate(self, response_text):
        t = response_text
        rubric = Rubric(self.max_score)

        earned = 0.0
        tool_call_blocks = re.findall(r'<tool_call>.*?</tool_call>', t, re.DOTALL)
        if tool_call_blocks:
            earned += 2.0
            parsed = self._extract_tool_calls(t)
            if len(parsed) >= 2:
                earned += 1.0
        rubric.add_criterion("Output format compliance", 3.0, earned)

        earned = 0.0
        tool_call_index = t.find('<tool_call>')
        before_tools = t[:tool_call_index] if tool_call_index != -1 else t
        before_tools_sample = before_tools[:1000]
        if re.search(r'<plan>.*?</plan>', before_tools_sample, re.DOTALL):
            earned += 1.5
        if re.search(r'(?i)(plan|step|first|then|finally|order|sequence)', before_tools_sample):
            earned += 0.5
        rubric.add_criterion("Planning / reasoning", 2.0, earned)

        calls = self._extract_tool_calls(t)
        call_names = [c.get("name") for c in calls if isinstance(c, dict)]
        args_list = [c.get("args", {}) for c in calls if isinstance(c, dict)]

        required_tools = {
            "get_weather",
            "search_flights",
            "book_hotel",
            "get_stock_price",
            "convert_currency",
            "send_email",
        }
        present_tools = set(call_names) & required_tools
        earned = (len(present_tools) / len(required_tools)) * 5.0 if present_tools else 0.0
        rubric.add_criterion("Required tools present", 5.0, earned)

        arg_score = 0.0
        for name, args in zip(call_names, args_list):
            if not isinstance(args, dict):
                continue
            if name == "get_weather":
                loc = args.get("location", "")
                if isinstance(loc, str) and "tokyo" in loc.lower():
                    arg_score += 0.5
                if args.get("unit", "").lower() in {"celsius", "c"}:
                    arg_score += 0.5
            elif name == "search_flights":
                origin = args.get("origin", "")
                dest = args.get("destination", "")
                date = args.get("date", "")
                if isinstance(origin, str) and "jfk" in origin.lower():
                    arg_score += 0.5
                if isinstance(dest, str) and "tokyo" in dest.lower():
                    arg_score += 0.5
                if re.match(r"^\d{4}-\d{2}-\d{2}$", str(date)):
                    arg_score += 0.5
            elif name == "book_hotel":
                city = args.get("city", "")
                check_in = args.get("check_in", "")
                check_out = args.get("check_out", "")
                guests = args.get("guests")
                if isinstance(city, str) and "tokyo" in city.lower():
                    arg_score += 0.5
                if re.match(r"^\d{4}-\d{2}-\d{2}$", str(check_in)) and re.match(r"^\d{4}-\d{2}-\d{2}$", str(check_out)):
                    arg_score += 0.5
                if isinstance(guests, int) and guests == 2:
                    arg_score += 0.5
            elif name == "get_stock_price":
                ticker = args.get("ticker", "")
                if isinstance(ticker, str) and ticker.upper() == "SONY":
                    arg_score += 0.5
            elif name == "convert_currency":
                amount = args.get("amount")
                from_curr = args.get("from_curr", "")
                to_curr = args.get("to_curr", "")
                if amount == 1000:
                    arg_score += 0.5
                if str(from_curr).upper() == "USD" and str(to_curr).upper() == "JPY":
                    arg_score += 0.5
            elif name == "send_email":
                to = args.get("to", "")
                subject = args.get("subject", "")
                if isinstance(to, str) and "alice@example.com" in to.lower():
                    arg_score += 0.5
                if isinstance(subject, str) and "tokyo" in subject.lower():
                    arg_score += 0.5
        rubric.add_criterion("Correct arguments", 8.0, arg_score)

        expected_order = [
            "get_weather",
            "search_flights",
            "book_hotel",
            "get_stock_price",
            "convert_currency",
            "send_email",
        ]
        order_matches = sum(1 for a, b in zip(call_names, expected_order) if a == b)
        earned = 0.0
        if len(call_names) >= 3 and order_matches > 0:
            earned = (order_matches / len(expected_order)) * 3.0
        rubric.add_criterion("Correct ordering / dependencies", 3.0, earned)

        has_weather = re.search(r'(?:weather|degrees|celsius|fahrenheit|sunny|rain|cloud)', t.lower())
        has_flight = re.search(r'(?:flight|jfk|tokyo|depart)', t.lower())
        has_hotel = re.search(r'(?:hotel|check.in|guests)', t.lower())
        has_stock = re.search(r'(?:price|shares|stock|sony)', t.lower())
        has_currency = re.search(r'(?:yen|jpy|usd|currency|exchange)', t.lower())
        has_email = re.search(r'(?:email|itinerary|alice)', t.lower())
        synthesis_hits = sum([bool(has_weather), bool(has_flight), bool(has_hotel), bool(has_stock), bool(has_currency), bool(has_email)])
        earned = round((synthesis_hits / 6.0) * 4.0, 1)
        rubric.add_criterion("Synthesis / final response", 4.0, earned)

        return rubric.results()

    def score(self, response_text):
        return self.evaluate(response_text)[0]
