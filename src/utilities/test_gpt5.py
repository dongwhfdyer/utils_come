import os
from dotenv import load_dotenv
from openai import OpenAI


def main() -> int:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DOUABAO_API_KEY")
    base_url = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("DOUABAO_BASE_URL")
        or "https://doubao.zwchat.cn/v1"
    )

    if not api_key:
        print("Missing OPENAI_API_KEY or DOUABAO_API_KEY in .env")
        return 1

    client = OpenAI(api_key=api_key, base_url=base_url)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Say a one-sentence greeting and mention today's weather in general terms.",
        },
    ]

    try:
        resp = client.chat.completions.create(
            model="gpt-5",
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
        )
        print(resp.choices[0].message.content.strip())
        return 0
    except Exception as exc:
        print(f"Error calling model: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())



