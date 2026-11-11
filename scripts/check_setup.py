
"""Check that Foundry and Azure AI Search work. Usage: python -m scripts.check_setup"""
from app import config


def main() -> None:
    #tests chat model
    reply = config.openai_client().chat.completions.create(
        model=config.CHAT_DEPLOYMENT,
        messages=[{"role": "user", "content": "Say hello in Norwegian in three words."}],
    )
    print("Chat:", reply.choices[0].message.content)

    #tests embed model
    vector = config.embed(["trapp"])[0]
    print("Embedding length:", len(vector), "(expected", config.EMBEDDING_DIMENSIONS, ")")

    #tests Azure AI search
    names = [index.name for index in config.index_client().list_indexes()]
    print("Search indexes:", names)
    print("All three services work.")


if __name__ == "__main__":
    main()
