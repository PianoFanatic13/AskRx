from uuid import uuid4

from backend.agent.graph import ask

_QUIT_COMMANDS = {"quit", "exit"}


def main() -> None:
    thread_id = str(uuid4())
    print("AskRx CLI - ask a question, or type 'quit' to exit.")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue
        if query.lower() in _QUIT_COMMANDS:
            break

        answer = ask(query, thread_id)
        print(answer.answer)
        print()


if __name__ == "__main__":
    main()
