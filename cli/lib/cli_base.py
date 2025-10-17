import argparse
from abc import ABC, abstractmethod
from typing import Any, Callable


class CommandHandler:
    def __init__(self, func: Callable[..., Any], description: str):
        self.func = func
        self.description = description
        self.args: list[tuple[str, dict[str, Any]]] = []

    def add_argument(self, *args: str, **kwargs: Any) -> "CommandHandler":
        self.args.append((args, kwargs))
        return self

    def execute(self, parsed_args: argparse.Namespace) -> Any:
        kwargs = {}
        for arg_names, _ in self.args:
            arg_name = arg_names[-1].lstrip("-").replace("-", "_")
            if hasattr(parsed_args, arg_name):
                kwargs[arg_name] = getattr(parsed_args, arg_name)
        return self.func(**kwargs)


class BaseCLI(ABC):
    def __init__(self, description: str):
        self.description = description
        self.commands: dict[str, CommandHandler] = {}

    def add_command(self, name: str, handler: CommandHandler) -> None:
        self.commands[name] = handler

    def command(self, name: str, description: str) -> Callable[[Callable[..., Any]], CommandHandler]:
        def decorator(func: Callable[..., Any]) -> CommandHandler:
            handler = CommandHandler(func, description)
            self.add_command(name, handler)
            return handler
        return decorator

    def run(self) -> None:
        parser = argparse.ArgumentParser(description=self.description)
        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        for name, handler in self.commands.items():
            subparser = subparsers.add_parser(name, help=handler.description)
            for arg_names, kwargs in handler.args:
                subparser.add_argument(*arg_names, **kwargs)

        args = parser.parse_args()

        if args.command in self.commands:
            try:
                result = self.commands[args.command].execute(args)
                if result is not None:
                    self.handle_result(args.command, result)
            except Exception as e:
                print(f"Error executing {args.command}: {e}")
        else:
            parser.print_help()

    @abstractmethod
    def handle_result(self, command: str, result: Any) -> None:
        pass