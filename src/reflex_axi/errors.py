class ReflexError(Exception):
    def __init__(self, code: str, message: str, help: str = "reflex-axi --help") -> None:
        super().__init__(message)
        self.code = code
        self.help = help
