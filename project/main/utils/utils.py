import ast


def type_of(obj: str):
    try:
        type_value = type(ast.literal_eval(obj)).__name__
        return type_value
    except (ValueError, SyntaxError):
        return type(obj).__name__
