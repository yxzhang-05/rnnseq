class colorstrings:
    BOLD = '\033[1m'
    CYAN = '\033[96m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    END = '\033[0m'

def CYAN_STR(text):
    return f"{colorstrings.CYAN}{text}{colorstrings.END}"

def RED_STR(text):
    return f"{colorstrings.RED}{text}{colorstrings.END}"

def END_STR(text):
    return text + colorstrings.END