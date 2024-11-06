BASE_DIR = ".."

aa = set("abcdefghiklmnpqrstuxvwyzv".upper())

def validate_enzyme(seq, alphabet=aa):
    "Checks that a sequence only contains values from an alphabet"
    leftover = set(seq.upper()) - alphabet
    return not leftover
