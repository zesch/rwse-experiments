import cassis

def read_to_cas(filepath: str) -> str:
    cas = cassis.Cas()
    with open(filepath) as f:
        sentences = f.readlines()
        for sentence in sentences:
            pass
