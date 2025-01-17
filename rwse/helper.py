import cassis

path = '../data/TypeSystem.xml'

with open(path, 'rb') as f:
    cassis.load_typesystem(f)