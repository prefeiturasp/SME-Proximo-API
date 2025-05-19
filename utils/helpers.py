def parse_str_list(value: str, cast_type=float):
    """Transforma string separada por vírgula em lista do tipo desejado"""
    if not value:
        return []
    return [cast_type(v.strip()) for v in value.split(",")]

def normalizar_componente(componente: str) -> str:
    """Mapeia o nome do componente para o código esperado"""
    mapa = {
        "Língua portuguesa": "LP",
        "Matemática": "MT",
        "Ciências da Natureza": "CN",
        "Ciências Humanas": "CH"
    }
    return mapa.get(componente.strip(), componente.strip())