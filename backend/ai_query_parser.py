from mo_sql_parsing import parse

def extract_keywords(where):
    if isinstance(where, dict):
        if "like" in where:
            value = where["like"]
            if isinstance(value, list) and len(value) == 2:
                literal = value[1]
                if isinstance(literal, dict) and "literal" in literal:
                    return str(literal["literal"])
                return str(literal)
            return str(value)
        elif "and" in where:
            return " ".join(extract_keywords(w) for w in where["and"])
        elif "or" in where:
            return " ".join(extract_keywords(w) for w in where["or"])
    return ""

def parse_sql_query(sql_query: str):
    try:
        parsed = parse(sql_query)
        print("QUERY PARSEADA: ", parsed)
        table_name = parsed.get("from") or parsed.get("FROM") or "Audio"
        limit = parsed.get("limit") or parsed.get("LIMIT") or 5
        where_clause = parsed.get("where") or parsed.get("WHERE") or {}
        select_clause = parsed.get("select") or parsed.get("SELECT") or []

        # Extraer campos seleccionados correctamente
        selected_fields = []
        if isinstance(select_clause, dict):
            selected_fields = [select_clause.get("value")]
        elif isinstance(select_clause, list):
            for f in select_clause:
                if isinstance(f, dict) and "value" in f:
                    selected_fields.append(f["value"])
                elif isinstance(f, str):
                    selected_fields.append(f)
        else:
            selected_fields = ["track_name", "track_artist"]

        query_text_raw = extract_keywords(where_clause)
        query_text = " ".join(str(query_text_raw).split())

        return table_name, query_text, limit, selected_fields

    except Exception as e:
        print(f"❌ Error parseando SQL: {e}")
        return "", "", 5, []
