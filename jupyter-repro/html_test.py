import json
from io import StringIO

import pandas as pd
from bs4 import BeautifulSoup


def _get_cell_html(nb_path, cell_index):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb.get('cells', [])

    if cell_index < 1 or cell_index > len(cells):
        return None

    cell = cells[cell_index - 1]
    if cell.get('cell_type') != 'code':
        return None

    for output in cell.get('outputs', []):
        data = output.get('data', {})
        if 'text/html' in data:
            return ''.join(data['text/html'])
    return None


def clean_table_html(html_content, table_index=0, drop_ellipsis_rows=True):
    if html_content is None:
        return None

    soup = BeautifulSoup(html_content, 'html.parser')
    tables = soup.find_all('table')
    if table_index < 0 or table_index >= len(tables):
        return None

    table = tables[table_index]

    if drop_ellipsis_rows:
        for tr in table.find_all('tr'):
            cells = tr.find_all(['th', 'td'])
            cell_texts = [c.get_text(strip=True) for c in cells]
            if cell_texts and all(text == '...' for text in cell_texts):
                tr.decompose()

    return str(table)


def parse_dataframe_with_bs4(
    nb_path,
    cell_index,
    table_index=0,
    skiprows=None,
    drop_ellipsis_rows=True,
    return_html=False,
):
    html_content = _get_cell_html(nb_path, cell_index)
    if html_content is None:
        return (None, None) if return_html else None

    cleaned_html = clean_table_html(
        html_content,
        table_index=table_index,
        drop_ellipsis_rows=drop_ellipsis_rows,
    )
    if cleaned_html is None:
        return (None, None) if return_html else None

    tables = pd.read_html(StringIO(cleaned_html), skiprows=skiprows)
    if not tables:
        return (None, cleaned_html) if return_html else None

    df = tables[0]
    if return_html:
        return df, cleaned_html
    return df

def get_dataframe(nb_path, cell_index):
    return parse_dataframe_with_bs4(
        nb_path,
        cell_index,
        table_index=0,
        skiprows=None,
        drop_ellipsis_rows=False,
        return_html=False,
    )


def get_html(nb_path, cell_index, as_display_object=True):
    html_content = _get_cell_html(nb_path, cell_index)
    if as_display_object:
        return html_content
    return clean_table_html(html_content)

    


    

if __name__ == "__main__":
        pass


