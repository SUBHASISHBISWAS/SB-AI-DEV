from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy import text
import numpy as np

def dataframe_to_database(df, table_name):
    engine = create_engine(f'sqlite:///:memory:', echo=False)
    df.to_sql(name=table_name, con=engine, index=False)
    return engine

def handle_response(response):
    query = response["choices"][0]["text"]
    if query.startswith(" "):
        query = "Select"+ query
    return query

def execute_query(engine, query):
    with engine.connect() as conn:
        result = conn.execute(text(query))
        return result.fetchall()

def grade(correct_answer_dict, answers):
    correct_answers = 0
    for question, answer in answers.items():
        if answer.upper() == correct_answer_dict[question].upper()[16]:
            correct_answers+=1
    grade = 100 * correct_answers / len(answers)

    if grade < 60:
        passed = "Not passed!"
    else:
        passed = "Passed!"
    return f"{correct_answers} out of {len(answers)} correct! You achieved: {grade} % : {passed}"


def check_for_duplicate_links(path_to_new_content, links):
    urls = [str(link.get("href")) for link in links]
    content_path = str(Path(*path_to_new_content.parts[-2:]))
    return content_path in urls
