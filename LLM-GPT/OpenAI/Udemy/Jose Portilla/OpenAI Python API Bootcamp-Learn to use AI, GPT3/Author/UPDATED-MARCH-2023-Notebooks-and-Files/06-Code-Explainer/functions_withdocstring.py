def check_for_duplicate_links(path_to_new_content, links):
    """
    Checks if the new content is already in the index.
    :param path_to_new_content: The path to the new content.
    :param links: The links in the index.
    :return: True if the new content is already in the index.
    """
    urls = [str(link.get("href")) for link in links]
    content_path = str(Path(*path_to_new_content.parts[-2:]))
    return content_path in urls


def dataframe_to_database(df, table_name):
    """
    This function takes a pandas dataframe and a table name as input and returns a sqlite engine.
    The function creates a sqlite engine in memory and stores the dataframe in the engine as a table.
    The function returns the engine.
    """
    engine = create_engine(f'sqlite:///:memory:', echo=False)
    df.to_sql(name=table_name, con=engine, index=False)
    return engine


def execute_query(engine, query):
    """
    Executes a query on a database engine.

    Parameters
    ----------
    engine : sqlalchemy.engine.base.Engine
        The database engine to execute the query on.
    query : str
        The query to execute.

    Returns
    -------
    list
        The result of the query.
    """
    with engine.connect() as conn:
        result = conn.execute(text(query))
        return result.fetchall()


def grade(correct_answer_dict, answers):
    """
    This function takes two arguments:
    correct_answer_dict: a dictionary with the correct answers to the questions
    answers: a dictionary with the answers given by the user

    The function returns a string with the number of correct answers, the total number of questions, the percentage of correct answers and whether the user passed or not.
    """
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


def handle_response(response):
    """
    This function takes in a response from the user and returns a query to be
    sent to the database.
    :param response: A response from the user
    :return: A query to be sent to the database
    """
    query = response["choices"][0]["text"]
    if query.startswith(" "):
        query = "Select"+ query
    return query
