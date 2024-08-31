""""""
import os
import pandas as pd
from sqlalchemy import create_engine
from contextlib import contextmanager


db_user: str = os.environ.get('USER', '')
db_password: str = os.environ.get('PWD', '')
db_host: str = os.environ.get('HOST', '')
db_port: int = int(os.environ.get('PORT', 3306))


@contextmanager
def db_session(engine):
    """Context manager for SQLAlchemy engine."""
    connection = engine.connect()
    try:
        yield connection
    finally:
        connection.close()


def import_data(engine, file_path) -> None:
    """Imports CSV data into a MySQL table with bulk insert."""
    try:
        df = pd.read_csv(file_path)
        table_name = os.path.splitext(os.path.basename(file_path))[0]

        with db_session(engine) as conn:
            df.to_sql(table_name, con=conn, if_exists='append',
                      index=False, method='multi')

        print(f"Data imported successfully into `{table_name}`!")
    except Exception as e:
        print({"error": str(e), "file": file_path})


def main() -> None:
    dir_path = input("Enter the directory path: ").strip(
        '"').replace("\\", "/")
    db_name = input("Enter the database name: ")

    engine = create_engine(
        f'mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

    file_paths = [os.path.join(root, file)
                  for root, _, files in os.walk(dir_path) for file in files if file.endswith(".csv")]

    if file_paths:
        for file_path in file_paths:
            import_data(engine, file_path)
    else:
        print("No CSV files found in the specified directory.")


if __name__ == "__main__":
    main()
