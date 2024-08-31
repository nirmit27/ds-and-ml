import configparser as cfgp
import mysql.connector as msc
from mysql.connector import Error as mscError


def connect():
    creds = fetch_config()
    connection = any

    if len(creds) == 4:
        host = creds[0]
        user = creds[1]
        pwd = creds[2]
        db = creds[3]

    if len(creds) < 4:
        host = input("Enter host name : ")
        db = input("Enter database name : ")
        user = input("Enter username : ")
        pwd = input("Enter password : ")

    try:
        connection = msc.connect(
            user=user,
            password=pwd,
            host=host,
            database=db
        )
        print(f"\nConnected successfully!")
        return connection

    except mscError as e:
        print(f"\nConnection error - {e}\n")
        return None


def query(c):
    if c is not None:
        cursor = c.cursor(dictionary=True)

        q = input("\nEnter your query :\n\n")
        cursor.execute(q)

        print(f"\n\nQuery ran successfully! Results :\n\n")
        result = cursor.fetchall()

        for row in result:
            print(row, end='\n\n')
    return


def fetch_config() -> list:
    creds = []

    config = cfgp.ConfigParser()
    config.read("sql.cfg")

    for key in config['SQLCONNECT']:
        creds.append(config['SQLCONNECT'][key])

    return creds


if __name__ == "__main__":
    connection = connect()
    query(connection)
