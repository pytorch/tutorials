#!/usr/bin/env python3
import os.path
import sqlite3
from typing import List, Optional, Tuple
import boto3
import pandas as pd
import awswrangler as wr

def run_command(cmd: str, cwd: Optional[str] = None) -> str:
    """
    Run a shell command.

    Args:
        cmd: Command to run
        cwd: Working directory
    Returns:
        Output of the command.
    """
    import shlex
    from subprocess import check_output

    return check_output(shlex.split(cmd), cwd=cwd).decode("utf-8")


def get_history(cwd: Optional[str] = None) -> List[List[str]]:
    """
    Get commit history from git.
    Args:
        cwd: Working directory
    Returns:
        List of commit hashes
    """
    lines = run_command(
        "git log --date=short --pretty=format:" '%h;"%an";%ad;"%s"' " --shortstat",
        cwd=cwd,
    ).split("\n")

    def parse_string(line: str) -> str:
        """
        Parse a line into a list of strings.
        Args:
            line: Line to parse
        Returns:
            List of strings
        """
        # Add missing deletions info
        if "deletion" not in line:
            line += ", 0 deletions(-)"
        elif "insertion" not in line:
            line = ",".join(
                [line.split(",")[0], " 0 insertions(+)", line.split(",")[-1]]
            )
        return line

    def do_replace(x: str) -> str:
        """
        Replace patterns from git log with empty string. This helps us get rid of unnecessary "insertions" and "deletions"
        and we'd like to have only numbers.
        Args:   x: String to replace
        Returns:
            Replaced string
        """
        for pattern in [
            "files changed",
            "file changed",
            "insertions(+)",
            "insertion(+)",
            "deletion(-)",
            "deletions(-)",
        ]:
            x = x.replace(f" {pattern}", "")
        return x

    title = None
    rc = []
    for line in lines:
        # Check for weird entries where subject has double quotes or similar issues
        if title is None:
            title = line.split(";", 3)
        # In the lines with stat, add 0 insertions or 0 deletions to make sure we don't break the table
        elif "files changed" in line.replace("file changed", "files changed"):
            stats = do_replace(parse_string(line)).split(",")
        elif len(line) == 0:
            rc.append(title + stats)
            title = None
        else:
            rc.append(title + ["0", "0", "0"])
            title = line.split(";", 3)
    return rc

def get_file_names(
    cwd: Optional[str] = None,
) -> List[Tuple[str, List[Tuple[str, int, int]]]]:
    """
    Get file names by using the git log command as well as the git log --numstat. This gives us
    the commit hash and the number of files changed in that commit, as well as the number of
    insertions and deletions.
    Args:   cwd: Working directory
    Returns:    List of tuples (commit_hash, files)
    """
    lines = run_command('git log --pretty="format:%h" --numstat', cwd=cwd).split("\n")
    rc = []
    commit_hash = ""
    files: List[Tuple[str, int, int]] = []
    for line in lines:
        if not line:
            # Git log uses empty line as separator between commits (except for oneline case)
            rc.append((commit_hash, files))
            commit_hash, files = "", []
        elif not commit_hash:
            # First line is commit short hash
            commit_hash = line
        elif len(line.split("\t")) != 3:
            # Encountered an empty commit
            assert len(files) == 0
            rc.append((commit_hash, files))
            commit_hash = line
        else:
            added, deleted, name = line.split("\t")
            # Special casing for binary files
            if added == "-":
                assert deleted == "-"
                files.append((name, -1, -1))
            else:
                files.append((name, int(added), int(deleted)))
    return rc


def connect_db(dfile: str) -> sqlite3.Connection:
    """
    Open a database connection to a file.
    Args:   dfile: Path to database
    Returns:    Database connection
    """
    db_connect = None
    try:
        db_connect = sqlite3.connect(dfile)
    except sqlite3.Error as error:
        print(error)
    return db_connect


def create_db_schema(handle: sqlite3.Connection) -> None:
    """
    Create the database schema. Specifying the table name here will allow us to drop the table and re-create it later.
    The create_table_statement statement speficies the table name, the column names and data types.
    Args:   handle: Database connection
    Returns:    None
    """
    delete_table = "DROP TABLE IF EXISTS commits;"
    create_table_statement = """CREATE TABLE IF NOT EXISTS commits(
                commit_id TEXT,
                author TEXT,
                date DATE,
                title TEXT,
                number_of_changed_files INT,
                lines_added INT,
                lines_deleted INT);
                """
    execute_statement(handle, delete_table)
    execute_statement(handle, create_table_statement)
    delete_table = "DROP TABLE IF EXISTS files;"
    create_table_statement = """CREATE TABLE IF NOT EXISTS files(
                commit_id TEXT,
                file_name TEXT,
                lines_added INT,
                lines_deleted INT);
                """
    execute_statement(handle, delete_table)
    execute_statement(handle, create_table_statement)


def create_dataframe(conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    """
    Create a pandas dataframe from the SQL database we created in create_db_schema.
    Args:   conn: Database connection
    Returns:    pandas dataframe
    """
    return pd.read_sql(f"SELECT * from {table_name}", conn, index_col=None)

def merge_dataframe(df1, df2) -> pd.DataFrame:
    """
    Merge two dataframes together by joining them on Commit ID.
    Args:   df1: pandas dataframe
    Returns:    pandas dataframe
    """
    df = pd.merge(left=df1, right=df2, left_on="commit_id", right_on="commit_id")
    return df

def print_dataframe() -> None:
    """
    Print the contents of the dataframe created in the previous step.
    Args:   None
    Returns:    None
    """
    conn = connect_db("test.db")
    df = merge_dataframe(
        create_dataframe(conn, "commits"), create_dataframe(conn, "files")
    )
    print(df)

def upload_df_to_dynamo(conn: sqlite3.Connection, df1, df2, table_name):
    df = pd.merge(
        left=df1[["commit_id", "author", "date", "title", "number_of_changed_files"]],
        right=df2,
        left_on="commit_id",
        right_on="commit_id",
    )
    print(df)
    wr.dynamodb.put_df(df, table_name)

def main() -> None:
    tutorials_dir = os.path.expanduser("~/tutorials")
    get_history_log = get_history(tutorials_dir)
    commits_to_files = get_file_names(tutorials_dir)
    db = "test.db"
    connect = connect_db(db)
    create_db_schema(connect)
    for entry in get_history_log:
        cursor = connect.cursor()
        cursor.execute("INSERT INTO commits VALUES (?, ?, ?, ?, ?, ?, ?)", entry)
        connect.commit()
    for entry in commits_to_files:
        cursor = connect.cursor()
        commit_id, files = entry
        for (fname, lines_added, lines_deleted) in files:
            cursor.execute(
                "INSERT INTO files VALUES (?, ?, ?, ?)",
                (commit_id, fname, lines_added, lines_deleted),
            )
        connect.commit()
    create_dataframe(connect, "files")
    create_dataframe(connect, "commits")
    upload_to_dynamo(connect, create_dataframe(connect, "commits"), create_dataframe(connect, "files"), 'torchci-tutorial-metadata')

if __name__ == "__main__":
    main()
