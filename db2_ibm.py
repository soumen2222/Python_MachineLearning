import ibm_db
import pandas
import ibm_db_dbi


# Replace the placeholder values with your actual Db2 hostname, username, and password:
# e.g.: "dashdb-txn-sbox-yp-dal09-04.services.dal.bluemix.net"
dsn_hostname = "dashdb-txn-sbox-yp-lon02-01.services.eu-gb.bluemix.net"
dsn_uid = "lrn61010"        # e.g. "abc12345"
dsn_pwd = "4s713hc-4w4nvbv4"      # e.g. "7dBZ3wWt9XN6$o0J"

dsn_driver = "{IBM DB2 ODBC DRIVER}"
dsn_database = "BLUDB"            # e.g. "BLUDB"
dsn_port = "50000"                # e.g. "50000"
dsn_protocol = "TCPIP"            # i.e. "TCPIP"


# DO NOT MODIFY THIS CELL. Just RUN it with Shift + Enter
# Create the dsn connection string
dsn = (
    "DRIVER={0};"
    "DATABASE={1};"
    "HOSTNAME={2};"
    "PORT={3};"
    "PROTOCOL={4};"
    "UID={5};"
    "PWD={6};").format(dsn_driver, dsn_database, dsn_hostname, dsn_port, dsn_protocol, dsn_uid, dsn_pwd)

# print the connection string to check correct values are specified
print(dsn)

# DO NOT MODIFY THIS CELL. Just RUN it with Shift + Enter
# Create database connection

try:
    conn = ibm_db.connect(dsn, "", "")
    print("Connected to database: ", dsn_database,
          "as user: ", dsn_uid, "on host: ", dsn_hostname)

except:
    print("Unable to connect: ", ibm_db.conn_errormsg())


# Retrieve Metadata for the Database Server
server = ibm_db.server_info(conn)

print("DBMS_NAME: ", server.DBMS_NAME)
print("DBMS_VER:  ", server.DBMS_VER)
print("DB_NAME:   ", server.DB_NAME)


# Retrieve Metadata for the Database Client / Driver
client = ibm_db.client_info(conn)

print("DRIVER_NAME:          ", client.DRIVER_NAME)
print("DRIVER_VER:           ", client.DRIVER_VER)
print("DATA_SOURCE_NAME:     ", client.DATA_SOURCE_NAME)
print("DRIVER_ODBC_VER:      ", client.DRIVER_ODBC_VER)
print("ODBC_VER:             ", client.ODBC_VER)
print("ODBC_SQL_CONFORMANCE: ", client.ODBC_SQL_CONFORMANCE)
print("APPL_CODEPAGE:        ", client.APPL_CODEPAGE)
print("CONN_CODEPAGE:        ", client.CONN_CODEPAGE)


"""

his notebook illustrates how to access your database instance using Python by following the steps below:

Import the ibm_db Python library
Identify and enter the database connection credentials
Create the database connection
Create a table
Insert data into the table
Query data from the table
Retrieve the result set into a pandas dataframe
Close the database connection
Notice: Please follow the instructions given in the first Lab of this course to Create a database service instance of Db2 on Cloud.

"""

"""
# Lets first drop the table INSTRUCTOR in case it exists from a previous attempt
dropQuery = "drop table INSTRUCTOR"

# Now execute the drop statment
dropStmt = ibm_db.exec_immediate(conn, dropQuery)


createQuery = "create table INSTRUCTOR(ID INTEGER PRIMARY KEY NOT NULL, FNAME VARCHAR(20), LNAME VARCHAR(20), CITY VARCHAR(20), CCODE CHAR(2))"

createStmt = ibm_db.exec_immediate(conn, createQuery)


insertQuery = "insert into INSTRUCTOR values (3, 'Rav', 'Ahuja', 'TORONTO', 'CA')"

insertStmt = ibm_db.exec_immediate(conn, insertQuery)


insertQuery2 = "insert into INSTRUCTOR values (4, 'Raul', 'Chong', 'Markham', 'CA'), (3, 'Hima', 'Vasudevan', 'Chicago', 'US')"

insertStmt2 = ibm_db.exec_immediate(conn, insertQuery2)

"""
# Construct the query that retrieves all rows from the INSTRUCTOR table
selectQuery = "select * from INSTRUCTOR"

# Execute the statement
selectStmt = ibm_db.exec_immediate(conn, selectQuery)

# Fetch the Dictionary (for the first row only)
ibm_db.fetch_both(selectStmt)

# Fetch the rest of the rows and print the ID and FNAME for those rows
while ibm_db.fetch_row(selectStmt) != False:
    print(" ID:",  ibm_db.result(selectStmt, 0),
          " FNAME:",  ibm_db.result(selectStmt, "FNAME"))


updateQuery = "update INSTRUCTOR set CITY='MOOSETOWN' where FNAME='Rav'"
updateStmt = ibm_db.exec_immediate(conn, updateQuery)

pconn = ibm_db_dbi.Connection(conn)
# query statement to retrieve all rows in INSTRUCTOR table
selectQuery = "select * from INSTRUCTOR"

# retrieve the query results into a pandas dataframe
pdf = pandas.read_sql(selectQuery, pconn)

# print just the LNAME for first row in the pandas data frame
print(pdf.LNAME[0])

print(pdf, pdf.shape)

ibm_db.close(conn)
