RESTORE FILELISTONLY FROM DISK='/var/opt/mssql/backups/AdventureWorks2017.bak'


RESTORE DATABASE AdventureWorks FROM DISK = '/var/opt/mssql/backups/AdventureWorks2017.bak'
WITH MOVE 'AdventureWorks2017'  to  '/var/opt/mssql/data/awd_2017.mdf',
MOVE 'AdventureWorks2017_log'  to  '/var/opt/mssql/data/awd_2017_log.ldf';
