Team - DataMiner

Team Members: 
Sayed Rahman (ssr8998)
Jingwei Ye (jy3555)
Neel Bhattacharyya (nb2772)
Shreya Shreya (ss13337)

Tar File contains:
-ReadMe.txt 
-DataGeneration.py
-Jar file
-Presentation
-**Provide access of CSV file in HDFS

Execution of DataGeneration.py on Greene:

	cmd: python DataGeneration.py --size=1000
	First Argument is the size of the data we want to create. When this argument is not provided, it will take 200000 as default value.

	Output: File named data0-4.csv will be created in the Greene & Peel path where we try to run the python file.

To move the CSV file to HDFS
	ssh to peel

	cmd: hfds dfs -put ./data0-4.csv <path_in_hdfs>data0-4.csv
	
Location of data stored in HDFS with access:
	hdfs:///user/ss13337/project/data0-4.csv

To run the jar file:
	cmd: 
	    spark-submit --class Main --master yarn <path_to_jar>/dataminer_2.11-1.0.jar <path_to_csv_in_hdfs>
	    yarn logs --applicationId <application_id> > result.txt
        
	Eg of Cmd: 
	    spark-submit --class Main --master yarn /home/ss13337/project/dataminer_2.11-1.0.jar hdfs:///user/ss13337/project/data0-4.csv
	    yarn logs --applicationId application_1619268950684_10168> result.txt

	Output: check result.txt, to see the output, it should have the normalized RMSE outputs for models somewhere in that file (mostly towards EOF). If not found, search for 'Normalized' in the result.txt file.