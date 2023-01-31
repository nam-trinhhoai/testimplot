/*
 *
 *
 *  Created on: 27 June 2022
 *      Author: l0359127
 */

#ifndef NEXTVISION_SRC_WIDGET_WGEOMECHANICS_WELLLOGSIO_H_
#define NEXTVISION_SRC_WIDGET_WGEOMECHANICS_WELLLOGSIO_H_

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <sstream>
#include <stdio.h>

class WellLogsIO
{

public:
	WellLogsIO();

	~WellLogsIO();

	bool readLASHeader(std::string fileName, std::vector<std::string> & logNames);
	bool readLASData(std::string fileName, std::vector<std::string> & logData, int & numPoints);
};

/*
 * Iterate through the header in the LAS file and
 * put the log names in a vector
 */
inline bool WellLogsIO::readLASHeader(std::string fileName, std::vector<std::string> & logNames)
{
    // Open the File
    std::ifstream in(fileName.c_str());

    // Check if object is valid
    if(!in)
    {
        std::cerr << "Cannot open the File : "<<fileName<<std::endl;
        return false;
    }

	// Reset the logNames
	logNames.clear();

	// Read header
    std::string str;
	int nBlock;
    while (std::getline(in, str))
    {
		// Determine the blocks
		if(str.find("~Version") != std::string::npos)
			nBlock = 1;	
		if(str.find("~Well") != std::string::npos)
			nBlock = 2;
		if(str.find("~Curve") != std::string::npos)
			nBlock = 3;	
		if(str.find("~Param") != std::string::npos)
			nBlock = 4;
		if(str.find("~A") != std::string::npos)
			break;

        // Search for line containing the log names and save them in a vector
        if((nBlock == 3) & (str.size() > 0) & (str.find("~") == std::string::npos) & (str.find("#") == std::string::npos))
		{	
			std::string logName = str.substr(0, str.find("."));
			boost::trim(logName);

            logNames.push_back(logName);
		}	
    } 

    // Close the LAS file
    in.close();

	return true;
}

inline bool WellLogsIO::readLASData(std::string fileName, std::vector<std::string> & logData, int & numPoints)
{
    // Open the File
    std::ifstream in(fileName.c_str());

    // Check if object is valid
    if(!in)
    {
        std::cerr << "Cannot open the File : "<<fileName<<std::endl;
        return false;
    }

	// Reset logData
	logData.clear();

	// Read data
    std::string str;
	int nBlock;
	numPoints = 0; // count number of acquisition depth points
    
    while (std::getline(in, str))
    {
		if(str.find("~A") != std::string::npos)
			nBlock = 5;

		// Search for the data block then store the values in a vector
		if((nBlock == 5) & (str.size() > 0) & (str.find("~") == std::string::npos) & (str.find("#") == std::string::npos))
		{

			std::istringstream iss(str);
		    std::string token;
			
    		while(std::getline(iss, token, ' '))
				if(token.size()>0)
        			logData.push_back(token);

			numPoints++;
		}		
    } 

    // Close the file
    in.close();

	return true;
}

#endif // NEXTVISION_SRC_WIDGET_WGEOMECHANICS_WELLLOGSIO_H_
