/*
 * CulturalCategory.cpp
 *
 *  Created on: Apr 2, 2020
 *      Author: Georges
 *
 *   Warning we are only adressing the mono survey3D problems
 *
 */

#include "culturalcategory.h"

#include <iostream>
#include <fstream>
#include <png.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>

#include <boost/algorithm/string.hpp>


#include "smtopo3ddesc.h"
#include <QByteArray>
#include <QRect>
#include <QDebug>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <boost/filesystem.hpp>

#include "sismagedbmanager.h"
#include "smsurvey3D.h"
#include "utils/ioutil.h"
#include "utils/stringutil.h"
#include "LayerSlice.h"
#include "rgblayerslice.h"
#include "fixedrgblayersfromdatasetandcube.h"
#include "cudargbimage.h"
#include "cudaimagepaletteholder.h"

namespace fs = boost::filesystem;

/**
 *Get or create for a specific catagory
 */
CulturalCategory::CulturalCategory(const std::string& culturalDirPath, const std::string& categoryName) :
		m_name( categoryName) {

	fs::path p(culturalDirPath);
	p /= "CATEGORIES";
	// Get or create CATEGIRIES directory
	if (!fs::exists(p) )
		fs:create_directory(p);

	//Find categoryName
	bool found = false;
	for( const auto & entry : fs::directory_iterator(p)) {
		std::string line;
		std::ifstream myfile( entry.path().c_str() );
		if (myfile.is_open()) {
			std::getline (myfile,line);
			std::string name = line.c_str();
			if (name.compare("NextVision") == 0) {
				// Category "NextVision"
				found = true;
				m_categoryFilePath = entry.path().c_str();
				m_sismageId = entry.path().stem().c_str();
				std::cout << "CATEGRIES NAME= " << m_sismageId << " File Path= " <<
						m_categoryFilePath	<< std::endl;
			}

			myfile.close();
		}
	}
	if (!found) {
		//WARNING SPECIFIQUE POUR LA CATEGORIE NextVision
		m_sismageId = "NxVision";
		p.append(m_sismageId);
		p.append(".cat");
		std::ofstream myFile( p.c_str() );
		if (myFile.is_open()) {
			myFile << m_name << std::endl;

			myFile.close();
		}
	}
}
