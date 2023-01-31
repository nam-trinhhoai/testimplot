/*
 * Culturals.cpp
 *
 *  Created on: Apr 2, 2020
 *      Author: l0222891
 *
 *   Warning we are only adressing the mono survey3D problems
 *
 */

#include "culturals.h"

#include <iostream>
#include <fstream>

//#include "SampleTypeBinder.h"
#include <QByteArray>
#include <QRect>
#include <QDebug>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <boost/filesystem.hpp>

#include "utils/ioutil.h"
#include "utils/stringutil.h"

namespace fs = boost::filesystem;


std::string CDAT(".cdat");
std::string GEOREF_IMAGE("GEOREF IMAGE");
std::string NEXT_VISION_STR("NextVision");
/**
 * Scan Culturals from CULTURAL DIRECTORY in project directory and select Layering of kind "layeringKind"
 *
 * dirName is the directory of CULTURAL inside Project directory / DATA
 */
Culturals::Culturals(const std::string& culturalsDirPath) :
		m_dirName( culturalsDirPath) {

	// dirName is in CULTURAL directory of Survey3D


	struct stat buffer;
	if ( access ( culturalsDirPath.c_str(), F_OK) == -1)
		return;

	// CATEGORIES
	fs::path categoriesPath(culturalsDirPath);
	categoriesPath /= "CATEGORIES";
	std::cout << "CATEGRIES DIRECTORY= " << categoriesPath << std::endl;

	// Get or create CATEGIRIES directory
	if (!boost::filesystem::exists(categoriesPath) )
		boost::filesystem::create_directory(categoriesPath);

	for( const auto & entry : fs::directory_iterator(categoriesPath)) {
		std::string line;
		std::ifstream myfile( entry.path().c_str() );
		if (myfile.is_open()) {
			std::getline (myfile,line);
			std::string name = line.c_str();
			if (name.compare(NEXT_VISION_STR) == 0) {
				// Category "NextVision"
				m_nextVisionCategory = entry.path().stem().c_str();
				std::cout << "CATEGRIES NAME= " << m_nextVisionCategory << std::endl;
			}

			myfile.close();
		}
	}

	// CULTURALS
	fs::path dp(culturalsDirPath);
	for( const auto & entry : fs::directory_iterator(dp)) {
		fs::path dp(entry);

		if (  CDAT.compare(dp.extension().c_str()) == 0 ) { // fichier .cdat
			Cultural* cultural = new Cultural(entry.path().c_str());

			//const std::string name = dp.stem().c_str();
//			std::cout << "Compare " << cultural->getCategory() << " / " <<
//					m_nextVisionCategory << std::endl;
			if ((m_nextVisionCategory.empty() ||
					cultural->getCategory().compare(m_nextVisionCategory) == 0) &&
					GEOREF_IMAGE.compare( cultural->getType() ) == 0) {
				m_culturals.push_back(cultural);
			}
			else {
				delete cultural;
			}
		}
	}
}

Culturals::~Culturals() {
	//TODO delete pour le vecteur
}

std::vector<std::string> Culturals::getNames(int dimW, int dimH) {
	std::vector<std::string> names;

	for ( int i = 0; i < m_culturals.size(); i++ ) {
		int w = m_culturals.at(i)->getDimW();
		int h = m_culturals.at(i)->getDimH();
		names.push_back(m_culturals.at(i)->getName());
	}
	return names;
}

Cultural* Culturals::getCultural(std::string name) {
	for ( int i = 0; i < m_culturals.size(); i++ ) {
		if (name.compare( m_culturals.at(i)->getName() ) == 0)
			return m_culturals.at(i);
	}
	return (Cultural*) nullptr;
}

const std::vector<Cultural*>& Culturals::getCulturals() const {
	return m_culturals;
}
