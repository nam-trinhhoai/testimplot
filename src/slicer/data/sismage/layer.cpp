/*
 * Layering.cpp
 *
 *  Created on: Apr 2, 2020
 *      Author: l0222891
 */

#include "layer.h"

#include <fstream>
#include <iostream>
#include <QString>

//#include "RawUtil.h"
#include "layerproperty.h"
#include "utils/stringutil.h"

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

Layer::Layer() {
}

Layer::~Layer() {
	for ( int i = 0; i < m_layerProperties.size(); i++ ) {
		delete ( m_layerProperties.at(i));
	}
}

bool Layer::setLayerDirectory(const std::string& dirName) {
	m_dirName = dirName;

	for( const auto & entry : fs::directory_iterator(dirName)) {
		std::cout << entry.path() << std::endl;

		if ( endsWith(entry.path().c_str(), "desc") == 1) {
			m_descFile = entry.path().c_str();
			readLayerDesc(entry.path().c_str());
		}
		else {
			// Property directory
			LayerProperty* prop = new LayerProperty( entry.path().c_str() );
			m_layerProperties.push_back(prop);
		}

	}
	// Le desc est OK on passe les dims aux LayerProperties

	for ( int i = 0; i < m_layerProperties.size(); i++ ) {
		m_layerProperties.at(i)->setDims(m_nbTraces, m_nbProfiles );
	}

	return true;
}

bool  Layer::readLayerDesc(const std::string& filename) {

	std::string line;
	std::ifstream myfile( filename.c_str() );
	if (myfile.is_open()) {
		while ( std::getline (myfile,line) ) {
			std::cout << " LAYER DESC --> " << line << std::endl;
			int len = line.length();
			const size_t b = line.find_last_of("=");
			if ( b >= len)
				continue;
			const std::string nameStr = line.substr(0, b);
			const std::string valueStr = (b+1 <= len-1) ? line.substr(b+1, len-1) : "";
			std::cout << "         Name: " << nameStr << " Value= " << valueStr << std::endl;
			if ( nameStr.compare("label") == 0) {
				m_label = valueStr.c_str();
			}
			else if ( nameStr.compare("extent.trace0") == 0) {
				m_trace0 = (int) atoi(valueStr.c_str());
			}
			else if ( nameStr.compare("extent.profile0") == 0) {
				m_profile0 = (int) atoi(valueStr.c_str());
			}
			else if ( nameStr.compare("extent.nbTraces") == 0) {
				m_nbTraces = (int) atoi(valueStr.c_str());
			}
			else if ( nameStr.compare("extent.nbProfiles") == 0) {
				m_nbProfiles = (int) atoi(valueStr.c_str());
			}
			else if ( nameStr.compare("kind") == 0) {
				m_kind = valueStr.c_str();
			}
		}
		myfile.close();
	}
	else {
		return false;
		//throw io::CubeIOException(std::string("Unable to open file ") + filename.c_str());
	}
	return true;
}

bool Layer::writeProperty(float *tab, std::string& propName) {

	bool found = false;
	for ( int i = 0; i < m_layerProperties.size(); i++ ) {
		if ( propName.compare(m_layerProperties.at(i)->getName()) == 0) {

			if ( m_layerProperties.at(i)->writeProperty(tab, propName) ) {
				std::cout << " Write property : " << propName << std::endl;
				found = true;
			}
			else {
				std::cout << " ***PROBLEM TO Write property : " << propName << std::endl;
			}
			break;
		}
	}
	if ( ! found )
		std::cout << " ***PROBLEM TO Write property : " << propName << std::endl;
	return found;
}

//void Layer::headerFromV2ToV1() {
//	if (m_storageVersion == 1 )
//		return;
//
///*
//#Thu Oct 15 14:28:19 CEST 2015
//name=Isochron bottom
//extent.j0=0
//kind=Time
//extent.nj=782
//extent.ni=484
//extent.i0=17
//origin.datasets=\t
//nullValue=-9999.0
//origin.geometries=\t
//
// */
//	std::fstream file;
//    file.open(m_dirName.c_str(), std::ios_base::out);
//
//    file << "#Thu Oct 15 14:28:19 CEST 2015" << std::endl;
//    file << "name=" << m_label << std::endl;
//    file << "kind=" << m_kind << std::endl;
//    file << "extent.i0=" << m_profile0 << std::endl;
//    file << "extent.j0=" << m_trace0 << std::endl;
//    file << "extent.ni=" << m_nbProfiles << std::endl;
//    file << "extent.nj=" << m_nbTraces << std::endl;
//    file << "nullValue=" << m_originGeometries << std::endl;
//    file << "origin.geometries=" << m_originGeometries << std::endl;
//
//    file.close();
//
//}

