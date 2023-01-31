/*
 * LayerPropertying.cpp
 *
 *  Created on: Apr 2, 2020
 *      Author: l0222891
 */

#include "layerproperty.h"

#include <fstream>
#include <iostream>
#include <QString>

//#include "RawUtil.h"
#include "utils/stringutil.h"
#include "ioutil.h"

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;


LayerProperty::LayerProperty(const std::string& dirName) {
	m_dirName = dirName;

	for( const auto & entry : fs::directory_iterator(dirName)) {
		std::cout << entry.path() << std::endl;

		if ( endsWith(entry.path().c_str(), "desc") == 1) {
			m_descFile = entry.path().c_str();
			readLayerPropertyDesc(entry.path().c_str());
		}
		else if ( endsWith(entry.path().c_str(), "values") == 1) {
			m_dataFile = entry.path().c_str();
		}
	}
}

LayerProperty::~LayerProperty() {
	// TODO Auto-generated destructor stub
}

bool  LayerProperty::readLayerPropertyDesc(const std::string& filename) {

	std::string line;
	std::ifstream myfile( filename.c_str() );
	if (myfile.is_open()) {
		while ( std::getline (myfile,line) ) {
			int len = line.length();
			const size_t b = line.find_last_of("=");
			if ( b >= len)
				continue;
			const std::string nameStr = line.substr(0, b);
			const std::string valueStr = (b+1 <= len-1) ? line.substr(b+1, len-1) : "";
			if ( nameStr.compare("name") == 0) {
				m_name = valueStr.c_str();
			}
			else if ( nameStr.compare("extent.j0") == 0) {
				m_j0 = (int) atoi(valueStr.c_str());
			}
			else if ( nameStr.compare("extent.i0") == 0) {
				m_i0 = (int) atoi(valueStr.c_str());
			}
			else if ( nameStr.compare("extent.ni") == 0) {
				m_ni = (int) atoi(valueStr.c_str());
			}
			else if ( nameStr.compare("extent.nj") == 0) {
				m_nj = (int) atoi(valueStr.c_str());
			}
			else if ( nameStr.compare("storageVersion") == 0) {
				m_storageVersion = 2;
			}
			else if ( nameStr.compare("nullValue") == 0) {
				m_nullValue = valueStr.c_str();
			}
			else if ( nameStr.compare("kind") == 0) {
				m_kind = valueStr.c_str();
			}
			else if ( nameStr.compare("origin.geometries") == 0) {
				m_originGeometries = valueStr.c_str();
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

/**
 * Les nouveaux desc version2 ne contiennent pas de dimensions
 */
void LayerProperty::setDims( int dimW, int dimH ) {
	m_nj = dimW;
	m_ni = dimH;
}

/**
 *
 */
bool LayerProperty::writeProperty(float *tab, const std::string& propName) {

	headerFromV2ToV1( );

	std::ofstream outfile(m_dataFile.c_str(), std::ios::out | std::ios::binary);

	if ( outfile.fail()) {
		std::cout << "Fail in opening for writing file: " << m_dataFile.c_str() << std::endl;
		return false;
	}
	int header[4];
	header[0] = 0;
	header[1] = 0;
	header[2] = m_ni; // inlines
	header[3] = m_nj; // xlines
	switch_list_endianness_inplace(&header, 4, 4);
	switch_list_endianness_inplace(tab, 4, m_ni * m_nj);

	outfile.write(reinterpret_cast<const char *>(header), 4 * sizeof(int));
	outfile.write(reinterpret_cast<const char *>(tab), m_ni * ((size_t)m_nj) *sizeof(float));

	outfile.close();
	return true;
}

void LayerProperty::headerFromV2ToV1() {
	if (m_storageVersion == 1 )
		return;

/*
#Thu Oct 15 14:28:19 CEST 2015
name=Isochron bottom
extent.j0=0
kind=Time
extent.nj=782
extent.ni=484
extent.i0=17
origin.datasets=\t
nullValue=-9999.0
origin.geometries=\t

 */
    std::ofstream file;
    file.open (m_descFile.c_str());

    file << "#Thu Oct 15 14:28:19 CEST 2015" << std::endl;
    file << "name=" << m_name << std::endl;
    file << "kind=" << m_kind << std::endl;
    file << "extent.i0=" << m_i0 << std::endl;
    file << "extent.j0=" << m_j0 << std::endl;
    file << "extent.ni=" << m_ni << std::endl;
    file << "extent.nj=" << m_nj << std::endl;
    file << "nullValue=" << m_nullValue << std::endl;
    file << "origin.geometries=" << m_originGeometries << std::endl;

    file.close();

}

