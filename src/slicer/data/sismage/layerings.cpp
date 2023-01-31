/*
 * Layerings.cpp
 *
 *  Created on: Apr 2, 2020
 *      Author: l0222891
 *
 *   Warning we are only adressing the mono survey3D problems
 *
 */

#include "layerings.h"

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

#include "layer.h"
#include "utils/ioutil.h"
#include "utils/stringutil.h"

namespace fs = boost::filesystem;


/**
 * Scan Layerings from LAYER DIRECTORY in Survey3D and select Layering of kind "layeringKind"
 *
 * dirName is the directory of Layerings inside one single survey3D
 */
Layerings::Layerings(const std::string& dirName, const std::string& layeringKind) :
		m_dirName( dirName), m_layeringKind( layeringKind ) {

	// dirName is in Layers directory of Survey3D
	// We read desc in DATA/LAYERS

	fs::path dp(dirName);

	struct stat buffer;
	if ( access ( dirName.c_str(), F_OK) == -1)
		return;

	for( const auto & entry : fs::directory_iterator(dp)) {
		std::cout << entry.path() << std::endl;
		Layering* layering = new Layering(entry.path().c_str());
		const std::string layeringName = dp.filename().c_str();
		if ( layeringKind.compare( layering->getType() ) == 0) {
			m_layerings.push_back(layering);
		}
	}
}

Layerings::~Layerings() {
	//TODO delete pour le vecteur
}

std::vector<std::string> Layerings::getNames() {
	std::vector<std::string> names;

	for ( int i = 0; i < m_layerings.size(); i++ ) {
		names.push_back(m_layerings.at(i)->getName());
	}
	return names;
}

Layering* Layerings::getLayering(std::string name) {
	for ( int i = 0; i < m_layerings.size(); i++ ) {
		if (name.compare( m_layerings.at(i)->getName() ) == 0)
			return m_layerings.at(i);
	}
	return (Layering*) nullptr;
}
