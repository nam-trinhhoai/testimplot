/*
 * Layerings.h
 *
 *  Created on: Apr 2, 2020
 *      Author: l0222891
 */

#ifndef TARUMAPP_SRC_DATA_LAYERINGS_H_
#define TARUMAPP_SRC_DATA_LAYERINGS_H_

#include "layering.h"

#include <string>
#include <vector>


class Layerings {
public:
	Layerings(const std::string& dirName, const std::string& layeringKind);
	virtual ~Layerings();

	std::vector<std::string> getNames();
	Layering* getLayering(std::string name);

private:

	std::string m_dirName;
	std::string m_layeringKind;

	// For more intensive use, it could be replaced by map
	std::vector<Layering*> m_layerings;
};

#endif /* TARUMAPP_SRC_DATA_LAYERINGS_H_ */
