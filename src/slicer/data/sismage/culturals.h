/*
 * Culturals.h
 *
 *  Created on: Apr 2, 2020
 *      Author: l0222891
 */

#ifndef TARUMAPP_SRC_DATA_CULTURALS_H_
#define TARUMAPP_SRC_DATA_CULTURALS_H_

#include "cultural.h"

#include <string>
#include <vector>

/**
 * Get Sismage Cultural for category "NextVision"
 */
class Culturals {
public:
	Culturals(const std::string& culturalsDirPath);
	virtual ~Culturals();

	std::vector<std::string> getNames(int dimW, int dimH);
	Cultural* getCultural(std::string name);

	const std::vector<Cultural*>& getCulturals() const;

private:

	std::string m_dirName;
	std::string m_nextVisionCategory;

	// For more intensive use, it could be replaced by map
	std::vector<Cultural*> m_culturals;
};

#endif /* TARUMAPP_SRC_DATA_CULTURALS_H_ */
