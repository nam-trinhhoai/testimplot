/*
 * CulturalCategory.h
 *
 *  Created on: Apr 2, 2020
 *      Author: l0222891
 */

#ifndef TARUMAPP_SRC_DATA_CulturalCategoryCATEGORY_H_
#define TARUMAPP_SRC_DATA_CulturalCategoryCATEGORY_H_

#include <string>

#include <QVector2D>



class CulturalCategory {
public:
	CulturalCategory(const std::string& culturalDirPath, const std::string& categoryName);
	virtual ~CulturalCategory() {};

	const std::string& getName() const {
		return m_name;
	}

	const std::string& getSismageId() const {
		return m_sismageId;
	}

private:


	std::string m_sismageId; // id of category
	std::string m_name; // name of category

	std::string m_categoryFilePath;


};


#endif /* TARUMAPP_SRC_DATA_CulturalCategoryCATEGORY_H_ */
