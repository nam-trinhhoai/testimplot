#ifndef SRC_INFORMATIONMANAGER_PROPERTYFILTERSPARSER_H
#define SRC_INFORMATIONMANAGER_PROPERTYFILTERSPARSER_H

#include "informationutils.h"

#include <QDateTime>
#include <QList>
#include <QMetaType>
#include <QString>
#include <QVariant>

#include <functional>
#include <map>

class IInformation;

class PropertyFiltersParser {
public:
	PropertyFiltersParser();
	PropertyFiltersParser(const QString& filtersText, const QString& separator);
	PropertyFiltersParser(const std::map<information::Property, QVariantList>& propertyFilters);
	PropertyFiltersParser(const PropertyFiltersParser& other);
	~PropertyFiltersParser();

	PropertyFiltersParser& operator=(const PropertyFiltersParser& other);

	bool isValid(const IInformation* information);

	static QString constructPropertiesParsingHelp(const std::list<information::Property>& properties);
	static QMetaType::Type extractType(information::Property property);
	static bool isCompatible(information::Property property, const QVariant& filter, const QVariant& value);
	static bool isDateListCompatible(const QVariant& filter, const QVariant& value);
	static bool isInformationPropertyValid(const IInformation* information, information::Property property, const QVariant& filter);
	static bool isStringCompatible(const QVariant& filter, const QVariant& value);
	static bool isStringListCompatible(const QVariant& filter, const QVariant& value);
	static QString propertyToString(const information::Property& property);
	static QString storageToString(information::StorageType storageType);
	static information::Property stringToProperty(const QString& txt, bool* ok=nullptr);

	static QVariant toVariant(const QList<QDateTime>& dates);

private:
	std::map<information::Property, QVariantList> m_propertyFilters;
};

class InformationPredicate {
public:
	static std::function<bool(const IInformation*, const IInformation*)> createComparator(information::Property property);
};

class PropertyPredicate {
public:
	static QDateTime getDateTime(const QVariant& value);
	static std::function<bool(const QVariant&, const QVariant&)> getPredicate(information::Property property);
	static bool dateComparator(const QVariant& first, const QVariant& second);
	static bool stringComparator(const QVariant& first, const QVariant& second);
	static bool stringListComparator(const QVariant& first, const QVariant& second);
};

#endif // SRC_INFORMATIONMANAGER_PROPERTYFILTERSPARSER_H
