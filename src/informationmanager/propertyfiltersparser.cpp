#include "propertyfiltersparser.h"

#include "iinformation.h"

#include <QDate>

PropertyFiltersParser::PropertyFiltersParser() {

}

PropertyFiltersParser::PropertyFiltersParser(const QString& filtersText, const QString& separator) {
	QStringList filters = filtersText.split(separator, Qt::SkipEmptyParts);

	for (int i=0; i<filters.size(); i++) {
		QString filterTxt = filters[i];

		QStringList filterKeys = filterTxt.split("=", Qt::KeepEmptyParts);
		bool valid = false;
		information::Property property;
		int filterStart = 1;
		if (filterKeys.size()>1) {
			property = stringToProperty(filterKeys[0], &valid);
			valid = true;
		} else if (filterKeys.size()==1) {
			property = information::Property::NAME;
			filterStart = 0;
			valid = true;
		}
		QVariantList values;
		if (valid) {
			for (int j=filterStart; j<filterKeys.size(); j++) {
				values.append(filterKeys[j]);
			}
		}
		valid = values.size()>0;
		if (valid) {
			auto it = m_propertyFilters.find(property);
			if (it!=m_propertyFilters.end()) {
				it->second.append(values);
			} else {
				m_propertyFilters[property] = values;
			}
		}
	}
}

PropertyFiltersParser::PropertyFiltersParser(const std::map<information::Property, QVariantList>& propertyFilters) :
		m_propertyFilters(propertyFilters) {

}

PropertyFiltersParser::PropertyFiltersParser(const PropertyFiltersParser& other) :
		m_propertyFilters(other.m_propertyFilters) {

}

PropertyFiltersParser::~PropertyFiltersParser() {

}

PropertyFiltersParser& PropertyFiltersParser::operator=(const PropertyFiltersParser& other) {
	m_propertyFilters = other.m_propertyFilters;
	return *this;
}

bool PropertyFiltersParser::isValid(const IInformation* information) {
	bool valid = true;

	auto it = m_propertyFilters.begin();
	while (valid && it!=m_propertyFilters.end()) {
		int i = 0;
		while (valid && i<it->second.size()) {
			valid = isInformationPropertyValid(information, it->first, it->second[i]);
			i++;
		}
		it++;
	}

	return valid;
}

QString PropertyFiltersParser::constructPropertiesParsingHelp(const std::list<information::Property>& properties) {
	QString msg;
	for (auto it=properties.begin(); it!=properties.end(); it++) {
		if (it!=properties.begin()) {
			msg += "\n";
		}

		information::Property property = *it;
		if (property == information::Property::AXIS_TYPE) {
			msg += "axis=depth or axis=time";
		} else if (property == information::Property::CREATION_DATE) {
			msg += "created=yyyy-MM-dd , created=HH:mm:ss or created=yyyy-MM-ddTHH:mm:ss";
		} else if (property == information::Property::DATA_TYPE) {
			msg += "dtype=int16 for example";
		} else if (property == information::Property::MODIFICATION_DATE) {
			msg += "modified=yyyy-MM-dd , created=HH:mm:ss or created=yyyy-MM-ddTHH:mm:ss";
		} else if (property == information::Property::NAME) {
			msg += "name=my_name or my_name without key";
		} else if (property == information::Property::OWNER) {
			msg += "owner=an_username";
		} else if (property == information::Property::STORAGE_TYPE) {
			msg += "storage=nextvision or storage=sismage";
		} else if (property == information::Property::VOLUME_TYPE) {
			msg += "volume=seismic , volume=rgt or volume=patch";
		} else if (property == information::Property::WELL_BORE) {
			msg += "wellbore=wellbore_name";
		} else if (property == information::Property::WELL_HEAD) {
			msg += "wellhead=wellhead_name";
		} else if (property == information::Property::WELL_KIND) {
			msg += "kind=log_kind";
		} else if (property == information::Property::WELL_LOG_NAME) {
			msg += "log=log_name";
		} else if (property == information::Property::WELL_PICK_NAME) {
			msg += "pick=pick_name";
		} else if (property == information::Property::WELL_TFP_NAME) {
			msg += "tfp=tfp_name";
		}
	}

	return msg;
}

QMetaType::Type PropertyFiltersParser::extractType(information::Property property) {
	QMetaType::Type type = QMetaType::UnknownType;
	switch (property) {
	case information::Property::AXIS_TYPE:
	case information::Property::NAME:
	case information::Property::STORAGE_TYPE:
	case information::Property::VOLUME_TYPE:
	case information::Property::WELL_BORE:
	case information::Property::WELL_HEAD:
		type = QMetaType::QString;
		break;
	case information::Property::CREATION_DATE:
	case information::Property::MODIFICATION_DATE:
		type = QMetaType::QVariantList;
		break;
	case information::Property::OWNER:
	case information::Property::WELL_KIND:
	case information::Property::WELL_LOG_NAME:
	case information::Property::WELL_PICK_NAME:
	case information::Property::WELL_TFP_NAME:
		type = QMetaType::QStringList;
		break;
	}

	return type;
}

bool PropertyFiltersParser::isCompatible(information::Property property, const QVariant& filter, const QVariant& value) {
	QMetaType::Type type = extractType(property);

	bool valid = false;
	if (type==QMetaType::QString) {
		valid = isStringCompatible(filter, value);
	} else if (type==QMetaType::QVariantList) {
		valid = isDateListCompatible(filter, value);
	} else if (type==QMetaType::QStringList) {
		valid = isStringListCompatible(filter, value);
	}
	return valid;
}

bool PropertyFiltersParser::isDateListCompatible(const QVariant& filter, const QVariant& values) {
	QString filterDate = filter.toString();
	QVariantList valueList = values.toList();
	bool valid = filter.canConvert<QString>() && values.canConvert<QVariantList>();
	if (valid) {
		valid = false;
		int i = 0;
		while (!valid && i<valueList.size()) {
			valid = valueList[i].canConvert<QDateTime>();
			if (valid) {
				QDateTime valueDate = valueList[i].toDateTime();
				QString dateStr = valueDate.toString("yyyy-MM-ddTHH:mm:ss");
				valid = dateStr.contains(filterDate, Qt::CaseInsensitive);
			}
			i++;
		}
	}
	return valid;
}

bool PropertyFiltersParser::isInformationPropertyValid(const IInformation* information,
		information::Property property, const QVariant& filter) {
	bool valid = information->hasProperty(property);

	if (valid) {
		valid = information->isCompatible(property, filter);
	}

	return valid;
}

bool PropertyFiltersParser::isStringCompatible(const QVariant& filter, const QVariant& value) {
	QString filterStr = filter.toString();
	QString valueStr = value.toString();
	bool valid = valueStr.contains(filterStr, Qt::CaseInsensitive);
	return valid;
}

bool PropertyFiltersParser::isStringListCompatible(const QVariant& filter, const QVariant& value) {
	QString filterStr = filter.toString();
	QStringList valueStr = value.toStringList();
	bool valid = false;
	int i = 0;
	while (!valid && i<valueStr.size()) {
		valid = valueStr[i].contains(filterStr, Qt::CaseInsensitive);
		i++;
	}
	return valid;
}

information::Property PropertyFiltersParser::stringToProperty(const QString& _txt, bool* ok) {
	bool valid = false;
	information::Property property;
	QString txt = _txt.toLower();

	if (txt.compare("axis")==0) {
		valid = true;
		property = information::Property::AXIS_TYPE;
	} else if (txt.compare("created")==0) {
		valid = true;
		property = information::Property::CREATION_DATE;
	} else if (txt.compare("dtype")==0) {
		valid = true;
		property = information::Property::DATA_TYPE;
	} else if (txt.compare("modified")==0) {
		valid = true;
		property = information::Property::MODIFICATION_DATE;
	} else if (txt.compare("name")==0) {
		valid = true;
		property = information::Property::NAME;
	} else if (txt.compare("owner")==0) {
		valid = true;
		property = information::Property::OWNER;
	} else if (txt.compare("storage")==0) {
		valid = true;
		property = information::Property::STORAGE_TYPE;
	} else if (txt.compare("volume")==0) {
		valid = true;
		property = information::Property::VOLUME_TYPE;
	} else if (txt.compare("wellbore")==0) {
		valid = true;
		property = information::Property::WELL_BORE;
	} else if (txt.compare("wellhead")==0) {
		valid = true;
		property = information::Property::WELL_HEAD;
	} else if (txt.compare("kind")==0) {
		valid = true;
		property = information::Property::WELL_KIND;
	} else if (txt.compare("log")==0) {
		valid = true;
		property = information::Property::WELL_LOG_NAME;
	}  else if (txt.compare("pick")==0) {
		valid = true;
		property = information::Property::WELL_PICK_NAME;
	} else if (txt.compare("tfp")==0) {
		valid = true;
		property = information::Property::WELL_TFP_NAME;
	}

	if (ok!=nullptr) {
		*ok = valid;
	}
	return property;
}

QString PropertyFiltersParser::propertyToString(const information::Property& property) {
	QString txt;

	switch (property) {
	case information::Property::AXIS_TYPE:
		txt = "axis";
		break;
	case information::Property::CREATION_DATE:
		txt = "created";
		break;
	case information::Property::DATA_TYPE:
		txt = "dtype";
		break;
	case information::Property::MODIFICATION_DATE:
			txt = "modified";
			break;
	case information::Property::NAME:
		txt = "name";
		break;
	case information::Property::OWNER:
		txt = "owner";
		break;
	case information::Property::STORAGE_TYPE:
		txt = "storage";
		break;
	case information::Property::VOLUME_TYPE:
		txt = "volume";
		break;
	case information::Property::WELL_BORE:
		txt = "wellbore";
		break;
	case information::Property::WELL_HEAD:
		txt = "wellhead";
		break;
	case information::Property::WELL_KIND:
		txt = "kind";
		break;
	case information::Property::WELL_LOG_NAME:
		txt = "log";
		break;
	case information::Property::WELL_PICK_NAME:
		txt = "pick";
		break;
	case information::Property::WELL_TFP_NAME:
		txt = "tfp";
		break;
	}

	return txt;
}

QString PropertyFiltersParser::storageToString(information::StorageType storageType) {
	QString txt;

	switch (storageType) {
	case information::StorageType::NEXTVISION:
		txt = "NextVision";
		break;
	case information::StorageType::SISMAGE:
		txt = "Sismage";
		break;
	}

	return txt;
}

QVariant PropertyFiltersParser::toVariant(const QList<QDateTime>& dates) {
	QVariantList variantList;

	for (auto it = dates.begin(); it!=dates.end(); it++) {
		variantList.append(*it);
	}

	return variantList;
}

QDateTime PropertyPredicate::getDateTime(const QVariant& value) {
	QDateTime outDate;
	if (value.canConvert<QDateTime>()) {
		outDate = value.toDateTime();
	} else if (value.canConvert<QVariantList>()) {
		QVariantList values = value.toList();
		for (int i=0; i<values.size(); i++) {
			QDateTime date = getDateTime(values[i]);

			// take the newest date of all the dates
			if (!outDate.isValid() || (date.isValid() && date>outDate)) {
				outDate = date;
			}
		}
	}
	return outDate;
}

std::function<bool(const IInformation*, const IInformation*)> InformationPredicate::createComparator(
		information::Property property) {
	std::function<bool(const QVariant&, const QVariant&)> variantPredicate = PropertyPredicate::getPredicate(property);

	std::function<bool(const IInformation*, const IInformation*)> fn =
			[property, variantPredicate](const IInformation* first, const IInformation* second) {
		QVariant firstVariant;
		if (first && first->hasProperty(property)) {
			firstVariant = first->property(property);
		}
		QVariant secondVariant;
		if (second && second->hasProperty(property)) {
			secondVariant = second->property(property);
		}
		return variantPredicate(firstVariant, secondVariant);
	};

	return fn;
}

std::function<bool(const QVariant&, const QVariant&)> PropertyPredicate::getPredicate(information::Property property) {
	QMetaType::Type type = PropertyFiltersParser::extractType(property);

	std::function<bool(const QVariant&, const QVariant&)> func;
	if (type==QMetaType::QVariantList) {
		func = dateComparator;
	} else if (type==QMetaType::QStringList) {
		func = stringListComparator;
	} else {
		// string case
		func = stringComparator;
	}
	return func;
}

bool PropertyPredicate::dateComparator(const QVariant& first, const QVariant& second) {
	QDateTime firstDate = getDateTime(first);
	QDateTime secondDate = getDateTime(second);
	bool firstValid = firstDate.isValid()>0;
	bool secondValid = secondDate.isValid()>0;

	bool compareRes;
	if (firstValid && secondValid) {
		compareRes = firstDate<secondDate;
	} else if (firstValid && !secondValid) {
		// valid date > invalid
		compareRes = false;
	} else {
		// invalid < valid date
		compareRes = true;
	}
	return compareRes;
}

bool PropertyPredicate::stringComparator(const QVariant& first, const QVariant& second) {
	QString firstStr = first.toString();
	QString secondStr = second.toString();
	return firstStr<secondStr;
}

bool PropertyPredicate::stringListComparator(const QVariant& first, const QVariant& second) {
	QStringList firstStrList = first.toStringList();
	QStringList secondStrList = second.toStringList();
	bool firstStrListValid = firstStrList.size()>0;
	bool secondStrListValid = secondStrList.size()>0;

	bool compareRes;
	if (firstStrListValid && secondStrListValid) {
		compareRes = firstStrList[0]<secondStrList[0];
	} else if (firstStrListValid && !secondStrListValid) {
		// valid str > invalid
		compareRes = false;
	} else {
		// invalid < valid str
		compareRes = true;
	}
	return compareRes;
}
