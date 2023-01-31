#ifndef SRC_INFORMATIONMANAGER_INFORMATIONPROPERTYWIDGETFACTORY_H
#define SRC_INFORMATIONMANAGER_INFORMATIONPROPERTYWIDGETFACTORY_H

#include "informationutils.h"

#include <functional>
#include <unordered_map>

class IInformationPropertyWidget;

class QWidget;


class InformationPropertyWidgetFactory {
public:
	InformationPropertyWidgetFactory();
	~InformationPropertyWidgetFactory();

	IInformationPropertyWidget* build(information::Property property, QWidget* parent=nullptr) const;
	void registerBuilder(information::Property property, std::function<IInformationPropertyWidget*(QWidget*)> builder);

private:
	std::unordered_map<information::Property, std::function<IInformationPropertyWidget*(QWidget*)>> m_builders;
};

#endif // SRC_INFORMATIONMANAGER_INFORMATIONPROPERTYWIDGETFACTORY_H
