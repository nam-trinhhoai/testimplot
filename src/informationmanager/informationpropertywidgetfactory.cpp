#include "iinformationpropertywidget.h"
#include "informationpropertywidgetfactory.h"

#include <QLineEdit>

InformationPropertyWidgetFactory::InformationPropertyWidgetFactory() {

}

InformationPropertyWidgetFactory::~InformationPropertyWidgetFactory() {

}

IInformationPropertyWidget* InformationPropertyWidgetFactory::build(information::Property property, QWidget* parent) const {
	IInformationPropertyWidget* holder = nullptr;
	auto it = m_builders.find(property);
	if (it!=m_builders.end()) {
		holder = it->second(parent);
	}
	return holder;
}

void InformationPropertyWidgetFactory::registerBuilder(information::Property property, std::function<IInformationPropertyWidget*(QWidget*)> builder) {
	m_builders[property] = builder;
}
